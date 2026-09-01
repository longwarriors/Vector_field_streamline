"""Build serializable 2D scenes from the reusable scientific core."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from vectorviz.core import Domain, SphericalExclusion, VectorField
from vectorviz.fields import (
    CircularLoopField,
    MagneticDipoleField,
    PointChargeField,
    UniformField,
)
from vectorviz.tracing import (
    FieldLineTracer,
    TerminationReason,
    TraceDirection,
    TraceOptions,
)

from .schemas import (
    DomainPayload,
    LinePayload,
    MetadataPayload,
    ScalarPayload,
    SceneRequest,
    SceneResponse,
    SourceInput,
    SourcePayload,
)

DOMAIN = Domain(lower=(-3.0, -3.0), upper=(3.0, 3.0))
SOURCE_RADIUS = 0.16
CURRENT_LOOP_RADIUS = 1.0
CURRENT_LOOP_CURRENT = 1.0
CURRENT_LOOP_EXCLUSION_RADIUS = 0.16

DEFAULT_TRACE_OPTIONS = TraceOptions(
    max_arc_length=18.0,
    max_step=0.09,
    rtol=2.0e-6,
    atol=1.0e-8,
    null_threshold=1.0e-14,
    output_step=0.045,
    method="DOP853",
)

CURRENT_LOOP_TRACE_OPTIONS = TraceOptions(
    max_arc_length=18.0,
    max_step=0.09,
    rtol=2.0e-6,
    atol=1.0e-8,
    null_threshold=1.0e-14,
    output_step=0.045,
    method="DOP853",
    closure_tolerance=0.005,
    closure_min_arc_length=0.6,
    closure_tangent_cosine=0.99,
)


class _PlanarMagneticDipoleField(VectorField):
    """The invariant z=0 plane of dipoles whose moments also lie in that plane."""

    def __init__(self, field: MagneticDipoleField) -> None:
        if np.any(field.positions[:, 2] != 0.0) or np.any(field.moments[:, 2] != 0.0):
            raise ValueError("the z=0 plane is invariant only for in-plane sources")
        self._field = field

    @property
    def dimension(self) -> int:
        return 2

    def evaluate(self, points: object) -> NDArray[np.float64]:
        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim == 0 or coordinates.shape[-1] != 2:
            raise ValueError(f"points must have shape (..., 2); got {coordinates.shape}.")
        embedded = np.zeros((*coordinates.shape[:-1], 3), dtype=float)
        embedded[..., :2] = coordinates
        return self._field.evaluate(embedded)[..., :2]


class _PlanarCircularLoopField(VectorField):
    """The invariant z=0 meridional plane of a y-axis circular loop."""

    def __init__(self, field: CircularLoopField) -> None:
        expected_normal = np.array((0.0, 1.0, 0.0))
        if not np.array_equal(field.center, np.zeros(3)) or not np.array_equal(
            field.normal, expected_normal
        ):
            raise ValueError("the z=0 meridional adapter requires a centered +y-axis loop")
        self._field = field

    @property
    def dimension(self) -> int:
        return 2

    def evaluate(self, points: object) -> NDArray[np.float64]:
        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim == 0 or coordinates.shape[-1] != 2:
            raise ValueError(f"points must have shape (..., 2); got {coordinates.shape}.")
        embedded = np.zeros((*coordinates.shape[:-1], 3), dtype=float)
        embedded[..., :2] = coordinates
        return self._field.evaluate(embedded)[..., :2]


@dataclass(frozen=True, slots=True)
class _SceneModel:
    field: VectorField
    sources: tuple[SourcePayload, ...]
    exclusions: tuple[SphericalExclusion, ...]
    seeds: NDArray[np.float64]
    direction: TraceDirection
    trace_options: TraceOptions
    scalar_label: str
    scalar_unit: str
    title: str
    field_model: str
    projection_note: str
    seed_mode: str
    reflect_y_symmetric_domain_exits: bool = False


def _default_sources(preset: str) -> list[SourceInput]:
    if preset == "electric_dipole":
        return [
            SourceInput(x=-0.85, y=0.0, kind="positive", strength=1.0),
            SourceInput(x=0.85, y=0.0, kind="negative", strength=-1.0),
        ]
    if preset == "magnetic_dipole":
        return [SourceInput(x=0.0, y=0.0, kind="dipole", strength=1.0)]
    return []


def _source_payloads(sources: list[SourceInput]) -> tuple[SourcePayload, ...]:
    payloads: list[SourcePayload] = []
    for source in sources:
        if source.kind in {"positive", "negative"}:
            strength_unit = "nC"
        elif source.kind == "dipole":
            strength_unit = "A·m²"
        else:
            raise ValueError("uniform sources do not have a localized source strength")
        payloads.append(
            SourcePayload(**source.model_dump(), strength_unit=strength_unit)
        )
    return tuple(payloads)


def _allocate_seed_counts(strengths: NDArray[np.float64], total: int) -> NDArray[np.int64]:
    source_count = int(strengths.size)
    if source_count == 0:
        raise ValueError("at least one seeding source is required")
    if source_count > total:
        raise ValueError("seed budget must provide at least one seed per seeding source")

    counts = np.ones(source_count, dtype=np.int64)
    remaining = total - source_count
    if remaining == 0:
        return counts

    weights = np.abs(strengths)
    if not np.any(weights):
        weights = np.ones_like(weights)
    quotas = weights / np.sum(weights) * remaining
    extras = np.floor(quotas).astype(np.int64)
    counts += extras
    unassigned = remaining - int(np.sum(extras))
    if unassigned:
        fractions = quotas - extras
        order = np.argsort(-fractions, kind="stable")
        counts[order[:unassigned]] += 1
    return counts


def _require_seed_budget(
    preset: str,
    source_label: str,
    source_count: int,
    density: int,
) -> None:
    if source_count > density:
        raise ValueError(
            f"{preset} 有 {source_count} 个{source_label}参与播种，"
            f"density 至少为 {source_count}"
        )


def _circle_seeds(
    centers: NDArray[np.float64],
    strengths: NDArray[np.float64],
    total: int,
) -> NDArray[np.float64]:
    counts = _allocate_seed_counts(strengths, total)
    groups: list[NDArray[np.float64]] = []
    radius = SOURCE_RADIUS + 2.0e-3
    for center, count in zip(centers, counts, strict=True):
        angles = np.linspace(0.0, 2.0 * np.pi, int(count), endpoint=False)
        offsets = radius * np.column_stack((np.cos(angles), np.sin(angles)))
        groups.append(center + offsets)
    return np.concatenate(groups, axis=0)


def _magnetic_seeds(
    centers: NDArray[np.float64],
    strengths: NDArray[np.float64],
    total: int,
) -> NDArray[np.float64]:
    counts = _allocate_seed_counts(strengths, total)
    groups: list[NDArray[np.float64]] = []
    radius = SOURCE_RADIUS + 2.0e-3
    for center, strength, count in zip(centers, strengths, counts, strict=True):
        # Seed only the hemisphere where B points away from the excluded source.
        angles = np.linspace(-1.43, 1.43, int(count))
        axis_sign = 1.0 if strength >= 0 else -1.0
        offsets = radius * np.column_stack((np.sin(angles), axis_sign * np.cos(angles)))
        groups.append(center + offsets)
    return np.concatenate(groups, axis=0)


def _current_loop_seeds(total: int) -> NDArray[np.float64]:
    """Cover distinct loop-flux contours symmetrically in the meridional plane."""

    pair_count = total // 2
    radii = np.linspace(0.12, 0.82, pair_count)
    paired = np.zeros((2 * pair_count, 2), dtype=float)
    paired[0::2, 0] = -radii
    paired[1::2, 0] = radii
    if total % 2 == 0:
        return paired
    axis_seed = np.array(((0.0, float(DOMAIN.lower[1]) + 1.0e-4),))
    return np.concatenate((axis_seed, paired), axis=0)


def _build_model(request: SceneRequest) -> _SceneModel:
    if request.preset == "current_loop":
        loop = CircularLoopField(
            CURRENT_LOOP_CURRENT,
            CURRENT_LOOP_RADIUS,
            normal=(0.0, 1.0, 0.0),
        )
        wire_centers = np.array(
            ((-CURRENT_LOOP_RADIUS, 0.0), (CURRENT_LOOP_RADIUS, 0.0)),
            dtype=float,
        )
        markers = (
            SourcePayload(
                x=-CURRENT_LOOP_RADIUS,
                y=0.0,
                kind="wire_out",
                strength=CURRENT_LOOP_CURRENT,
                strength_unit="A",
            ),
            SourcePayload(
                x=CURRENT_LOOP_RADIUS,
                y=0.0,
                kind="wire_into",
                strength=CURRENT_LOOP_CURRENT,
                strength_unit="A",
            ),
        )
        return _SceneModel(
            field=_PlanarCircularLoopField(loop),
            sources=markers,
            exclusions=(
                SphericalExclusion(wire_centers, CURRENT_LOOP_EXCLUSION_RADIUS),
            ),
            seeds=_current_loop_seeds(request.density),
            direction=TraceDirection.FORWARD,
            trace_options=CURRENT_LOOP_TRACE_OPTIONS,
            scalar_label="|B|",
            scalar_unit="T",
            title="圆形电流线圈的磁力线",
            field_model="三维理想圆形电流线圈在 z=0 子午面上的限制",
            projection_note=(
                "z=0 子午面是该轴对称场的不变平面，所示曲线是真实三维磁力线。"
            ),
            seed_mode=(
                "在圆环两侧的环内赤道段镜像等距覆盖播种；奇数预算另含轴线。"
                "线密度不代表磁通或磁感应强度。"
            ),
            reflect_y_symmetric_domain_exits=True,
        )

    inputs = list(
        _default_sources(request.preset) if request.sources is None else request.sources
    )
    payloads = _source_payloads(inputs)

    if request.preset == "electric_dipole":
        active = [source for source in inputs if source.kind in {"positive", "negative"}]
        centers = np.array([[source.x, source.y] for source in active], dtype=float)
        signed_strengths = np.array([source.strength for source in active], dtype=float)
        if not np.any(signed_strengths > 0) or not np.any(signed_strengths < 0):
            raise ValueError("electric dipole needs at least one positive and one negative source")
        field = PointChargeField(signed_strengths * 1.0e-9, centers)
        positive = signed_strengths > 0
        _require_seed_budget(
            request.preset,
            "正电荷",
            int(np.count_nonzero(positive)),
            request.density,
        )
        seeds = _circle_seeds(centers[positive], signed_strengths[positive], request.density)
        return _SceneModel(
            field=field,
            sources=payloads,
            exclusions=(SphericalExclusion(centers, SOURCE_RADIUS),),
            seeds=seeds,
            direction=TraceDirection.FORWARD,
            trace_options=DEFAULT_TRACE_OPTIONS,
            scalar_label="|E|",
            scalar_unit="V/m",
            title="电偶极子的电场线",
            field_model="三维点电荷场在 z=0 对称平面上的限制",
            projection_note="该平面法向场分量为零，所示曲线是真实场线，不是投影流线。",
            seed_mode="从正电荷排除面的覆盖播种；线密度默认不代表场强。",
        )

    if request.preset == "magnetic_dipole":
        active = [source for source in inputs if source.kind == "dipole"]
        centers = np.array([[source.x, source.y] for source in active], dtype=float)
        strengths = np.array([source.strength for source in active], dtype=float)
        _require_seed_budget(
            request.preset,
            "磁偶极子",
            len(active),
            request.density,
        )
        moments = np.column_stack((np.zeros_like(strengths), strengths, np.zeros_like(strengths)))
        positions = np.column_stack((centers, np.zeros(len(centers))))
        field = _PlanarMagneticDipoleField(MagneticDipoleField(moments, positions))
        return _SceneModel(
            field=field,
            sources=payloads,
            exclusions=(SphericalExclusion(centers, SOURCE_RADIUS),),
            seeds=_magnetic_seeds(centers, strengths, request.density),
            direction=TraceDirection.FORWARD,
            trace_options=DEFAULT_TRACE_OPTIONS,
            scalar_label="|B|",
            scalar_unit="T",
            title="磁偶极子的磁力线",
            field_model="三维理想磁偶极场在 z=0 对称平面上的限制",
            projection_note="偶极矩位于切片内且法向场为零，所示曲线是真实磁力线。",
            seed_mode="从偶极子北半排除面覆盖播种；线密度默认不代表磁感应强度。",
        )

    field = UniformField((1.0, 0.28))
    y = np.linspace(DOMAIN.lower[1] + 0.12, DOMAIN.upper[1] - 0.12, request.density)
    x = np.full_like(y, DOMAIN.lower[0] + 1.0e-4)
    return _SceneModel(
        field=field,
        sources=payloads,
        exclusions=(),
        seeds=np.column_stack((x, y)),
        direction=TraceDirection.FORWARD,
        trace_options=DEFAULT_TRACE_OPTIONS,
        scalar_label="|E|",
        scalar_unit="V/m",
        title="匀强电场",
        field_model="二维常向量场",
        projection_note="这是原生二维向量场，曲线与电场方向处处相切。",
        seed_mode="从左边界等距覆盖播种；线密度不表示场强。",
    )


def _sample_scalar(model: _SceneModel, resolution: int) -> ScalarPayload:
    x = np.linspace(DOMAIN.lower[0], DOMAIN.upper[0], resolution)
    # Canvas rows run from top to bottom, so y is sampled in descending order.
    y = np.linspace(DOMAIN.upper[1], DOMAIN.lower[1], resolution)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = np.stack((xx, yy), axis=-1)
    values = np.linalg.norm(model.field.evaluate(points), axis=-1)
    mask = ~np.isfinite(values)
    for exclusion in model.exclusions:
        mask |= exclusion.margin(points) <= 0.0
    valid = values[~mask & (values > 0.0)]
    if valid.size:
        vmin, vmax = np.percentile(valid, [3.0, 98.5])
        vmin = float(max(vmin, np.finfo(float).tiny))
        vmax = float(max(vmax, vmin * (1.0 + 1.0e-12)))
        scale: Literal["linear", "log"] = "log" if vmax / vmin > 8.0 else "linear"
    else:
        vmin, vmax, scale = 0.0, 1.0, "linear"
    serial_values = np.where(mask, vmin, np.clip(values, vmin, vmax))
    return ScalarPayload(
        nx=resolution,
        ny=resolution,
        values=serial_values.ravel().tolist(),
        mask=mask.ravel().tolist(),
        scale=scale,
        label=model.scalar_label,
        unit=model.scalar_unit,
        vmin=vmin,
        vmax=vmax,
    )


def _trace_lines(model: _SceneModel) -> tuple[list[LinePayload], Counter[str]]:
    tracer = FieldLineTracer(
        model.field,
        domain=DOMAIN,
        options=model.trace_options,
        exclusions=model.exclusions,
    )
    lines: list[LinePayload] = []
    terminations: Counter[str] = Counter()
    for seed in model.seeds:
        result = tracer.trace(seed, direction=model.direction)
        branch = result.forward if model.direction is TraceDirection.FORWARD else result.backward
        if branch is None:
            continue
        terminations[branch.termination.value] += 1
        finite = np.all(np.isfinite(result.points), axis=1)
        points = result.points[finite]
        if (
            model.reflect_y_symmetric_domain_exits
            and branch.termination is TerminationReason.DOMAIN_EXIT
            and seed[1] == 0.0
        ):
            # Across the loop plane, (Bx, By)(x, -y) = (-Bx, By)(x, y).
            # Reversing the reflected forward branch therefore preserves the
            # displayed +B ordering while completing the missing half-line.
            lower_to_seed = points[:0:-1].copy()
            lower_to_seed[:, 1] *= -1.0
            points = np.concatenate((lower_to_seed, points), axis=0)
        if points.shape[0] < 2:
            continue
        lines.append(
            LinePayload(
                points=[(float(point[0]), float(point[1])) for point in points],
                direction=1 if model.direction is TraceDirection.FORWARD else -1,
                termination=branch.termination.value,
            )
        )
    return lines, terminations


def build_scene(request: SceneRequest) -> SceneResponse:
    """Compute one complete scene for the browser client."""

    model = _build_model(request)
    lines, terminations = _trace_lines(model)
    return SceneResponse(
        domain=DomainPayload(
            x=(float(DOMAIN.lower[0]), float(DOMAIN.upper[0])),
            y=(float(DOMAIN.lower[1]), float(DOMAIN.upper[1])),
            coordinate_system="cartesian",
            unit="m",
        ),
        scalar=_sample_scalar(model, request.resolution),
        lines=lines,
        sources=list(model.sources),
        metadata=MetadataPayload(
            title=model.title,
            projection_note=model.projection_note,
            field_model=model.field_model,
            seed_mode=model.seed_mode,
            termination_counts=dict(terminations),
        ),
    )


__all__ = ["build_scene"]
