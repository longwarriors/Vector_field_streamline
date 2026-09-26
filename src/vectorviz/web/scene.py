"""Build serializable 2D scenes from the reusable scientific core."""

from __future__ import annotations

import math
import threading
from collections import Counter, OrderedDict
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
    TraceBranch,
    TraceDirection,
    TraceOptions,
    TraceResult,
)

from .schemas import (
    ELECTRIC_PRESETS,
    DomainPayload,
    LinePayload,
    MetadataPayload,
    ScalarPayload,
    SceneRequest,
    SceneResponse,
    SeedMode,
    SourceInput,
    SourcePayload,
)
from .seeding import (
    TraceJob,
    current_loop_equal_flux_jobs,
    electric_source_jobs,
    halbach_rail_jobs,
    magnetic_source_jobs,
    single_dipole_equatorial_jobs,
)

DOMAIN = Domain(lower=(-3.0, -3.0), upper=(3.0, 3.0))
SOURCE_RADIUS = 0.16
SOURCE_SEED_CLEARANCE = 2.0e-3
SOURCE_SEED_RADIUS = SOURCE_RADIUS + SOURCE_SEED_CLEARANCE
MIN_SOURCE_SEPARATION = SOURCE_RADIUS + SOURCE_SEED_RADIUS
CURRENT_LOOP_RADIUS = 1.0
CURRENT_LOOP_CURRENT = 1.0
CURRENT_LOOP_EXCLUSION_RADIUS = 0.16
HALBACH_SOURCE_COUNT = 8
HALBACH_X_EXTENT = 2.1
ELECTRIC_ARRANGEMENT_RADIUS = 0.9
# Point-charge scenes are built from 1 nC charges about 1 m apart, so |E|
# is of order 10 V/m; a line is considered to have reached a null point
# once |E| falls seven orders of magnitude below that, about 1e-7 m from
# a linear null. This is a numerical cut-off in V/m, not a physical zero.
ELECTRIC_NULL_THRESHOLD = 1.0e-6

DEFAULT_TRACE_OPTIONS = TraceOptions(
    max_arc_length=18.0,
    max_step=0.09,
    rtol=2.0e-6,
    atol=1.0e-8,
    null_threshold=1.0e-14,
    output_step=0.045,
    method="DOP853",
)

ELECTRIC_TRACE_OPTIONS = TraceOptions(
    max_arc_length=18.0,
    max_step=0.09,
    rtol=2.0e-6,
    atol=1.0e-8,
    null_threshold=ELECTRIC_NULL_THRESHOLD,
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


# Scene title and field-model text for each point-charge preset; an edited
# arrangement other than the dipole no longer claims its named geometry.
ELECTRIC_TITLES: dict[str, tuple[str, str]] = {
    "electric_dipole": ("电偶极子的电场线", "三维点电荷场在 z=0 对称平面上的限制"),
    "electric_quadrupole": (
        "电四极子的电场线",
        "正方形顶点上正负交替的四个点电荷在 z=0 对称平面上的限制",
    ),
    "electric_hexagon": (
        "六个等量正电荷的电场线",
        "正六边形顶点上六个等量正点电荷在 z=0 对称平面上的限制",
    ),
    "electric_hexagon_alternating": (
        "三对交替正负电荷的电场线",
        "正六边形顶点上正负交替的六个点电荷在 z=0 对称平面上的限制",
    ),
    "edited": ("可编辑点电荷组的电场线", "可编辑的三维点电荷场在 z=0 对称平面上的限制"),
}


@dataclass(frozen=True, slots=True)
class _SceneModel:
    field: VectorField
    sources: tuple[SourcePayload, ...]
    exclusions: tuple[SphericalExclusion, ...]
    trace_jobs: tuple[TraceJob, ...]
    trace_options: TraceOptions
    scalar_label: str
    scalar_unit: str
    title: str
    field_model: str
    projection_note: str
    seed_mode: SeedMode
    seed_description: str
    reflect_y_symmetric_domain_exits: bool = False
    suppress_electric_return_pairs: bool = False

    @property
    def seeds(self) -> NDArray[np.float64]:
        """Return a compatibility snapshot of the planned seed coordinates."""

        return np.asarray([job.seed for job in self.trace_jobs], dtype=float).reshape(-1, 2)


def _charge(x: float, y: float, sign: int) -> SourceInput:
    kind = "positive" if sign > 0 else "negative"
    return SourceInput(x=float(x), y=float(y), kind=kind, strength=float(sign))


def _hexagon_charges(signs: tuple[int, ...]) -> list[SourceInput]:
    """Six charges on a regular hexagon, mirror-exact about both axes.

    The vertices are built from one rounded pair of coordinates so that the
    field is exactly symmetric in floating point; a charge aimed at the
    centre then stays on its symmetry line instead of drifting off it.
    """

    radius = ELECTRIC_ARRANGEMENT_RADIUS
    half = radius / 2.0
    height = radius * math.sqrt(3.0) / 2.0
    vertices = (
        (radius, 0.0),
        (half, height),
        (-half, height),
        (-radius, 0.0),
        (-half, -height),
        (half, -height),
    )
    return [_charge(x, y, sign) for (x, y), sign in zip(vertices, signs, strict=True)]


def _default_sources(preset: str) -> list[SourceInput]:
    if preset == "electric_dipole":
        return [
            SourceInput(x=-0.85, y=0.0, kind="positive", strength=1.0),
            SourceInput(x=0.85, y=0.0, kind="negative", strength=-1.0),
        ]
    if preset == "electric_quadrupole":
        side = ELECTRIC_ARRANGEMENT_RADIUS
        return [
            _charge(side, side, 1),
            _charge(-side, side, -1),
            _charge(-side, -side, 1),
            _charge(side, -side, -1),
        ]
    if preset == "electric_hexagon":
        return _hexagon_charges((1, 1, 1, 1, 1, 1))
    if preset == "electric_hexagon_alternating":
        return _hexagon_charges((1, -1, 1, -1, 1, -1))
    if preset == "magnetic_dipole":
        return [SourceInput(x=0.0, y=0.0, kind="dipole", strength=1.0)]
    if preset == "halbach_array":
        positions = np.linspace(-HALBACH_X_EXTENT, HALBACH_X_EXTENT, HALBACH_SOURCE_COUNT)
        return [
            SourceInput(
                x=float(position),
                y=0.0,
                kind="dipole",
                strength=1.0,
                angle_deg=float(90 * index % 360),
            )
            for index, position in enumerate(positions)
        ]
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
        payloads.append(SourcePayload(**source.model_dump(), strength_unit=strength_unit))
    return tuple(payloads)


def _require_source_separation(
    indexed_sources: list[tuple[int, SourceInput]],
) -> None:
    for offset, (first_index, first) in enumerate(indexed_sources):
        for second_index, second in indexed_sources[offset + 1 :]:
            distance = float(np.hypot(first.x - second.x, first.y - second.y))
            if distance <= MIN_SOURCE_SEPARATION:
                raise ValueError(
                    f"sources[{first_index}] 与 sources[{second_index}] 的中心距离必须大于 "
                    f"{MIN_SOURCE_SEPARATION:g} m"
                )


def _require_seed_budget(
    preset: str,
    source_label: str,
    source_count: int,
    density: int,
) -> None:
    if source_count > density:
        raise ValueError(
            f"{preset} 有 {source_count} 个{source_label}参与播种，density 至少为 {source_count}"
        )


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
            exclusions=(SphericalExclusion(wire_centers, CURRENT_LOOP_EXCLUSION_RADIUS),),
            trace_jobs=tuple(
                current_loop_equal_flux_jobs(
                    loop,
                    request.density,
                    axis_y=float(DOMAIN.lower[1]) + 1.0e-4,
                )
            ),
            trace_options=CURRENT_LOOP_TRACE_OPTIONS,
            scalar_label="|B|",
            scalar_unit="T",
            title="圆形电流线圈的磁力线",
            field_model="三维理想圆形电流线圈在 z=0 子午面上的限制",
            projection_note=("z=0 子午面是该轴对称场的不变平面，所示曲线是真实三维磁力线。"),
            seed_mode=SeedMode.EQUAL_FLUX,
            seed_description=(
                "相邻非轴线代表相等磁通间隔；奇数预算另含一条不带等通量权重的"
                "轴线特征线。线数不表示磁感应强度本身。"
            ),
            reflect_y_symmetric_domain_exits=True,
        )

    inputs = list(_default_sources(request.preset) if request.sources is None else request.sources)
    payloads = _source_payloads(inputs)

    if request.preset in ELECTRIC_PRESETS:
        indexed_active = [
            (index, source)
            for index, source in enumerate(inputs)
            if source.kind in {"positive", "negative"}
        ]
        _require_source_separation(indexed_active)
        active = [source for _, source in indexed_active]
        centers = np.array([[source.x, source.y] for source in active], dtype=float)
        signed_strengths = np.array([source.strength for source in active], dtype=float)
        if request.preset == "electric_dipole" and (
            not np.any(signed_strengths > 0) or not np.any(signed_strengths < 0)
        ):
            raise ValueError("electric dipole needs at least one positive and one negative source")
        field = PointChargeField(signed_strengths * 1.0e-9, centers)
        title, field_model = ELECTRIC_TITLES[request.preset]
        if request.sources is not None and request.preset != "electric_dipole":
            title, field_model = ELECTRIC_TITLES["edited"]
        _require_seed_budget(
            request.preset,
            "电荷",
            len(active),
            request.density,
        )
        return _SceneModel(
            field=field,
            sources=payloads,
            exclusions=(SphericalExclusion(centers, SOURCE_RADIUS),),
            trace_jobs=tuple(
                electric_source_jobs(indexed_active, request.density, SOURCE_SEED_RADIUS)
            ),
            trace_options=ELECTRIC_TRACE_OPTIONS,
            scalar_label="|E|",
            scalar_unit="V/m",
            title=title,
            field_model=field_model,
            projection_note="该平面法向场分量为零，所示曲线是真实场线，不是投影流线。",
            seed_mode=SeedMode.COVERAGE,
            seed_description=(
                "从所有正负电荷排除面按绝对强度共同分配预算并向外播种；"
                "同一源对已有正向代表时抑制负向返线。线密度默认不代表场强。"
            ),
            suppress_electric_return_pairs=True,
        )

    if request.preset in {"magnetic_dipole", "halbach_array"}:
        indexed_active = [
            (index, source)
            for index, source in enumerate(inputs)
            if source.kind == "dipole" and source.strength != 0.0
        ]
        if not indexed_active:
            raise ValueError("至少需要一个非零磁偶极")
        _require_source_separation(indexed_active)
        active = [source for _, source in indexed_active]
        centers = np.array([[source.x, source.y] for source in active], dtype=float)
        strengths = np.array([source.strength for source in active], dtype=float)
        angles = np.deg2rad([source.angle_deg for source in active])
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        planar_moments = strengths[:, np.newaxis] * directions
        moments = np.column_stack((planar_moments, np.zeros_like(strengths)))
        positions = np.column_stack((centers, np.zeros(len(centers))))
        field = _PlanarMagneticDipoleField(MagneticDipoleField(moments, positions))
        is_halbach = request.preset == "halbach_array"
        is_default_halbach = is_halbach and request.sources is None
        is_single_magnetic = not is_halbach and len(indexed_active) == 1
        if is_default_halbach:
            trace_jobs = tuple(
                halbach_rail_jobs(
                    request.density,
                    x_extent=HALBACH_X_EXTENT,
                    y_offset=0.45,
                )
            )
            seed_mode = SeedMode.COVERAGE
            seed_description = (
                "在阵列强、弱场两侧 y=±0.45 m 的平行轨道覆盖播种；"
                "线密度默认不代表磁感应强度。"
            )
        elif is_single_magnetic:
            source_index, source = indexed_active[0]
            trace_jobs = tuple(
                single_dipole_equatorial_jobs(
                    source_index,
                    source,
                    request.density,
                    DOMAIN,
                    SOURCE_SEED_RADIUS,
                )
            )
            seed_mode = SeedMode.COVERAGE
            seed_description = (
                "沿随磁矩旋转的赤道线等距覆盖播种并双向追踪；线密度默认不代表磁感应强度。"
            )
        else:
            _require_seed_budget(
                request.preset,
                "磁偶极子",
                len(active),
                request.density,
            )
            trace_jobs = tuple(
                magnetic_source_jobs(indexed_active, request.density, SOURCE_SEED_RADIUS)
            )
            seed_mode = SeedMode.COVERAGE
            seed_description = (
                "从每个偶极子的实际磁矩外向半球覆盖播种；"
                "线密度默认不代表磁感应强度。"
            )
        return _SceneModel(
            field=field,
            sources=payloads,
            exclusions=(SphericalExclusion(centers, SOURCE_RADIUS),),
            trace_jobs=trace_jobs,
            trace_options=DEFAULT_TRACE_OPTIONS,
            scalar_label="|B|",
            scalar_unit="T",
            title=(
                "线性 Halbach 阵列的一侧增强磁场"
                if is_default_halbach
                else "可编辑面内磁偶极子阵列"
                if is_halbach
                else "磁偶极子的磁力线"
            ),
            field_model=(
                "八个面内理想点磁偶极子的线性 Halbach 阵列"
                if is_default_halbach
                else "可编辑面内理想点磁偶极子阵列"
                if is_halbach
                else "三维理想磁偶极场在 z=0 对称平面上的限制"
            ),
            projection_note=(
                "所有偶极矩与源位置都位于 z=0 不变平面，所示曲线是真实磁力线。"
                if is_halbach
                else "偶极矩位于切片内且法向场为零，所示曲线是真实磁力线。"
            ),
            seed_mode=seed_mode,
            seed_description=seed_description,
        )

    field = UniformField((1.0, 0.28))
    y = np.linspace(DOMAIN.lower[1] + 0.12, DOMAIN.upper[1] - 0.12, request.density)
    x = np.full_like(y, DOMAIN.lower[0] + 1.0e-4)
    return _SceneModel(
        field=field,
        sources=payloads,
        exclusions=(),
        trace_jobs=tuple(
            TraceJob((float(seed_x), float(seed_y)), TraceDirection.FORWARD)
            for seed_x, seed_y in zip(x, y, strict=True)
        ),
        trace_options=DEFAULT_TRACE_OPTIONS,
        scalar_label="|E|",
        scalar_unit="V/m",
        title="匀强电场",
        field_model="二维常向量场",
        projection_note="这是原生二维向量场，曲线与电场方向处处相切。",
        seed_mode=SeedMode.COVERAGE,
        seed_description="从左边界等距覆盖播种；线密度不表示场强。",
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
    serial_values = [
        None if masked else float(value)
        for value, masked in zip(values.ravel(), mask.ravel(), strict=True)
    ]
    return ScalarPayload(
        nx=resolution,
        ny=resolution,
        values=serial_values,
        mask=mask.ravel().tolist(),
        scale=scale,
        label=model.scalar_label,
        unit=model.scalar_unit,
        vmin=vmin,
        vmax=vmax,
    )


@dataclass(frozen=True, slots=True)
class _TraceCandidate:
    job: TraceJob
    line: LinePayload
    terminal_source_index: int | None


@dataclass(frozen=True, slots=True)
class _TraceSummary:
    lines: list[LinePayload]
    termination_counts: Counter[str]
    start_termination_counts: Counter[str]
    suppressed_count: int


TRACE_CACHE_MAX_ENTRIES = 32
TRACE_CACHE_MAX_BYTES = 64 * 1024 * 1024
# Estimated CPython footprint of cached lines: each point is a list slot
# holding a tuple of two floats (8 + 56 + 2 * 24 bytes); each line adds its
# model, list and strings.
_CACHED_POINT_BYTES = 112
_CACHED_LINE_BYTES = 512


class _TraceCache:
    """Bounded LRU memo of traced lines, keyed by every input tracing reads.

    Tracing depends on the preset, the seed budget and the source list but not
    on the sampling resolution, so a resolution change reuses the lines. A hit
    returns the summary computed for an identical key, and each response gets
    its own copies of the lines, so a caller that edits a response cannot
    change later ones. Byte sizes are estimates of the Python objects held and
    only bound memory use.
    """

    def __init__(self, max_entries: int, max_bytes: int) -> None:
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._entries: OrderedDict[str, tuple[_TraceSummary, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def estimated_bytes(summary: _TraceSummary) -> int:
        points = sum(len(line.points) for line in summary.lines)
        return _CACHED_POINT_BYTES * points + _CACHED_LINE_BYTES * len(summary.lines)

    def get(self, key: str) -> _TraceSummary | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry[0]

    def put(self, key: str, summary: _TraceSummary) -> None:
        size = self.estimated_bytes(summary)
        if size > self._max_bytes:
            return
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            self._entries[key] = (summary, size)
            self._bytes += size
            while len(self._entries) > self._max_entries or self._bytes > self._max_bytes:
                _, (_, evicted_size) = self._entries.popitem(last=False)
                self._bytes -= evicted_size

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


_TRACE_CACHE = _TraceCache(TRACE_CACHE_MAX_ENTRIES, TRACE_CACHE_MAX_BYTES)


def _trace_cache_key(request: SceneRequest) -> str:
    """Everything in the request except the sampling resolution."""

    return request.model_dump_json(exclude={"resolution"})


def _trace_branches(
    result: TraceResult,
    direction: TraceDirection,
) -> tuple[TraceBranch, TraceBranch | None]:
    if direction is TraceDirection.FORWARD:
        if result.forward is None:  # pragma: no cover - tracer contract guard
            raise RuntimeError("forward trace did not return a forward branch")
        return result.forward, None
    if direction is TraceDirection.BACKWARD:
        if result.backward is None:  # pragma: no cover - tracer contract guard
            raise RuntimeError("backward trace did not return a backward branch")
        return result.backward, None
    if result.forward is None or result.backward is None:  # pragma: no cover
        raise RuntimeError("bidirectional trace did not return both branches")
    return result.forward, result.backward


def _electric_terminal_source_index(
    model: _SceneModel,
    branch: TraceBranch,
) -> int | None:
    if branch.termination is not TerminationReason.EXCLUSION_HIT:
        return None
    indexed_charges = [
        (index, source)
        for index, source in enumerate(model.sources)
        if source.kind in {"positive", "negative"}
    ]
    if not indexed_charges:  # pragma: no cover - electric model invariant
        return None
    terminal = np.asarray(branch.terminal_point, dtype=float)
    return min(
        indexed_charges,
        key=lambda item: abs(
            float(np.hypot(terminal[0] - item[1].x, terminal[1] - item[1].y))
            - SOURCE_RADIUS
        ),
    )[0]


def _is_source_kind(model: _SceneModel, index: int | None, kind: str) -> bool:
    return index is not None and 0 <= index < len(model.sources) and model.sources[index].kind == kind


def _trace_lines(model: _SceneModel) -> _TraceSummary:
    tracer = FieldLineTracer(
        model.field,
        domain=DOMAIN,
        options=model.trace_options,
        exclusions=model.exclusions,
    )
    terminations: Counter[str] = Counter()
    start_terminations: Counter[str] = Counter()
    candidates: list[_TraceCandidate] = []
    for job in model.trace_jobs:
        seed = np.asarray(job.seed, dtype=float)
        result = tracer.trace(seed, direction=job.direction)
        branch, start_branch = _trace_branches(result, job.direction)
        terminations[branch.termination.value] += 1
        if start_branch is not None:
            start_terminations[start_branch.termination.value] += 1
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
        line = LinePayload(
            points=[(float(point[0]), float(point[1])) for point in points],
            direction=-1 if job.direction is TraceDirection.BACKWARD else 1,
            termination=branch.termination.value,
            start_termination=(
                start_branch.termination.value if start_branch is not None else None
            ),
        )
        terminal_source_index = (
            _electric_terminal_source_index(model, branch)
            if model.suppress_electric_return_pairs
            else None
        )
        candidates.append(
            _TraceCandidate(
                job=job,
                line=line,
                terminal_source_index=terminal_source_index,
            )
        )

    covered_pairs: set[tuple[int, int]] = set()
    if model.suppress_electric_return_pairs:
        for candidate in candidates:
            origin = candidate.job.origin_source_index
            terminal = candidate.terminal_source_index
            if (
                candidate.job.direction is TraceDirection.FORWARD
                and _is_source_kind(model, origin, "positive")
                and _is_source_kind(model, terminal, "negative")
            ):
                assert origin is not None and terminal is not None
                covered_pairs.add((origin, terminal))

    lines: list[LinePayload] = []
    suppressed_count = 0
    for candidate in candidates:
        origin = candidate.job.origin_source_index
        terminal = candidate.terminal_source_index
        suppress = (
            model.suppress_electric_return_pairs
            and candidate.job.direction is TraceDirection.BACKWARD
            and _is_source_kind(model, origin, "negative")
            and _is_source_kind(model, terminal, "positive")
            and (terminal, origin) in covered_pairs
        )
        if suppress:
            suppressed_count += 1
        else:
            lines.append(candidate.line)

    return _TraceSummary(
        lines=lines,
        termination_counts=terminations,
        start_termination_counts=start_terminations,
        suppressed_count=suppressed_count,
    )


def build_scene(request: SceneRequest) -> SceneResponse:
    """Compute one complete scene for the browser client."""

    model = _build_model(request)
    cache_key = _trace_cache_key(request)
    traces = _TRACE_CACHE.get(cache_key)
    if traces is None:
        traces = _trace_lines(model)
        _TRACE_CACHE.put(cache_key, traces)
    return SceneResponse(
        domain=DomainPayload(
            x=(float(DOMAIN.lower[0]), float(DOMAIN.upper[0])),
            y=(float(DOMAIN.lower[1]), float(DOMAIN.upper[1])),
            coordinate_system="cartesian",
            unit="m",
        ),
        scalar=_sample_scalar(model, request.resolution),
        # Copy each cached line and its point list; the points are tuples.
        lines=[line.model_copy(update={"points": list(line.points)}) for line in traces.lines],
        sources=list(model.sources),
        metadata=MetadataPayload(
            title=model.title,
            projection_note=model.projection_note,
            field_model=model.field_model,
            seed_mode=model.seed_mode,
            seed_description=model.seed_description,
            termination_counts=dict(traces.termination_counts),
            start_termination_counts=dict(traces.start_termination_counts),
            suppressed_count=traces.suppressed_count,
            rendered_line_count=len(traces.lines),
        ),
    )


__all__ = ["build_scene"]
