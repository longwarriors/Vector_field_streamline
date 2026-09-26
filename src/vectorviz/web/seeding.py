"""Pure seed planners for the browser scene orchestrator."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from vectorviz.core import Domain
from vectorviz.fields import ChargedRingField, CircularLoopField, DielectricSphereField
from vectorviz.tracing import TraceDirection


class _SourceLike(Protocol):
    x: float
    y: float
    kind: str
    strength: float
    angle_deg: float | None


IndexedSources = Iterable[tuple[int, _SourceLike]]


@dataclass(frozen=True, slots=True)
class TraceJob:
    """One seed and the branch orientation that should be traced from it."""

    seed: tuple[float, float]
    direction: TraceDirection
    origin_source_index: int | None = None

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.seed, dtype=float)
        if coordinates.shape != (2,) or not np.all(np.isfinite(coordinates)):
            raise ValueError("seed must contain exactly two finite coordinates")
        origin = self.origin_source_index
        if origin is not None:
            if isinstance(origin, bool):
                raise TypeError("origin_source_index must be an integer or None")
            origin = operator.index(origin)
            if origin < 0:
                raise ValueError("origin_source_index must be non-negative")
        object.__setattr__(self, "seed", (float(coordinates[0]), float(coordinates[1])))
        object.__setattr__(self, "direction", TraceDirection.coerce(self.direction))
        object.__setattr__(self, "origin_source_index", origin)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _source_index(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("source index must be an integer")
    try:
        index = operator.index(value)
    except TypeError as error:
        raise TypeError("source index must be an integer") from error
    if index < 0:
        raise ValueError("source index must be non-negative")
    return index


def _source_geometry(source: _SourceLike) -> tuple[float, float, float]:
    x = float(source.x)
    y = float(source.y)
    strength = float(source.strength)
    if not np.all(np.isfinite((x, y, strength))):
        raise ValueError("source coordinates and strength must be finite")
    return x, y, strength


def allocate_seed_counts(strengths: ArrayLike, total: int) -> NDArray[np.int64]:
    """Allocate an integer budget with one seed per source and stable remainders."""

    values = np.asarray(strengths, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("strengths must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("strengths must be finite")
    budget = _positive_integer(total, "total")
    source_count = int(values.size)
    if source_count > budget:
        raise ValueError("seed budget must provide at least one seed per source")

    counts = np.ones(source_count, dtype=np.int64)
    remaining = budget - source_count
    if remaining == 0:
        return counts

    weights = np.abs(values)
    scale = float(np.max(weights))
    normalized = np.ones_like(weights) if scale == 0.0 else weights / scale
    quotas = normalized / np.sum(normalized) * remaining
    extras = np.floor(quotas).astype(np.int64)
    counts += extras
    unassigned = remaining - int(np.sum(extras))
    if unassigned:
        fractions = quotas - extras
        order = np.argsort(-fractions, kind="stable")
        counts[order[:unassigned]] += 1
    return counts


def _materialize_sources(indexed_sources: IndexedSources) -> tuple[tuple[int, _SourceLike], ...]:
    sources = tuple((_source_index(index), source) for index, source in indexed_sources)
    if not sources:
        raise ValueError("at least one seeding source is required")
    return sources


def _circle_offsets(count: int, radius: float) -> NDArray[np.float64]:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return radius * np.column_stack((np.cos(angles), np.sin(angles)))


def electric_source_jobs(
    indexed_sources: IndexedSources,
    total: int,
    seed_radius: float,
) -> list[TraceJob]:
    """Seed every positive and negative charge, tracing away from each charge."""

    sources = _materialize_sources(indexed_sources)
    radius = _finite_positive(seed_radius, "seed_radius")
    strengths: list[float] = []
    for _index, source in sources:
        _x, _y, strength = _source_geometry(source)
        valid_positive = source.kind == "positive" and strength > 0.0
        valid_negative = source.kind == "negative" and strength < 0.0
        if not (valid_positive or valid_negative):
            raise ValueError("electric sources must have matching nonzero kind and strength")
        strengths.append(strength)
    counts = allocate_seed_counts(strengths, total)

    jobs: list[TraceJob] = []
    for (index, source), count in zip(sources, counts, strict=True):
        center = np.asarray((float(source.x), float(source.y)))
        direction = TraceDirection.FORWARD if source.kind == "positive" else TraceDirection.BACKWARD
        for seed in center + _circle_offsets(int(count), radius):
            jobs.append(TraceJob(tuple(seed), direction, index))
    return jobs


def magnetic_source_jobs(
    indexed_sources: IndexedSources,
    total: int,
    seed_radius: float,
) -> list[TraceJob]:
    """Seed generic dipoles on the hemisphere outward from each actual moment."""

    sources = _materialize_sources(indexed_sources)
    radius = _finite_positive(seed_radius, "seed_radius")
    strengths: list[float] = []
    angles: list[float] = []
    for _index, source in sources:
        _x, _y, strength = _source_geometry(source)
        if source.kind != "dipole" or strength == 0.0:
            raise ValueError("magnetic seeding sources must be nonzero dipoles")
        angle = float(source.angle_deg) if source.angle_deg is not None else np.nan
        if not np.isfinite(angle):
            raise ValueError("dipole angle_deg must be finite")
        strengths.append(strength)
        angles.append(angle)
    counts = allocate_seed_counts(strengths, total)

    jobs: list[TraceJob] = []
    for (index, source), strength, angle, count in zip(
        sources, strengths, angles, counts, strict=True
    ):
        center = np.asarray((float(source.x), float(source.y)))
        parameter_angle = np.deg2rad(angle)
        parameter_axis = np.asarray((np.cos(parameter_angle), np.sin(parameter_angle)))
        actual_axis = parameter_axis if strength > 0.0 else -parameter_axis
        perpendicular = np.asarray((actual_axis[1], -actual_axis[0]))
        coverage_angles = (
            np.asarray((0.0,)) if int(count) == 1 else np.linspace(-1.43, 1.43, int(count))
        )
        offsets = radius * (
            np.cos(coverage_angles)[:, np.newaxis] * actual_axis
            + np.sin(coverage_angles)[:, np.newaxis] * perpendicular
        )
        jobs.extend(
            TraceJob(tuple(seed), TraceDirection.FORWARD, index) for seed in center + offsets
        )
    return jobs


def _ray_limit(
    center: NDArray[np.float64],
    direction: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> float:
    limits: list[float] = []
    for coordinate, component, low, high in zip(center, direction, lower, upper, strict=True):
        if component > 0.0:
            limits.append(float((high - coordinate) / component))
        elif component < 0.0:
            limits.append(float((low - coordinate) / component))
    if not limits:  # pragma: no cover - perpendicular unit vectors are never zero
        raise ValueError("equatorial direction must be nonzero")
    return min(limits)


def _equatorial_distances(start: float, stop: float, count: int) -> NDArray[np.float64]:
    if count == 0:
        return np.empty(0, dtype=float)
    return np.linspace(start, stop, count)


def single_dipole_equatorial_jobs(
    index: int,
    source: _SourceLike,
    total: int,
    domain: Domain,
    seed_radius: float,
    domain_inset: float = 1.0e-4,
) -> list[TraceJob]:
    """Cover both rays of a single dipole's equatorial line with BOTH jobs."""

    source_index = _source_index(index)
    budget = _positive_integer(total, "total")
    radius = _finite_positive(seed_radius, "seed_radius")
    inset = float(domain_inset)
    if not np.isfinite(inset) or inset < 0.0:
        raise ValueError("domain_inset must be a finite non-negative number")
    if not isinstance(domain, Domain) or domain.dimension != 2:
        raise ValueError("domain must be a two-dimensional Domain")
    x, y, strength = _source_geometry(source)
    if source.kind != "dipole" or strength == 0.0:
        raise ValueError("single-dipole equatorial seeding requires a nonzero dipole")
    angle = float(source.angle_deg) if source.angle_deg is not None else np.nan
    if not np.isfinite(angle):
        raise ValueError("dipole angle_deg must be finite")

    lower = np.asarray(domain.lower, dtype=float) + inset
    upper = np.asarray(domain.upper, dtype=float) - inset
    if np.any(lower >= upper):
        raise ValueError("domain_inset leaves no seedable domain")
    center = np.asarray((x, y), dtype=float)
    if np.any(center < lower) or np.any(center > upper):
        raise ValueError("dipole center must lie inside the inset domain")
    parameter_angle = np.deg2rad(angle)
    moment_axis = np.asarray((np.cos(parameter_angle), np.sin(parameter_angle)))
    perpendicular = np.asarray((-moment_axis[1], moment_axis[0]))
    directions = (perpendicular, -perpendicular)
    limits = np.asarray(
        [_ray_limit(center, direction, lower, upper) for direction in directions],
        dtype=float,
    )
    if np.any(limits < radius):
        raise ValueError("both equatorial rays must extend beyond seed_radius")

    if budget == 1:
        counts = np.zeros(2, dtype=np.int64)
        counts[int(np.argmax(limits - radius))] = 1
    else:
        counts = allocate_seed_counts(limits - radius, budget)

    jobs: list[TraceJob] = []
    for direction, limit, count in zip(directions, limits, counts, strict=True):
        distances = _equatorial_distances(radius, float(limit), int(count))
        points = center + distances[:, np.newaxis] * direction
        points = np.minimum(np.maximum(points, lower), upper)
        jobs.extend(TraceJob(tuple(seed), TraceDirection.BOTH, source_index) for seed in points)
    return jobs


def halbach_rail_jobs(
    total: int,
    x_extent: float = 2.1,
    y_offset: float = 0.45,
) -> list[TraceJob]:
    """Place BOTH-direction coverage jobs on upper and lower Halbach rails."""

    budget = _positive_integer(total, "total")
    extent = _finite_positive(x_extent, "x_extent")
    offset = _finite_positive(y_offset, "y_offset")
    upper_count = (budget + 1) // 2
    lower_count = budget // 2
    jobs: list[TraceJob] = []
    for count, y in ((upper_count, offset), (lower_count, -offset)):
        jobs.extend(
            TraceJob((float(x), y), TraceDirection.BOTH)
            for x in np.linspace(-extent, extent, count)
        )
    return jobs


def _loop_flux_at_radius(loop: CircularLoopField, radius: float) -> float:
    point = np.array(loop.center, dtype=float, copy=True)
    point[0] += radius
    flux = float(np.asarray(loop.flux_function(point)))
    if not np.isfinite(flux):  # pragma: no cover - finite off the wire by construction
        raise ValueError("loop flux must be finite throughout the seed interval")
    return flux


def _loop_flux_residual(
    radius: float,
    loop: CircularLoopField,
    target: float,
) -> float:
    return _loop_flux_at_radius(loop, radius) - target


def current_loop_equal_flux_jobs(
    loop: CircularLoopField,
    total: int,
    axis_y: float,
    inner_radius: float = 0.12,
    outer_radius: float = 0.82,
) -> list[TraceJob]:
    """Invert equally spaced public flux targets into mirrored meridional jobs."""

    if not isinstance(loop, CircularLoopField):
        raise TypeError("loop must be a CircularLoopField")
    budget = _positive_integer(total, "total")
    axis_coordinate = float(axis_y)
    if not np.isfinite(axis_coordinate):
        raise ValueError("axis_y must be finite")
    inner = _finite_positive(inner_radius, "inner_radius")
    outer = _finite_positive(outer_radius, "outer_radius")
    if inner >= outer or outer >= loop.radius:
        raise ValueError("loop seed radii must satisfy 0 < inner_radius < outer_radius < radius")
    normal = np.asarray(loop.normal)
    if normal[0] != 0.0 or abs(normal[1]) != 1.0 or normal[2] != 0.0:
        raise ValueError("loop normal must be parallel to the web y-axis")

    pair_count = budget // 2
    jobs: list[TraceJob] = []
    center_x = float(loop.center[0])
    center_y = float(loop.center[1])
    if budget % 2:
        jobs.append(TraceJob((center_x, axis_coordinate), TraceDirection.FORWARD))
    if pair_count == 0:
        return jobs

    inner_flux = _loop_flux_at_radius(loop, inner)
    outer_flux = _loop_flux_at_radius(loop, outer)
    if inner_flux == outer_flux:  # pragma: no cover - psi is strictly monotonic on the equator
        raise ValueError("loop flux must vary across the seed interval")
    targets = np.linspace(inner_flux, outer_flux, pair_count)
    for target in targets:
        radius = brentq(
            _loop_flux_residual,
            inner,
            outer,
            args=(loop, float(target)),
        )
        jobs.append(TraceJob((center_x - radius, center_y), TraceDirection.FORWARD))
        jobs.append(TraceJob((center_x + radius, center_y), TraceDirection.FORWARD))
    return jobs


def _ring_flux_on_seed_circle(angle: float, ring: ChargedRingField, seed_radius: float) -> float:
    """Public flux function at ``angle`` around the +x ring cross-section."""

    point = np.array(ring.center, dtype=float, copy=True)
    point[0] += ring.radius + seed_radius * math.cos(angle)
    point[1] += seed_radius * math.sin(angle)
    flux = float(np.asarray(ring.flux_function(point)))
    if not np.isfinite(flux):  # pragma: no cover - finite off the filament by construction
        raise ValueError("ring flux must be finite on the seed circle")
    return flux


def _ring_flux_residual(
    angle: float,
    ring: ChargedRingField,
    seed_radius: float,
    target: float,
) -> float:
    return _ring_flux_on_seed_circle(angle, ring, seed_radius) - target


def charged_ring_equal_flux_jobs(
    ring: ChargedRingField,
    total: int,
    seed_radius: float,
) -> list[TraceJob]:
    """Seed both ring cross-sections at equal steps of the public flux function.

    Around the +x cross-section the flux function falls monotonically from
    ``+Q/(4 pi eps0)`` just above the outer equator to ``-Q/(4 pi eps0)`` just
    below it, so ``total // 2`` targets at the midpoints of equal flux steps
    are inverted to angles on the seed circle and mirrored to the -x section.
    An odd budget adds the outer equatorial ray, the separatrix between the
    upper and lower half-planes, which carries no equal-flux weight.
    """

    if not isinstance(ring, ChargedRingField):
        raise TypeError("ring must be a ChargedRingField")
    budget = _positive_integer(total, "total")
    radius = _finite_positive(seed_radius, "seed_radius")
    if radius >= ring.radius:
        raise ValueError("seed_radius must be smaller than the ring radius")
    normal = np.asarray(ring.normal)
    if normal[0] != 0.0 or abs(normal[1]) != 1.0 or normal[2] != 0.0:
        raise ValueError("ring normal must be parallel to the web y-axis")

    center_x = float(ring.center[0])
    center_y = float(ring.center[1])
    pair_count = budget // 2
    jobs: list[TraceJob] = []
    if budget % 2:
        jobs.append(TraceJob((center_x + ring.radius + radius, center_y), TraceDirection.FORWARD))
    if pair_count == 0:
        return jobs

    # The z -> 0+ value on the outer cut; the seed circle spans (-cut, +cut).
    cut_flux = _ring_flux_on_seed_circle(0.0, ring, radius)
    targets = cut_flux * (-1.0 + (2.0 * np.arange(pair_count) + 1.0) / pair_count)
    for target in targets:
        if target == 0.0:
            # The zero-flux line is the inward equator by symmetry. It runs
            # into the saddle at the centre, which any off-axis rounding would
            # deflect, so place it on the axis exactly instead of via a root.
            offset_x = ring.radius - radius
            offset_y = 0.0
        else:
            angle = brentq(
                _ring_flux_residual,
                1.0e-9,
                2.0 * np.pi - 1.0e-9,
                args=(ring, radius, float(target)),
            )
            offset_x = ring.radius + radius * math.cos(angle)
            offset_y = radius * math.sin(angle)
        jobs.append(TraceJob((center_x - offset_x, center_y + offset_y), TraceDirection.FORWARD))
        jobs.append(TraceJob((center_x + offset_x, center_y + offset_y), TraceDirection.FORWARD))
    return jobs


def _sphere_flux_at_height(
    height: float,
    sphere: DielectricSphereField,
    edge_x: float,
    center_y: float,
) -> float:
    """Public displacement-flux function on the seeding edge at ``height``."""

    point = (edge_x, center_y + height, 0.0)
    return float(np.asarray(sphere.flux_function(point)))


def _sphere_flux_residual(
    height: float,
    sphere: DielectricSphereField,
    edge_x: float,
    center_y: float,
    target: float,
) -> float:
    return _sphere_flux_at_height(height, sphere, edge_x, center_y) - target


def sphere_equal_flux_jobs(
    sphere: DielectricSphereField,
    total: int,
    domain: Domain,
    x_inset: float = 1.0e-4,
    y_inset: float = 0.12,
) -> list[TraceJob]:
    """Seed the upstream edge at equal steps of the displacement flux function.

    The applied field must point along +x with the sphere centre in the
    z=0 plane, so the web plane is a meridional plane. ``total // 2`` targets
    are spaced equally from one step up to the flux at the edge's largest
    usable height, inverted to heights and mirrored about the axis; an odd
    budget adds the axis line, which is the zero-flux member of the family.
    """

    if not isinstance(sphere, DielectricSphereField):
        raise TypeError("sphere must be a DielectricSphereField")
    budget = _positive_integer(total, "total")
    if not isinstance(domain, Domain) or domain.dimension != 2:
        raise ValueError("domain must be a two-dimensional Domain")
    applied = np.asarray(sphere.applied_field)
    if applied[1] != 0.0 or applied[2] != 0.0 or applied[0] <= 0.0:
        raise ValueError("the applied field must point along +x")
    if sphere.center[2] != 0.0:
        raise ValueError("the sphere centre must lie in the z=0 plane")
    inset_x = float(x_inset)
    inset_y = float(y_inset)
    if not np.isfinite(inset_x) or inset_x < 0.0 or not np.isfinite(inset_y) or inset_y < 0.0:
        raise ValueError("insets must be finite and non-negative")
    edge_x = float(domain.lower[0]) + inset_x
    center_y = float(sphere.center[1])
    height_limit = min(
        float(domain.upper[1]) - center_y,
        center_y - float(domain.lower[1]),
    ) - inset_y
    if height_limit <= 0.0:
        raise ValueError("the sphere centre must leave seedable height on both sides")
    if np.hypot(edge_x - float(sphere.center[0]), 0.0) <= sphere.radius:
        raise ValueError("the seeding edge must lie outside the sphere")

    pair_count = budget // 2
    jobs: list[TraceJob] = []
    if budget % 2:
        jobs.append(TraceJob((edge_x, center_y), TraceDirection.FORWARD))
    if pair_count == 0:
        return jobs

    top_flux = _sphere_flux_at_height(height_limit, sphere, edge_x, center_y)
    targets = np.linspace(top_flux / pair_count, top_flux, pair_count)
    for target in targets:
        height = brentq(
            _sphere_flux_residual,
            0.0,
            height_limit,
            args=(sphere, edge_x, center_y, float(target)),
        )
        jobs.append(TraceJob((edge_x, center_y - height), TraceDirection.FORWARD))
        jobs.append(TraceJob((edge_x, center_y + height), TraceDirection.FORWARD))
    return jobs


__all__ = [
    "TraceJob",
    "allocate_seed_counts",
    "charged_ring_equal_flux_jobs",
    "current_loop_equal_flux_jobs",
    "electric_source_jobs",
    "halbach_rail_jobs",
    "magnetic_source_jobs",
    "single_dipole_equatorial_jobs",
    "sphere_equal_flux_jobs",
]
