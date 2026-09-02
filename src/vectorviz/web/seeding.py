"""Pure seed planners for the browser scene orchestrator."""

from __future__ import annotations

import operator
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from vectorviz.core import Domain
from vectorviz.fields import CircularLoopField
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
    if not limits:
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
    if not np.isfinite(flux):
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
    if inner_flux == outer_flux:
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


__all__ = [
    "TraceJob",
    "allocate_seed_counts",
    "current_loop_equal_flux_jobs",
    "electric_source_jobs",
    "halbach_rail_jobs",
    "magnetic_source_jobs",
    "single_dipole_equatorial_jobs",
]
