"""Conservative checks for built-in exclusions hidden inside an ODE step.

The check follows the solver's dense polynomial, not the chord between output
points.  Bernstein control points bound that polynomial; subdivision tightens
the bounds without evaluating the field again.  Only the built-in Euclidean
geometries are inspected.  A custom exclusion's signed ``margin`` alone does
not supply a distance bound, so its existing solve_ivp event remains in charge.

Bounds include a floating-point reconstruction allowance.  Searching ahead
for a crossing also checks the entire intervening curve: every potentially
touched primitive must have decreasing distance.  An unresolved contact is
reported as a failure rather than being skipped for a later entry.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any

import numpy as np
from scipy.optimize import brentq

from .core import ExclusionRegion, SphericalExclusion, ToroidalExclusion

_EPS = float(np.finfo(float).eps)
_DEGREES = {"RK23": 3, "RK45": 4, "DOP853": 7, "Radau": 3, "BDF": 5, "LSODA": 12}


@cache
def _bernstein_transform(degree: int) -> tuple[np.ndarray, np.ndarray]:
    nodes = 0.5 * (1.0 - np.cos(np.pi * np.arange(degree + 1) / degree))
    matrix = np.array([
        [math.comb(degree, j) * t**j * (1.0 - t) ** (degree - j)
         for j in range(degree + 1)]
        for t in nodes
    ])
    return nodes, np.linalg.inv(matrix)


def _split(controls: np.ndarray, fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    work = controls.copy()
    left = [work[0].copy()]
    right = [work[-1].copy()]
    while len(work) > 1:
        work = (
            0.5 * (work[:-1] + work[1:]) if fraction == 0.5
            else (1.0 - fraction) * work[:-1] + fraction * work[1:]
        )
        left.append(work[0].copy())
        right.append(work[-1].copy())
    return np.asarray(left), np.asarray(right[::-1])


def _box_min_norm(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    closest = np.maximum(np.maximum(lower, -upper), 0.0)
    return np.linalg.norm(closest, axis=-1)


def _lower_margins(
    controls: np.ndarray,
    origin: np.ndarray,
    exclusion: SphericalExclusion | ToroidalExclusion,
) -> np.ndarray:
    """A lower bound over the convex hull of the Bernstein controls."""

    if isinstance(exclusion, SphericalExclusion):
        local = (origin - exclusion.centers)[:, None, :] + controls[None, :, :]
        distances = _box_min_norm(local.min(axis=1), local.max(axis=1))
        return distances - exclusion.radii
    local = (origin - exclusion.center) + controls
    axial = local @ exclusion.normal
    radial = local - axial[:, None] * exclusion.normal
    lower, upper = radial.min(axis=0), radial.max(axis=0)
    rho_lower = float(_box_min_norm(lower, upper))
    rho_upper = float(np.linalg.norm(np.maximum(np.abs(lower), np.abs(upper))))
    radial_gap = max(rho_lower - exclusion.major_radius, exclusion.major_radius - rho_upper, 0.0)
    axial_gap = max(float(axial.min()), -float(axial.max()), 0.0)
    return np.array([math.hypot(radial_gap, axial_gap) - exclusion.minor_radius])


def _product_bounds(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    products = a[:, None, ...] * b[None, :, ...]
    return np.stack((products.min(axis=(0, 1)), products.max(axis=(0, 1))))


def _approaching(
    controls: np.ndarray,
    origin: np.ndarray,
    region: SphericalExclusion | ToroidalExclusion,
    velocity: np.ndarray,
) -> np.ndarray:
    """Prove decreasing squared distance on a restricted Bernstein curve.

    A strict negative upper bound rules out entry/exit/re-entry inside a root
    bracket. Missing that proof is inconclusive, never permission to skip a
    preceding contact. The positive derivative scale (degree / duration) is
    omitted because only its sign matters.
    """

    if isinstance(region, SphericalExclusion):
        displacement = (origin - region.centers)[:, None, :] + controls[None, :, :]
        position_box = np.stack((displacement.min(axis=1), displacement.max(axis=1)))
        velocity_box = np.stack((velocity.min(axis=0), velocity.max(axis=0)))[:, None, :]
        terms = _product_bounds(position_box, velocity_box)
        upper = terms[1].sum(axis=-1)
        rounding = 64.0 * _EPS * np.max(np.abs(terms), axis=0).sum(axis=-1)
        return upper < -rounding
    displacement = origin - region.center + controls
    z = displacement @ region.normal
    radial = displacement - z[:, None] * region.normal
    dz = velocity @ region.normal
    dr = velocity - dz[:, None] * region.normal
    radial_box = np.stack((radial.min(axis=0), radial.max(axis=0)))
    rho_min = float(_box_min_norm(*radial_box))
    if rho_min <= 0.0:
        return np.array([False])
    rho_max = float(np.linalg.norm(np.max(np.abs(radial_box), axis=0)))
    radial_dot = _product_bounds(radial_box, np.stack((dr.min(axis=0), dr.max(axis=0))))
    radial_dot = radial_dot.sum(axis=-1)
    factor = 1.0 - region.major_radius / np.array([rho_min, rho_max])
    radial_term = _product_bounds(factor, radial_dot)
    axial_term = _product_bounds(np.array([z.min(), z.max()]), np.array([dz.min(), dz.max()]))
    upper = float(radial_term[1] + axial_term[1])
    rounding = 64.0 * _EPS * float(np.max(np.abs(radial_term)) + np.max(np.abs(axial_term)))
    return np.array([upper < -rounding])


def first_exclusion_event(
    solution: Any,
    exclusions: tuple[ExclusionRegion, ...] | list[ExclusionRegion],
    method: str,
    *,
    end_parameter: float | None = None,
    budget: list[int] | None = None,
    known_exclusion_parameter: float | None = None,
) -> tuple[float, bool] | None:
    """Return the earliest hidden entry, or the earliest unresolved interval.

    ``True`` requires a non-positive margin witness and a bracketed root;
    ``False`` asks the caller to stop with ``solver_failure``.  ``budget[0]``
    counts remaining subdivisions and may be shared across integration chunks.
    Clearly safe accepted steps do not spend it.  Existing terminal exclusion
    roots are retained verbatim only if the ambiguous prefix also passes the
    approach check.  The degree bounds cover SciPy's named solve_ivp methods; the dense
    interpolant and the ODE solver's own accuracy remain the numerical model.
    """

    # Each sphere needs its own roundoff scale: an unrelated far-away centre
    # must not inflate the uncertainty of a nearby obstacle in the same batch.
    geometries = tuple(
        region for region in exclusions
        if isinstance(region, SphericalExclusion | ToroidalExclusion)
    )
    if not geometries or len(solution.t) < 2:
        return None
    if solution.sol is None or method not in _DEGREES:
        return float(solution.t[0]), False
    remaining = budget if budget is not None else [8192]
    end = float(solution.t[-1]) if end_parameter is None else min(
        float(solution.t[-1]), float(end_parameter)
    )
    nodes, transform = _bernstein_transform(_DEGREES[method])
    initial = float(solution.t[0])
    geometry_scales = np.concatenate([
        np.max(np.abs(region.centers), axis=1) + region.radii
        if isinstance(region, SphericalExclusion)
        else np.array([float(np.max(np.abs(region.center))) + region.major_radius + region.minor_radius])
        for region in geometries
    ])

    def margin(parameter: float) -> float:
        point = np.asarray(solution.sol(parameter), dtype=float)
        return min(float(region.margin(point)) for region in geometries)

    initial_margin = margin(initial)
    for start_value, stop_value in zip(solution.t[:-1], solution.t[1:], strict=True):
        start, stop = float(start_value), min(float(stop_value), end)
        if start >= end:
            break
        if stop <= start:
            continue
        values = np.asarray(solution.sol(start + (stop - start) * nodes), dtype=float).T
        origin = values[0].copy()
        relative = values - origin
        controls = transform @ relative
        derivatives = np.diff(controls, axis=0)
        # Translation before reconstruction keeps the condition allowance local.
        amplification = float(np.max(np.abs(transform) @ np.max(np.abs(relative), axis=1)))
        magnitude = float(np.max(np.abs(values)))
        guards = 16.0 * _EPS * np.maximum(
            geometry_scales, max(magnitude, amplification, np.finfo(float).tiny)
        )
        time_tolerance = 64.0 * _EPS * max(abs(start), abs(stop), np.finfo(float).tiny)
        # A shallow entry can have a rounding-size distance residual well
        # before its root. This is a search window for penetration evidence,
        # NOT a tolerance for identifying two distinct roots as the same one.
        probe_span = max(time_tolerance, 16.0 * math.sqrt(_EPS) * (stop - start))

        # Bind this accepted step explicitly so the helpers cannot observe a
        # later iteration if their use is moved during future maintenance.
        def restrict(
            polygon: np.ndarray, left: float, right: float,
            *, start: float = start, stop: float = stop,
        ) -> np.ndarray:
            begin = (left - start) / (stop - start)
            finish = (right - start) / (stop - start)
            _, suffix = _split(polygon, begin)
            restricted, _ = _split(suffix, (finish - begin) / (1.0 - begin))
            return restricted

        def valid_bracket(
            left: float, right: float, *, departing: bool = False,
            controls: np.ndarray = controls, derivatives: np.ndarray = derivatives,
            origin: np.ndarray = origin, guards: np.ndarray = guards,
        ) -> bool:
            # Restrict the ORIGINAL step, never extrapolate a tiny leaf. Every
            # primitive possibly touched by this longer bracket must approach
            # monotonically; otherwise a later negative sample could hide an
            # earlier narrow obstacle or another tube section of the same torus.
            restricted = restrict(controls, left, right)
            # Differentiate BEFORE restriction. Subtracting positions on a
            # rounding-size interval can erase or reverse a valid derivative.
            velocity = restrict(derivatives, left, right)
            if departing:
                restricted, velocity = restricted[::-1], -velocity[::-1]
            lower = np.concatenate([_lower_margins(restricted, origin, region) for region in geometries])
            decreasing = np.concatenate([
                _approaching(restricted, origin, region, velocity) for region in geometries
            ])
            return bool(np.all((lower > guards) | decreasing))

        stack = [(start, stop, controls, 0)]
        while stack:
            left, right, polygon, depth = stack.pop()
            bounds = np.concatenate([_lower_margins(polygon, origin, region) for region in geometries])
            possible = ~np.isfinite(bounds) | (bounds <= guards)
            if not np.any(possible):
                continue
            guard = float(np.max(guards[possible]))
            # The RHS has unit speed. Allow coordinate rounding as well as
            # parameter rounding, without introducing an absolute length unit.
            endpoint_tolerance = max(time_tolerance, 2.0 * guard)
            if remaining[0] <= 0:
                return left, False
            remaining[0] -= 1
            middle = left + 0.5 * (right - left)
            if right - left <= time_tolerance or middle in (left, right) or depth >= 60:
                left_margin, middle_margin, right_margin = margin(left), margin(middle), margin(right)
                witnesses = [
                    parameter for parameter, value in (
                        (left, left_margin), (middle, middle_margin), (right, right_margin)
                    ) if value < 0.0
                ]
                if witnesses:
                    witness = witnesses[0]
                    bracket_start = left
                    if margin(bracket_start) > 0.0 and witness > bracket_start:
                        if not valid_bracket(bracket_start, witness):
                            return left, False
                        event = float(brentq(
                            margin, bracket_start, witness,
                            xtol=float(np.nextafter(0.0, 1.0)), rtol=4.0 * _EPS,
                        ))
                    else:
                        event = bracket_start
                    if (
                        known_exclusion_parameter is not None
                        and event >= known_exclusion_parameter
                    ):
                        return None
                    return event, True
                if (
                    known_exclusion_parameter is not None
                    and known_exclusion_parameter <= stop
                    and 0.0 <= known_exclusion_parameter - left <= probe_span
                    and (
                        known_exclusion_parameter == left
                        or valid_bracket(left, known_exclusion_parameter)
                    )
                ):
                    return None
                # A surface seed departing outwards shares its boundary point
                # with every nested box. Discard only its rounding-size prefix;
                # any sampled penetration above is still a confirmed entry.
                if (
                    abs(initial_margin) <= guard
                    and left - initial <= endpoint_tolerance
                    and min(left_margin, middle_margin, right_margin) >= 0.0
                    and right_margin > left_margin
                    and valid_bracket(left, right, departing=True)
                ):
                    continue
                # At a shallow entry the reconstruction allowance can touch
                # the surface a little before the real root. Look for actual
                # penetration within the permitted parameter neighbourhood,
                # rather than spending the budget on machine-sized leaves.
                probe_end = min(stop, left + probe_span)
                for probe in np.linspace(right, probe_end, 17):
                    if margin(float(probe)) < 0.0:
                        # A probe may be past the closest point of a thin
                        # obstacle. Refine its inside witness, but never move
                        # the prefix that still needs geometric verification.
                        search_left, witness = left, float(probe)
                        for _refinement in range(16):
                            if valid_bracket(left, witness):
                                break
                            middle_probe = search_left + 0.5 * (witness - search_left)
                            if margin(middle_probe) < 0.0:
                                witness = middle_probe
                            else:
                                search_left = middle_probe
                        else:
                            return left, False
                        event = float(brentq(
                            margin, left, witness,
                            xtol=float(np.nextafter(0.0, 1.0)), rtol=4.0 * _EPS,
                        ))
                        if (
                            known_exclusion_parameter is not None
                            and event >= known_exclusion_parameter
                        ):
                            return None
                        return event, True
                # The earliest ambiguous interval already prevents a claim of
                # safe passage. Enumerating neighbouring machine-size leaves
                # cannot justify skipping it and can consume the whole budget.
                return left, False
            left_polygon, right_polygon = _split(polygon)
            stack.append((middle, right, right_polygon, depth + 1))
            stack.append((left, middle, left_polygon, depth + 1))
    return None
