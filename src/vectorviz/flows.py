"""Time-dependent velocity fields and physical-time particle integration.

Spatial coordinates are in metres, velocity in m/s, and time in seconds for
``OscillatingFlowField``. The generic interfaces also support other consistent
unit systems. Streamlines use ``freeze`` and the separate normalized tracer;
pathlines always solve dx/dt = u(x, t), including through zero velocity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from numbers import Real

import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import DOP853

from .core import Domain, FloatArray, VectorField, _as_points


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar.")
    return result


class TimeDependentField(ABC):
    """Batch-first velocity field at one finite scalar physical time."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of spatial coordinates and velocity components."""

    @abstractmethod
    def evaluate(self, points: ArrayLike, time: float) -> FloatArray:
        """Return velocities with the same ``(..., dimension)`` shape as points."""

    def freeze(self, time: float) -> VectorField:
        """Adapt this field to the static field API at one fixed time."""
        return _FrozenVelocity(self, _finite_scalar(time, "time"))


@dataclass(frozen=True, slots=True)
class _FrozenVelocity(VectorField):
    field: TimeDependentField
    time: float

    @property
    def dimension(self) -> int:
        return self.field.dimension

    def evaluate(self, points: ArrayLike) -> FloatArray:
        return self.field.evaluate(points, self.time)


@dataclass(frozen=True, slots=True)
class OscillatingFlowField(TimeDependentField):
    """Spatially uniform 2D flow ``(U, V + A sin(omega * t))``.

    Speeds ``U``, ``V`` and ``A`` are in m/s and ``omega`` in rad/s.
    The model is incompressible and has no spatial singularities. ``A=0``
    gives a steady control. Invalid coordinates produce all-NaN vectors.
    """

    longitudinal_speed: float = 1.0
    transverse_speed: float = 0.0
    amplitude: float = 0.6
    angular_frequency: float = 1.0

    def __post_init__(self) -> None:
        for name in ("longitudinal_speed", "transverse_speed", "amplitude", "angular_frequency"):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name))
        if self.angular_frequency < 0:
            raise ValueError("angular_frequency must be non-negative.")

    @property
    def dimension(self) -> int:
        return 2

    def evaluate(self, points: ArrayLike, time: float) -> FloatArray:
        time = _finite_scalar(time, "time")
        if np.iscomplexobj(points):
            raise ValueError("points must contain real coordinates.")
        coordinates = _as_points(points, 2)
        result = np.empty_like(coordinates)
        result[..., 0] = self.longitudinal_speed
        with np.errstate(invalid="ignore", over="ignore"):
            result[..., 1] = self.transverse_speed if self.amplitude == 0 else (
                self.transverse_speed + self.amplitude * np.sin(self.angular_frequency * time)
            )
        result[~np.all(np.isfinite(coordinates), axis=-1)] = np.nan
        result[~np.all(np.isfinite(result), axis=-1)] = np.nan
        return result


class TimeTerminationReason(StrEnum):
    """Why a physical-time particle trajectory stopped."""

    TIME_END = "time_end"
    DOMAIN_EXIT = "domain_exit"
    SEED_OUTSIDE_DOMAIN = "seed_outside_domain"
    NONFINITE_FIELD = "nonfinite_field"
    SOLVER_FAILURE = "solver_failure"


@dataclass(frozen=True, slots=True)
class TimeOptions:
    """DOP853 controls; steps are in time and ``atol`` in coordinate units.

    ``output_step=None`` retains accepted steps. Otherwise output is uniform
    in time plus the actual terminal point. ``max_steps`` bounds solver work;
    an output request exceeding one million samples is rejected.
    """

    max_step: float = 0.1
    rtol: float = 1e-8
    atol: float = 1e-10
    output_step: float | None = 0.05
    max_steps: int = 10000

    def __post_init__(self) -> None:
        for name in ("max_step", "rtol", "atol", "output_step"):
            value = getattr(self, name)
            if name == "output_step" and value is None:
                continue
            value = _finite_scalar(value, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer.")


@dataclass(frozen=True, slots=True)
class PathlineResult:
    """One particle's positions in strictly increasing physical-time order.

    A zero-duration or immediately terminated trace contains just its seed.
    Non-finite evaluation terminates at the last verified finite state; it
    does not claim to locate a singularity or invent a boundary velocity.
    """

    points: FloatArray
    times: FloatArray
    termination: TimeTerminationReason
    nfev: int
    message: str


@dataclass(frozen=True, slots=True)
class StreaklineResult:
    """Particles from one fixed site, ordered by strictly increasing release time.

    Only ``valid`` positions have reached the common observation time. Failed
    or escaped particles retain their last finite positions and arrival times
    for audit, and must not be connected across gaps when drawing a streakline.
    """

    positions: FloatArray
    release_times: FloatArray
    arrival_times: FloatArray
    terminations: tuple[TimeTerminationReason, ...]
    valid: NDArray[np.bool_]
    nfev: int


class _NonfiniteVelocity(Exception):
    pass


def _first_domain_exit(dense: object, domain: Domain) -> tuple[float, FloatArray] | None:
    """Find the first outward crossing of DOP853's degree-seven interpolant.

    A conservative coefficient bound makes steps wholly inside the rectangle
    cheap. Near faces, polynomial roots partition the step into intervals of
    constant sign, so leaving and returning inside one step is not missed.
    ``F`` and ``y_old`` are SciPy's DOP853 continuous-extension coefficients.
    """
    coefficients = dense.F
    start = dense.y_old
    bound = np.sum(np.abs(coefficients), axis=0)
    if np.all(start - bound > domain.lower) and np.all(start + bound < domain.upper):
        return None
    # Reproduce the nested continuous extension in increasing power order.
    powers = np.zeros((1, domain.dimension))
    for index, coefficient in enumerate(reversed(coefficients)):
        powers[0] += coefficient
        expanded = np.zeros((len(powers) + 1, domain.dimension))
        if index % 2 == 0:
            expanded[1:] = powers
        else:
            expanded[:-1] = powers
            expanded[1:] -= powers
        powers = expanded
    powers[0] += start
    first: tuple[float, int, float] | None = None
    for axis in range(domain.dimension):
        for face, sign in ((domain.lower[axis], -1), (domain.upper[axis], 1)):
            margin = sign * powers[:, axis].copy()
            margin[0] -= sign * face
            roots = poly.polyroots(poly.polytrim(margin))
            real = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-9 and 0 < r.real < 1)
            breaks = [0.0, *real, 1.0]
            # Ignore only roundoff in the reconstructed polynomial. Geometry
            # is never shifted; the accepted root is on the original face.
            scale = max(abs(float(start[axis])), abs(float(face)), float(np.sum(np.abs(margin))))
            tolerance = 64 * np.finfo(float).eps * scale
            for left, right in pairwise(breaks):
                if poly.polyval((left + right) / 2, margin) > tolerance:
                    if first is None or left < first[0]:
                        first = (left, axis, float(face))
                    break
            # A trajectory arriving at a face exactly at the final instant
            # has reached time_end; an exit requires some outside interval.
    if first is None:
        return None
    fraction, axis, face = first
    time = dense.t_old + fraction * (dense.t - dense.t_old)
    point = np.asarray(dense(time), dtype=float)
    point[axis] = face
    return float(time), np.clip(point, domain.lower, domain.upper)


class PathlineTracer:
    """Integrate unnormalized velocity in increasing physical time with DOP853.

    The optional domain is closed: particles on a face may move inward or
    remain stationary. The first outward crossing of an accepted step's
    continuous interpolant terminates the particle, even if it later returns.
    No exclusion geometry, material interfaces or stochastic diffusion are
    implied by this API.
    """

    def __init__(
        self, field: TimeDependentField, *, domain: Domain | None = None,
        options: TimeOptions | None = None,
    ) -> None:
        if not isinstance(field, TimeDependentField):
            raise ValueError("field must implement TimeDependentField.")
        if domain is not None and domain.dimension != field.dimension:
            raise ValueError("domain and field dimensions must match.")
        self.field = field
        self.domain = domain
        self.options = options if options is not None else TimeOptions()

    def trace(self, seed: ArrayLike, start_time: float, end_time: float) -> PathlineResult:
        """Trace one finite seed over ``start_time <= end_time``."""
        start_time = _finite_scalar(start_time, "start_time")
        end_time = _finite_scalar(end_time, "end_time")
        if end_time < start_time or not np.isfinite(end_time - start_time):
            raise ValueError("end_time must be >= start_time with a finite duration.")
        if np.iscomplexobj(seed):
            raise ValueError("seed must contain real coordinates.")
        seed = np.array(_as_points(seed, self.field.dimension, name="seed"), copy=True)
        if seed.ndim != 1 or not np.all(np.isfinite(seed)):
            raise ValueError("seed must be one finite coordinate vector.")
        output_step = self.options.output_step
        if output_step is not None and (end_time - start_time) / output_step > 1_000_000:
            raise ValueError("output_step would produce more than one million samples.")
        points = [seed]
        times = [start_time]
        nfev = 0
        verified_time, verified_point = start_time, seed

        def finish(reason: TimeTerminationReason, message: str) -> PathlineResult:
            if verified_time > times[-1]:
                times.append(verified_time)
                points.append(verified_point.copy())
            return PathlineResult(np.array(points), np.array(times), reason, nfev, message)

        if self.domain is not None and not self.domain.contains(seed):
            return finish(TimeTerminationReason.SEED_OUTSIDE_DOMAIN, "seed is outside the domain")
        if start_time == end_time:
            return finish(TimeTerminationReason.TIME_END, "observation time reached")

        def velocity(time: float, point: FloatArray) -> FloatArray:
            nonlocal nfev
            nfev += 1
            raw = self.field.evaluate(point, time)
            if np.iscomplexobj(raw):
                raise ValueError("velocity must contain real components.")
            value = np.array(raw, dtype=float, copy=True)
            if value.shape != point.shape:
                raise ValueError("velocity shape must match the point shape.")
            if not np.all(np.isfinite(value)) or not np.all(np.isfinite(point)):
                raise _NonfiniteVelocity
            return value

        try:
            solver = DOP853(
                velocity, start_time, seed, end_time,
                max_step=self.options.max_step, rtol=self.options.rtol, atol=self.options.atol,
            )
        except _NonfiniteVelocity:
            return finish(TimeTerminationReason.NONFINITE_FIELD, "non-finite initial velocity")

        next_sample = 1
        for _ in range(self.options.max_steps):
            try:
                message = solver.step()
                if solver.status == "failed":
                    return finish(TimeTerminationReason.SOLVER_FAILURE, message or "solver failed")
                dense = solver.dense_output()
            except _NonfiniteVelocity:
                # DOP853 evaluates extra stages for dense output *after*
                # accepting a step. Its new endpoint is not yet domain-checked.
                return finish(TimeTerminationReason.NONFINITE_FIELD, "non-finite velocity during step")
            exit_point = _first_domain_exit(dense, self.domain) if self.domain is not None else None
            terminal_time, terminal_point = exit_point or (float(solver.t), solver.y.copy())
            if output_step is not None:
                while start_time + next_sample * output_step < terminal_time:
                    sample_time = start_time + next_sample * output_step
                    if sample_time > times[-1]:
                        times.append(sample_time)
                        points.append(np.array(dense(sample_time), copy=True))
                    next_sample += 1
            if (
                output_step is None or exit_point is not None or solver.status == "finished"
            ) and terminal_time > times[-1]:
                times.append(terminal_time)
                points.append(terminal_point)
            if exit_point is not None:
                return finish(TimeTerminationReason.DOMAIN_EXIT, "first outward domain crossing")
            verified_time, verified_point = float(solver.t), solver.y.copy()
            if solver.status == "finished":
                return finish(TimeTerminationReason.TIME_END, "observation time reached")
        return finish(TimeTerminationReason.SOLVER_FAILURE, "maximum accepted-step budget reached")

    def streakline(
        self, site: ArrayLike, release_times: ArrayLike, observation_time: float,
    ) -> StreaklineResult:
        """Release independent particles from a fixed site and observe them together."""
        observation_time = _finite_scalar(observation_time, "observation_time")
        if np.iscomplexobj(release_times):
            raise ValueError("release_times must be real.")
        releases = np.array(release_times, dtype=float, copy=True)
        if (
            releases.ndim != 1 or releases.size == 0 or not np.all(np.isfinite(releases))
            or np.any(np.diff(releases) <= 0) or np.any(releases > observation_time)
        ):
            raise ValueError("release_times must be nonempty, finite, strictly increasing and <= observation_time.")
        paths = [self.trace(site, float(release), observation_time) for release in releases]
        terminations = tuple(path.termination for path in paths)
        return StreaklineResult(
            positions=np.array([path.points[-1] for path in paths]),
            release_times=releases,
            arrival_times=np.array([path.times[-1] for path in paths]),
            terminations=terminations,
            valid=np.array([reason is TimeTerminationReason.TIME_END for reason in terminations]),
            nfev=sum(path.nfev for path in paths),
        )


__all__ = [
    "OscillatingFlowField", "PathlineResult", "PathlineTracer", "StreaklineResult",
    "TimeDependentField", "TimeOptions", "TimeTerminationReason",
]
