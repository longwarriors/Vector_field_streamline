"""Adaptive, event-aware field-line integration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from .core import Domain, FloatArray, SphericalExclusion, VectorField, _as_points


class TraceDirection(StrEnum):
    """Requested orientation relative to the field vector."""

    FORWARD = "forward"
    BACKWARD = "backward"
    BOTH = "both"

    @classmethod
    def coerce(cls, value: TraceDirection | str) -> TraceDirection:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as error:
            choices = ", ".join(direction.value for direction in cls)
            raise ValueError(f"direction must be one of: {choices}.") from error


class TerminationReason(StrEnum):
    """Why integration of one trace branch stopped."""

    DOMAIN_EXIT = "domain_exit"
    NULL_FIELD = "null_field"
    NONFINITE_FIELD = "nonfinite_field"
    MAX_ARC_LENGTH = "max_arc_length"
    SOLVER_FAILURE = "solver_failure"
    SEED_OUTSIDE_DOMAIN = "seed_outside_domain"
    EXCLUSION_HIT = "exclusion_hit"


@dataclass(frozen=True, slots=True)
class TraceOptions:
    """Numerical controls for normalized field-line integration.

    ``null_threshold`` is an absolute field-magnitude threshold in the field's
    own units.  It defines an event surface; the tracer never adds an epsilon
    to the norm or invents a direction inside this surface.
    """

    max_arc_length: float = 20.0
    max_step: float = 0.1
    first_step: float | None = None
    rtol: float = 1.0e-7
    atol: float = 1.0e-9
    null_threshold: float = 1.0e-12
    output_step: float | None = None
    method: str = "DOP853"

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_arc_length) or self.max_arc_length <= 0:
            raise ValueError("max_arc_length must be finite and positive.")
        if not np.isfinite(self.max_step) or self.max_step <= 0:
            raise ValueError("max_step must be finite and positive.")
        if self.first_step is not None and (
            not np.isfinite(self.first_step) or self.first_step <= 0
        ):
            raise ValueError("first_step must be finite and positive when supplied.")
        if not np.isfinite(self.rtol) or self.rtol <= 0:
            raise ValueError("rtol must be finite and positive.")
        if not np.isfinite(self.atol) or self.atol <= 0:
            raise ValueError("atol must be finite and positive.")
        if not np.isfinite(self.null_threshold) or self.null_threshold < 0:
            raise ValueError("null_threshold must be finite and non-negative.")
        if self.output_step is not None and (
            not np.isfinite(self.output_step) or self.output_step <= 0
        ):
            raise ValueError("output_step must be finite and positive when supplied.")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a non-empty solve_ivp method name.")


@dataclass(frozen=True, slots=True)
class TraceBranch:
    """One branch, ordered from its seed toward its terminal point."""

    direction: TraceDirection
    points: FloatArray
    arc_length: FloatArray
    field_magnitude: FloatArray
    termination: TerminationReason
    message: str
    nfev: int

    @property
    def terminal_point(self) -> FloatArray:
        return self.points[-1]


@dataclass(frozen=True, slots=True)
class TraceResult:
    """A complete trace and its independently inspectable branches.

    For a bidirectional trace, ``points`` are ordered from the backward terminal
    point through the seed to the forward terminal point. ``arc_length`` is the
    cumulative geometric length along that displayed ordering.
    """

    seed: FloatArray
    points: FloatArray
    arc_length: FloatArray
    field_magnitude: FloatArray
    forward: TraceBranch | None
    backward: TraceBranch | None

    @property
    def terminations(self) -> dict[TraceDirection, TerminationReason]:
        result: dict[TraceDirection, TerminationReason] = {}
        if self.forward is not None:
            result[TraceDirection.FORWARD] = self.forward.termination
        if self.backward is not None:
            result[TraceDirection.BACKWARD] = self.backward.termination
        return result


def _cumulative_length(points: FloatArray) -> FloatArray:
    if points.shape[0] <= 1:
        return np.zeros(points.shape[0], dtype=float)
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segments)))


class FieldLineTracer:
    """Trace geometric field lines with adaptive normalized integration.

    The solved equation is `dx/ds = ±F(x)/||F(x)||`.

    Thus integration length is decoupled from field magnitude. At or below the
    configured null threshold the right-hand side is exactly zero and a
    terminal event is located; no epsilon regularization carries a line through
    a point where its direction is undefined.
    """

    def __init__(
        self,
        field: VectorField,
        domain: Domain | None = None,
        options: TraceOptions | None = None,
        exclusions: Iterable[SphericalExclusion] = (),
    ) -> None:
        if not isinstance(field, VectorField):
            raise TypeError("field must implement VectorField.")
        if domain is not None and domain.dimension != field.dimension:
            raise ValueError("domain and field dimensions must match.")
        exclusion_values = tuple(exclusions)
        if any(exclusion.dimension != field.dimension for exclusion in exclusion_values):
            raise ValueError("exclusion geometry and field dimensions must match.")
        self.field = field
        self.domain = domain
        self.options = options if options is not None else TraceOptions()
        self.exclusions = exclusion_values

    def _field_at(self, point: FloatArray) -> tuple[FloatArray, float]:
        vector = np.asarray(self.field.evaluate(point), dtype=float)
        if vector.shape != (self.field.dimension,):
            raise ValueError(
                "VectorField.evaluate violated its contract: "
                f"expected {(self.field.dimension,)}, got {vector.shape}."
            )
        return vector, float(np.linalg.norm(vector))

    def _branch_without_integration(
        self,
        seed: FloatArray,
        direction: TraceDirection,
        reason: TerminationReason,
        message: str,
    ) -> TraceBranch:
        _, magnitude = self._field_at(seed)
        return TraceBranch(
            direction=direction,
            points=seed[np.newaxis, :].copy(),
            arc_length=np.zeros(1, dtype=float),
            field_magnitude=np.array([magnitude], dtype=float),
            termination=reason,
            message=message,
            nfev=0,
        )

    def _integrate_branch(self, seed: FloatArray, direction: TraceDirection) -> TraceBranch:
        sign = 1.0 if direction is TraceDirection.FORWARD else -1.0
        options = self.options

        seed_vector, seed_magnitude = self._field_at(seed)
        if not np.all(np.isfinite(seed_vector)) or not np.isfinite(seed_magnitude):
            return self._branch_without_integration(
                seed,
                direction,
                TerminationReason.NONFINITE_FIELD,
                "the field is non-finite at the seed",
            )
        if seed_magnitude <= options.null_threshold:
            return self._branch_without_integration(
                seed,
                direction,
                TerminationReason.NULL_FIELD,
                "the field direction is undefined at the seed",
            )

        encountered_nonfinite = False

        def rhs(_parameter: float, point: FloatArray) -> FloatArray:
            nonlocal encountered_nonfinite
            vector, magnitude = self._field_at(point)
            if not np.all(np.isfinite(vector)) or not np.isfinite(magnitude):
                encountered_nonfinite = True
                return np.zeros_like(point)
            if magnitude <= options.null_threshold:
                # Exactly stop motion within the null event surface.  This is
                # intentionally not epsilon regularization.
                return np.zeros_like(point)
            return sign * vector / magnitude

        def null_event(_parameter: float, point: FloatArray) -> float:
            vector, magnitude = self._field_at(point)
            if not np.all(np.isfinite(vector)) or not np.isfinite(magnitude):
                return 1.0
            return magnitude - options.null_threshold

        null_event.terminal = True  # type: ignore[attr-defined]
        null_event.direction = -1.0  # type: ignore[attr-defined]

        def finite_event(_parameter: float, point: FloatArray) -> float:
            vector, magnitude = self._field_at(point)
            return 1.0 if np.all(np.isfinite(vector)) and np.isfinite(magnitude) else -1.0

        finite_event.terminal = True  # type: ignore[attr-defined]
        finite_event.direction = -1.0  # type: ignore[attr-defined]

        events: list[Any] = []
        event_reasons: list[TerminationReason] = []
        if self.domain is not None:

            def domain_event(_parameter: float, point: FloatArray) -> float:
                return float(self.domain.margin(point))

            domain_event.terminal = True  # type: ignore[attr-defined]
            domain_event.direction = -1.0  # type: ignore[attr-defined]
            events.append(domain_event)
            event_reasons.append(TerminationReason.DOMAIN_EXIT)

        events.append(null_event)
        event_reasons.append(TerminationReason.NULL_FIELD)
        events.append(finite_event)
        event_reasons.append(TerminationReason.NONFINITE_FIELD)
        if self.exclusions:

            def exclusion_event(_parameter: float, point: FloatArray) -> float:
                return float(min(exclusion.margin(point) for exclusion in self.exclusions))

            exclusion_event.terminal = True  # type: ignore[attr-defined]
            exclusion_event.direction = -1.0  # type: ignore[attr-defined]
            events.append(exclusion_event)
            event_reasons.append(TerminationReason.EXCLUSION_HIT)

        solution = solve_ivp(
            rhs,
            (0.0, options.max_arc_length),
            seed,
            method=options.method,
            rtol=options.rtol,
            atol=options.atol,
            max_step=options.max_step,
            first_step=options.first_step,
            dense_output=options.output_step is not None,
            events=events,
        )

        if options.output_step is not None and solution.sol is not None:
            end_parameter = float(solution.t[-1])
            parameters = np.arange(0.0, end_parameter, options.output_step)
            if parameters.size == 0 or parameters[0] != 0.0:
                parameters = np.insert(parameters, 0, 0.0)
            if not np.isclose(parameters[-1], end_parameter):
                parameters = np.append(parameters, end_parameter)
            else:
                parameters[-1] = end_parameter
            points = np.asarray(solution.sol(parameters).T, dtype=float)
        else:
            points = np.asarray(solution.y.T, dtype=float)

        magnitudes = np.linalg.norm(self.field.evaluate(points), axis=-1)
        if solution.status == 1:
            triggered_index = next(
                (
                    index
                    for index, event_times in enumerate(solution.t_events)
                    if event_times.size > 0
                ),
                None,
            )
            termination = (
                event_reasons[triggered_index]
                if triggered_index is not None
                else TerminationReason.SOLVER_FAILURE
            )
        elif solution.status == 0:
            termination = TerminationReason.MAX_ARC_LENGTH
        else:
            termination = TerminationReason.SOLVER_FAILURE

        if encountered_nonfinite and termination is TerminationReason.SOLVER_FAILURE:
            termination = TerminationReason.NONFINITE_FIELD

        message = solution.message
        if termination is TerminationReason.NONFINITE_FIELD:
            message = "integration encountered a non-finite field value"

        return TraceBranch(
            direction=direction,
            points=points,
            arc_length=_cumulative_length(points),
            field_magnitude=np.asarray(magnitudes, dtype=float),
            termination=termination,
            message=message,
            nfev=int(solution.nfev),
        )

    def trace(
        self,
        seed: ArrayLike,
        direction: TraceDirection | str = TraceDirection.BOTH,
    ) -> TraceResult:
        """Trace from ``seed`` in one direction or in both directions."""

        seed_array = _as_points(seed, self.field.dimension, name="seed")
        if seed_array.ndim != 1:
            raise ValueError("seed must describe exactly one point.")
        if not np.all(np.isfinite(seed_array)):
            raise ValueError("seed coordinates must be finite.")
        requested = TraceDirection.coerce(direction)

        outside = self.domain is not None and not bool(self.domain.contains(seed_array))
        inside_exclusion = any(exclusion.margin(seed_array) < 0 for exclusion in self.exclusions)

        def build_branch(branch_direction: TraceDirection) -> TraceBranch:
            if outside:
                return self._branch_without_integration(
                    seed_array,
                    branch_direction,
                    TerminationReason.SEED_OUTSIDE_DOMAIN,
                    "the seed is outside the tracing domain",
                )
            if inside_exclusion:
                return self._branch_without_integration(
                    seed_array,
                    branch_direction,
                    TerminationReason.EXCLUSION_HIT,
                    "the seed is inside an excluded source region",
                )
            return self._integrate_branch(seed_array, branch_direction)

        forward = (
            build_branch(TraceDirection.FORWARD)
            if requested in (TraceDirection.FORWARD, TraceDirection.BOTH)
            else None
        )
        backward = (
            build_branch(TraceDirection.BACKWARD)
            if requested in (TraceDirection.BACKWARD, TraceDirection.BOTH)
            else None
        )

        if forward is not None and backward is not None:
            points = np.concatenate((backward.points[:0:-1], forward.points), axis=0)
        elif forward is not None:
            points = forward.points.copy()
        elif backward is not None:
            points = backward.points.copy()
        else:  # pragma: no cover - guarded by TraceDirection
            raise RuntimeError("no trace branch was requested")

        magnitudes = np.linalg.norm(self.field.evaluate(points), axis=-1)
        return TraceResult(
            seed=seed_array.copy(),
            points=points,
            arc_length=_cumulative_length(points),
            field_magnitude=np.asarray(magnitudes, dtype=float),
            forward=forward,
            backward=backward,
        )


def trace_field_line(
    field: VectorField,
    seed: ArrayLike,
    *,
    domain: Domain | None = None,
    options: TraceOptions | None = None,
    direction: TraceDirection | str = TraceDirection.BOTH,
    exclusions: Iterable[SphericalExclusion] = (),
) -> TraceResult:
    """Convenience wrapper around `FieldLineTracer`."""

    return FieldLineTracer(
        field,
        domain=domain,
        options=options,
        exclusions=exclusions,
    ).trace(seed, direction=direction)


__all__ = [
    "FieldLineTracer",
    "TerminationReason",
    "TraceBranch",
    "TraceDirection",
    "TraceOptions",
    "TraceResult",
    "trace_field_line",
]
