"""Behavioral tests for adaptive, event-aware field-line tracing."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vectorviz import (
    Domain,
    FieldLineTracer,
    SphericalExclusion,
    TerminationReason,
    TraceDirection,
    TraceOptions,
    UniformField,
    VectorField,
    trace_field_line,
)


class _LinearSinkField(VectorField):
    """A smooth 2D test field with a null at the origin."""

    @property
    def dimension(self) -> int:
        return 2

    def evaluate(self, points: ArrayLike) -> NDArray[np.float64]:
        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim == 0 or coordinates.shape[-1] != 2:
            raise ValueError("points must have shape (..., 2)")
        return -coordinates


def _options(**overrides: float | str | None) -> TraceOptions:
    values: dict[str, float | str | None] = {
        "max_arc_length": 5.0,
        "max_step": 0.04,
        "rtol": 1.0e-9,
        "atol": 1.0e-11,
        "null_threshold": 1.0e-10,
        "output_step": 0.025,
        "method": "DOP853",
    }
    values.update(overrides)
    return TraceOptions(**values)  # type: ignore[arg-type]


def test_uniform_field_traces_straight_to_both_domain_faces() -> None:
    result = trace_field_line(
        UniformField((3.0, 0.0)),
        seed=(0.25, -0.4),
        domain=Domain((-1.0, -1.0), (1.0, 1.0)),
        options=_options(),
        direction="both",
    )

    assert result.forward is not None
    assert result.backward is not None
    assert result.forward.termination is TerminationReason.DOMAIN_EXIT
    assert result.backward.termination is TerminationReason.DOMAIN_EXIT
    np.testing.assert_allclose(result.points[:, 1], -0.4, atol=2.0e-10)
    np.testing.assert_allclose(result.points[0], (-1.0, -0.4), atol=2.0e-8)
    np.testing.assert_allclose(result.points[-1], (1.0, -0.4), atol=2.0e-8)
    assert np.all(np.diff(result.arc_length) >= 0.0)
    np.testing.assert_allclose(result.arc_length[-1], 2.0, atol=2.0e-8)
    np.testing.assert_allclose(result.field_magnitude, 3.0)


def test_null_field_event_stops_on_threshold_surface() -> None:
    threshold = 0.075
    tracer = FieldLineTracer(
        _LinearSinkField(),
        domain=Domain((-1.0, -1.0), (1.0, 1.0)),
        options=_options(null_threshold=threshold),
    )

    result = tracer.trace((0.5, 0.0), direction=TraceDirection.FORWARD)

    assert result.forward is not None
    assert result.forward.termination is TerminationReason.NULL_FIELD
    np.testing.assert_allclose(result.forward.terminal_point, (threshold, 0.0), atol=2.0e-7)
    np.testing.assert_allclose(result.forward.field_magnitude[-1], threshold, atol=2.0e-7)


def test_seed_at_null_stops_without_solver_work() -> None:
    result = trace_field_line(
        UniformField((0.0, 0.0)),
        seed=(0.0, 0.0),
        options=_options(null_threshold=1.0e-6),
        direction="forward",
    )

    assert result.forward is not None
    assert result.forward.termination is TerminationReason.NULL_FIELD
    assert result.forward.nfev == 0
    np.testing.assert_array_equal(result.points, [[0.0, 0.0]])


def test_source_exclusion_terminates_at_surface() -> None:
    exclusion = SphericalExclusion((0.0, 0.0), 0.2)
    result = trace_field_line(
        UniformField((1.0, 0.0)),
        seed=(-0.8, 0.0),
        domain=Domain((-1.0, -1.0), (1.0, 1.0)),
        options=_options(),
        direction="forward",
        exclusions=(exclusion,),
    )

    assert result.forward is not None
    assert result.forward.termination is TerminationReason.EXCLUSION_HIT
    np.testing.assert_allclose(result.forward.terminal_point, (-0.2, 0.0), atol=2.0e-8)


def test_seed_inside_exclusion_is_rejected_without_integration() -> None:
    result = trace_field_line(
        UniformField((1.0, 0.0)),
        seed=(0.0, 0.0),
        exclusions=(SphericalExclusion((0.0, 0.0), 0.2),),
        direction="backward",
    )

    assert result.backward is not None
    assert result.backward.termination is TerminationReason.EXCLUSION_HIT
    assert result.backward.nfev == 0
