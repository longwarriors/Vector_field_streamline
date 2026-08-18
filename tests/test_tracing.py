"""Behavioral tests for adaptive, event-aware field-line tracing."""

from __future__ import annotations

import numpy as np
import pytest
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


class _RotationalField(VectorField):
    """The analytic unit-speed orbit benchmark F=(-y, x)."""

    @property
    def dimension(self) -> int:
        return 2

    def evaluate(self, points: ArrayLike) -> NDArray[np.float64]:
        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim == 0 or coordinates.shape[-1] != 2:
            raise ValueError("points must have shape (..., 2)")
        return np.stack((-coordinates[..., 1], coordinates[..., 0]), axis=-1)


class _ParabolicFlowField(VectorField):
    """A near-seed pass whose tangent turns away from the seed tangent."""

    @property
    def dimension(self) -> int:
        return 2

    def evaluate(self, points: ArrayLike) -> NDArray[np.float64]:
        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim == 0 or coordinates.shape[-1] != 2:
            raise ValueError("points must have shape (..., 2)")
        return np.stack((np.ones_like(coordinates[..., 0]), 4.0 * coordinates[..., 0]), axis=-1)


class _CubicFlowField(VectorField):
    """A smooth benchmark whose field lines satisfy y=x^3/3+C."""

    @property
    def dimension(self) -> int:
        return 2

    def evaluate(self, points: ArrayLike) -> NDArray[np.float64]:
        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim == 0 or coordinates.shape[-1] != 2:
            raise ValueError("points must have shape (..., 2)")
        return np.stack(
            (np.ones_like(coordinates[..., 0]), coordinates[..., 0] ** 2),
            axis=-1,
        )


class _NonfiniteField(VectorField):
    @property
    def dimension(self) -> int:
        return 2

    def evaluate(self, points: ArrayLike) -> NDArray[np.float64]:
        coordinates = np.asarray(points, dtype=float)
        return np.full_like(coordinates, np.nan)


class _BadShapeField(VectorField):
    @property
    def dimension(self) -> int:
        return 2

    def evaluate(self, points: ArrayLike) -> NDArray[np.float64]:
        coordinates = np.asarray(points, dtype=float)
        return np.zeros((*coordinates.shape[:-1], 3), dtype=float)


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
    assert np.count_nonzero(np.all(result.points == result.seed, axis=1)) == 1


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


def test_tracer_reuses_configuration_for_multiple_seeds() -> None:
    tracer = FieldLineTracer(
        UniformField((1.0, 0.0)),
        domain=Domain((-1.0, -1.0), (1.0, 1.0)),
        options=_options(),
    )

    first = tracer.trace((0.0, -0.5), direction="forward")
    second = tracer.trace((0.0, 0.5), direction="forward")

    np.testing.assert_allclose(first.points[-1], (1.0, -0.5), atol=2.0e-8)
    np.testing.assert_allclose(second.points[-1], (1.0, 0.5), atol=2.0e-8)
    assert first.terminations == {TraceDirection.FORWARD: TerminationReason.DOMAIN_EXIT}
    assert second.terminations == {TraceDirection.FORWARD: TerminationReason.DOMAIN_EXIT}


def test_outside_and_nonfinite_seeds_stop_without_solver_work() -> None:
    outside = trace_field_line(
        UniformField((1.0, 0.0)),
        seed=(2.0, 0.0),
        domain=Domain((-1.0, -1.0), (1.0, 1.0)),
        direction="forward",
    )
    nonfinite = trace_field_line(
        _NonfiniteField(),
        seed=(0.0, 0.0),
        direction="forward",
    )

    assert outside.forward is not None
    assert outside.forward.termination is TerminationReason.SEED_OUTSIDE_DOMAIN
    assert outside.forward.nfev == 0
    assert nonfinite.forward is not None
    assert nonfinite.forward.termination is TerminationReason.NONFINITE_FIELD
    assert nonfinite.forward.nfev == 0


def test_tracer_rejects_invalid_collaborators_seeds_and_field_output() -> None:
    with pytest.raises(TypeError, match="VectorField"):
        FieldLineTracer(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dimensions must match"):
        FieldLineTracer(UniformField((1.0, 0.0)), domain=Domain((0.0,), (1.0,)))
    with pytest.raises(ValueError, match="dimensions must match"):
        FieldLineTracer(
            UniformField((1.0, 0.0)),
            exclusions=(SphericalExclusion((0.0, 0.0, 0.0), 0.1),),
        )

    tracer = FieldLineTracer(UniformField((1.0, 0.0)))
    with pytest.raises(ValueError, match="exactly one point"):
        tracer.trace(((0.0, 0.0), (1.0, 0.0)))
    with pytest.raises(ValueError, match="finite"):
        tracer.trace((np.nan, 0.0))
    with pytest.raises(ValueError, match="violated its contract"):
        FieldLineTracer(_BadShapeField()).trace((0.0, 0.0), direction="forward")


def test_trace_direction_coercion_and_stable_string_values() -> None:
    assert TraceDirection.coerce(TraceDirection.BACKWARD) is TraceDirection.BACKWARD
    assert TraceDirection.coerce("FORWARD") is TraceDirection.FORWARD
    assert str(TerminationReason.CLOSED_LOOP) == "closed_loop"
    with pytest.raises(ValueError, match="forward, backward, both"):
        TraceDirection.coerce("sideways")


def test_rotational_field_converges_to_analytic_orbit_when_controls_are_refined() -> None:
    field = _RotationalField()
    arc_length = 1.5
    expected = np.array([np.cos(arc_length), np.sin(arc_length)])

    coarse = trace_field_line(
        field,
        seed=(1.0, 0.0),
        options=TraceOptions(
            max_arc_length=arc_length,
            max_step=1.0,
            rtol=1.0e-2,
            atol=1.0e-4,
        ),
        direction="forward",
    )
    refined = trace_field_line(
        field,
        seed=(1.0, 0.0),
        options=TraceOptions(
            max_arc_length=arc_length,
            max_step=0.05,
            rtol=1.0e-10,
            atol=1.0e-12,
        ),
        direction="forward",
    )

    assert coarse.forward is not None
    assert refined.forward is not None
    coarse_error = np.linalg.norm(coarse.forward.terminal_point - expected)
    refined_error = np.linalg.norm(refined.forward.terminal_point - expected)
    # DOP853's order-eight refinement should improve this smooth analytic orbit
    # by far more than two orders without relying on pixel-level comparisons.
    assert refined_error < coarse_error / 100.0
    assert refined_error < 1.0e-10


def test_tangent_residual_converges_with_output_spacing() -> None:
    field = _CubicFlowField()

    def maximum_residual(output_step: float) -> float:
        result = trace_field_line(
            field,
            seed=(-1.0, 0.0),
            options=_options(
                max_arc_length=2.0,
                max_step=0.02,
                output_step=output_step,
            ),
            direction="forward",
        )
        segments = np.diff(result.points, axis=0)
        midpoints = 0.5 * (result.points[:-1] + result.points[1:])
        vectors = field.evaluate(midpoints)
        cross = segments[:, 0] * vectors[:, 1] - segments[:, 1] * vectors[:, 0]
        residual = np.abs(cross) / (
            np.linalg.norm(segments, axis=1) * np.linalg.norm(vectors, axis=1)
        )
        return float(np.max(residual))

    coarse = maximum_residual(0.2)
    refined = maximum_residual(0.05)

    # A chord's midpoint tangent error is second order in output spacing for
    # this cubic curve: reducing spacing fourfold should improve it ~16-fold.
    assert refined < coarse / 10.0
    assert refined < 3.0e-4


def test_rotational_field_stops_after_one_closed_orbit() -> None:
    tolerance = 2.0e-3
    result = trace_field_line(
        _RotationalField(),
        seed=(1.0, 0.0),
        options=_options(
            max_arc_length=8.0,
            closure_tolerance=tolerance,
            closure_min_arc_length=4.0,
            closure_tangent_cosine=0.99,
        ),
        direction="forward",
    )

    assert result.forward is not None
    assert result.forward.termination is TerminationReason.CLOSED_LOOP
    assert np.linalg.norm(result.forward.terminal_point - result.seed) <= tolerance * 1.001
    np.testing.assert_allclose(result.forward.arc_length[-1], 2.0 * np.pi, atol=3.0e-3)
    # Closure tolerance is a geometric acceptance test, not an implicit step
    # size. This bound catches the former O(1 / tolerance) RHS explosion.
    assert result.forward.nfev < 5_000


def test_closure_detection_is_opt_in() -> None:
    result = trace_field_line(
        _RotationalField(),
        seed=(1.0, 0.0),
        options=_options(max_arc_length=7.0),
        direction="forward",
    )

    assert result.forward is not None
    assert result.forward.termination is TerminationReason.MAX_ARC_LENGTH


def test_closure_detection_does_not_override_domain_exit() -> None:
    result = trace_field_line(
        UniformField((1.0, 0.0)),
        seed=(0.99, 0.0),
        domain=Domain((-1.0, -1.0), (1.0, 1.0)),
        options=_options(
            closure_tolerance=0.02,
            closure_min_arc_length=0.1,
            closure_tangent_cosine=0.99,
        ),
        direction="forward",
    )

    assert result.forward is not None
    assert result.forward.termination is TerminationReason.DOMAIN_EXIT


def test_physical_event_at_closure_chunk_boundary_is_not_lost() -> None:
    result = trace_field_line(
        UniformField((1.0, 0.0)),
        seed=(0.0, 0.0),
        domain=Domain((-1.0, -1.0), (0.64, 1.0)),
        options=_options(
            closure_tolerance=0.02,
            closure_min_arc_length=0.1,
            closure_tangent_cosine=0.99,
        ),
        direction="forward",
    )

    assert result.forward is not None
    assert result.forward.termination is TerminationReason.DOMAIN_EXIT
    np.testing.assert_allclose(result.forward.terminal_point, (0.64, 0.0), atol=2.0e-8)


def test_physical_event_wins_when_it_coincides_with_closure_candidate() -> None:
    candidate_x = 1.0 + np.sqrt(14.0) / 4.0
    candidate_y = 2.0 * candidate_x**2
    result = trace_field_line(
        _ParabolicFlowField(),
        seed=(-2.0, 8.0),
        domain=Domain((-3.0, -1.0), (candidate_x, 9.0)),
        options=_options(
            max_arc_length=20.0,
            closure_tolerance=4.1,
            closure_min_arc_length=9.0,
            closure_tangent_cosine=-1.0,
        ),
        direction="forward",
    )

    assert result.forward is not None
    assert result.forward.termination is TerminationReason.DOMAIN_EXIT
    np.testing.assert_allclose(
        result.forward.terminal_point,
        (candidate_x, candidate_y),
        atol=2.0e-8,
    )


def test_closed_loop_supports_adaptive_solver_output_across_chunks() -> None:
    result = trace_field_line(
        _RotationalField(),
        seed=(1.0, 0.0),
        options=_options(
            max_arc_length=8.0,
            output_step=None,
            closure_tolerance=2.0e-3,
            closure_min_arc_length=4.0,
            closure_tangent_cosine=0.99,
        ),
        direction="forward",
    )

    assert result.forward is not None
    assert result.forward.termination is TerminationReason.CLOSED_LOOP
    assert np.all(np.diff(result.forward.arc_length) >= 0.0)
    assert np.linalg.norm(result.forward.terminal_point - result.seed) <= 2.0e-3
    np.testing.assert_allclose(result.forward.arc_length[-1], 2.0 * np.pi, atol=3.0e-3)


def test_closure_detection_rejects_a_near_pass_with_wrong_tangent() -> None:
    seed = np.array((-2.0, 8.0))
    return_point = np.array((2.0, 8.0))
    seed_tangent = _ParabolicFlowField().evaluate(seed)
    return_tangent = _ParabolicFlowField().evaluate(return_point)
    tangent_cosine = float(
        np.dot(seed_tangent, return_tangent)
        / (np.linalg.norm(seed_tangent) * np.linalg.norm(return_tangent))
    )
    assert np.linalg.norm(return_point - seed) < 4.1
    assert tangent_cosine < 0.99

    result = trace_field_line(
        _ParabolicFlowField(),
        seed=seed,
        options=_options(
            max_arc_length=20.0,
            closure_tolerance=4.1,
            closure_min_arc_length=9.0,
            closure_tangent_cosine=0.99,
        ),
        direction="forward",
    )

    assert result.forward is not None
    assert result.forward.termination is TerminationReason.MAX_ARC_LENGTH


def test_bidirectional_closed_orbit_is_not_duplicated_in_merged_points() -> None:
    result = trace_field_line(
        _RotationalField(),
        seed=(1.0, 0.0),
        options=_options(
            max_arc_length=8.0,
            closure_tolerance=2.0e-3,
            closure_min_arc_length=4.0,
            closure_tangent_cosine=0.99,
        ),
        direction="both",
    )

    assert result.forward is not None
    assert result.backward is not None
    assert result.forward.termination is TerminationReason.CLOSED_LOOP
    assert result.backward.termination is TerminationReason.CLOSED_LOOP
    assert result.points.shape[0] < result.forward.points.shape[0] + result.backward.points.shape[0]
    np.testing.assert_allclose(result.points[0], result.seed)
    assert np.linalg.norm(result.points[-1] - result.seed) <= 2.0e-3 * 1.001
    np.testing.assert_allclose(result.arc_length[-1], 2.0 * np.pi, atol=3.0e-3)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"closure_tolerance": 0.0, "closure_min_arc_length": 1.0}, "closure_tolerance"),
        ({"closure_tolerance": 1.0e-3}, "closure_min_arc_length"),
        (
            {"closure_tolerance": 1.0e-3, "closure_min_arc_length": 5.0},
            "less than max_arc_length",
        ),
        (
            {"closure_tolerance": 0.1, "closure_min_arc_length": 0.2},
            "greater than twice closure_tolerance",
        ),
        ({"closure_tangent_cosine": 1.0}, "closure_tangent_cosine"),
        ({"closure_tangent_cosine": 1.01}, "closure_tangent_cosine"),
    ],
)
def test_trace_options_reject_invalid_closure_controls(
    overrides: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _options(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_arc_length": 0.0}, "max_arc_length"),
        ({"max_step": np.inf}, "max_step"),
        ({"first_step": -1.0}, "first_step"),
        ({"rtol": 0.0}, "rtol"),
        ({"atol": np.nan}, "atol"),
        ({"null_threshold": -1.0}, "null_threshold"),
        ({"output_step": 0.0}, "output_step"),
        ({"method": ""}, "method"),
    ],
)
def test_trace_options_reject_invalid_numerical_controls(
    overrides: dict[str, float | str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _options(**overrides)
