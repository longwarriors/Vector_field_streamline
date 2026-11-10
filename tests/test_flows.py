"""Independent physical-time and common-observation-time flow benchmarks."""

from __future__ import annotations

import numpy as np
import pytest

from vectorviz import (
    Domain,
    FieldLineTracer,
    OscillatingFlowField,
    PathlineTracer,
    TimeDependentField,
    TimeOptions,
    TimeTerminationReason,
    TraceOptions,
)


class CallableFlow(TimeDependentField):
    def __init__(self, function):
        self.function = function

    @property
    def dimension(self):
        return 2

    def evaluate(self, points, time):
        return self.function(np.asarray(points), time)


def test_oscillating_field_batch_shape_freezing_and_coordinate_policy() -> None:
    field = OscillatingFlowField(2, -0.3, 0.7, 2)
    points = np.zeros((2, 3, 2))
    expected = np.broadcast_to((2, 0.4), points.shape)
    np.testing.assert_allclose(field.evaluate(points, np.pi / 4), expected, atol=1e-15)
    frozen = field.freeze(np.pi / 4)
    assert frozen.dimension == 2
    np.testing.assert_array_equal(frozen(points), field.evaluate(points, np.pi / 4))
    assert field.evaluate(np.empty((0, 2)), 0).shape == (0, 2)
    assert np.isnan(field.evaluate(((np.nan, 0), (0, np.inf)), 0)).all()
    assert np.isfinite(field.evaluate((1e300, -1e300), 0)).all()
    overflow = OscillatingFlowField(angular_frequency=1e308)
    assert np.isnan(overflow.evaluate((0, 0), 1e308)).all()
    with pytest.raises(ValueError, match="shape"):
        field.evaluate((1, 2, 3), 0)
    with pytest.raises(ValueError, match="real"):
        field.evaluate((1j, 0), 0)
    with pytest.raises(ValueError, match="time"):
        field.freeze(np.nan)


@pytest.mark.parametrize("time", [np.inf, np.nan, True, 1j, [1], "1"])
def test_field_rejects_non_scalar_or_nonfinite_time(time) -> None:
    with pytest.raises(ValueError, match="time"):
        OscillatingFlowField().evaluate((0, 0), time)


@pytest.mark.parametrize("kwargs", [
    {"longitudinal_speed": np.nan}, {"transverse_speed": np.inf}, {"amplitude": True},
    {"amplitude": 1j}, {"angular_frequency": -1}, {"angular_frequency": [1]},
])
def test_field_rejects_invalid_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        OscillatingFlowField(**kwargs)


def test_zero_amplitude_steady_limit_ignores_overflowing_unused_phase() -> None:
    field = OscillatingFlowField(2, -0.3, 0, 1e308)
    np.testing.assert_array_equal(field.evaluate((0, 0), 1e308), (2, -0.3))


def test_pathline_uses_speed_and_physical_time_against_independent_integrals() -> None:
    seed = np.array((-1.2, 0.2))
    start, end = 0.4, 2.7
    field = OscillatingFlowField(1.7, -0.2, 0.8, 1.3)
    result = PathlineTracer(field).trace(seed, start, end)
    t = result.times
    expected = np.column_stack((
        seed[0] + 1.7 * (t - start),
        seed[1] - 0.2 * (t - start) + 0.8 / 1.3 * (np.cos(1.3 * start) - np.cos(1.3 * t)),
    ))
    np.testing.assert_allclose(result.points, expected, atol=2e-11, rtol=0)
    assert result.termination is TimeTerminationReason.TIME_END
    assert (t[0], t[-1]) == (start, end)
    assert np.all(np.diff(t) > 0)
    np.testing.assert_allclose(np.diff(t)[:-1], 0.05, atol=1e-15)
    assert result.nfev > 0
    seed[:] = 99
    np.testing.assert_array_equal(result.points[0], (-1.2, 0.2))
    # Normalized tracing would travel only 2.3 units. Physical x displacement
    # alone is 3.91 m, and doubling the entire velocity doubles displacement.
    faster = PathlineTracer(OscillatingFlowField(3.4, -0.4, 1.6, 1.3)).trace((-1.2, 0.2), start, end)
    np.testing.assert_allclose(faster.points[-1] - faster.points[0], 2 * (expected[-1] - expected[0]), atol=1e-11)


def test_streakline_is_common_time_snapshot_and_three_unsteady_curves_differ() -> None:
    field = OscillatingFlowField()
    site = np.array((-2.0, 0.0))
    releases = np.linspace(0, 4, 33)
    tracer = PathlineTracer(field)
    streak = tracer.streakline(site, releases, 4)
    expected = np.column_stack((-2 + 4 - releases, 0.6 * (np.cos(releases) - np.cos(4))))
    np.testing.assert_allclose(streak.positions, expected, atol=2e-12)
    np.testing.assert_array_equal(streak.arrival_times, np.full(33, 4))
    assert streak.valid.all()
    assert set(streak.terminations) == {TimeTerminationReason.TIME_END}
    path = tracer.trace(site, 0, 4)
    # Compare at shared x=0 m: age=2 s on the streakline, t=2 s along
    # the pathline, and a linear frozen streamline at T=4 s.
    streak_y = streak.positions[16, 1]
    path_y = path.points[np.argmin(abs(path.times - 2)), 1]
    stream_y = 0.6 * np.sin(4) * 2
    assert min(abs(streak_y - path_y), abs(streak_y - stream_y), abs(path_y - stream_y)) > 0.1
    line = FieldLineTracer(field.freeze(4), options=TraceOptions(max_arc_length=4)).trace(site)
    np.testing.assert_allclose(line.points[:, 1], 0.6 * np.sin(4) * (line.points[:, 0] + 2), atol=2e-14)
    releases[:] = 99
    assert streak.release_times[0] == 0


def test_steady_limit_has_same_geometric_curve_for_all_three_definitions() -> None:
    field = OscillatingFlowField(2, 0.5, 0)
    site = (-2, -0.3)
    tracer = PathlineTracer(field)
    path = tracer.trace(site, 0.5, 2)
    streak = tracer.streakline(site, np.linspace(0.5, 2, 9), 2)
    line = FieldLineTracer(field.freeze(1.2), options=TraceOptions(max_arc_length=5)).trace(site)
    for points in (path.points, streak.positions, line.points):
        np.testing.assert_allclose(points[:, 1] + 0.3, 0.25 * (points[:, 0] + 2), atol=2e-14)
    np.testing.assert_allclose(streak.positions[0], path.points[-1], atol=1e-14)


def test_zero_velocity_does_not_stop_a_particle_before_it_starts_moving() -> None:
    field = CallableFlow(lambda p, t: np.array((t, 0)))
    result = PathlineTracer(field).trace((0, 0), 0, 2)
    assert result.termination is TimeTerminationReason.TIME_END
    np.testing.assert_allclose(result.points[:, 0], result.times**2 / 2, atol=1e-13)
    stationary = PathlineTracer(OscillatingFlowField(0, 0, 0)).trace((1, 2), 0, 2)
    assert stationary.termination is TimeTerminationReason.TIME_END
    np.testing.assert_array_equal(stationary.points, np.broadcast_to((1, 2), stationary.points.shape))


def test_domain_exit_time_not_arc_length_and_streak_masks_escaped_particles() -> None:
    tracer = PathlineTracer(
        OscillatingFlowField(2, 0, 0), domain=Domain((-1, -1), (1, 1)),
        options=TimeOptions(output_step=0.13),
    )
    result = tracer.trace((0, 0), 0.2, 2)
    assert result.termination is TimeTerminationReason.DOMAIN_EXIT
    np.testing.assert_allclose(result.times[-1], 0.7, atol=3e-15)
    np.testing.assert_array_equal(result.points[-1], (1, 0))
    assert np.all(np.diff(result.times) > 0)
    streak = tracer.streakline((0, 0), (0, 0.4, 0.8, 1), 1)
    np.testing.assert_array_equal(streak.valid, (False, False, True, True))
    np.testing.assert_allclose(streak.arrival_times, (0.5, 0.9, 1, 1), atol=1e-14)
    np.testing.assert_allclose(streak.positions[:, 0], (1, 1, 0.4, 0), atol=1e-14)


@pytest.mark.parametrize("seed,speed,reason,expected", [
    ((-1, 0), 1, "time_end", (-0.5, 0)),
    ((1, 0), 1, "domain_exit", (1, 0)),
    ((1, 0), -1, "time_end", (0.5, 0)),
    ((1, 0), 0, "time_end", (1, 0)),
    ((-1, 1), 1, "time_end", (-0.5, 1)),
    ((1.01, 0), -1, "seed_outside_domain", (1.01, 0)),
])
def test_closed_domain_boundary_can_start_inward_or_remain_stationary(seed, speed, reason, expected) -> None:
    tracer = PathlineTracer(OscillatingFlowField(speed, 0, 0), domain=Domain((-1, -1), (1, 1)))
    result = tracer.trace(seed, 0, 0.5)
    assert result.termination == reason
    np.testing.assert_allclose(result.points[-1], expected, atol=3e-15)


def test_domain_detects_leaving_and_reentering_inside_an_accepted_step() -> None:
    # x(t)=4t(1-t): both t=0 and t=1 are inside x<=0.5, but the first
    # outward crossing is (1-sqrt(1/2))/2. The integrator may take large
    # steps because the quadratic solution has zero truncation error.
    flow = CallableFlow(lambda p, t: np.array((4 - 8 * t, 0)))
    result = PathlineTracer(
        flow, domain=Domain((-1, -1), (0.5, 1)),
        options=TimeOptions(max_step=1, output_step=None),
    ).trace((0, 0), 0, 1)
    assert result.termination is TimeTerminationReason.DOMAIN_EXIT
    np.testing.assert_allclose(result.times[-1], (1 - np.sqrt(0.5)) / 2, atol=1e-13)
    np.testing.assert_array_equal(result.points[-1], (0.5, 0))


def test_tangent_contact_is_not_an_exit_and_endpoint_contact_reaches_time_end() -> None:
    flow = CallableFlow(lambda p, t: np.array((2 - 2 * t, 0)))
    result = PathlineTracer(flow, domain=Domain((-1, -1), (1, 1))).trace((0, 0), 0, 2)
    assert result.termination is TimeTerminationReason.TIME_END
    np.testing.assert_allclose(result.points[-1], (0, 0), atol=1e-14)
    straight = PathlineTracer(OscillatingFlowField(1, 0, 0), domain=Domain((-1, -1), (1, 1)))
    exact = straight.trace((0, 0), 0, 1)
    assert exact.termination is TimeTerminationReason.TIME_END
    np.testing.assert_allclose(exact.points[-1], (1, 0), atol=1e-14)


def test_nonfinite_velocity_preserves_last_accepted_finite_state() -> None:
    flow = CallableFlow(lambda p, t: np.array((1, np.nan if t > 0.5 else 0)))
    result = PathlineTracer(flow, options=TimeOptions(output_step=2)).trace((0, 0), 0, 2)
    assert result.termination is TimeTerminationReason.NONFINITE_FIELD
    assert 0 < result.times[-1] <= 0.5
    np.testing.assert_allclose(result.points[-1], (result.times[-1], 0), atol=1e-14)
    assert np.isfinite(result.points).all()
    first = PathlineTracer(CallableFlow(lambda p, t: (np.inf, 0))).trace((0, 0), 0, 1)
    assert first.termination is TimeTerminationReason.NONFINITE_FIELD
    assert first.times.tolist() == [0]


def test_accepted_step_budget_returns_solver_failure_with_auditable_last_point() -> None:
    result = PathlineTracer(
        OscillatingFlowField(1, 0, 0), options=TimeOptions(max_steps=1, output_step=2),
    ).trace((0, 0), 0, 3)
    assert result.termination is TimeTerminationReason.SOLVER_FAILURE
    assert 0 < result.times[-1] < 3
    np.testing.assert_allclose(result.points[-1], (result.times[-1], 0), atol=1e-15)


def test_zero_duration_and_single_release_are_single_particle_snapshots() -> None:
    tracer = PathlineTracer(OscillatingFlowField())
    path = tracer.trace((1, 2), 0.7, 0.7)
    assert path.termination is TimeTerminationReason.TIME_END
    assert path.nfev == 0
    np.testing.assert_array_equal(path.points, [[1, 2]])
    np.testing.assert_array_equal(path.times, [0.7])
    streak = tracer.streakline((1, 2), [0.7], 0.7)
    assert streak.valid.tolist() == [True]
    np.testing.assert_array_equal(streak.positions, [[1, 2]])


@pytest.mark.parametrize("kwargs", [
    {"max_step": 0}, {"rtol": -1}, {"atol": np.inf}, {"output_step": np.nan},
    {"output_step": False}, {"max_steps": 0}, {"max_steps": True}, {"max_steps": 1.5},
])
def test_time_options_reject_invalid_controls(kwargs) -> None:
    with pytest.raises(ValueError):
        TimeOptions(**kwargs)


@pytest.mark.parametrize("seed,start,end", [
    ((0, 0), 2, 1), ((0, 0), np.nan, 1), ((0, 0), 0, np.inf),
    ((0, 0), -1e308, 1e308), ((np.nan, 0), 0, 1), ((0, 0, 0), 0, 1),
    (((0, 0),), 0, 1), ((1j, 0), 0, 1),
])
def test_pathline_rejects_invalid_initial_value_problems(seed, start, end) -> None:
    with pytest.raises(ValueError):
        PathlineTracer(OscillatingFlowField()).trace(seed, start, end)


@pytest.mark.parametrize("releases,observation", [
    ([], 2), ([0, 0], 2), ([1, 0], 2), ([0, 3], 2), ([0, np.nan], 2),
    ([[0]], 2), ([1j], 2), ([0], np.inf),
])
def test_streakline_rejects_invalid_release_schedule(releases, observation) -> None:
    with pytest.raises(ValueError):
        PathlineTracer(OscillatingFlowField()).streakline((0, 0), releases, observation)


def test_mismatched_domain_invalid_velocity_contract_and_excessive_output_rejected() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        PathlineTracer(OscillatingFlowField(), domain=Domain((-1,), (1,)))
    with pytest.raises(ValueError, match="TimeDependentField"):
        PathlineTracer(OscillatingFlowField().freeze(0))
    with pytest.raises(ValueError, match="shape"):
        PathlineTracer(CallableFlow(lambda p, t: [1])).trace((0, 0), 0, 1)
    with pytest.raises(ValueError, match="real"):
        PathlineTracer(CallableFlow(lambda p, t: [1j, 0])).trace((0, 0), 0, 1)
    with pytest.raises(ValueError, match="million"):
        PathlineTracer(OscillatingFlowField(), options=TimeOptions(output_step=1e-8)).trace((0, 0), 0, 1)


def test_sampling_and_repeated_traces_are_reproducible_and_do_not_alias_field_buffer() -> None:
    buffer = np.zeros(2)

    def reused(p, t):
        buffer[:] = (1, t)
        return buffer

    tracer = PathlineTracer(CallableFlow(reused), options=TimeOptions(output_step=None))
    first = tracer.trace((0, 0), -0.2, 0.7)
    second = tracer.trace((0, 0), -0.2, 0.7)
    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.times, second.times)
    np.testing.assert_allclose(first.points[:, 0], first.times + 0.2, atol=1e-14)
    np.testing.assert_allclose(first.points[:, 1], (first.times**2 - 0.04) / 2, atol=1e-14)
    first.points[:] = 99
    assert second.points[0, 0] == 0


def test_tiny_coordinate_domain_still_locates_first_exit() -> None:
    tracer = PathlineTracer(
        OscillatingFlowField(1e-16, 0, 0),
        domain=Domain((-1e-16, -1e-16), (1e-16, 1e-16)),
        options=TimeOptions(atol=1e-30),
    )
    result = tracer.trace((0, 0), 0, 2)
    assert result.termination is TimeTerminationReason.DOMAIN_EXIT
    np.testing.assert_allclose(result.times[-1], 1, atol=1e-13)
    np.testing.assert_array_equal(result.points[-1], (1e-16, 0))


def test_nonfinite_dense_stage_does_not_keep_unverified_outside_endpoint() -> None:
    # The first step for this zero seed/constant velocity is 1e-4 s. Its
    # continuous extension evaluates an additional RHS at t=1e-5 s after
    # accepting x=1e-4 m, which lies outside the supplied small domain.
    flow = CallableFlow(lambda p, t: (np.nan if abs(t - 1e-5) < 1e-18 else 1, 0))
    result = PathlineTracer(
        flow, domain=Domain((-1e-3, -1), (5e-5, 1)),
        options=TimeOptions(output_step=1),
    ).trace((0, 0), 0, 1)
    assert result.termination is TimeTerminationReason.NONFINITE_FIELD
    np.testing.assert_array_equal(result.points, [[0, 0]])
    np.testing.assert_array_equal(result.times, [0])


def test_solver_step_failure_retains_last_verified_state_despite_sparse_output() -> None:
    # x'=x², x(0)=1 has a finite-time pole at t=1. The adaptive solver must
    # stop before it, retaining finite progress even when output spacing=2 s.
    flow = CallableFlow(lambda p, t: (p[0] ** 2, 0))
    result = PathlineTracer(flow, options=TimeOptions(output_step=2)).trace((1, 0), 0, 2)
    assert result.termination is TimeTerminationReason.SOLVER_FAILURE
    assert 0.99 < result.times[-1] < 1.00001
    assert result.points[-1, 0] > 1e6
    assert np.isfinite(result.points).all()
