"""Independent geometric oracles for events hidden inside an accepted ODE step."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vectorviz import (
    Domain,
    MagneticDipoleField,
    SphericalExclusion,
    TerminationReason,
    ToroidalExclusion,
    TraceOptions,
    UniformField,
    VectorField,
    trace_field_line,
)
from vectorviz._trace_geometry import first_exclusion_event


@pytest.mark.parametrize("output_step", [None, 0.002])
@pytest.mark.parametrize("shape", ["sphere", "torus"])
def test_thin_exclusion_stops_at_first_analytic_entry(shape: str, output_step: float | None) -> None:
    # Both ends of an accepted step can be outside even when it crossed the
    # entire obstacle. The torus can hide TWO separate crossings in one step.
    if shape == "sphere":
        exclusion = SphericalExclusion((0.0, 0.0), 0.01)
        dimension, entry = 2, -0.01
    else:
        exclusion = ToroidalExclusion((0, 0, 0), (0, 0, 1), 0.02, 0.001)
        dimension, entry = 3, -0.021
    seed = np.zeros(dimension)
    seed[0] = -0.8
    vector = np.zeros(dimension)
    vector[0] = 1.0
    result = trace_field_line(
        UniformField(vector), seed,
        domain=Domain(-np.ones(dimension), np.ones(dimension)),
        exclusions=(exclusion,),
        options=TraceOptions(output_step=output_step),
        direction="forward",
    )
    branch = result.forward
    assert branch is not None
    assert branch.termination is TerminationReason.EXCLUSION_HIT
    expected = np.zeros(dimension)
    expected[0] = entry
    np.testing.assert_allclose(branch.terminal_point, expected, atol=2e-10, rtol=0)
    assert np.all(branch.points[:, 0] <= entry + 2e-10)
    assert len(branch.points) == len(branch.arc_length) == len(branch.field_magnitude)
    assert np.all(np.diff(branch.arc_length) > 0)


@pytest.mark.parametrize("offset", [-1e-9, 1e-9])
def test_grazing_line_distinguishes_entry_from_near_miss(offset: float) -> None:
    height = 1.0 + offset
    result = trace_field_line(
        UniformField((1, 0)), (-0.8, height),
        domain=Domain((-1, -2), (1, 2)),
        exclusions=(SphericalExclusion((0, 0), 1.0),),
        options=TraceOptions(output_step=0.002), direction="forward",
    )
    branch = result.forward
    assert branch is not None
    if offset < 0:
        # Stable analytic quadratic root; independent of the event locator.
        entry = -np.sqrt((1.0 - height) * (1.0 + height))
        assert branch.termination is TerminationReason.EXCLUSION_HIT
        np.testing.assert_allclose(branch.terminal_point, (entry, height), atol=2e-9, rtol=0)
    else:
        assert branch.termination is TerminationReason.DOMAIN_EXIT
        np.testing.assert_allclose(branch.terminal_point, (1.0, height), atol=2e-10, rtol=0)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_hidden_entry_rotates_translates_and_scales_with_geometry(scale: float) -> None:
    direction = np.array((0.6, 0.8))
    center = scale * np.array((0.3, -0.2))
    result = trace_field_line(
        UniformField(direction), center - 0.8 * scale * direction,
        exclusions=(SphericalExclusion(center, 0.01 * scale),),
        options=TraceOptions(
            max_arc_length=2 * scale, max_step=0.1 * scale, atol=1e-12 * scale,
        ), direction="forward",
    )
    assert result.forward is not None
    assert result.forward.termination is TerminationReason.EXCLUSION_HIT
    np.testing.assert_allclose(
        result.forward.terminal_point, center - 0.01 * scale * direction,
        atol=2e-10 * scale, rtol=0,
    )


class _Rotation(VectorField):
    dimension = 2

    def evaluate(self, points: object) -> np.ndarray:
        coordinates = np.asarray(points, dtype=float)
        return np.stack((-coordinates[..., 1], coordinates[..., 0]), axis=-1)


@pytest.mark.parametrize("radial_offset", [0.0, -0.003])
def test_curved_trace_uses_the_curve_instead_of_a_straight_chord(radial_offset: float) -> None:
    angle, radius = 0.8, 0.001
    center = (1 + radial_offset) * np.array((np.cos(angle), np.sin(angle)))
    result = trace_field_line(
        _Rotation(), (1.0, 0.0), exclusions=(SphericalExclusion(center, radius),),
        options=TraceOptions(max_arc_length=1.5, max_step=0.3, rtol=1e-10, atol=1e-12),
        direction="forward",
    )
    assert result.forward is not None
    if radial_offset == 0:
        entry_angle = angle - 2 * np.arcsin(radius / 2)
        assert result.forward.termination is TerminationReason.EXCLUSION_HIT
        np.testing.assert_allclose(
            result.forward.terminal_point, (np.cos(entry_angle), np.sin(entry_angle)), atol=2e-9,
        )
    else:
        assert result.forward.termination is TerminationReason.MAX_ARC_LENGTH


def test_first_entry_is_independent_of_exclusion_order_and_precedes_domain_exit() -> None:
    # Force a step containing both obstacles, listed in the opposite order.
    result = trace_field_line(
        UniformField((1, 0)), (-0.05, 0),
        domain=Domain((-0.1, -0.1), (0.2, 0.1)),
        exclusions=(SphericalExclusion(((0.02, 0), (0.0, 0)), 0.001),),
        options=TraceOptions(first_step=0.1, max_step=0.1), direction="forward",
    )
    assert result.forward is not None
    assert result.forward.termination is TerminationReason.EXCLUSION_HIT
    np.testing.assert_allclose(result.forward.terminal_point, (-0.001, 0), atol=2e-10, rtol=0)


@pytest.mark.parametrize("method", ["DOP853", "RK45", "RK23", "Radau", "BDF", "LSODA"])
def test_hidden_entry_with_each_supported_dense_interpolant(method: str) -> None:
    result = trace_field_line(
        UniformField((1, 0)), (-0.05, 0),
        exclusions=(SphericalExclusion((0, 0), 0.001),),
        options=TraceOptions(
            method=method, first_step=0.1, max_step=0.1, max_arc_length=0.3,
        ), direction="forward",
    )
    assert result.forward is not None
    assert result.forward.termination is TerminationReason.EXCLUSION_HIT
    np.testing.assert_allclose(result.forward.terminal_point, (-0.001, 0), atol=2e-10, rtol=0)


def test_a_very_short_trace_keeps_its_seed_and_nonzero_length() -> None:
    seed = np.array((-0.010000001, 0.0))
    result = trace_field_line(
        UniformField((1, 0)), seed,
        exclusions=(SphericalExclusion((0, 0), 0.01),),
        options=TraceOptions(output_step=0.01), direction="forward",
    )
    assert result.forward is not None
    assert result.forward.termination is TerminationReason.EXCLUSION_HIT
    np.testing.assert_array_equal(result.points[0], seed)
    assert len(result.points) == 2
    assert result.forward.arc_length[-1] == pytest.approx(1e-9, abs=2e-15)


def test_unresolved_tangency_is_reported_instead_of_claiming_a_safe_passage() -> None:
    result = trace_field_line(
        UniformField((1, 0)), (-0.8, 1),
        domain=Domain((-1, -2), (1, 2)),
        exclusions=(SphericalExclusion((0, 0), 1),), direction="forward",
    )
    assert result.forward is not None
    assert result.forward.termination is TerminationReason.SOLVER_FAILURE
    assert "exclusion contact" in result.forward.message


def test_hidden_entry_is_not_merged_with_a_later_obstacles_known_event() -> None:
    # The large sphere gives solve_ivp a visible event at x=0. The tiny sphere
    # has a distinct, fully representable entry before it; proximity in time
    # does not make those two obstacles the same boundary.
    result = trace_field_line(
        UniformField((1, 0)), (-0.5, 0),
        exclusions=(SphericalExclusion(((-1e-7, 0), (1, 0)), (1e-8, 1)),),
        options=TraceOptions(first_step=1, max_step=1, max_arc_length=2),
        direction="forward",
    )
    assert result.forward is not None
    assert result.forward.termination is TerminationReason.EXCLUSION_HIT
    np.testing.assert_allclose(result.forward.terminal_point, (-1.1e-7, 0), atol=2e-14, rtol=0)


def test_two_entries_of_one_torus_are_not_merged_as_the_same_known_root() -> None:
    # A single region can also have distinct entries: the accepted step skips
    # the left tube and ends inside the right one, which produces a later root.
    major, minor = 5e-8, 5e-9
    result = trace_field_line(
        UniformField((1, 0, 0)), (-0.5, 0, 0),
        exclusions=(ToroidalExclusion((0, 0, 0), (0, 0, 1), major, minor),),
        options=TraceOptions(first_step=0.5 + major, max_step=1, max_arc_length=2),
        direction="forward",
    )
    assert result.forward is not None
    assert result.forward.termination is TerminationReason.EXCLUSION_HIT
    np.testing.assert_allclose(
        result.forward.terminal_point, (-(major + minor), 0, 0), atol=2e-14, rtol=0,
    )


@pytest.mark.parametrize("scale", [1e-12, 1e-15])
def test_first_entry_has_no_absolute_parameter_resolution_floor(scale: float) -> None:
    # The normalized geometry is well resolved in float64 at either scale.
    # A time tolerance based on max(1, t) can instead collapse both crossings
    # into a single leaf and choose the second sphere's entry.
    result = trace_field_line(
        UniformField((1, 0)), (-0.05 * scale, 0),
        exclusions=(SphericalExclusion(((-0.02 * scale, 0), (0, 0)), 0.001 * scale),),
        options=TraceOptions(
            first_step=0.1 * scale, max_step=0.1 * scale,
            max_arc_length=0.3 * scale, atol=1e-12 * scale,
        ),
        direction="forward",
    )
    assert result.forward is not None
    assert result.forward.termination is TerminationReason.EXCLUSION_HIT
    np.testing.assert_allclose(result.forward.terminal_point / scale, (-0.021, 0), atol=2e-10, rtol=0)


def test_known_root_does_not_bypass_an_unresolved_earlier_obstacle() -> None:
    result = trace_field_line(
        UniformField((1, 0)), (-0.5, 0),
        exclusions=(SphericalExclusion(((-5e-14, 0), (1, 0)), (1e-14, 1)),),
        options=TraceOptions(first_step=1, max_step=1, max_arc_length=2), direction="forward",
    )
    branch = result.forward
    assert branch is not None
    assert branch.termination in (TerminationReason.EXCLUSION_HIT, TerminationReason.SOLVER_FAILURE)
    # An unresolved earlier contact can stop conservatively, but a later known
    # event is no justification for silently crossing it, even inside a guard.
    assert branch.terminal_point[0] <= -6e-14 + 1e-15


def test_known_dipole_entry_remains_resolvable_on_a_rounding_size_prefix() -> None:
    moment = (np.cos(np.pi / 2), np.sin(np.pi / 2), 0)
    result = trace_field_line(
        MagneticDipoleField(moment), (-0.162, 9.91963907309356e-18, 0),
        exclusions=(SphericalExclusion((0, 0, 0), 0.16),),
        options=TraceOptions(
            max_arc_length=18, max_step=0.09, rtol=2e-6, atol=1e-8,
            output_step=0.045, null_threshold=1e-14,
        ), direction="forward",
    )
    assert result.forward is not None
    assert result.forward.termination is TerminationReason.EXCLUSION_HIT
    # A smooth inward crossing is well conditioned even when the final
    # geometric interval is too short for differencing its rounded positions.
    assert np.linalg.norm(result.forward.terminal_point) == pytest.approx(0.16, abs=2e-14)


@pytest.mark.parametrize("batched", [False, True])
def test_distant_geometry_does_not_make_a_resolved_near_miss_ambiguous(batched: bool) -> None:
    height = 1.0 + 1e-9
    if batched:
        exclusions = (SphericalExclusion(((0, 0), (1e8, 1e8)), 1),)
    else:
        exclusions = (SphericalExclusion((0, 0), 1), SphericalExclusion((1e8, 1e8), 1))
    result = trace_field_line(
        UniformField((1, 0)), (-0.8, height),
        domain=Domain((-1, -2), (1, 2)), exclusions=exclusions, direction="forward",
    )
    assert result.forward is not None
    assert result.forward.termination is TerminationReason.DOMAIN_EXIT
    np.testing.assert_allclose(result.forward.terminal_point, (1, height), atol=2e-10, rtol=0)


@pytest.mark.parametrize("scale", [1e-15, 1.0])
@pytest.mark.parametrize("sign", [-1, 1])
def test_a_surface_seed_respects_departure_direction_at_different_scales(scale: float, sign: int) -> None:
    result = trace_field_line(
        UniformField((sign, 0)), (-scale, 0),
        exclusions=(SphericalExclusion((0, 0), scale),),
        options=TraceOptions(
            max_step=0.1 * scale, max_arc_length=0.5 * scale, atol=1e-12 * scale,
        ),
        direction="forward",
    )
    assert result.forward is not None
    expected_reason = TerminationReason.MAX_ARC_LENGTH if sign < 0 else TerminationReason.EXCLUSION_HIT
    assert result.forward.termination is expected_reason
    expected_x = -1.5 if sign < 0 else -1.0
    np.testing.assert_allclose(result.forward.terminal_point / scale, (expected_x, 0), atol=2e-12, rtol=0)


@pytest.mark.parametrize("shape", ["spheres", "torus"])
def test_lookahead_does_not_jump_over_a_thin_earlier_obstacle(shape: str) -> None:
    if shape == "spheres":
        first_entry = -1e-7 - 1e-13
        exclusion = SphericalExclusion(((-1e-7, 0), (-7e-8, 0)), (1e-13, 3e-9))
        direction, seed = (1, 0), (-0.5, 0)
    else:
        major, minor = 2**-27 - 1.8e-13, 1e-13
        first_entry = -(major + minor)
        exclusion = ToroidalExclusion((0, 0, 0), (0, 0, 1), major, minor)
        direction, seed = (1, 0, 0), (-0.5, 0, 0)
    result = trace_field_line(
        UniformField(direction), seed, exclusions=(exclusion,),
        options=TraceOptions(first_step=1, max_step=1, max_arc_length=2),
        direction="forward",
    )
    assert result.forward is not None
    branch = result.forward
    if branch.termination is TerminationReason.EXCLUSION_HIT:
        expected = np.zeros(len(seed))
        expected[0] = first_entry
        np.testing.assert_allclose(branch.terminal_point, expected, atol=2e-14, rtol=0)
    else:
        # Failure before an unresolved first contact is honest; jumping over it
        # and claiming the second sphere's entry is the first one is not.
        assert branch.termination is TerminationReason.SOLVER_FAILURE
        assert branch.terminal_point[0] <= first_entry + 2e-14
        assert "exclusion contact" in branch.message


def test_surface_departure_does_not_skip_another_obstacle_in_its_prefix() -> None:
    result = trace_field_line(
        UniformField((-1, 0)), (0, 0),
        exclusions=(SphericalExclusion(((1, 0), (-1e-14, 0)), (1, 1e-15)),),
        options=TraceOptions(first_step=1, max_step=1, max_arc_length=2), direction="forward",
    )
    branch = result.forward
    assert branch is not None
    # Moving away from the initial surface says nothing about another tiny
    # obstacle between this leaf's samples. Preserve its unresolved prefix.
    assert branch.termination in (TerminationReason.EXCLUSION_HIT, TerminationReason.SOLVER_FAILURE)
    assert branch.terminal_point[0] >= -9e-15 - 2e-16


def test_leaf_witness_does_not_skip_an_earlier_entry_at_a_late_parameter() -> None:
    # Simulate a late accepted step with an exact straight-line interpolant.
    # Its parameter-resolution leaf can contain two separate obstacles.
    start = 1e4
    solution = SimpleNamespace(
        t=np.array([start, start + 0.1]),
        sol=lambda t: np.array([np.asarray(t) - start - 0.05, np.zeros_like(t)]),
    )
    centers = (0.00981923487761708, 0.00981923492383675)
    radii = (4.200180135140079e-12, 1.7756060972215785e-11)
    event = first_exclusion_event(
        solution, (SphericalExclusion(((centers[0], 0), (centers[1], 0)), radii),), "RK45",
    )
    assert event is not None
    # A negative leaf sample only proves penetration, not that a bracketed
    # root finder cannot jump over the first narrow negative interval.
    assert solution.sol(event[0])[0] <= centers[0] - radii[0] + 1e-11


@pytest.mark.parametrize("direction, sign", [("forward", 1), ("backward", -1)])
@pytest.mark.parametrize("offset", [-1e-6, 0.0, 1e-6])
def test_radau_exclusion_near_an_interface_stops_at_its_first_entry(
    direction: str, sign: int, offset: float,
) -> None:
    entry, radius = 0.1 + offset, 1e-5
    result = trace_field_line(
        UniformField((1, 0)), (0, 0),
        interfaces=(SphericalExclusion((0, 0), 0.1),),
        exclusions=(SphericalExclusion((sign * (entry + radius), 0), radius),),
        options=TraceOptions(
            method="Radau", first_step=0.1, max_step=0.1,
            max_arc_length=0.5, output_step=0.02,
        ), direction=direction,
    )
    branch = result.forward if sign > 0 else result.backward
    assert branch is not None
    # Straight-line geometry fixes the first entry independently of solver
    # events, including an exclusion before, at, or after the restart surface.
    assert branch.termination is TerminationReason.EXCLUSION_HIT
    np.testing.assert_allclose(branch.terminal_point, (sign * entry, 0), atol=2e-12, rtol=0)
    assert np.all(np.diff(branch.arc_length) > 0)
