"""Independent retarded-potential, symmetry and radiation-law checks."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import c, epsilon_0

from vectorviz import (
    Domain,
    ElectromagneticSample,
    FieldLineTracer,
    OscillatingElectricDipole,
    SphericalExclusion,
    TerminationReason,
    TraceOptions,
)
from vectorviz.electromagnetism import SPEED_OF_LIGHT, VACUUM_PERMEABILITY


def test_static_and_low_frequency_limits_against_electrostatic_dipole() -> None:
    moment = np.array((0.3, -0.2, 1.1)) * 1e-9
    points = np.array(((0.4, 0.8, -1.1), (2, -1, 0.3)))
    r = np.linalg.norm(points, axis=-1)
    expected = (
        3 * points * (points @ moment)[:, None] / r[:, None]**5
        - moment / r[:, None]**3
    ) / (4 * np.pi * epsilon_0)
    static = OscillatingElectricDipole(moment, angular_frequency=0, phase=0.4)
    values = static.evaluate(points, -1e300)
    np.testing.assert_allclose(values.electric, expected * np.cos(0.4), rtol=5e-15)
    np.testing.assert_array_equal(values.magnetic, np.zeros_like(points))
    np.testing.assert_array_equal(values.poynting, np.zeros_like(points))
    low = OscillatingElectricDipole(moment, angular_frequency=1e-3).evaluate(points, 0.1)
    np.testing.assert_allclose(low.electric, expected, rtol=1e-8)
    assert np.max(np.abs(low.magnetic)) < 1e-22


def test_fields_match_independent_numerical_derivatives_of_retarded_potentials() -> None:
    # Differentiate scalar/vector potentials, rather than reimplementing E/B.
    field = OscillatingElectricDipole((0.4e-9, -0.7e-9, 1e-9), 2e8, 0.31, (0.1, -0.2, 0.3))
    point, time = np.array((0.7, 0.9, -1.2)), 0.7e-8

    def potentials(position, observation):
        displacement = position - field.center
        radius = np.linalg.norm(displacement)
        phase = field.angular_frequency * (observation - radius / c) + field.phase
        p = field.moment * np.cos(phase)
        dp = -field.moment * field.angular_frequency * np.sin(phase)
        phi = np.dot(displacement, p / radius**3 + dp / (c * radius**2)) / (4 * np.pi * epsilon_0)
        vector = dp / (4 * np.pi * epsilon_0 * c**2 * radius)
        return phi, vector

    step, dt = 2e-5, 2e-13
    gradient = np.empty(3)
    jacobian = np.empty((3, 3))
    for axis in range(3):
        delta = np.eye(3)[axis] * step
        plus, minus = potentials(point + delta, time), potentials(point - delta, time)
        gradient[axis] = (plus[0] - minus[0]) / (2 * step)
        jacobian[:, axis] = (plus[1] - minus[1]) / (2 * step)
    da = (potentials(point, time + dt)[1] - potentials(point, time - dt)[1]) / (2 * dt)
    curl = np.array((jacobian[2, 1] - jacobian[1, 2], jacobian[0, 2] - jacobian[2, 0], jacobian[1, 0] - jacobian[0, 1]))
    result = field.evaluate(point, time)
    np.testing.assert_allclose(result.electric, -gradient - da, rtol=2e-9)
    np.testing.assert_allclose(result.magnetic, curl, rtol=2e-9)


def test_axis_equator_retarded_phase_and_transverse_symmetries() -> None:
    field = OscillatingElectricDipole()
    radius, p, omega = 0.8, 1e-9, field.angular_frequency
    # Source retarded phase is zero; derivative is zero but radiation is not.
    values = field.evaluate(((0, 0, radius), (radius, 0, 0), (0, radius, 0)), radius / c)
    scale = p / (4 * np.pi * epsilon_0)
    np.testing.assert_allclose(values.electric[0], (0, 0, 2 * scale / radius**3), rtol=1e-14)
    np.testing.assert_array_equal(values.magnetic[0], (0, 0, 0))
    np.testing.assert_array_equal(values.poynting[0], (0, 0, 0))
    expected_e = scale * (omega**2 / (c**2 * radius) - 1 / radius**3)
    np.testing.assert_allclose(values.electric[1:], ((0, 0, expected_e), (0, 0, expected_e)), atol=1e-14)
    expected_b = VACUUM_PERMEABILITY * p * omega**2 / (4 * np.pi * c * radius)
    np.testing.assert_allclose(values.magnetic[1:], ((0, -expected_b, 0), (expected_b, 0, 0)), atol=1e-22)
    # At retarded pi/2 the electrostatic and radiation parts vanish; this
    # isolates the induction terms, and fails if time is not retarded.
    quarter = field.evaluate((radius, 0, 0), radius / c + np.pi / (2 * omega))
    np.testing.assert_allclose(quarter.electric, (0, 0, scale * omega / (c * radius**2)), atol=2e-14)
    np.testing.assert_allclose(quarter.magnetic, (0, -VACUUM_PERMEABILITY * p * omega / (4 * np.pi * radius**2), 0), atol=1e-22)


def test_near_and_far_distance_scaling_and_far_wave_relation() -> None:
    field = OscillatingElectricDipole()
    # Compare equal retarded phases to avoid confusing decay with oscillation.
    near = [field.evaluate((radius, 0, 0), radius / c).electric[2] for radius in (1e-4, 2e-4)]
    np.testing.assert_allclose(near[0] / near[1], 8, rtol=2e-7)
    radii = (1e5, 2e5)
    far = [field.evaluate((radius, 0, 0), radius / c) for radius in radii]
    np.testing.assert_allclose(far[0].electric[2] / far[1].electric[2], 2, rtol=3e-11)
    for value in far:
        np.testing.assert_allclose(np.cross((1, 0, 0), value.electric) / c, value.magnetic, rtol=3e-11)
        assert value.electric[0] == value.magnetic[0] == 0
        assert value.poynting[0] > 0
        np.testing.assert_array_equal(value.poynting[1:], (0, 0))


def test_instantaneous_energy_can_return_inward_but_cycle_average_is_radiative() -> None:
    field = OscillatingElectricDipole()
    omega = field.angular_frequency
    radius = 0.02 * c / omega
    near = field.evaluate((radius, 0, 0), radius / c + np.pi / (4 * omega))
    assert near.poynting[0] < 0
    # The full fields give the radial sin²(theta)/r² cycle mean at *any*
    # radius outside the source; angular quadrature then gives total power.
    cosines, weights = np.polynomial.legendre.leggauss(12)
    normals = np.column_stack((np.sqrt(1 - cosines**2), np.zeros(12), cosines))
    for radius in (0.1, 7):
        average = np.mean([
            field.evaluate(radius * normals, phase / omega).poynting
            for phase in np.linspace(0, 2 * np.pi, 128, endpoint=False)
        ], axis=0)
        coefficient = 1e-18 * omega**4 / (32 * np.pi**2 * epsilon_0 * c**3 * radius**2)
        np.testing.assert_allclose(average, coefficient * (1 - cosines**2)[:, None] * normals, rtol=2e-11, atol=1e-12)
        total = 2 * np.pi * radius**2 * np.dot(weights, np.sum(average * normals, axis=-1))
        expected_power = 1e-18 * omega**4 / (12 * np.pi * epsilon_0 * c**3)
        np.testing.assert_allclose(total, expected_power, rtol=1e-12)


def test_rotation_translation_batch_shape_and_freezing() -> None:
    moment, center = np.array((0, 0, 1e-9)), np.array((0.4, -0.2, 0.1))
    field = OscillatingElectricDipole(moment, center=center)
    moment[:] = 99
    center[:] = 99
    points = np.array([[[0.8, -0.2, 0.6], [0.7, 0.3, -0.2]]])
    rotation = np.array(((0, 1, 0), (0, 0, 1), (1, 0, 0)))
    rotated = OscillatingElectricDipole(rotation @ field.moment, center=rotation @ field.center)
    sample = field.evaluate(points, -1.2e-9)
    assert isinstance(sample, ElectromagneticSample)
    assert field.dimension == 3
    assert not field.moment.flags.writeable and not field.center.flags.writeable
    rotated_sample = rotated.evaluate(points @ rotation.T, -1.2e-9)
    for name in ("electric", "magnetic", "poynting"):
        values = getattr(sample, name)
        assert values.shape == points.shape
        np.testing.assert_allclose(getattr(rotated_sample, name), values @ rotation.T, rtol=5e-15)
        frozen = field.freeze(-1.2e-9, name)
        assert frozen.dimension == 3
        np.testing.assert_array_equal(frozen(points), values)
        assert getattr(field.evaluate(np.empty((2, 0, 3)), 0), name).shape == (2, 0, 3)
    np.testing.assert_allclose(sample.poynting, np.cross(sample.electric, sample.magnetic) / VACUUM_PERMEABILITY)
    assert c == SPEED_OF_LIGHT
    assert VACUUM_PERMEABILITY * epsilon_0 * c**2 == pytest.approx(1, abs=3e-16)


def test_singular_nonfinite_and_overflow_policy_keeps_exact_zeros() -> None:
    field = OscillatingElectricDipole()
    points = ((0, 0, 0), (np.inf, 1, 0), (0, np.nan, 0), (1e-300, 0, 0))
    for name in ("electric", "magnetic", "poynting"):
        assert np.isnan(getattr(field.evaluate(points, 0), name)).all()
        assert np.isnan(getattr(OscillatingElectricDipole(angular_frequency=1e308).evaluate((1, 0, 0), 1e308), name)).all()
        zero = OscillatingElectricDipole((0, 0, 0), angular_frequency=1e308)
        sample = zero.evaluate(((0, 0, 0), (1e-300, 0, 0), (1, 0, 0)), 1e308)
        assert np.isnan(getattr(sample, name)[0]).all()
        np.testing.assert_array_equal(getattr(sample, name)[1:], np.zeros((2, 3)))
    with pytest.raises(ValueError, match="shape"):
        field.evaluate((1, 2), 0)
    with pytest.raises(ValueError, match="real"):
        field.evaluate((1j, 0, 0), 0)
    with pytest.raises(ValueError, match="observable"):
        field.freeze(0, "energy")


@pytest.mark.parametrize("time", [True, np.bool_(False), np.nan, np.inf, "1", [1], 1j])
def test_observation_time_is_a_finite_real_scalar(time) -> None:
    field = OscillatingElectricDipole()
    with pytest.raises(ValueError, match="time"):
        field.evaluate((1, 0, 0), time)
    with pytest.raises(ValueError, match="time"):
        field.freeze(time)


@pytest.mark.parametrize("kwargs", [
    {"moment": (1, 2)}, {"moment": (1, np.inf, 0)}, {"moment": (1j, 0, 0)},
    {"center": (1, 2)}, {"center": (1, np.nan, 0)},
    {"angular_frequency": -1}, {"angular_frequency": True}, {"angular_frequency": np.inf},
    {"phase": np.nan}, {"phase": "1"},
])
def test_invalid_source_parameters_are_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        OscillatingElectricDipole(**kwargs)


def test_frozen_magnetic_lines_are_equatorial_circles_and_preserve_time() -> None:
    field = OscillatingElectricDipole()
    line = FieldLineTracer(
        field.freeze(1.25e-9, "magnetic"),
        options=TraceOptions(max_arc_length=8, max_step=0.1, output_step=0.1,
                             closure_tolerance=1e-5, closure_min_arc_length=1),
    ).trace((0.8, 0, 0))
    assert line.forward.termination is TerminationReason.CLOSED_LOOP
    np.testing.assert_allclose(np.linalg.norm(line.points[:, :2], axis=-1), 0.8, atol=2e-9)
    np.testing.assert_array_equal(line.points[:, 2], np.zeros(len(line.points)))
    angle = np.unwrap(np.arctan2(line.points[:, 1], line.points[:, 0]))
    assert abs(angle[-1] - angle[0]) == pytest.approx(2 * np.pi, abs=2e-8)


def test_frozen_traces_respect_exclusion_and_zero_or_undefined_seed() -> None:
    field = OscillatingElectricDipole(angular_frequency=0)
    tracer = FieldLineTracer(field.freeze(0),
                            domain=Domain((-2, -2, -2), (2, 2, 2)),
                            exclusions=(SphericalExclusion((0, 0, 0), 0.2),))
    trace = tracer.trace((0, 0, 0.8), direction="both")
    assert trace.forward.termination is TerminationReason.DOMAIN_EXIT
    assert trace.backward.termination is TerminationReason.EXCLUSION_HIT
    np.testing.assert_allclose(trace.backward.points[-1], (0, 0, 0.2), atol=1e-14)
    source = FieldLineTracer(field.freeze(0)).trace((0, 0, 0))
    assert source.forward.termination is TerminationReason.NONFINITE_FIELD
    zero = FieldLineTracer(field.freeze(0, "magnetic")).trace((1, 0, 0))
    assert zero.forward.termination is TerminationReason.NULL_FIELD
    np.testing.assert_array_equal(zero.points, ((1, 0, 0),))
