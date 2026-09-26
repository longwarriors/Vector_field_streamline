"""Independent numerical contracts for the uniformly charged ring field."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import epsilon_0
from scipy.integrate import quad

from vectorviz import ChargedRingField


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=float)
    unit /= np.linalg.norm(unit)
    x, y, z = unit
    cross = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _coulomb_ring_oracle(
    points: np.ndarray,
    *,
    charge: float,
    radius: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    permittivity: float,
    order: int = 2048,
) -> np.ndarray:
    """Integrate Coulomb's law around the ring without using the closed form.

    The integrand is smooth and periodic in the ring angle, so the midpoint
    rule converges spectrally away from the filament.
    """

    coordinates = np.asarray(points, dtype=float)
    original_shape = coordinates.shape
    flat_points = coordinates.reshape(-1, 3)
    center_array = np.asarray(center, dtype=float)
    normal_array = np.asarray(normal, dtype=float)
    normal_array /= np.linalg.norm(normal_array)
    reference = np.zeros(3)
    reference[int(np.argmin(np.abs(normal_array)))] = 1.0
    first_basis = np.cross(reference, normal_array)
    first_basis /= np.linalg.norm(first_basis)
    second_basis = np.cross(normal_array, first_basis)
    angles = 2.0 * np.pi * (np.arange(order) + 0.5) / order
    sources = center_array + radius * (
        np.cos(angles)[:, np.newaxis] * first_basis + np.sin(angles)[:, np.newaxis] * second_basis
    )
    element_charge = charge / order
    values = np.empty_like(flat_points)
    for index, point in enumerate(flat_points):
        displacement = point - sources
        distance = np.linalg.norm(displacement, axis=1)
        values[index] = np.sum(displacement / distance[:, np.newaxis] ** 3, axis=0)
    return (element_charge / (4.0 * np.pi * permittivity) * values).reshape(original_shape)


def test_charged_ring_matches_axis_solution_and_preserves_batch_shape() -> None:
    charge = 2.3e-9
    radius = 0.7
    ring = ChargedRingField(charge, radius)
    axial = np.array((-1.2, -0.3, 0.0, 0.45, 2.0))
    points = np.zeros((5, 1, 3))
    points[:, 0, 2] = axial

    values = ring.evaluate(points)

    expected = charge / (4.0 * np.pi * epsilon_0) * axial / (radius**2 + axial**2) ** 1.5
    assert values.shape == points.shape
    np.testing.assert_allclose(values[:, 0, 2], expected, rtol=1.0e-14)
    np.testing.assert_array_equal(values[:, 0, :2], 0.0)


def test_charged_ring_reports_the_ideal_ring_as_nan_without_softening() -> None:
    ring = ChargedRingField(1.0e-9, 1.0)
    zero_ring = ChargedRingField(0.0, 1.0)
    on_ring = np.array(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (np.cos(0.3), np.sin(0.3), 0.0)))
    beside_ring = np.array(((1.0, 0.0, 1.0e-6), (1.0 + 1.0e-6, 0.0, 0.0)))

    assert np.all(np.isnan(ring.evaluate(on_ring)))
    assert np.all(np.isnan(zero_ring.evaluate(on_ring)))
    assert np.all(np.isnan(ring.flux_function(on_ring)))
    beside = ring.evaluate(beside_ring)
    assert np.all(np.isfinite(beside))
    # Just off the filament the field is that of a line charge: E ~ lambda / (2 pi eps0 d).
    line_charge = 1.0e-9 / (2.0 * np.pi * 1.0)
    np.testing.assert_allclose(
        np.linalg.norm(beside, axis=1),
        line_charge / (2.0 * np.pi * epsilon_0 * 1.0e-6),
        rtol=2.0e-5,
    )


def test_charged_ring_uses_the_regular_near_axis_limit() -> None:
    ring = ChargedRingField(1.3e-9, 0.9)
    axial = 0.4
    boundary = ring._NEAR_AXIS_RATIO * np.hypot(0.9, axial)
    inside = np.array((0.999 * boundary, 0.0, axial))
    outside = np.array((1.001 * boundary, 0.0, axial))
    oracle = _coulomb_ring_oracle(
        np.array((inside, outside)), charge=1.3e-9, radius=0.9, permittivity=epsilon_0
    )

    values = ring.evaluate(np.array((inside, outside)))

    np.testing.assert_allclose(values[:, 2], oracle[:, 2], rtol=1.0e-12)
    np.testing.assert_allclose(values[:, 0], oracle[:, 0], rtol=1.0e-8)
    assert values[0, 0] < 0.0 < values[1, 2]
    axis_value = ring.evaluate((0.0, 0.0, axial))
    assert np.all(np.isfinite(axis_value))
    assert axis_value[0] == axis_value[1] == 0.0


def test_analytic_ring_matches_an_independent_coulomb_oracle() -> None:
    parameters = {
        "charge": -3.1e-9,
        "radius": 0.8,
        "center": (0.2, -0.1, 0.3),
        "normal": (0.3, -0.4, 0.8),
        "permittivity": 2.0 * epsilon_0,
    }
    ring = ChargedRingField(**parameters)  # type: ignore[arg-type]
    generator = np.random.default_rng(7)
    points = generator.uniform(-2.0, 2.0, size=(40, 3))
    # Keep the oracle's midpoint rule in its spectral regime.
    axis = np.asarray(parameters["normal"], dtype=float)
    axis /= np.linalg.norm(axis)
    displacement = points - np.asarray(parameters["center"])
    axial = displacement @ axis
    radial = np.linalg.norm(displacement - axial[:, np.newaxis] * axis, axis=1)
    points = points[np.hypot(radial - 0.8, axial) > 0.15]
    assert points.shape[0] > 25

    expected = _coulomb_ring_oracle(points, **parameters)  # type: ignore[arg-type]

    np.testing.assert_allclose(ring.evaluate(points), expected, rtol=1.0e-10, atol=1.0e-12)


def test_charged_ring_converges_to_its_point_charge_far_field() -> None:
    charge = 1.0e-9
    ring = ChargedRingField(charge, 1.0)
    directions = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.6, 0.0, 0.8), (0.0, -0.8, 0.6)))
    distance = 300.0
    points = distance * directions

    values = ring.evaluate(points)

    expected = charge / (4.0 * np.pi * epsilon_0 * distance**2) * directions
    # The leading correction is the ring's quadrupole moment, of order (a/r)^2.
    np.testing.assert_allclose(values, expected, rtol=3.0e-5)


def test_charged_ring_is_covariant_under_rigid_rotation() -> None:
    rotation = _rotation(np.array((1.0, 2.0, -0.5)), 0.9)
    center = np.array((0.1, -0.2, 0.3))
    normal = np.array((0.2, 0.3, 0.9))
    base = ChargedRingField(1.0e-9, 0.6, center=center, normal=normal)
    rotated = ChargedRingField(1.0e-9, 0.6, center=rotation @ center, normal=rotation @ normal)
    points = np.array(((0.5, 0.2, 0.9), (-0.7, 0.4, -0.2), (0.15, -0.9, 0.35)))

    expected = base.evaluate(points) @ rotation.T

    np.testing.assert_allclose(rotated.evaluate(points @ rotation.T), expected, rtol=1.0e-12)
    np.testing.assert_allclose(
        rotated.flux_function(points @ rotation.T), base.flux_function(points), rtol=1.0e-12
    )


def test_ring_flux_function_generates_the_field_and_has_its_symmetries() -> None:
    charge = 1.7e-9
    ring = ChargedRingField(charge, 0.9)
    points = np.array(((0.3, 0.0, 0.2), (1.4, 0.0, 0.5), (0.5, 0.0, -0.9), (2.0, 0.0, 1.5)))
    step = 1.0e-5

    psi = ring.flux_function(points)
    values = ring.evaluate(points)

    assert psi.shape == (4,)
    for point, value in zip(points, values, strict=True):
        radial_step = np.array((step, 0.0, 0.0))
        axial_step = np.array((0.0, 0.0, step))
        radial_derivative = (
            ring.flux_function(point + radial_step) - ring.flux_function(point - radial_step)
        ) / (2.0 * step)
        axial_derivative = (
            ring.flux_function(point + axial_step) - ring.flux_function(point - axial_step)
        ) / (2.0 * step)
        np.testing.assert_allclose(value[2], radial_derivative / point[0], rtol=3.0e-9)
        np.testing.assert_allclose(value[0], -axial_derivative / point[0], rtol=3.0e-9)
    mirrored = points * np.array((1.0, 1.0, -1.0))
    np.testing.assert_allclose(ring.flux_function(mirrored), -psi, rtol=1.0e-14)
    assert ring.flux_function((0.0, 0.0, 0.5)) == 0.0
    assert ring.flux_function((0.5, 0.0, 0.0)) == 0.0
    # Outside the ring the plane z = 0 is the cut: half the total flux went up.
    np.testing.assert_allclose(
        ring.flux_function((1.5, 0.0, 0.0)), charge / (4.0 * np.pi * epsilon_0), rtol=1.0e-9
    )
    np.testing.assert_allclose(
        ring.flux_function((1.5, 0.0, -1.0e-12)), -charge / (4.0 * np.pi * epsilon_0), rtol=1.0e-9
    )


def test_ring_flux_function_matches_direct_disk_flux_quadrature() -> None:
    ring = ChargedRingField(1.7e-9, 0.9)

    def disk_flux(rho: float, z: float) -> float:
        integrand = lambda r: r * float(ring.evaluate((r, 0.0, z))[2])  # noqa: E731
        breakpoints = [0.9] if rho > 0.9 else None
        value, _error = quad(integrand, 0.0, rho, points=breakpoints, limit=400)
        return value

    for rho, z in ((0.3, 0.2), (1.4, 0.5), (0.5, -0.9), (0.9, 0.3), (0.95, 0.01), (2.0, 0.05)):
        np.testing.assert_allclose(ring.flux_function((rho, 0.0, z)), disk_flux(rho, z), rtol=1.0e-9)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ChargedRingField(np.nan, 1.0),
        lambda: ChargedRingField(np.inf, 1.0),
        lambda: ChargedRingField(1.0, 0.0),
        lambda: ChargedRingField(1.0, -1.0),
        lambda: ChargedRingField(1.0, np.inf),
        lambda: ChargedRingField(1.0, 1.0, center=(0.0, 0.0)),
        lambda: ChargedRingField(1.0, 1.0, center=(0.0, 0.0, np.nan)),
        lambda: ChargedRingField(1.0, 1.0, normal=(0.0, 0.0, 0.0)),
        lambda: ChargedRingField(1.0, 1.0, normal=(0.0, 1.0)),
        lambda: ChargedRingField(1.0, 1.0, permittivity=0.0),
        lambda: ChargedRingField(1.0, 1.0, permittivity=np.inf),
    ],
)
def test_ring_fields_reject_invalid_physical_and_numerical_parameters(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


def test_ring_geometry_is_normalized_and_read_only() -> None:
    ring = ChargedRingField(1.0e-9, 1.0, center=(1.0, 2.0, 3.0), normal=(0.0, 3.0, 4.0))

    assert ring.dimension == 3
    assert ring.charge == 1.0e-9
    assert ring.permittivity == epsilon_0
    np.testing.assert_allclose(ring.normal, (0.0, 0.6, 0.8))
    with pytest.raises(ValueError):
        ring.center[0] = 0.0
    with pytest.raises(ValueError):
        ring.normal[0] = 1.0
