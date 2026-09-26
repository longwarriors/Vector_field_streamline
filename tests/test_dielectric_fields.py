"""Contracts for the dielectric and conducting sphere in a uniform field."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vectorviz import DielectricSphereField


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=float)
    unit /= np.linalg.norm(unit)
    x, y, z = unit
    cross = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _unit_directions(count: int, seed: int) -> np.ndarray:
    directions = np.random.default_rng(seed).normal(size=(count, 3))
    return directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]


def test_dielectric_sphere_is_uniform_inside_and_uniform_plus_dipole_outside() -> None:
    applied = np.array((0.0, 0.0, 2.5))
    radius = 0.8
    permittivity = 5.0
    sphere = DielectricSphereField(applied, radius, relative_permittivity=permittivity)
    generator = np.random.default_rng(11)
    points = np.concatenate(
        (
            0.75 * radius * _unit_directions(20, seed=1) * generator.uniform(0.0, 1.0, size=(20, 1)),
            generator.uniform(-2.0, 2.0, size=(60, 3)),
        )
    )
    distances = np.linalg.norm(points, axis=1)

    values = sphere.evaluate(points)

    interior = distances < radius
    assert 20 <= np.count_nonzero(interior) < points.shape[0]
    np.testing.assert_allclose(
        values[interior],
        np.broadcast_to(3.0 / (permittivity + 2.0) * applied, values[interior].shape),
        rtol=1.0e-15,
    )
    alpha = (permittivity - 1.0) / (permittivity + 2.0)
    unit = points[~interior] / distances[~interior][:, np.newaxis]
    dipole = alpha * radius**3 / distances[~interior][:, np.newaxis] ** 3 * (
        3.0 * (unit @ applied)[:, np.newaxis] * unit - applied
    )
    np.testing.assert_allclose(values[~interior], applied + dipole, rtol=1.0e-13)
    assert sphere.polarizability == pytest.approx(alpha)
    np.testing.assert_allclose(sphere.interior_field, 3.0 / 7.0 * applied)
    assert values.shape == points.shape
    assert sphere.evaluate((0.0, 0.0, 3.0)).shape == (3,)


def test_unit_permittivity_reduces_to_the_applied_field() -> None:
    sphere = DielectricSphereField((1.0, -2.0, 0.5), 1.0, relative_permittivity=1.0)
    points = np.random.default_rng(2).uniform(-2.0, 2.0, size=(20, 3))

    np.testing.assert_array_equal(sphere.evaluate(points), np.tile((1.0, -2.0, 0.5), (20, 1)))
    assert sphere.polarizability == 0.0


@pytest.mark.parametrize("permittivity", [1.0, 4.0, 80.0, math.inf])
def test_sphere_satisfies_the_interface_conditions(permittivity: float) -> None:
    applied = np.array((1.0, 0.0, 0.0))
    sphere = DielectricSphereField(applied, 1.0, relative_permittivity=permittivity)
    normals = _unit_directions(12, seed=5)
    offset = 1.0e-9

    outside = sphere.evaluate(normals * (1.0 + offset))
    inside = sphere.evaluate(normals * (1.0 - offset))

    # Tangential E is continuous; normal D is continuous.
    np.testing.assert_allclose(np.cross(outside, normals), np.cross(inside, normals), atol=1.0e-8)
    normal_outside = np.einsum("ij,ij->i", outside, normals)
    normal_inside = np.einsum("ij,ij->i", inside, normals)
    if math.isinf(permittivity):
        np.testing.assert_array_equal(inside, 0.0)
        np.testing.assert_allclose(np.cross(outside, normals), 0.0, atol=1.0e-8)
        np.testing.assert_allclose(normal_outside, 3.0 * (normals @ applied), rtol=1.0e-7)
    else:
        np.testing.assert_allclose(normal_outside, permittivity * normal_inside, rtol=1.0e-7)


def test_sphere_flux_function_is_continuous_and_generates_the_field() -> None:
    permittivity = 4.0
    sphere = DielectricSphereField((1.0, 0.0, 0.0), 1.0, relative_permittivity=permittivity)
    normals = _unit_directions(8, seed=9)

    np.testing.assert_allclose(
        sphere.flux_function(normals * (1.0 + 1.0e-9)),
        sphere.flux_function(normals * (1.0 - 1.0e-9)),
        rtol=1.0e-7,
    )
    step = 1.0e-6
    for point, scale in (
        (np.array((0.3, 0.4, 0.0)), permittivity),
        (np.array((-0.2, 0.9, 0.0)), permittivity),
        (np.array((1.5, 0.7, 0.0)), 1.0),
        (np.array((-1.1, 1.6, 0.0)), 1.0),
    ):
        axial_step = np.array((step, 0.0, 0.0))
        radial_step = np.array((0.0, step, 0.0))
        axial_derivative = (
            sphere.flux_function(point + axial_step) - sphere.flux_function(point - axial_step)
        ) / (2.0 * step)
        radial_derivative = (
            sphere.flux_function(point + radial_step) - sphere.flux_function(point - radial_step)
        ) / (2.0 * step)
        value = sphere.evaluate(point)
        rho = point[1]
        np.testing.assert_allclose(radial_derivative / rho, scale * value[0], rtol=1.0e-8)
        np.testing.assert_allclose(-axial_derivative / rho, scale * value[1], rtol=1.0e-8, atol=1.0e-12)
    assert sphere.flux_function((0.5, 0.0, 0.0)) == 0.0
    assert sphere.flux_function((2.0, 0.0, 0.0)) == 0.0


def test_conducting_sphere_has_no_interior_field_and_no_interior_flux() -> None:
    sphere = DielectricSphereField((0.0, 1.0, 0.0), 0.5, center=(1.0, 0.0, 0.0), relative_permittivity=math.inf)

    assert sphere.is_conductor
    assert sphere.polarizability == 1.0
    np.testing.assert_array_equal(sphere.interior_field, 0.0)
    np.testing.assert_array_equal(sphere.evaluate((1.1, 0.2, 0.1)), 0.0)
    assert sphere.flux_function((1.1, 0.2, 0.1)) == 0.0
    # The far field is the applied field plus a dipole of strength a^3 E0.
    far = sphere.evaluate((1.0, 4.0, 0.0))
    np.testing.assert_allclose(far, (0.0, 1.0 + 2.0 * 0.5**3 / 4.0**3, 0.0), rtol=1.0e-14)


def test_sphere_is_covariant_under_rigid_rotation() -> None:
    rotation = _rotation(np.array((0.3, -1.0, 0.7)), 1.1)
    applied = np.array((0.4, -0.2, 0.9))
    center = np.array((0.2, 0.1, -0.3))
    base = DielectricSphereField(applied, 0.7, center=center, relative_permittivity=3.0)
    rotated = DielectricSphereField(
        rotation @ applied, 0.7, center=rotation @ center, relative_permittivity=3.0
    )
    points = np.array(((0.5, 0.2, 0.9), (0.3, 0.1, -0.2), (-1.4, 0.9, 0.35)))

    np.testing.assert_allclose(
        rotated.evaluate(points @ rotation.T), base.evaluate(points) @ rotation.T, rtol=1.0e-12
    )
    np.testing.assert_allclose(
        rotated.flux_function(points @ rotation.T), base.flux_function(points), rtol=1.0e-12
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DielectricSphereField((0.0, 0.0, 0.0), 1.0, relative_permittivity=2.0),
        lambda: DielectricSphereField((np.nan, 0.0, 0.0), 1.0, relative_permittivity=2.0),
        lambda: DielectricSphereField((1.0, 0.0), 1.0, relative_permittivity=2.0),
        lambda: DielectricSphereField((1.0, 0.0, 0.0), 0.0, relative_permittivity=2.0),
        lambda: DielectricSphereField((1.0, 0.0, 0.0), np.inf, relative_permittivity=2.0),
        lambda: DielectricSphereField((1.0, 0.0, 0.0), 1.0, center=(0.0, 0.0), relative_permittivity=2.0),
        lambda: DielectricSphereField((1.0, 0.0, 0.0), 1.0, center=(0.0, np.inf, 0.0), relative_permittivity=2.0),
        lambda: DielectricSphereField((1.0, 0.0, 0.0), 1.0, relative_permittivity=0.5),
        lambda: DielectricSphereField((1.0, 0.0, 0.0), 1.0, relative_permittivity=np.nan),
    ],
)
def test_sphere_rejects_invalid_parameters(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


def test_sphere_geometry_is_read_only() -> None:
    sphere = DielectricSphereField((2.0, 0.0, 0.0), 1.5, center=(1.0, 2.0, 3.0), relative_permittivity=4.0)

    assert sphere.dimension == 3
    assert sphere.radius == 1.5
    assert sphere.relative_permittivity == 4.0
    assert not sphere.is_conductor
    with pytest.raises(ValueError):
        sphere.applied_field[0] = 0.0
    with pytest.raises(ValueError):
        sphere.center[0] = 0.0
