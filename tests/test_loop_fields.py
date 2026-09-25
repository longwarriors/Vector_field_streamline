"""Independent numerical contracts for circular-current-loop fields."""

from __future__ import annotations

import numpy as np
import pytest

from vectorviz import CircularLoopField, MagneticDipoleField


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return a proper 3D rotation from Rodrigues' formula."""

    unit = np.asarray(axis, dtype=float)
    unit /= np.linalg.norm(unit)
    x, y, z = unit
    cross = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _biot_savart_oracle(
    points: np.ndarray,
    *,
    current: float,
    radius: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    permeability: float,
    order: int = 256,
) -> np.ndarray:
    """Integrate the defining line integral without using the closed form."""

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
    nodes, weights = np.polynomial.legendre.leggauss(order)
    angles = np.pi * (nodes + 1.0)
    weights = np.pi * weights
    cosine = np.cos(angles)
    sine = np.sin(angles)
    radial_directions = (
        cosine[:, np.newaxis] * first_basis + sine[:, np.newaxis] * second_basis
    )
    tangents = -sine[:, np.newaxis] * first_basis + cosine[:, np.newaxis] * second_basis
    sources = center_array + radius * radial_directions
    values = np.empty_like(flat_points)

    for index, point in enumerate(flat_points):
        displacement = point - sources
        integrand = (
            radius
            * np.cross(tangents, displacement)
            / np.linalg.norm(displacement, axis=1)[:, np.newaxis] ** 3
        )
        integral = np.sum(weights[:, np.newaxis] * integrand, axis=0)
        values[index] = permeability * current / (4.0 * np.pi) * integral

    return values.reshape(original_shape)


def test_circular_loop_matches_axis_solution_and_preserves_batch_shape() -> None:
    current = 2.3
    radius = 0.7
    permeability = 1.6
    center = np.array((0.2, -0.3, 0.4))
    normal = np.array((0.0, 0.0, 1.0))
    axial_offsets = np.array((0.0, 0.5, -1.2, 2.0)).reshape(2, 2)
    points = center + axial_offsets[..., np.newaxis] * normal
    field = CircularLoopField(
        current=current,
        radius=radius,
        center=center,
        normal=normal,
        permeability=permeability,
    )

    values = field.evaluate(points)

    expected_magnitude = (
        permeability
        * current
        * radius**2
        / (2.0 * (radius**2 + axial_offsets**2) ** 1.5)
    )
    assert values.shape == points.shape
    assert values.dtype == np.float64
    np.testing.assert_allclose(
        values,
        expected_magnitude[..., np.newaxis] * normal,
        rtol=4.0e-15,
        atol=2.0e-15,
    )


def test_circular_loop_reports_the_ideal_wire_as_nan_without_softening() -> None:
    field = CircularLoopField(current=0.0, radius=1.25, permeability=1.0)

    values = field.evaluate(((1.25, 0.0, 0.0), (1.25, 0.0, 1.0e-12)))

    assert np.all(np.isnan(values[0]))
    np.testing.assert_array_equal(values[1], (0.0, 0.0, 0.0))

    normal = np.array((1.0, 1.0, 1.0))
    radial_direction = np.array((1.0, -1.0, 0.0)) / np.sqrt(2.0)
    rotated = CircularLoopField(1.0, 1.25, normal=normal, permeability=1.0)
    assert np.all(np.isnan(rotated.evaluate(1.25 * radial_direction)))
    assert np.all(np.isfinite(rotated.evaluate(1.25 * radial_direction + 1.0e-12 * normal)))


def test_circular_loop_uses_the_regular_near_axis_limit() -> None:
    current = -1.7
    radius = 0.8
    permeability = 1.3
    rho = 1.0e-10
    z = 0.35
    field = CircularLoopField(
        current=current,
        radius=radius,
        permeability=permeability,
    )

    value = field.evaluate((rho, 0.0, z))

    s = radius**2 + z**2
    coefficient = permeability * current * radius**2 / 2.0
    axis_field = coefficient / s**1.5
    axis_second_derivative = 3.0 * coefficient * (4.0 * z**2 - radius**2) / s**3.5
    axis_third_derivative = (
        15.0 * coefficient * z * (3.0 * radius**2 - 4.0 * z**2) / s**4.5
    )
    expected_radial = (
        3.0 * coefficient * z * rho / (2.0 * s**2.5)
        + axis_third_derivative * rho**3 / 16.0
    )
    expected_axial = axis_field - axis_second_derivative * rho**2 / 4.0
    np.testing.assert_allclose(
        value,
        (expected_radial, 0.0, expected_axial),
        rtol=2.0e-14,
        atol=1.0e-25,
    )

    near_branch_rho = 5.0e-4 * np.sqrt(s)
    near_branch_point = np.array((near_branch_rho, 0.0, z))
    oracle = _biot_savart_oracle(
        near_branch_point,
        current=current,
        radius=radius,
        permeability=permeability,
    )
    np.testing.assert_allclose(
        field.evaluate(near_branch_point),
        oracle,
        rtol=2.0e-10,
        atol=2.0e-14 * abs(axis_field),
    )


def test_circular_loop_stays_finite_when_m_rounds_to_one() -> None:
    distance = 1.0e-9
    field = CircularLoopField(current=1.0, radius=1.0, permeability=1.0)

    value = field.evaluate((1.0 + distance, 0.0, distance))

    # Locally the loop approaches a straight +y-directed wire. At equal radial
    # and axial offsets its leading field components have equal magnitude.
    leading = 1.0 / (4.0 * np.pi * distance)
    assert np.all(np.isfinite(value))
    np.testing.assert_allclose(value, (leading, 0.0, -leading), rtol=5.0e-7, atol=0.0)


def test_analytic_loop_matches_an_independent_biot_savart_oracle() -> None:
    parameters = {
        "current": -1.4,
        "radius": 0.85,
        "center": (0.15, -0.2, 0.3),
        "normal": (0.3, -0.4, 0.5),
        "permeability": 1.1,
    }
    analytic = CircularLoopField(**parameters)
    center = np.asarray(parameters["center"])
    normal = np.asarray(parameters["normal"], dtype=float)
    normal /= np.linalg.norm(normal)
    candidates = np.random.default_rng(20260818).uniform(-1.5, 1.5, size=(80, 3)) + center
    displacement = candidates - center
    axial = displacement @ normal
    radial = np.linalg.norm(displacement - axial[:, np.newaxis] * normal, axis=1)
    wire_distance = np.hypot(radial - parameters["radius"], axial)
    points = candidates[wire_distance >= 0.25 * parameters["radius"]][:12].reshape(2, 2, 3, 3)

    expected = _biot_savart_oracle(points, **parameters)
    actual = analytic.evaluate(points)

    assert expected.shape == points.shape
    np.testing.assert_allclose(actual, expected, rtol=3.0e-11, atol=2.0e-12)


def test_biot_savart_oracle_matches_axis_solution_and_self_converges() -> None:
    current = 1.8
    radius = 0.6
    permeability = 0.9
    z = np.array((-1.1, 0.0, 0.75))
    points = np.column_stack((np.zeros_like(z), np.zeros_like(z), z))

    values = _biot_savart_oracle(
        points,
        current=current,
        radius=radius,
        permeability=permeability,
    )

    expected_z = permeability * current * radius**2 / (2.0 * (radius**2 + z**2) ** 1.5)
    np.testing.assert_allclose(values[:, :2], 0.0, atol=3.0e-13)
    np.testing.assert_allclose(values[:, 2], expected_z, rtol=3.0e-12, atol=0.0)

    challenging = np.array((0.55, 0.0, 0.15))
    coarse = _biot_savart_oracle(
        challenging,
        current=current,
        radius=radius,
        permeability=permeability,
        order=64,
    )
    refined = _biot_savart_oracle(
        challenging,
        current=current,
        radius=radius,
        permeability=permeability,
        order=256,
    )
    reference = CircularLoopField(
        current=current,
        radius=radius,
        permeability=permeability,
    ).evaluate(challenging)
    assert np.linalg.norm(refined - reference) < np.linalg.norm(coarse - reference) / 100.0
    np.testing.assert_allclose(refined, reference, rtol=3.0e-11, atol=3.0e-12)


def test_circular_loop_converges_to_its_magnetic_dipole_far_field() -> None:
    current = 2.1
    radius = 0.4
    permeability = 1.7
    loop = CircularLoopField(
        current=current,
        radius=radius,
        permeability=permeability,
    )
    dipole = MagneticDipoleField(
        moment=(0.0, 0.0, current * np.pi * radius**2),
        permeability=permeability,
    )
    direction = np.array((2.0, -1.0, 3.0))
    direction /= np.linalg.norm(direction)
    points = np.array((40.0 * radius * direction, 80.0 * radius * direction))

    actual = loop.evaluate(points)
    expected = dipole.evaluate(points)

    relative_errors = np.linalg.norm(actual - expected, axis=1) / np.linalg.norm(expected, axis=1)
    assert relative_errors[1] < 1.6e-4
    assert 3.8 < relative_errors[0] / relative_errors[1] < 4.2

    extreme_point = 1.0e8 * radius * direction
    np.testing.assert_allclose(
        loop.evaluate(extreme_point),
        dipole.evaluate(extreme_point),
        rtol=3.0e-14,
        atol=0.0,
    )


def test_circular_loop_is_covariant_under_rigid_rotation() -> None:
    rotation = _rotation(np.array((0.2, -0.7, 0.4)), 0.83)
    center = np.array((0.3, -0.2, 0.5))
    normal = np.array((0.4, 0.1, 0.9))
    points = np.array(((0.8, -0.6, 0.2), (-0.4, 0.7, 1.1), (1.2, 0.3, -0.5)))
    field = CircularLoopField(1.6, 0.75, center=center, normal=normal, permeability=1.2)
    rotated = CircularLoopField(
        1.6,
        0.75,
        center=rotation @ center,
        normal=rotation @ normal,
        permeability=1.2,
    )

    expected = field.evaluate(points) @ rotation.T
    actual = rotated.evaluate(points @ rotation.T)

    np.testing.assert_allclose(actual, expected, rtol=8.0e-14, atol=2.0e-15)


def test_flux_function_has_the_axis_limit_and_generates_the_loop_field() -> None:
    field = CircularLoopField(1.3, 0.9, permeability=1.1)
    point = np.array((0.55, 0.0, 0.35))
    step = 2.0e-5
    radial_step = np.array((step, 0.0, 0.0))
    axial_step = np.array((0.0, 0.0, step))

    psi = field.flux_function(np.array(((0.0, 0.0, 0.2), point)))
    radial_derivative = (
        field.flux_function(point + radial_step)
        - field.flux_function(point - radial_step)
    ) / (2.0 * step)
    axial_derivative = (
        field.flux_function(point + axial_step)
        - field.flux_function(point - axial_step)
    ) / (2.0 * step)
    value = field.evaluate(point)

    assert psi.shape == (2,)
    assert psi.dtype == np.float64
    assert psi[0] == 0.0
    np.testing.assert_allclose(value[0], -axial_derivative / point[0], rtol=3.0e-9)
    np.testing.assert_allclose(value[2], radial_derivative / point[0], rtol=3.0e-9)
    assert np.isnan(field.flux_function((0.9, 0.0, 0.0)))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CircularLoopField(np.nan, 1.0),
        lambda: CircularLoopField(np.inf, 1.0),
        lambda: CircularLoopField(1.0, 0.0),
        lambda: CircularLoopField(1.0, -1.0),
        lambda: CircularLoopField(1.0, np.inf),
        lambda: CircularLoopField(1.0, 1.0, center=(0.0, 0.0)),
        lambda: CircularLoopField(1.0, 1.0, center=(0.0, 0.0, np.nan)),
        lambda: CircularLoopField(1.0, 1.0, normal=(0.0, 0.0, 0.0)),
        lambda: CircularLoopField(1.0, 1.0, normal=(0.0, 1.0)),
        lambda: CircularLoopField(1.0, 1.0, permeability=0.0),
        lambda: CircularLoopField(1.0, 1.0, permeability=np.inf),
    ],
)
def test_loop_fields_reject_invalid_physical_and_numerical_parameters(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


def test_loop_geometry_is_normalized_and_read_only() -> None:
    field = CircularLoopField(2.0, 0.7, center=(0.1, 0.2, 0.3), normal=(0.0, 0.0, 4.0))

    np.testing.assert_array_equal(field.center, (0.1, 0.2, 0.3))
    np.testing.assert_array_equal(field.normal, (0.0, 0.0, 1.0))
    assert field.current == 2.0
    assert field.radius == 0.7
    assert not field.center.flags.writeable
    assert not field.normal.flags.writeable
    with pytest.raises(ValueError, match="shape"):
        field.evaluate((1.0, 2.0))

    huge_normal = CircularLoopField(1.0, 1.0, normal=(1.0e308, 1.0e308, 1.0e308))
    np.testing.assert_allclose(huge_normal.normal, np.ones(3) / np.sqrt(3.0), rtol=2.0e-16)



def _point_in_loop_frame(
    loop: CircularLoopField, rho: float, z: float, angle: float
) -> np.ndarray:
    radial_axis = np.cross(loop.normal, (1.0, 0.0, 0.0))
    radial_axis /= np.linalg.norm(radial_axis)
    other_axis = np.cross(loop.normal, radial_axis)
    direction = np.cos(angle) * radial_axis + np.sin(angle) * other_axis
    return loop.center + rho * direction + z * loop.normal


def test_single_point_evaluation_matches_the_one_row_batch_bit_for_bit() -> None:
    # The tracer evaluates one point per solver stage through a scalar fast
    # path; it must return exactly the bytes the batch code returns for that
    # point in every region: general, near the wire, far field (small m), near
    # the axis, on the axis, on the filament, on each branch threshold and
    # where the squared wire distance underflows to zero.
    rng = np.random.default_rng(20260926)
    loops = (
        CircularLoopField(1.0, 1.0, normal=(0.0, 1.0, 0.0)),
        CircularLoopField(-2.5, 0.7, center=(0.2, -0.1, 0.3), normal=(0.3, 0.4, 0.866)),
    )
    for loop in loops:
        radius = loop.radius
        general = zip(
            rng.uniform(0.0, 3.0, size=64),
            rng.uniform(-3.0, 3.0, size=64),
            rng.uniform(0.0, 2.0 * np.pi, size=64),
            strict=True,
        )
        near_wire = zip(
            10.0 ** rng.uniform(-7.0, -0.5, size=32),
            rng.uniform(0.0, 2.0 * np.pi, size=32),
            rng.uniform(0.0, 2.0 * np.pi, size=32),
            strict=True,
        )
        far_field = zip(
            rng.uniform(20.0, 200.0, size=16),
            rng.uniform(-200.0, 200.0, size=16),
            strict=True,
        )
        cases = [(float(rho), float(z), float(angle)) for rho, z, angle in general]
        cases += [
            (radius + distance * np.cos(phase), distance * np.sin(phase), angle)
            for distance, phase, angle in near_wire
        ]
        cases += [(float(rho), float(z), 0.7) for rho, z in far_field]
        cases += [(1.0e-5 * radius, z, 0.3) for z in (-2.0, 0.0, 0.4)]
        cases += [(0.0, z, 0.0) for z in (-1.0, 0.0, 2.0)]
        cases += [(radius, 0.0, angle) for angle in (0.0, 1.0)]

        for rho, z, angle in cases:
            _assert_single_point_matches_batch(loop, _point_in_loop_frame(loop, rho, z, angle))

    # On the aligned unit loop rho and z are exact, so these points sit on the
    # branch thresholds: rho = 1e-3 * sqrt(a^2 + z^2) (near axis), wire/q = 0.1
    # and 4 a rho / q = 1e-2. Their float neighbours fall on either side.
    unit_loop = CircularLoopField(1.0, 1.0)
    near_axis = (1.0e-3, 0.0)
    wire_ratio = (1.5, float(np.nextafter(np.sqrt(5.0 / 12.0), np.inf)))
    small_parameter = (1.0, float(np.sqrt(396.0)))
    assert near_axis[0] == 1.0e-3 * np.sqrt(1.0 + near_axis[1] ** 2)
    assert (0.25 + wire_ratio[1] * wire_ratio[1]) / (6.25 + wire_ratio[1] * wire_ratio[1]) == 0.1
    assert 4.0 / (4.0 + small_parameter[1] * small_parameter[1]) == 1.0e-2
    for rho, z, varied in ((*near_axis, 0), (*wire_ratio, 2), (*small_parameter, 2)):
        point = np.array([rho, 0.0, z])
        for direction in (-np.inf, np.inf):
            neighbour = point.copy()
            neighbour[varied] = np.nextafter(point[varied], direction)
            _assert_single_point_matches_batch(unit_loop, neighbour)
        _assert_single_point_matches_batch(unit_loop, point)

    # (radius - rho)^2 + z^2 underflows to zero off the filament; the batch
    # path then divides by zero and returns nan, as v0.3.2 did.
    tiny_loop = CircularLoopField(1.0, 1.0e-160)
    with np.errstate(divide="ignore", invalid="ignore"):
        _assert_single_point_matches_batch(tiny_loop, np.array([1.0000001e-160, 0.0, 1.0e-167]))


def _assert_single_point_matches_batch(loop: CircularLoopField, point: np.ndarray) -> None:
    single = loop.evaluate(point)
    batch = loop.evaluate(point[np.newaxis, :])[0]
    assert single.shape == (3,)
    assert single.tobytes() == batch.tobytes(), (point, single, batch)
