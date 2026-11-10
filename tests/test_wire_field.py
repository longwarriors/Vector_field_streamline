"""Physical identities and singular geometry for infinite parallel wires."""

import numpy as np
import pytest
from scipy.constants import mu_0

from vectorviz import InfiniteWireField


def test_wire_right_hand_rule_inverse_radius_and_axial_invariance() -> None:
    field = InfiniteWireField(3.0)
    points = np.array(((2.0, 0.0, 0.0), (0.0, -4.0, 91.0), (-1.0, 0.0, -8.0)))
    coefficient = mu_0 * 3.0 / (2.0 * np.pi)
    expected = coefficient * np.array(((0.0, 0.5, 0.0), (0.25, 0.0, 0.0), (0.0, -1.0, 0.0)))
    np.testing.assert_allclose(field(points), expected, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(InfiniteWireField(-3.0)(points), -expected, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(
        InfiniteWireField(3.0, direction=(0, 0, -7))(points), -expected, rtol=2e-15, atol=0.0
    )


def test_parallel_wire_superposition_cancellation_and_scalar_broadcast() -> None:
    positions = ((-1.0, 0.0, 0.0), (1.0, 0.0, 10.0))
    same = InfiniteWireField(2.0, positions, permeability=2.0 * np.pi)
    opposite = InfiniteWireField((2.0, -2.0), positions, permeability=2.0 * np.pi)
    np.testing.assert_array_equal(same((0.0, 0.0, 0.0)), np.zeros(3))
    np.testing.assert_allclose(opposite((0.0, 0.0, 0.0)), (0.0, 4.0, 0.0))
    points = np.array(((0.5, 0.7, -2.0), (1.4, -0.2, 4.0)))
    independent = sum(
        InfiniteWireField(current, position, permeability=2.0 * np.pi)(points)
        for current, position in zip((2.0, -2.0), positions, strict=True)
    )
    np.testing.assert_allclose(opposite(points), independent, rtol=2e-15, atol=0.0)


def test_wire_rotation_translation_batch_and_single_point_agree() -> None:
    angle = 0.71
    rotation = np.array(
        ((np.cos(angle), 0, np.sin(angle)), (0, 1, 0), (-np.sin(angle), 0, np.cos(angle)))
    )
    translation = np.array((0.2, -0.6, 1.5))
    points = np.array([[[1.0, 2.0, -3.0], [-0.1, 0.4, 0.7]], [[2.2, 0.0, 0.0], [4, 3, 2]]])
    original = InfiniteWireField(1.5)
    rotated = InfiniteWireField(1.5, translation, direction=rotation[:, 2] * 4.0)
    transformed = points @ rotation.T + translation
    values = rotated(transformed)
    assert values.shape == points.shape
    assert values.dtype == np.float64
    assert values.flags.writeable
    np.testing.assert_allclose(values, original(points) @ rotation.T, rtol=2e-14, atol=1e-22)
    for point, value in zip(transformed.reshape(-1, 3), values.reshape(-1, 3), strict=True):
        np.testing.assert_allclose(rotated(point), value, rtol=2e-15, atol=1e-22)
    assert rotated(np.empty((2, 0, 3))).shape == (2, 0, 3)


def test_singular_entire_wire_including_zero_current_and_rotated_geometry() -> None:
    field = InfiniteWireField((1.0, 0.0), ((0, 0, 0), (1, 2, 3)))
    points = np.array(((0, 0, -5), (1, 2, 900), (1, 2, 3)))
    assert np.isnan(field(points)).all()
    for point in points:
        assert np.isnan(field(point)).all()
    rotated = InfiniteWireField(0.0, (2.0, 3.0, -1.0), direction=(1.0, 2.0, 3.0))
    on_wire = rotated.positions[0] + 4.0 * rotated.direction
    assert np.isnan(rotated(on_wire)).all()
    assert np.isnan(rotated(on_wire[None, :])).all()
    np.testing.assert_array_equal(rotated(on_wire + np.array((0, 0, 1e-10))), np.zeros(3))


def test_wire_preserves_nonfinite_samples_without_contaminating_valid_neighbors() -> None:
    field = InfiniteWireField(1.0)
    missing = (np.nan, 0.0, 1.0)
    assert np.isnan(field(missing)).all()
    values = field((missing, (1.0, 0.0, 1.0)))
    assert np.isnan(values[0]).all()
    np.testing.assert_array_equal(values[1], field((1.0, 0.0, 1.0)))


def test_wire_roundoff_bound_does_not_hide_resolved_small_radial_distance() -> None:
    field = InfiniteWireField(1.0, permeability=2.0 * np.pi)
    # A large independent axial coordinate must not create a thick exclusion tube.
    point = (1e-20, 0.0, 1e20)
    np.testing.assert_allclose(field(point), (0.0, 1e20, 0.0), rtol=2e-15)
    np.testing.assert_allclose(field([point])[0], field(point), rtol=2e-15)
    # The 1/r law remains representable even when r squared would overflow/underflow.
    for radius in (1e-200, 1e200):
        np.testing.assert_allclose(field((radius, 0.0, 0.0)), (0.0, 1.0 / radius, 0.0))


def test_wire_collinearity_roundoff_bound_stays_finite_at_large_coordinates() -> None:
    field = InfiniteWireField(1.0, direction=(1.0, 1.0, 0.0), permeability=2.0 * np.pi)
    point = (1.7e308, 1.0e308, 0.0)
    # The perpendicular distance is (1.7e308 - 1e308) / sqrt(2),
    # so the unit-prefactor field points in -z and is still representable.
    expected = np.array((0.0, 0.0, -np.sqrt(2.0) / 0.7e308))
    np.testing.assert_allclose(field(point), expected, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(field([point])[0], expected, rtol=2e-15, atol=0.0)


def test_wire_geometry_is_copied_readonly_and_direction_normalization_is_scaled() -> None:
    positions = np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
    currents = np.array((1.0, -1.0))
    direction = np.array((1e300, 0.0, 1e300))
    field = InfiniteWireField(currents, positions, direction=direction, permeability=1.0)
    positions[:] = 0.0
    currents[:] = 0.0
    direction[:] = 0.0
    assert field.dimension == 3
    assert field.permeability == 1.0
    np.testing.assert_array_equal(field.positions, ((1, 2, 3), (4, 5, 6)))
    np.testing.assert_array_equal(field.currents, (1.0, -1.0))
    np.testing.assert_allclose(field.direction, np.array((1, 0, 1)) / np.sqrt(2))
    for array in (field.positions, field.currents, field.direction):
        assert not array.flags.writeable


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": ()},
        {"position": ((0, 0),)},
        {"position": np.empty((0, 3))},
        {"position": (0, np.inf, 0)},
        {"current": (1, 2)},
        {"current": [[1]]},
        {"current": np.nan},
        {"direction": (0, 1)},
        {"direction": (0, 0, 0)},
        {"direction": (0, np.nan, 1)},
        {"permeability": 0},
        {"permeability": -1},
        {"permeability": np.inf},
        {"permeability": [1]},
    ],
)
def test_wire_rejects_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        InfiniteWireField(**{"current": 1.0, **kwargs})


@pytest.mark.parametrize("points", [1.0, (1, 2), np.zeros((2, 4))])
def test_wire_rejects_wrong_coordinate_shape(points: object) -> None:
    with pytest.raises(ValueError, match="shape"):
        InfiniteWireField(1.0)(points)
