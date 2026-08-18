"""Analytic correctness and batch-contract tests for vector fields."""

from __future__ import annotations

import numpy as np
import pytest

from vectorviz import (
    CompositeField,
    MagneticDipoleField,
    PointChargeField,
    UniformField,
)


def test_uniform_field_preserves_single_and_grid_shapes() -> None:
    field = UniformField((1.5, -2.0))

    single = field.evaluate((9.0, 4.0))
    grid = field(np.zeros((3, 4, 2)))

    assert single.shape == (2,)
    assert grid.shape == (3, 4, 2)
    assert single.dtype == np.float64
    assert grid.dtype == np.float64
    np.testing.assert_allclose(single, (1.5, -2.0))
    np.testing.assert_allclose(grid, np.broadcast_to((1.5, -2.0), grid.shape))
    # Evaluation must return writable storage rather than a broadcast view.
    assert grid.flags.writeable


def test_point_charge_matches_inverse_square_law_for_batches() -> None:
    # Choosing epsilon=1/(4*pi) makes the Coulomb prefactor exactly one.
    field = PointChargeField(
        charge=2.0,
        position=(0.0, 0.0, 0.0),
        permittivity=1.0 / (4.0 * np.pi),
    )
    points = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, 0.0, -4.0], [1.0, 2.0, 2.0]],
        ]
    )

    values = field.evaluate(points)

    assert values.shape == points.shape
    assert values.dtype == np.float64
    expected = 2.0 * points / np.linalg.norm(points, axis=-1)[..., None] ** 3
    np.testing.assert_allclose(values, expected, rtol=1.0e-13, atol=0.0)


def test_point_charge_superposition_and_singularity_are_explicit() -> None:
    field = PointChargeField(
        charge=(1.0, -1.0),
        position=((-1.0, 0.0), (1.0, 0.0)),
        permittivity=1.0 / (4.0 * np.pi),
    )

    # At the origin both contributions point in +x and have unit magnitude.
    np.testing.assert_allclose(field.evaluate((0.0, 0.0)), (2.0, 0.0))
    assert np.all(np.isnan(field.evaluate((-1.0, 0.0))))


def test_magnetic_dipole_axis_and_equator_values() -> None:
    # Choosing mu=4*pi makes the dipole prefactor exactly one.
    field = MagneticDipoleField(
        moment=(0.0, 0.0, 1.0),
        permeability=4.0 * np.pi,
    )
    points = np.array(((0.0, 0.0, 2.0), (2.0, 0.0, 0.0)))

    values = field.evaluate(points)

    assert values.shape == points.shape
    assert values.dtype == np.float64
    np.testing.assert_allclose(values[0], (0.0, 0.0, 0.25), atol=1.0e-15)
    np.testing.assert_allclose(values[1], (0.0, 0.0, -0.125), atol=1.0e-15)
    assert np.all(np.isnan(field.evaluate((0.0, 0.0, 0.0))))


def test_magnetic_dipole_has_inverse_cube_scaling_and_rotation_covariance() -> None:
    moment = np.array((0.3, -0.4, 0.5))
    point = np.array((1.2, -0.7, 0.9))
    field = MagneticDipoleField(moment)
    rotation = np.array(
        (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )

    value = field.evaluate(point)
    scaled = field.evaluate(3.0 * point)
    rotated = MagneticDipoleField(rotation @ moment).evaluate(rotation @ point)

    np.testing.assert_allclose(scaled, value / 3.0**3, rtol=2.0e-15, atol=0.0)
    np.testing.assert_allclose(rotated, rotation @ value, rtol=2.0e-15, atol=1.0e-30)


def test_composite_field_applies_weights_and_preserves_batch_shape() -> None:
    first = UniformField((1.0, 2.0))
    second = UniformField((-4.0, 3.0))
    field = CompositeField(first, second, weights=(2.0, -0.5))
    points = np.zeros((2, 3, 2))

    values = field.evaluate(points)

    assert values.shape == points.shape
    np.testing.assert_allclose(values, np.broadcast_to((4.0, 2.5), points.shape))
    assert field.fields == (first, second)
    np.testing.assert_allclose(field.weights, (2.0, -0.5))


def test_composite_rejects_dimension_or_weight_mismatch() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        CompositeField(UniformField((1.0, 0.0)), UniformField((1.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="one value per field"):
        CompositeField([UniformField((1.0, 0.0))], weights=(1.0, 2.0))


@pytest.mark.parametrize(
    "vector",
    [(), ((1.0, 2.0),), (1.0, np.nan), (1.0, np.inf)],
)
def test_uniform_field_rejects_invalid_vectors(vector: object) -> None:
    with pytest.raises(ValueError):
        UniformField(vector)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PointChargeField(1.0, ()),
        lambda: PointChargeField((1.0, 2.0), ((0.0, 0.0),)),
        lambda: PointChargeField(((1.0,),), (0.0, 0.0)),
        lambda: PointChargeField(np.nan, (0.0, 0.0)),
        lambda: PointChargeField(1.0, (0.0, np.inf)),
        lambda: PointChargeField(1.0, (0.0, 0.0), permittivity=0.0),
        lambda: PointChargeField(1.0, (0.0, 0.0), permittivity=np.inf),
    ],
)
def test_point_charge_rejects_invalid_source_data(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MagneticDipoleField((1.0, 0.0)),
        lambda: MagneticDipoleField((1.0, 0.0, 0.0), position=(0.0, 0.0)),
        lambda: MagneticDipoleField(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            position=((0.0, 0.0, 0.0),) * 3,
        ),
        lambda: MagneticDipoleField((np.nan, 0.0, 0.0)),
        lambda: MagneticDipoleField((1.0, 0.0, 0.0), position=(0.0, 0.0, np.inf)),
        lambda: MagneticDipoleField((1.0, 0.0, 0.0), permeability=0.0),
    ],
)
def test_magnetic_dipole_rejects_invalid_source_data(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


def test_composite_field_rejects_empty_invalid_and_ambiguous_inputs() -> None:
    field = UniformField((1.0, 0.0))

    with pytest.raises(ValueError, match="at least one VectorField"):
        CompositeField([])
    with pytest.raises(ValueError, match="at least one VectorField"):
        CompositeField([object()])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="additional positional"):
        CompositeField([field], field)
    with pytest.raises(ValueError, match="weights must be finite"):
        CompositeField(field, weights=(np.nan,))

    unweighted = CompositeField(field)
    np.testing.assert_array_equal(unweighted.weights, (1.0,))
    assert unweighted.weights.dtype == np.float64


@pytest.mark.parametrize(
    "field",
    [
        UniformField((1.0, 0.0)),
        PointChargeField(1.0, (0.0, 0.0)),
        MagneticDipoleField((0.0, 0.0, 1.0)),
    ],
)
def test_field_contract_rejects_wrong_coordinate_dimension(field: object) -> None:
    with pytest.raises(ValueError, match="shape"):
        field.evaluate((1.0,))  # type: ignore[union-attr]
