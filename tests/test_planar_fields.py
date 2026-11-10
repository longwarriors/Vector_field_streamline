"""Independent polynomial and ODE benchmarks for the true planar models."""

from __future__ import annotations

import numpy as np
import pytest

from vectorviz import Domain, FieldLineTracer, TerminationReason, TraceOptions
from vectorviz.planar_fields import LaplaceField, SlopeField


@pytest.mark.parametrize("order", [1, 2, 3])
def test_laplace_low_orders_match_cartesian_polynomials(order: int) -> None:
    points = np.array([[[0, 0], [1, 2]], [[-2, 1], [3, -2]]])
    x, y = points[..., 0], points[..., 1]
    expected = {
        1: (np.full_like(x, -1), np.zeros_like(y)),
        2: (-x, y),
        3: (y**2 - x**2, 2 * x * y),
    }[order]
    potentials = {1: x, 2: (x**2 - y**2) / 2, 3: x**3 / 3 - x * y**2}
    field = LaplaceField(order)
    actual = field(points)
    assert actual.shape == points.shape
    assert actual.dtype == np.float64
    assert actual.flags.writeable
    np.testing.assert_allclose(actual, np.stack(expected, axis=-1), atol=1e-14)
    np.testing.assert_allclose(field.potential(points), potentials[order], atol=1e-14)
    np.testing.assert_allclose(field(points[0, 1]), actual[0, 1], atol=0)


def test_laplace_rotation_translation_and_length_units() -> None:
    angle = 0.71
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = np.array((2.1, -0.7))
    local = np.array(((0.2, -0.4), (0.8, 0.1), (-0.5, 0.6)))
    field = LaplaceField(4, -2.5, length_scale=3, angle=angle, center=center)
    baseline = LaplaceField(4)
    transformed = local @ rotation.T * 3 + center
    np.testing.assert_allclose(field(transformed), -2.5 * baseline(local) @ rotation.T, atol=1e-14)
    np.testing.assert_allclose(
        field.potential(transformed), -7.5 * baseline.potential(local), atol=1e-14
    )
    center[:] = 99
    np.testing.assert_array_equal(field.center, (2.1, -0.7))
    assert not field.center.flags.writeable
    assert (field.dimension, field.order, field.strength, field.length_scale, field.angle) == (
        2,
        4,
        -2.5,
        3,
        angle,
    )


def test_laplace_field_is_curl_free_divergence_free_and_negative_gradient() -> None:
    field = LaplaceField(5, 0.8, length_scale=1.4, angle=-0.31, center=(0.2, -0.1))
    points = np.array(((0.4, 0.8), (-0.7, 0.3), (1.1, -0.2)))
    h = 1e-5
    offsets = h * np.eye(2)
    jacobian = np.stack([(field(points + d) - field(points - d)) / (2 * h) for d in offsets])
    gradient = np.stack(
        [(field.potential(points + d) - field.potential(points - d)) / (2 * h) for d in offsets],
        axis=-1,
    )
    np.testing.assert_allclose(jacobian[0, :, 0] + jacobian[1, :, 1], 0, atol=2e-10)
    np.testing.assert_allclose(jacobian[0, :, 1] - jacobian[1, :, 0], 0, atol=2e-10)
    np.testing.assert_allclose(field(points), -gradient, atol=1e-10)


@pytest.mark.parametrize("order", [2, 3])
def test_laplace_traces_preserve_harmonic_conjugate(order: int) -> None:
    field = LaplaceField(order)
    tracer = FieldLineTracer(
        field,
        domain=Domain((-2, -2), (2, 2)),
        options=TraceOptions(max_arc_length=12, max_step=0.05, rtol=1e-9, atol=1e-11),
    )
    result = tracer.trace((0.7, 0.3), direction="both")
    x, y = result.points.T
    invariant = x * y if order == 2 else x**2 * y - y**3 / 3
    assert np.ptp(invariant) < 2e-8
    assert np.all(np.diff(field.potential(result.points)) <= 1e-10)
    assert result.forward.termination is TerminationReason.DOMAIN_EXIT
    assert result.backward.termination is TerminationReason.DOMAIN_EXIT
    null = tracer.trace((0, 0))
    assert null.forward.termination is TerminationReason.NULL_FIELD


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"order": 0}, "positive integer"),
        ({"order": 2.5}, "positive integer"),
        ({"order": True}, "positive integer"),
        ({"strength": np.nan}, "strength"),
        ({"length_scale": 0}, "length_scale"),
        ({"length_scale": np.inf}, "length_scale"),
        ({"angle": np.inf}, "angle"),
        ({"center": (0, 0, 0)}, "center"),
        ({"center": (0, np.nan)}, "center"),
    ],
)
def test_laplace_rejects_invalid_parameters(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        LaplaceField(**kwargs)


def test_laplace_nonfinite_overflow_zero_strength_and_empty_batches() -> None:
    points = np.array(((np.nan, 0), (0, np.inf), (1e200, 1e200)))
    field = LaplaceField(5)
    assert np.isnan(field(points)).all()
    assert np.isnan(field.potential(points)).all()
    zero = LaplaceField(5, 0)
    np.testing.assert_array_equal(zero(points[-1]), (0, 0))
    assert zero.potential(points[-1]) == 0
    assert np.isnan(zero(points[:2])).all()
    assert np.isnan(zero.potential(points[:2])).all()
    assert field(np.empty((3, 0, 2))).shape == (3, 0, 2)
    assert field.potential(np.empty((3, 0, 2))).shape == (3, 0)
    with pytest.raises(ValueError, match="shape"):
        field((1, 2, 3))
    with pytest.raises(ValueError, match="shape"):
        field.potential(1)


def test_slope_batch_contract_constant_broadcast_and_nonfinite_values() -> None:
    points = np.array([[[0, 0], [1, 2]], [[-2, 1], [3, -2]]])
    field = SlopeField(lambda x, y: x - y)
    assert field.dimension == 2
    actual = field(points)
    assert actual.shape == points.shape
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual[..., 0], 1)
    np.testing.assert_array_equal(actual[..., 1], points[..., 0] - points[..., 1])
    np.testing.assert_array_equal(field(points[0, 1]), (1, -1))
    constant = SlopeField(lambda x, y: 0)
    np.testing.assert_array_equal(constant((0, 0)), (1, 0))
    assert constant(np.empty((0, 2))).shape == (0, 2)
    assert np.isnan(constant(((np.nan, 0), (0, np.inf)))).all()
    assert np.isnan(SlopeField(lambda x, y: 1 / x)((0, 1))).all()


def test_slope_rejects_noncallables_bad_output_shape_and_coordinates() -> None:
    with pytest.raises(ValueError, match="callable"):
        SlopeField("x-y")
    with pytest.raises(ValueError, match="batch shape"):
        SlopeField(lambda x, y: np.zeros(3))(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="shape"):
        SlopeField(lambda x, y: x)((1, 2, 3))


def test_slope_rejects_complex_values_instead_of_silently_discarding_them() -> None:
    field = SlopeField(lambda x, y: np.full(x.shape, 1 + 2j))
    with pytest.raises(ValueError, match="real"):
        field((0, 0))


@pytest.mark.parametrize("kind", ["linear", "logistic", "equilibrium"])
def test_slope_traces_match_closed_form_ode_solutions(kind: str) -> None:
    field = (
        SlopeField(lambda x, y: x - y) if kind == "linear" else SlopeField(lambda x, y: y * (1 - y))
    )
    seed = (0, 0 if kind == "equilibrium" else 0.4)
    result = FieldLineTracer(
        field,
        domain=Domain((-1, -3), (1, 3)),
        options=TraceOptions(max_arc_length=10, max_step=0.04, rtol=1e-10, atol=1e-12),
    ).trace(seed, direction="both")
    x, y = result.points.T
    if kind == "linear":
        expected = x - 1 + 1.4 * np.exp(-x)
    elif kind == "logistic":
        expected = 1 / (1 + 1.5 * np.exp(-x))
    else:
        expected = np.zeros_like(x)
    np.testing.assert_allclose(y, expected, atol=2e-9)
    assert np.all(np.diff(x) > 0)
    assert result.forward.termination is TerminationReason.DOMAIN_EXIT
    assert result.backward.termination is TerminationReason.DOMAIN_EXIT
