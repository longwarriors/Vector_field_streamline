"""Analytic and interface contracts of the ideal transverse magnetic shield."""

from __future__ import annotations

import numpy as np
import pytest

from vectorviz import Domain, FieldLineTracer, SphericalExclusion, TerminationReason, TraceOptions
from vectorviz.material_fields import MagneticShieldCylinderField


def test_shield_matches_independently_solved_four_boundary_equations() -> None:
    a, b, mu = 0.65, 1.2, 17.0
    field = MagneticShieldCylinderField((1.0, 0.0), a, b, relative_permeability=mu)
    # Magnetic scalar potentials -H0*(A*x), -H0*(C*x+D*x/r²),
    # -H0*(x+E*x/r²); independently solve H_t/B_n at a and b.
    matrix = np.array(
        (
            (1, -1, -1 / a**2, 0),
            (1, -mu, mu / a**2, 0),
            (0, 1, 1 / b**2, -1 / b**2),
            (0, mu, -mu / b**2, 1 / b**2),
        )
    )
    interior, uniform, shell_dipole, exterior_dipole = np.linalg.solve(matrix, (0, 0, 1, 1))
    points = np.array(((0, 0), (0.2, -0.4), (0.75, 0.3), (-0.4, 0.9), (1.4, 0.3), (2, -3)))
    expected = []
    for point in points:
        r2 = float(point @ point)
        if r2 < a**2:
            expected.append((interior, 0.0))
            continue
        dipole = np.array((1.0, 0.0)) / r2 - 2 * point[0] * point / r2**2
        if r2 < b**2:
            expected.append(mu * (np.array((uniform, 0.0)) + shell_dipole * dipole))
        else:
            expected.append(np.array((1.0, 0.0)) + exterior_dipole * dipole)
    np.testing.assert_allclose(field.evaluate(points), expected, rtol=2e-14, atol=1e-15)
    assert field.shielding_factor == pytest.approx(1 / interior, rel=1e-14)
    np.testing.assert_allclose(field.interior_field, (interior, 0), rtol=1e-14)
    assert field.evaluate((0, 0)).shape == (2,)
    assert field.evaluate(points.reshape(2, 3, 2)).shape == (2, 3, 2)
    assert field.evaluate(np.empty((0, 2))).shape == (0, 2)


@pytest.mark.parametrize("mu", [1.0, 4.0, 100.0, 1e5])
def test_both_boundaries_keep_normal_b_and_tangential_h_continuous(mu: float) -> None:
    field = MagneticShieldCylinderField((0.6, -0.8), relative_permeability=mu)
    angles = np.linspace(0.13, 2 * np.pi + 0.13, 16, endpoint=False)
    normals = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tangents = np.stack((-normals[:, 1], normals[:, 0]), axis=-1)
    for radius, inside_mu, outside_mu in ((0.7, 1, mu), (1.0, mu, 1)):
        inside = field.evaluate(radius * (1 - 1e-10) * normals)
        outside = field.evaluate(radius * (1 + 1e-10) * normals)
        np.testing.assert_allclose(
            np.sum(inside * normals, axis=-1),
            np.sum(outside * normals, axis=-1),
            atol=3e-9,
            rtol=2e-8,
        )
        np.testing.assert_allclose(
            np.sum(inside * tangents, axis=-1) / inside_mu,
            np.sum(outside * tangents, axis=-1) / outside_mu,
            atol=3e-9,
            rtol=2e-8,
        )
        np.testing.assert_allclose(
            field.flux_function(radius * (1 - 1e-10) * normals),
            field.flux_function(radius * (1 + 1e-10) * normals),
            atol=3e-9,
        )


def test_surface_values_are_from_the_radially_outward_side() -> None:
    field = MagneticShieldCylinderField((1.0, 0.0), relative_permeability=100)
    # At the top of either circle the applied field is tangential.
    at_inner = field.evaluate((0, field.inner_radius))
    np.testing.assert_allclose(at_inner, field.interior_field * 100, rtol=1e-14)
    at_outer = field.evaluate((0, field.outer_radius))
    just_inside_outer = field.evaluate((0, field.outer_radius - 1e-10))
    np.testing.assert_allclose(at_outer, just_inside_outer / 100, rtol=1e-9)


def test_flux_function_generates_b_in_each_region() -> None:
    field = MagneticShieldCylinderField((0.4, -0.7), center=(0.1, -0.2))
    points = np.array(((0.2, 0.1), (0.85, 0.2), (1.4, -0.8)))
    dx, dy = np.eye(2) * 1e-6
    d_dx = (field.flux_function(points + dx) - field.flux_function(points - dx)) / 2e-6
    d_dy = (field.flux_function(points + dy) - field.flux_function(points - dy)) / 2e-6
    np.testing.assert_allclose(np.stack((d_dy, -d_dx), axis=-1), field.evaluate(points), rtol=1e-8)
    assert field.flux_function(field.center) == 0.0


def test_uniform_thin_shell_high_permeability_and_far_field_limits() -> None:
    points = np.array(((0, 0), (0.3, -0.2), (0.8, 0.2), (1, 0), (2, 3)))
    uniform = MagneticShieldCylinderField((1, -2), relative_permeability=1)
    np.testing.assert_allclose(uniform.evaluate(points), np.tile((1, -2), (5, 1)), atol=1e-15)
    assert uniform.shielding_factor == 1
    thin = MagneticShieldCylinderField((1, 0), 1 - 1e-12, 1, relative_permeability=100)
    assert thin.shielding_factor == pytest.approx(1, abs=5e-11)
    strong = MagneticShieldCylinderField((1, 0), relative_permeability=1e8)
    assert strong.shielding_factor / 1e8 == pytest.approx((1 - 0.7**2) / 4, rel=1e-7)
    assert np.linalg.norm(strong.interior_field) < 1e-7
    # The transverse top of the wall concentrates flux, despite the cavity attenuation.
    assert strong.evaluate((0, 0.85))[0] > 5
    far = strong.evaluate((1e6, 0))
    np.testing.assert_allclose(far, (1, 0), atol=1e-12)


def test_extreme_finite_permeability_and_close_radii_do_not_overflow() -> None:
    thin_radius = np.nextafter(1.0, 0.0)
    field = MagneticShieldCylinderField((1, 0), thin_radius, 1, relative_permeability=1e308)
    points = np.array(((0, 0), (0, thin_radius), (0, 1), (1e200, 1e200)))
    values = field.evaluate(points)
    assert np.all(np.isfinite(values))
    assert np.isfinite(field.shielding_factor)
    np.testing.assert_allclose(values[0], (1 / field.shielding_factor, 0), rtol=1e-14)
    np.testing.assert_allclose(values[1, 0] / 1e308, values[0, 0], rtol=1e-14)
    np.testing.assert_allclose(values[-1], (1, 0), atol=1e-15)


def test_shield_rotates_translates_and_scales_with_its_geometry() -> None:
    angle = 0.91
    rotation = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    center = np.array((1.1, -0.4))
    points = np.array(((0, 0), (0.2, 0.4), (0.8, 0.2), (1.5, 0.3)))
    base = MagneticShieldCylinderField((0.6, -0.2))
    transformed = MagneticShieldCylinderField(
        rotation @ base.applied_field,
        2 * base.inner_radius,
        2 * base.outer_radius,
        center,
    )
    mapped = 2 * points @ rotation.T + center
    np.testing.assert_allclose(
        transformed.evaluate(mapped), base.evaluate(points) @ rotation.T, rtol=1e-14
    )
    np.testing.assert_allclose(
        transformed.flux_function(mapped), 2 * base.flux_function(points), atol=1e-20
    )


@pytest.mark.parametrize("method", ["DOP853", "Radau"])
def test_traces_cross_both_material_interfaces_and_conserve_flux(method: str) -> None:
    field = MagneticShieldCylinderField()
    tracer = FieldLineTracer(
        field,
        domain=Domain((-2, -2), (2, 2)),
        interfaces=tuple(SphericalExclusion((0, 0), radius) for radius in (0.7, 1.0)),
        options=TraceOptions(
            method=method,
            max_arc_length=8,
            max_step=0.06,
            rtol=2e-7,
            atol=1e-10,
            null_threshold=1e-16,
            output_step=0.025,
        ),
    )
    result = tracer.trace((0, 0.2), direction="both")
    assert result.forward.termination is TerminationReason.DOMAIN_EXIT
    assert result.backward.termination is TerminationReason.DOMAIN_EXIT
    radii = np.linalg.norm(result.points, axis=-1)
    for radius in (0.7, 1.0):
        assert np.count_nonzero(np.abs(radii - radius) < 1e-9) == 2
    # Absolute flux error in T m; normalized by the 1 microtesla far field
    # this is below 2e-7 m, including both material-interface restarts.
    assert (
        np.max(np.abs(field.flux_function(result.points) - field.flux_function((0, 0.2)))) < 2e-13
    )


def test_oblique_inner_surface_seed_exits_without_spurious_interface_sliding() -> None:
    field = MagneticShieldCylinderField()
    seed = np.array((0.4949747468305833, 0.4949747468305833))
    tracer = FieldLineTracer(
        field,
        domain=Domain((-3, -3), (3, 3)),
        interfaces=tuple(SphericalExclusion((0, 0), radius) for radius in (0.7, 1.0)),
        options=TraceOptions(
            method="DOP853",
            max_arc_length=18,
            max_step=0.035,
            rtol=2e-7,
            atol=1e-9,
            null_threshold=1e-14,
            output_step=0.045,
        ),
    )
    result = tracer.trace(seed, direction="both")
    assert result.forward.termination is TerminationReason.DOMAIN_EXIT
    assert result.backward.termination is TerminationReason.DOMAIN_EXIT
    assert np.max(np.abs(field.flux_function(result.points) - field.flux_function(seed))) < 2e-13


@pytest.mark.parametrize(
    "kwargs",
    [
        {"applied_field": (0, 0)},
        {"applied_field": (1, 2, 3)},
        {"applied_field": (np.nan, 0)},
        {"center": (1,)},
        {"center": (0, np.inf)},
        {"inner_radius": 0},
        {"inner_radius": -1},
        {"outer_radius": 0.7},
        {"outer_radius": np.inf},
        {"relative_permeability": 0.99},
        {"relative_permeability": np.inf},
        {"relative_permeability": np.nan},
    ],
)
def test_shield_rejects_invalid_model_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        MagneticShieldCylinderField(**kwargs)


def test_shield_point_dimensions_nonfinite_policy_and_read_only_geometry() -> None:
    applied, center = np.array((1.0, 0.0)), np.array((0.0, 0.0))
    field = MagneticShieldCylinderField(applied, center=center)
    applied[:] = center[:] = 9
    np.testing.assert_array_equal(field.applied_field, (1, 0))
    np.testing.assert_array_equal(field.center, (0, 0))
    with pytest.raises(ValueError):
        field.applied_field[0] = 3
    with pytest.raises(ValueError):
        field.center[0] = 3
    with pytest.raises(ValueError, match="shape"):
        field.evaluate((0, 0, 0))
    assert field.dimension == 2
    assert field.relative_permeability == 100
    with np.errstate(invalid="ignore"):
        assert np.all(np.isnan(field.evaluate(((np.nan, 0), (np.inf, 0)))))
        assert np.all(np.isnan(field.flux_function(((np.nan, 0), (np.inf, 0)))))
