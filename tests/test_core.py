"""Tests for geometry primitives shared by field models and tracers."""

from __future__ import annotations

import numpy as np
import pytest

from vectorviz import Domain, SphericalExclusion, ToroidalExclusion


def test_domain_reports_geometry_and_batch_containment() -> None:
    domain = Domain(lower=(-2.0, -1.0), upper=(4.0, 3.0))

    assert domain.dimension == 2
    np.testing.assert_allclose(domain.extent, (6.0, 4.0))
    np.testing.assert_allclose(domain.center, (1.0, 1.0))
    assert bool(domain.contains((0.0, 0.0)))
    assert bool(domain.contains((-2.0, 3.0)))  # The domain is closed.

    contained = domain.contains(
        np.array(
            [
                [[0.0, 0.0], [4.0, 3.0]],
                [[4.01, 0.0], [-2.01, 0.0]],
            ]
        )
    )
    assert contained.shape == (2, 2)
    np.testing.assert_array_equal(contained, [[True, True], [False, False]])
    assert bool(domain.contains((4.01, 0.0), atol=0.02))
    assert domain.lower.dtype == np.float64
    assert domain.upper.dtype == np.float64
    with pytest.raises(ValueError, match="atol must be non-negative"):
        domain.contains((0.0, 0.0), atol=-1.0e-12)


def test_domain_margin_has_expected_sign_and_readonly_bounds() -> None:
    domain = Domain(lower=(-1.0, -2.0), upper=(2.0, 2.0))

    np.testing.assert_allclose(
        domain.margin(((0.0, 0.0), (-1.0, 1.0), (2.5, 0.0))),
        (1.0, 0.0, -0.5),
    )
    assert not domain.lower.flags.writeable
    assert not domain.upper.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        domain.lower[0] = 99.0


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ((0.0,), (0.0,)),
        ((0.0, 1.0), (1.0,)),
        ((0.0,), (np.inf,)),
        ((), ()),
    ],
)
def test_domain_rejects_invalid_bounds(lower: tuple[float, ...], upper: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        Domain(lower=lower, upper=upper)


def test_spherical_exclusion_margin_for_one_and_many_sources() -> None:
    one = SphericalExclusion(centers=(0.0, 0.0), radii=0.5)
    assert one.dimension == 2
    np.testing.assert_allclose(
        one.margin(((0.0, 0.0), (0.5, 0.0), (1.0, 0.0))),
        (-0.5, 0.0, 0.5),
    )

    many = SphericalExclusion(centers=((-1.0, 0.0), (2.0, 0.0)), radii=(0.25, 0.5))
    # The result is the signed distance to the nearest excluded surface.
    np.testing.assert_allclose(
        many.margin(((-1.0, 0.0), (1.5, 0.0), (0.0, 0.0))),
        (-0.25, 0.0, 0.75),
    )
    assert not many.centers.flags.writeable
    assert not many.radii.flags.writeable


@pytest.mark.parametrize(
    ("centers", "radii"),
    [
        (((0.0, 0.0), (1.0, 0.0)), (0.1,)),
        ((0.0, 0.0), 0.0),
        ((0.0, np.nan), 0.1),
        ([], 0.1),
        ([[]], 0.1),
    ],
)
def test_spherical_exclusion_rejects_invalid_geometry(centers: object, radii: object) -> None:
    with pytest.raises(ValueError):
        SphericalExclusion(centers=centers, radii=radii)


def test_toroidal_exclusion_margin_and_readonly_geometry() -> None:
    exclusion = ToroidalExclusion(
        center=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 4.0),
        major_radius=2.0,
        minor_radius=0.25,
    )

    assert exclusion.dimension == 3
    np.testing.assert_allclose(
        exclusion.margin(
            (
                (2.0, 0.0, 0.0),
                (2.25, 0.0, 0.0),
                (2.0, 0.0, 0.25),
                (0.0, 0.0, 0.0),
                (2.5, 0.0, 0.0),
            )
        ),
        (-0.25, 0.0, 0.0, 1.75, 0.25),
        atol=2.0e-16,
    )
    np.testing.assert_array_equal(exclusion.normal, (0.0, 0.0, 1.0))
    assert not exclusion.center.flags.writeable
    assert not exclusion.normal.flags.writeable


def test_toroidal_exclusion_is_covariant_under_rotation_and_translation() -> None:
    rotation = np.array(
        (
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=2.0e-16)
    center = np.array((0.3, -0.2, 0.5))
    translation = np.array((-0.4, 0.7, 0.2))
    normal = np.array((0.0, 0.0, 1.0))
    points = np.array(((1.4, -0.2, 0.5), (0.8, 0.1, 0.9), (-0.2, -0.5, 0.5)))
    reference = ToroidalExclusion(center, normal, 1.1, 0.18)
    transformed = ToroidalExclusion(
        rotation @ center + translation,
        rotation @ normal,
        1.1,
        0.18,
    )

    expected = reference.margin(points)
    actual = transformed.margin(points @ rotation.T + translation)

    np.testing.assert_allclose(actual, expected, rtol=3.0e-15, atol=3.0e-16)


def test_meridional_torus_slice_is_exactly_two_circular_exclusions() -> None:
    major_radius = 1.2
    minor_radius = 0.16
    torus = ToroidalExclusion(
        center=(0.0, 0.0, 0.0),
        normal=(0.0, 1.0, 0.0),
        major_radius=major_radius,
        minor_radius=minor_radius,
    )
    cross_section = SphericalExclusion(
        centers=((-major_radius, 0.0), (major_radius, 0.0)),
        radii=minor_radius,
    )
    points = np.array(
        ((-1.4, 0.0), (-1.2, 0.1), (0.0, 0.0), (1.2, 0.0), (1.5, -0.2))
    )
    embedded = np.column_stack((points[:, 0], points[:, 1], np.zeros(points.shape[0])))

    np.testing.assert_allclose(torus.margin(embedded), cross_section.margin(points), atol=0.0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ToroidalExclusion((0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 0.1),
        lambda: ToroidalExclusion((0.0, 0.0, np.nan), (0.0, 0.0, 1.0), 1.0, 0.1),
        lambda: ToroidalExclusion((0.0, 0.0, 0.0), (0.0, 0.0), 1.0, 0.1),
        lambda: ToroidalExclusion((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, 0.1),
        lambda: ToroidalExclusion((0.0, 0.0, 0.0), (0.0, 0.0, np.inf), 1.0, 0.1),
        lambda: ToroidalExclusion((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.0, 0.1),
        lambda: ToroidalExclusion((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), np.inf, 0.1),
        lambda: ToroidalExclusion((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 0.0),
        lambda: ToroidalExclusion((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, 1.0),
    ],
)
def test_toroidal_exclusion_rejects_invalid_geometry(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]
