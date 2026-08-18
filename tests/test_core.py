"""Tests for geometry primitives shared by field models and tracers."""

from __future__ import annotations

import numpy as np
import pytest

from vectorviz import Domain, SphericalExclusion


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
    ],
)
def test_spherical_exclusion_rejects_invalid_geometry(centers: object, radii: object) -> None:
    with pytest.raises(ValueError):
        SphericalExclusion(centers=centers, radii=radii)
