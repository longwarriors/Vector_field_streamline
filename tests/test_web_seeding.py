from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

from vectorviz import (
    ChargedRingField,
    CircularLoopField,
    DielectricSphereField,
    Domain,
    TraceDirection,
)
from vectorviz.web.schemas import SourceInput
from vectorviz.web.seeding import (
    TraceJob,
    allocate_seed_counts,
    charged_ring_equal_flux_jobs,
    current_loop_equal_flux_jobs,
    electric_source_jobs,
    halbach_rail_jobs,
    magnetic_source_jobs,
    single_dipole_equatorial_jobs,
    sphere_equal_flux_jobs,
)


def _seed_array(jobs: list[TraceJob]) -> np.ndarray:
    return np.asarray([job.seed for job in jobs], dtype=float)


def test_trace_job_is_an_immutable_value_object() -> None:
    job = TraceJob(
        seed=(1.25, -0.5),
        direction=TraceDirection.BACKWARD,
        origin_source_index=7,
    )

    assert job.seed == (1.25, -0.5)
    assert job.direction is TraceDirection.BACKWARD
    assert job.origin_source_index == 7
    with pytest.raises(FrozenInstanceError):
        job.seed = (0.0, 0.0)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("strengths", "total", "expected"),
    [
        ([3.0, -2.0, 1.0], 10, [5, 3, 2]),
        ([1.0, 1.0, 1.0], 5, [2, 2, 1]),
        ([0.0, 0.0], 5, [3, 2]),
        ([9.0, 1.0, 1.0], 3, [1, 1, 1]),
    ],
)
def test_allocate_seed_counts_reserves_one_then_uses_stable_largest_remainders(
    strengths: list[float],
    total: int,
    expected: list[int],
) -> None:
    counts = allocate_seed_counts(strengths, total)

    np.testing.assert_array_equal(counts, expected)
    assert counts.dtype == np.int64
    assert int(np.sum(counts)) == total
    assert np.all(counts >= 1)


@pytest.mark.parametrize(
    ("strengths", "total"),
    [([], 1), ([1.0, 2.0], 1), ([1.0, np.nan], 2), ([[1.0]], 1), ([1.0], 1.5)],
)
def test_allocate_seed_counts_rejects_invalid_budgets(
    strengths: object,
    total: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        allocate_seed_counts(strengths, total)  # type: ignore[arg-type]


def test_electric_jobs_budget_all_charges_and_trace_outward() -> None:
    indexed_sources = [
        (4, SourceInput(x=1.0, y=-0.5, kind="positive", strength=2.0)),
        (9, SourceInput(x=-1.0, y=0.25, kind="negative", strength=-1.0)),
    ]

    jobs = electric_source_jobs(indexed_sources, total=6, seed_radius=0.2)

    assert len(jobs) == 6
    assert [job.origin_source_index for job in jobs] == [4, 4, 4, 4, 9, 9]
    assert [job.direction for job in jobs] == [
        TraceDirection.FORWARD,
        TraceDirection.FORWARD,
        TraceDirection.FORWARD,
        TraceDirection.FORWARD,
        TraceDirection.BACKWARD,
        TraceDirection.BACKWARD,
    ]
    centers = np.asarray([(1.0, -0.5)] * 4 + [(-1.0, 0.25)] * 2)
    np.testing.assert_allclose(
        np.linalg.norm(_seed_array(jobs) - centers, axis=1),
        0.2,
        rtol=0.0,
        atol=2.0e-16,
    )
    assert np.all(np.isfinite(_seed_array(jobs)))


def test_generic_magnetic_jobs_follow_the_actual_signed_moment() -> None:
    sources = [
        (2, SourceInput(x=0.3, y=-0.2, kind="dipole", strength=1.0, angle_deg=30.0)),
        (7, SourceInput(x=-0.4, y=0.5, kind="dipole", strength=-3.0, angle_deg=30.0)),
    ]

    jobs = magnetic_source_jobs(sources, total=6, seed_radius=0.2)

    assert len(jobs) == 6
    assert [job.origin_source_index for job in jobs] == [2, 2, 7, 7, 7, 7]
    assert all(job.direction is TraceDirection.FORWARD for job in jobs)
    points = _seed_array(jobs)
    centers = np.asarray([(0.3, -0.2)] * 2 + [(-0.4, 0.5)] * 4)
    axes = np.deg2rad(30.0)
    positive_axis = np.asarray((np.cos(axes), np.sin(axes)))
    actual_axes = np.asarray([positive_axis] * 2 + [-positive_axis] * 4)
    radial = points - centers
    np.testing.assert_allclose(
        np.linalg.norm(radial, axis=1),
        0.2,
        rtol=0.0,
        atol=2.0e-16,
    )
    assert np.all(np.einsum("ij,ij->i", radial, actual_axes) > 0.0)
    assert np.all(np.isfinite(points))


def test_generic_magnetic_jobs_are_rotation_and_translation_covariant() -> None:
    base = SourceInput(x=0.0, y=0.0, kind="dipole", strength=2.0, angle_deg=10.0)
    shifted = SourceInput(x=1.1, y=-0.6, kind="dipole", strength=2.0, angle_deg=80.0)
    rotation_angle = np.deg2rad(70.0)
    rotation = np.asarray(
        (
            (np.cos(rotation_angle), -np.sin(rotation_angle)),
            (np.sin(rotation_angle), np.cos(rotation_angle)),
        )
    )

    base_points = _seed_array(magnetic_source_jobs([(0, base)], 5, 0.2))
    shifted_points = _seed_array(magnetic_source_jobs([(3, shifted)], 5, 0.2))

    expected = base_points @ rotation.T + np.asarray((1.1, -0.6))
    np.testing.assert_allclose(shifted_points, expected, rtol=0.0, atol=3.0e-16)


def test_single_dipole_equatorial_jobs_cover_both_rays_inside_the_domain() -> None:
    domain = Domain(lower=(-3.0, -3.0), upper=(3.0, 3.0))
    source = SourceInput(
        x=0.0,
        y=0.0,
        kind="dipole",
        strength=1.0,
        angle_deg=0.0,
    )

    jobs = single_dipole_equatorial_jobs(
        5,
        source,
        total=6,
        domain=domain,
        seed_radius=0.2,
        domain_inset=0.1,
    )

    assert len(jobs) == 6
    assert all(job.direction is TraceDirection.BOTH for job in jobs)
    assert all(job.origin_source_index == 5 for job in jobs)
    points = _seed_array(jobs)
    np.testing.assert_allclose(points[:, 0], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.sort(points[:, 1]),
        [-2.9, -1.55, -0.2, 0.2, 1.55, 2.9],
        rtol=0.0,
        atol=5.0e-16,
    )
    assert np.all(domain.contains(points))
    assert np.all(np.linalg.norm(points, axis=1) >= 0.2)


def test_single_dipole_equatorial_geometry_ignores_strength_sign() -> None:
    domain = Domain(lower=(-3.0, -3.0), upper=(3.0, 3.0))
    positive = SourceInput(
        x=0.0,
        y=0.0,
        kind="dipole",
        strength=2.0,
        angle_deg=37.0,
    )
    negative = positive.model_copy(update={"strength": -2.0})

    positive_jobs = single_dipole_equatorial_jobs(0, positive, 7, domain, 0.2)
    negative_jobs = single_dipole_equatorial_jobs(0, negative, 7, domain, 0.2)

    assert positive_jobs == negative_jobs


def test_single_dipole_equatorial_jobs_rotate_and_translate_with_the_geometry() -> None:
    base_domain = Domain(lower=(-3.0, -3.0), upper=(3.0, 3.0))
    base = SourceInput(x=0.0, y=0.0, kind="dipole", strength=1.0, angle_deg=0.0)
    translated_domain = Domain(lower=(-1.8, -3.7), upper=(4.2, 2.3))
    translated = SourceInput(
        x=1.2,
        y=-0.7,
        kind="dipole",
        strength=1.0,
        angle_deg=0.0,
    )
    rotated = SourceInput(x=0.0, y=0.0, kind="dipole", strength=1.0, angle_deg=90.0)

    base_points = _seed_array(
        single_dipole_equatorial_jobs(0, base, 6, base_domain, 0.2, domain_inset=0.1)
    )
    translated_points = _seed_array(
        single_dipole_equatorial_jobs(
            0,
            translated,
            6,
            translated_domain,
            0.2,
            domain_inset=0.1,
        )
    )
    rotated_points = _seed_array(
        single_dipole_equatorial_jobs(
            0,
            rotated,
            6,
            base_domain,
            0.2,
            domain_inset=0.1,
        )
    )
    rotation = np.asarray(((0.0, -1.0), (1.0, 0.0)))

    np.testing.assert_allclose(
        translated_points,
        base_points + np.asarray((1.2, -0.7)),
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(rotated_points, base_points @ rotation.T, atol=5.0e-16)


def test_single_dipole_equatorial_jobs_respect_rotated_inset_boundaries() -> None:
    domain = Domain(lower=(-2.0, -1.5), upper=(3.0, 2.5))
    source = SourceInput(
        x=0.7,
        y=-0.2,
        kind="dipole",
        strength=1.0,
        angle_deg=31.0,
    )

    jobs = single_dipole_equatorial_jobs(
        3,
        source,
        total=9,
        domain=domain,
        seed_radius=0.16,
        domain_inset=0.05,
    )

    points = _seed_array(jobs)
    lower = np.asarray(domain.lower) + 0.05
    upper = np.asarray(domain.upper) - 0.05
    center = np.asarray((source.x, source.y))
    moment_axis = np.asarray(
        (np.cos(np.deg2rad(source.angle_deg)), np.sin(np.deg2rad(source.angle_deg)))
    )
    perpendicular = np.asarray((-moment_axis[1], moment_axis[0]))
    signed_distances = (points - center) @ perpendicular

    assert len(jobs) == 9
    assert np.all(points >= lower)
    assert np.all(points <= upper)
    np.testing.assert_allclose((points - center) @ moment_axis, 0.0, atol=5.0e-16)
    assert np.any(signed_distances > 0.0)
    assert np.any(signed_distances < 0.0)
    assert np.all(np.abs(signed_distances) >= 0.16)
    assert any(np.any(point == lower) or np.any(point == upper) for point in points)


def test_single_dipole_equatorial_budget_tracks_each_seedable_ray_length() -> None:
    domain = Domain(lower=(-1.0, -2.0), upper=(1.0, 4.0))
    source = SourceInput(x=0, y=0, kind="dipole", strength=1, angle_deg=0)

    jobs = single_dipole_equatorial_jobs(0, source, 7, domain, 0.2, domain_inset=0)

    points = _seed_array(jobs)
    np.testing.assert_allclose(
        points[:4, 1],
        np.linspace(0.2, 4.0, 4),
        rtol=0.0,
        atol=5.0e-16,
    )
    np.testing.assert_allclose(
        points[4:, 1],
        -np.linspace(0.2, 2.0, 3),
        rtol=0.0,
        atol=5.0e-16,
    )


@pytest.mark.parametrize("total", [6, 7])
def test_halbach_rails_split_the_budget_and_cover_each_full_extent(total: int) -> None:
    jobs = halbach_rail_jobs(total, x_extent=2.1, y_offset=0.45)
    upper_count = (total + 1) // 2
    lower_count = total // 2

    assert len(jobs) == total
    assert all(job.direction is TraceDirection.BOTH for job in jobs)
    assert all(job.origin_source_index is None for job in jobs)
    np.testing.assert_array_equal(
        _seed_array(jobs[:upper_count]),
        np.column_stack(
            (
                np.linspace(-2.1, 2.1, upper_count),
                np.full(upper_count, 0.45),
            )
        ),
    )
    np.testing.assert_array_equal(
        _seed_array(jobs[upper_count:]),
        np.column_stack(
            (
                np.linspace(-2.1, 2.1, lower_count),
                np.full(lower_count, -0.45),
            )
        ),
    )


def test_halbach_rails_still_return_an_exact_small_budget() -> None:
    jobs = halbach_rail_jobs(3)

    assert len(jobs) == 3
    assert [job.seed[1] for job in jobs] == [0.45, 0.45, -0.45]
    assert np.all(np.isfinite(_seed_array(jobs)))


def test_current_loop_jobs_are_mirrored_and_equispaced_in_public_flux() -> None:
    loop = CircularLoopField(1.0, 1.0, normal=(0.0, 1.0, 0.0))

    jobs = current_loop_equal_flux_jobs(loop, total=6, axis_y=-2.9999)

    assert len(jobs) == 6
    assert all(job.direction is TraceDirection.FORWARD for job in jobs)
    assert all(job.origin_source_index is None for job in jobs)
    points = _seed_array(jobs)
    np.testing.assert_allclose(points[0::2, 0], -points[1::2, 0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(points[:, 1], 0.0, rtol=0.0, atol=0.0)
    positive_radii = points[1::2, 0]
    np.testing.assert_allclose(positive_radii[[0, -1]], [0.12, 0.82], atol=2.0e-15)
    assert positive_radii[1] == pytest.approx(0.643707870282, abs=2.0e-12)
    assert not np.allclose(np.diff(positive_radii), np.diff(positive_radii)[0])
    flux = loop.flux_function(np.column_stack((positive_radii, np.zeros(3), np.zeros(3))))
    np.testing.assert_allclose(np.diff(flux), np.diff(flux)[0], rtol=2.0e-11)


def test_odd_current_loop_budget_adds_one_translated_axis_feature() -> None:
    loop = CircularLoopField(
        1.0,
        1.0,
        center=(0.3, -0.2, 0.4),
        normal=(0.0, 1.0, 0.0),
    )

    odd_jobs = current_loop_equal_flux_jobs(loop, total=7, axis_y=-2.7)
    even_jobs = current_loop_equal_flux_jobs(loop, total=6, axis_y=-2.7)

    assert odd_jobs[0] == TraceJob((0.3, -2.7), TraceDirection.FORWARD)
    assert odd_jobs[1:] == even_jobs
    points = _seed_array(even_jobs)
    np.testing.assert_allclose(points[0::2, 0] + points[1::2, 0], 0.6, atol=2.0e-15)
    np.testing.assert_allclose(points[:, 1], -0.2, rtol=0.0, atol=0.0)
    positive = points[1::2]
    embedded = np.column_stack((positive[:, 0], positive[:, 1], np.full(3, 0.4)))
    flux = loop.flux_function(embedded)
    np.testing.assert_allclose(np.diff(flux), np.diff(flux)[0], rtol=2.0e-11)


def test_charged_ring_jobs_are_mirrored_and_equispaced_in_public_flux() -> None:
    ring = ChargedRingField(1.0e-9, 1.0, normal=(0.0, 1.0, 0.0))

    jobs = charged_ring_equal_flux_jobs(ring, total=8, seed_radius=0.162)

    assert len(jobs) == 8
    assert all(job.direction is TraceDirection.FORWARD for job in jobs)
    assert all(job.origin_source_index is None for job in jobs)
    points = _seed_array(jobs)
    np.testing.assert_allclose(points[0::2, 0], -points[1::2, 0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(points[0::2, 1], points[1::2, 1], rtol=0.0, atol=0.0)
    right = points[1::2]
    np.testing.assert_allclose(np.hypot(right[:, 0] - 1.0, right[:, 1]), 0.162, rtol=1.0e-14)
    flux = ring.flux_function(np.column_stack((right, np.zeros(4))))
    cut = float(ring.flux_function((1.162, 0.0, 0.0)))
    np.testing.assert_allclose(flux, cut * np.array((-0.75, -0.25, 0.25, 0.75)), rtol=1.0e-9)
    # Equal flux is not equal angle: the seeds crowd toward the outer equator.
    angles = np.arctan2(right[:, 1], right[:, 0] - 1.0)
    assert not np.allclose(np.diff(np.sort(angles)), np.diff(np.sort(angles))[0])


def test_odd_charged_ring_budget_adds_the_outer_equatorial_ray() -> None:
    ring = ChargedRingField(1.0e-9, 1.0, center=(0.3, -0.2, 0.0), normal=(0.0, 1.0, 0.0))

    odd_jobs = charged_ring_equal_flux_jobs(ring, total=7, seed_radius=0.162)
    even_jobs = charged_ring_equal_flux_jobs(ring, total=6, seed_radius=0.162)

    assert odd_jobs[0] == TraceJob((1.462, -0.2), TraceDirection.FORWARD)
    assert odd_jobs[1:] == even_jobs
    points = _seed_array(even_jobs)
    np.testing.assert_allclose(points[0::2, 0] + points[1::2, 0], 0.6, atol=2.0e-15)
    # Three targets straddle zero flux: the middle pair is the inward equator.
    np.testing.assert_allclose(points[2:4], ((0.3 - 0.838, -0.2), (0.3 + 0.838, -0.2)), atol=1.0e-9)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: charged_ring_equal_flux_jobs(CircularLoopField(1, 1, normal=(0, 1, 0)), 8, 0.162),
        lambda: charged_ring_equal_flux_jobs(ChargedRingField(1e-9, 1, normal=(0, 1, 0)), 0, 0.162),
        lambda: charged_ring_equal_flux_jobs(ChargedRingField(1e-9, 1, normal=(0, 1, 0)), 8, 1.0),
        lambda: charged_ring_equal_flux_jobs(ChargedRingField(1e-9, 1, normal=(0, 0, 1)), 8, 0.162),
    ],
)
def test_charged_ring_planner_rejects_invalid_inputs(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


def test_sphere_jobs_are_mirrored_and_equispaced_in_displacement_flux() -> None:
    sphere = DielectricSphereField((1.0, 0.0, 0.0), 1.0, relative_permittivity=4.0)
    domain = Domain(lower=(-3.0, -3.0), upper=(3.0, 3.0))

    jobs = sphere_equal_flux_jobs(sphere, total=8, domain=domain)

    assert len(jobs) == 8
    assert all(job.direction is TraceDirection.FORWARD for job in jobs)
    points = _seed_array(jobs)
    np.testing.assert_allclose(points[:, 0], -2.9999)
    np.testing.assert_allclose(points[0::2, 1], -points[1::2, 1], rtol=0.0, atol=0.0)
    heights = points[1::2, 1]
    assert heights[-1] == pytest.approx(2.88)
    flux = sphere.flux_function(np.column_stack((points[1::2], np.zeros(4))))
    np.testing.assert_allclose(flux, flux[-1] * np.array((0.25, 0.5, 0.75, 1.0)), rtol=1.0e-9)
    # Equal flux is not equal height: the uniform far field makes flux grow as y^2.
    assert not np.allclose(np.diff(heights), np.diff(heights)[0])


def test_odd_sphere_budget_adds_the_axis_line() -> None:
    sphere = DielectricSphereField((2.0, 0.0, 0.0), 0.5, center=(0.4, -0.3, 0.0), relative_permittivity=3.0)
    domain = Domain(lower=(-3.0, -3.0), upper=(3.0, 3.0))

    odd_jobs = sphere_equal_flux_jobs(sphere, total=7, domain=domain)
    even_jobs = sphere_equal_flux_jobs(sphere, total=6, domain=domain)

    assert odd_jobs[0] == TraceJob((-2.9999, -0.3), TraceDirection.FORWARD)
    assert odd_jobs[1:] == even_jobs
    points = _seed_array(even_jobs)
    np.testing.assert_allclose(points[0::2, 1] + points[1::2, 1], -0.6, atol=1.0e-15)
    # The usable height is limited by the nearer domain edge below the centre.
    assert points[1::2, 1][-1] == pytest.approx(-0.3 + (2.7 - 0.12))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: sphere_equal_flux_jobs(CircularLoopField(1, 1), 8, Domain(lower=(-3, -3), upper=(3, 3))),
        lambda: sphere_equal_flux_jobs(
            DielectricSphereField((1, 0, 0), 1, relative_permittivity=2), 0, Domain(lower=(-3, -3), upper=(3, 3))
        ),
        lambda: sphere_equal_flux_jobs(
            DielectricSphereField((0, 1, 0), 1, relative_permittivity=2), 8, Domain(lower=(-3, -3), upper=(3, 3))
        ),
        lambda: sphere_equal_flux_jobs(
            DielectricSphereField((1, 0, 0), 1, center=(0, 0, 0.5), relative_permittivity=2),
            8,
            Domain(lower=(-3, -3), upper=(3, 3)),
        ),
        lambda: sphere_equal_flux_jobs(
            DielectricSphereField((1, 0, 0), 4, relative_permittivity=2), 8, Domain(lower=(-3, -3), upper=(3, 3))
        ),
        lambda: sphere_equal_flux_jobs(
            DielectricSphereField((1, 0, 0), 1, relative_permittivity=2), 8, Domain(lower=(-3, -3, -3), upper=(3, 3, 3))
        ),
    ],
)
def test_sphere_planner_rejects_invalid_inputs(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "jobs",
    [
        lambda: electric_source_jobs(
            [
                (0, SourceInput(x=-1, y=0, kind="positive", strength=1)),
                (1, SourceInput(x=1, y=0, kind="negative", strength=-1)),
            ],
            8,
            0.2,
        ),
        lambda: magnetic_source_jobs(
            [(0, SourceInput(x=0, y=0, kind="dipole", strength=1, angle_deg=90))],
            8,
            0.2,
        ),
        lambda: single_dipole_equatorial_jobs(
            0,
            SourceInput(x=0, y=0, kind="dipole", strength=1, angle_deg=90),
            8,
            Domain(lower=(-3, -3), upper=(3, 3)),
            0.2,
        ),
        lambda: halbach_rail_jobs(8),
        lambda: current_loop_equal_flux_jobs(CircularLoopField(1, 1, normal=(0, 1, 0)), 8, -2.9999),
        lambda: charged_ring_equal_flux_jobs(ChargedRingField(1e-9, 1, normal=(0, 1, 0)), 8, 0.162),
        lambda: sphere_equal_flux_jobs(
            DielectricSphereField((1, 0, 0), 1, relative_permittivity=4),
            8,
            Domain(lower=(-3, -3), upper=(3, 3)),
        ),
    ],
)
def test_every_planner_returns_exactly_the_finite_requested_budget(jobs: object) -> None:
    planned = jobs()  # type: ignore[operator]

    assert len(planned) == 8
    assert np.all(np.isfinite(_seed_array(planned)))


@dataclass
class _UnvalidatedSource:
    """A source-like object that bypasses request validation on purpose."""

    x: float
    y: float
    kind: str
    strength: float
    angle_deg: float | None = None


_DOMAIN = Domain(lower=(-3.0, -3.0), upper=(3.0, 3.0))
_DIPOLE = SourceInput(x=0.0, y=0.0, kind="dipole", strength=1.0, angle_deg=90.0)
_LOOP = CircularLoopField(1.0, 1.0, normal=(0.0, 1.0, 0.0))


@pytest.mark.parametrize(
    ("factory", "exception", "match"),
    [
        (lambda: TraceJob((0.0, np.nan), TraceDirection.FORWARD), ValueError, "two finite"),
        (lambda: TraceJob((0.0, 0.0, 0.0), TraceDirection.FORWARD), ValueError, "two finite"),
        (lambda: TraceJob((0.0, 0.0), TraceDirection.FORWARD, True), TypeError, "integer or None"),
        (lambda: TraceJob((0.0, 0.0), TraceDirection.FORWARD, -1), ValueError, "non-negative"),
        (lambda: allocate_seed_counts((1.0,), True), TypeError, "must be an integer"),
        (lambda: allocate_seed_counts((1.0,), 0), ValueError, "must be positive"),
        (lambda: halbach_rail_jobs("4"), TypeError, "must be an integer"),
        (lambda: halbach_rail_jobs(4, x_extent=np.inf), ValueError, "finite positive"),
        (
            lambda: electric_source_jobs(
                [(0, SourceInput(x=0.0, y=0.0, kind="positive"))], 4, 0.0
            ),
            ValueError,
            "finite positive",
        ),
        (
            lambda: electric_source_jobs(
                [(True, SourceInput(x=0.0, y=0.0, kind="positive"))], 4, 0.2
            ),
            TypeError,
            "source index must be an integer",
        ),
        (
            lambda: electric_source_jobs(
                [("0", SourceInput(x=0.0, y=0.0, kind="positive"))], 4, 0.2
            ),
            TypeError,
            "source index must be an integer",
        ),
        (
            lambda: electric_source_jobs(
                [(-1, SourceInput(x=0.0, y=0.0, kind="positive"))], 4, 0.2
            ),
            ValueError,
            "source index must be non-negative",
        ),
        (lambda: electric_source_jobs([], 4, 0.2), ValueError, "at least one seeding source"),
        (
            lambda: electric_source_jobs(
                [(0, SourceInput(x=0.0, y=0.0, kind="dipole"))], 4, 0.2
            ),
            ValueError,
            "matching nonzero kind",
        ),
        (
            lambda: electric_source_jobs(
                [(0, _UnvalidatedSource(np.nan, 0.0, "positive", 1.0))], 4, 0.2
            ),
            ValueError,
            "must be finite",
        ),
        (
            lambda: magnetic_source_jobs(
                [(0, SourceInput(x=0.0, y=0.0, kind="dipole", strength=0.0))], 4, 0.2
            ),
            ValueError,
            "nonzero dipoles",
        ),
        (
            lambda: magnetic_source_jobs(
                [(0, _UnvalidatedSource(0.0, 0.0, "dipole", 1.0, None))], 4, 0.2
            ),
            ValueError,
            "angle_deg must be finite",
        ),
        (
            lambda: single_dipole_equatorial_jobs(0, _DIPOLE, 4, _DOMAIN, 0.2, domain_inset=-1.0),
            ValueError,
            "domain_inset",
        ),
        (
            lambda: single_dipole_equatorial_jobs(0, _DIPOLE, 4, Domain((0.0,), (1.0,)), 0.2),
            ValueError,
            "two-dimensional",
        ),
        (
            lambda: single_dipole_equatorial_jobs(
                0, SourceInput(x=0.0, y=0.0, kind="dipole", strength=0.0), 4, _DOMAIN, 0.2
            ),
            ValueError,
            "nonzero dipole",
        ),
        (
            lambda: single_dipole_equatorial_jobs(
                0, _UnvalidatedSource(0.0, 0.0, "dipole", 1.0, None), 4, _DOMAIN, 0.2
            ),
            ValueError,
            "angle_deg must be finite",
        ),
        (
            lambda: single_dipole_equatorial_jobs(0, _DIPOLE, 4, _DOMAIN, 0.2, domain_inset=3.0),
            ValueError,
            "no seedable domain",
        ),
        (
            lambda: single_dipole_equatorial_jobs(
                0,
                SourceInput(x=2.8, y=0.0, kind="dipole"),
                4,
                Domain((-1.0, -1.0), (1.0, 1.0)),
                0.2,
            ),
            ValueError,
            "inside the inset domain",
        ),
        (
            lambda: single_dipole_equatorial_jobs(
                0, SourceInput(x=2.8, y=0.0, kind="dipole", angle_deg=90.0), 4, _DOMAIN, 0.2
            ),
            ValueError,
            "extend beyond seed_radius",
        ),
        (lambda: current_loop_equal_flux_jobs(object(), 4, -2.9), TypeError, "CircularLoopField"),
        (lambda: current_loop_equal_flux_jobs(_LOOP, 4, np.nan), ValueError, "axis_y must be finite"),
        (
            lambda: current_loop_equal_flux_jobs(_LOOP, 4, -2.9, inner_radius=0.9, outer_radius=0.5),
            ValueError,
            "inner_radius < outer_radius",
        ),
        (
            lambda: current_loop_equal_flux_jobs(
                CircularLoopField(1.0, 1.0, normal=(0.0, 0.0, 1.0)), 4, -2.9
            ),
            ValueError,
            "parallel to the web y-axis",
        ),
    ],
)
def test_seed_planners_reject_invalid_inputs(
    factory: object, exception: type[Exception], match: str
) -> None:
    with pytest.raises(exception, match=match):
        factory()  # type: ignore[operator]


def test_single_seed_budgets_fall_back_to_one_deterministic_job() -> None:
    dipole = SourceInput(x=1.0, y=0.0, kind="dipole", strength=1.0, angle_deg=90.0)

    equatorial = single_dipole_equatorial_jobs(0, dipole, 1, _DOMAIN, 0.2)
    loop = current_loop_equal_flux_jobs(_LOOP, 1, -2.9)

    # The longer -x ray receives the only equatorial seed; the loop keeps its axis job.
    assert len(equatorial) == 1
    assert equatorial[0].seed[0] < 1.0
    assert equatorial[0].direction is TraceDirection.BOTH
    assert len(loop) == 1
    assert loop[0].seed == (0.0, -2.9)
