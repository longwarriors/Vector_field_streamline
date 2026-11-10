"""Physics and scene contracts for the fixed magnetostatic current presets."""

from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.constants import mu_0

from vectorviz import CircularLoopField, InfiniteWireField
from vectorviz.tracing import FieldLineTracer, TerminationReason
from vectorviz.web.app import create_app
from vectorviz.web.scene import (
    DOMAIN,
    _build_model,
    _PlanarCurrentField,
    build_scene,
)
from vectorviz.web.schemas import SceneRequest, SeedInput, SeedMode
from vectorviz.web.seeding import coverage_segment_jobs

PRESETS = (
    "straight_wire",
    "parallel_wires",
    "antiparallel_wires",
    "solenoid",
    "helmholtz_coils",
    "anti_helmholtz_coils",
)


@pytest.mark.parametrize("preset", PRESETS)
def test_current_preset_contract_and_manual_seed(preset: str) -> None:
    with TestClient(create_app()) as client:
        payload = {
            "preset": preset,
            "density": 6,
            "resolution": 32,
            "seeds": [{"x": 0.4, "y": 0.3}],
        }
        response = client.post("/api/scene", json=payload)
        assert response.status_code == 200
        scene = response.json()
        assert scene["vector"]["label"] == "B"
        assert scene["vector"]["unit"] == scene["scalar"]["unit"] == "T"
        assert scene["metadata"]["seed_mode"] == SeedMode.COVERAGE
        assert sum(scene["metadata"]["termination_counts"].values()) == 6
        assert scene["metadata"]["manual_seed_count"] == 1
        assert sum("seed_index" in line for line in scene["lines"]) == 1
        assert "solver_failure" not in scene["metadata"]["termination_counts"]
        assert "max_arc_length" not in scene["metadata"]["termination_counts"]
        assert (
            len(scene["sources"])
            == {
                "straight_wire": 1,
                "parallel_wires": 2,
                "antiparallel_wires": 2,
                "solenoid": 16,
                "helmholtz_coils": 4,
                "anti_helmholtz_coils": 4,
            }[preset]
        )
        assert all(source["strength"] == 1.0 for source in scene["sources"])
        assert all(source["strength_unit"] == "A" for source in scene["sources"])
        assert all(
            mask == (u is None) == (v is None)
            for mask, u, v in zip(
                scene["scalar"]["mask"], scene["vector"]["u"], scene["vector"]["v"], strict=True
            )
        )
        assert (
            client.post(
                "/api/scene", json={**payload, "sources": [{"x": 0, "y": 0, "kind": "dipole"}]}
            ).status_code
            == 422
        )


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("density", [6, 7, 18, 40])
def test_current_seed_budget_is_unique_finite_and_outside_wires(preset: str, density: int) -> None:
    model = _build_model(SceneRequest(preset=preset, density=density))
    assert len(model.seeds) == density
    assert len(np.unique(model.seeds, axis=0)) == density
    assert np.all(DOMAIN.contains(model.seeds))
    assert all(np.all(exclusion.margin(model.seeds) > 0) for exclusion in model.exclusions)
    assert np.all(np.isfinite(model.field.evaluate(model.seeds)))
    assert np.all(np.linalg.norm(model.field.evaluate(model.seeds), axis=1) > 1e-14)


@pytest.mark.parametrize("preset", ["solenoid", "helmholtz_coils", "anti_helmholtz_coils"])
def test_coil_axis_matches_sum_of_analytic_loop_axis_fields(preset: str) -> None:
    model = _build_model(SceneRequest(preset=preset))
    ys = np.array((-2.0, -0.23, 0.0, 0.51, 2.7))
    actual = model.field.evaluate(np.stack((np.zeros_like(ys), ys), axis=-1))
    if preset == "solenoid":
        centers, currents = np.linspace(-1.2, 1.2, 8), np.ones(8)
    else:
        centers = np.array((-0.5, 0.5))
        currents = np.array((1.0, -1.0 if preset == "anti_helmholtz_coils" else 1.0))
    expected = mu_0 / 2 * np.sum(currents / (1 + (ys[:, None] - centers) ** 2) ** 1.5, axis=1)
    np.testing.assert_allclose(actual[:, 1], expected, rtol=2e-14, atol=1e-22)
    np.testing.assert_array_equal(actual[:, 0], 0.0)


def test_helmholtz_flat_center_and_anti_helmholtz_quadrupole() -> None:
    symmetric = _build_model(SceneRequest(preset="helmholtz_coils")).field
    np.testing.assert_allclose(symmetric.evaluate((0, 0))[1], mu_0 * (4 / 5) ** 1.5, rtol=1e-14)
    h = 1e-3
    b = symmetric.evaluate(((0, -h), (0, 0), (0, h)))[:, 1]
    assert abs(b[0] - 2 * b[1] + b[2]) / b[1] < 3e-12
    opposed = _build_model(SceneRequest(preset="anti_helmholtz_coils")).field
    np.testing.assert_array_equal(opposed.evaluate((0, 0)), 0.0)
    expected_gradient = -3 * mu_0 * 0.5 / 1.25**2.5
    np.testing.assert_allclose(opposed.evaluate((0, h))[1] / h, expected_gradient, rtol=2e-6)
    np.testing.assert_allclose(opposed.evaluate((h, 0))[0] / h, -expected_gradient / 2, rtol=2e-6)


@pytest.mark.parametrize("preset", ["straight_wire", "parallel_wires", "antiparallel_wires"])
def test_wire_lines_preserve_their_analytic_flux_contours(preset: str) -> None:
    scene = build_scene(SceneRequest(preset=preset, density=7, resolution=32))
    for line in scene.lines:
        points = np.asarray(line.points)
        if preset == "straight_wire":
            radii = np.linalg.norm(points, axis=1)
            assert np.ptp(radii) < 2e-5
            # A closed branch travels exactly one revolution, not two.
            phase = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
            assert abs(abs(phase[-1] - phase[0]) - 2 * np.pi) < 0.03
        else:
            r_left = np.linalg.norm(points - (-0.8, 0.0), axis=1)
            r_right = np.linalg.norm(points - (0.8, 0.0), axis=1)
            flux = np.log(r_left) + (1 if preset == "parallel_wires" else -1) * np.log(r_right)
            assert np.ptp(flux) < 2e-4


@pytest.mark.parametrize(
    "preset,density",
    [
        ("solenoid", 18),
        ("solenoid", 40),
        ("helmholtz_coils", 7),
        ("anti_helmholtz_coils", 40),
    ],
)
def test_coil_coverage_contains_complete_returns_without_failed_traces(
    preset: str, density: int
) -> None:
    scene = build_scene(SceneRequest(preset=preset, density=density, resolution=32))
    counts = scene.metadata.termination_counts
    assert counts.get("closed_loop", 0) >= density // 2
    assert set(counts) <= {"closed_loop", "domain_exit"}
    model = _build_model(SceneRequest(preset=preset, density=density))
    for line in scene.lines:
        points = np.asarray(line.points)
        assert all(np.min(exclusion.margin(points)) > -1e-7 for exclusion in model.exclusions)


def test_small_manual_coil_orbit_finishes_after_one_turn() -> None:
    model = _build_model(SceneRequest(preset="anti_helmholtz_coils"))
    # The old single-loop preset's 0.6 m minimum misses this 0.575 m orbit.
    # Reproducing that setting verifies the regression would double the line.
    seed = (0.918, -0.5)
    previous = FieldLineTracer(
        model.field,
        DOMAIN,
        replace(model.trace_options, closure_min_arc_length=0.6),
        model.exclusions,
    ).trace(seed, direction="forward")
    current = FieldLineTracer(model.field, DOMAIN, model.trace_options, model.exclusions).trace(
        seed, direction="forward"
    )
    assert current.forward.termination == TerminationReason.CLOSED_LOOP
    assert 0.5 < current.arc_length[-1] < 0.6
    assert previous.arc_length[-1] > 1.0
    assert np.isclose(previous.arc_length[-1], 2 * current.arc_length[-1], rtol=1e-3)


def test_solenoid_closed_return_preserves_the_summed_flux_function() -> None:
    scene = build_scene(SceneRequest(preset="solenoid", density=18, resolution=32))
    points = np.asarray(
        next(line.points for line in scene.lines if line.termination == "closed_loop")
    )
    embedded = np.column_stack((points, np.zeros(len(points))))
    flux = sum(
        CircularLoopField(1, 1, center=(0, y, 0), normal=(0, 1, 0)).flux_function(embedded)
        for y in np.linspace(-1.2, 1.2, 8)
    )
    assert np.ptp(flux) / abs(np.mean(flux)) < 2e-5


@pytest.mark.parametrize("preset", ["parallel_wires", "anti_helmholtz_coils"])
def test_manual_seed_at_current_null_has_no_line(preset: str) -> None:
    scene = build_scene(
        SceneRequest(preset=preset, density=6, resolution=32, seeds=[SeedInput(x=0, y=0)])
    )
    assert scene.metadata.manual_termination_counts == {"null_field": 1}
    assert not any(line.seed_index is not None for line in scene.lines)


def test_current_adapter_rejects_planes_that_are_not_invariant() -> None:
    with pytest.raises(ValueError, match="perpendicular"):
        _PlanarCurrentField(InfiniteWireField(1, direction=(1, 0, 0)))
    for loops in [
        (),
        (CircularLoopField(1, 1),),
        (CircularLoopField(1, 1, center=(1, 0, 0), normal=(0, 1, 0)),),
    ]:
        with pytest.raises(ValueError, match="share the y axis"):
            _PlanarCurrentField(loops)
    field = _PlanarCurrentField(InfiniteWireField(1))
    for points in [0, [1, 2, 3]]:
        with pytest.raises(ValueError, match="shape"):
            field.evaluate(points)
    np.testing.assert_allclose(field.evaluate((1, 0)), (0, mu_0 / (2 * np.pi)))


@pytest.mark.parametrize(
    "segments,weights",
    [
        ([], None),
        ([[[0, 0], [0, 0]]], None),
        ([[[0, 0], [1, np.nan]]], None),
        ([[[0, 0], [1, 0]]], [1, 2]),
        ([[[0, 0], [1, 0]]], [0]),
    ],
)
def test_coverage_segments_reject_invalid_geometry_or_weights(segments, weights) -> None:
    with pytest.raises(ValueError):
        coverage_segment_jobs(segments, 6, weights=weights)
