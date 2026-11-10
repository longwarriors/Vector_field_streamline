"""End-to-end contracts for mathematical fields and the ideal magnetic shell."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from vectorviz.web.app import create_app
from vectorviz.web.scene import _build_model, build_scene
from vectorviz.web.schemas import RegionPayload, SceneRequest, SeedInput

PRESETS = (
    "laplace_saddle",
    "laplace_multipole",
    "slope_linear",
    "slope_logistic",
    "magnetic_shield_cylinder",
)


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("density", [6, 7, 40])
def test_new_scene_budgets_cover_distinct_valid_orbits(preset: str, density: int) -> None:
    request = SceneRequest(preset=preset, density=density, resolution=32)
    model = _build_model(request)
    assert len(np.unique(model.seeds, axis=0)) == density
    assert np.all(np.linalg.norm(model.field.evaluate(model.seeds), axis=-1) > 1e-8)
    assert all(
        np.all(np.abs(interface.margin(model.seeds)) > 1e-4) for interface in model.interfaces
    )
    scene = build_scene(request)
    assert len(scene.lines) == density
    assert scene.metadata.termination_counts == {"domain_exit": density}
    assert scene.metadata.start_termination_counts == {"domain_exit": density}
    assert scene.metadata.seed_mode == "coverage"
    assert scene.sources == []
    assert not any(scene.scalar.mask)
    # A streamfunction independently identifies orbits in each half/sector;
    # the tracer must preserve it across both shell surfaces as well.
    if preset == "magnetic_shield_cylinder":
        for line in scene.lines:
            flux = model.field.flux_function(line.points)
            assert np.ptp(flux) < 5e-10  # T m; <0.05% of B0 * outer radius
        radii = np.linalg.norm(model.seeds, axis=1)
        assert np.any(radii < 0.7) and np.any((radii > 0.7) & (radii < 1)) and np.any(radii > 1)
    elif preset.startswith("laplace_"):
        order = 2 if preset == "laplace_saddle" else 3
        for line in scene.lines:
            xy = np.asarray(line.points)
            invariant = np.imag((xy[:, 0] + 1j * xy[:, 1]) ** order) / order
            assert np.ptp(invariant) < 3e-5
    elif preset == "slope_logistic":
        for equilibrium in (0.0, 1.0):
            assert any(np.all(np.asarray(line.points)[:, 1] == equilibrium) for line in scene.lines)


@pytest.mark.parametrize("preset", PRESETS)
def test_new_api_scenes_keep_units_vectors_manual_accounting_and_fixed_geometry(
    preset: str,
) -> None:
    with TestClient(create_app()) as client:
        request = {
            "preset": preset,
            "density": 6,
            "resolution": 32,
            "seeds": [{"x": 0.2, "y": 0.3}],
        }
        response = client.post("/api/scene", json=request)
        assert response.status_code == 200
        scene = response.json()
        shield = preset == "magnetic_shield_cylinder"
        assert scene["domain"]["unit"] == ("m" if shield else "1")
        assert scene["scalar"]["unit"] == scene["vector"]["unit"] == ("T" if shield else "1")
        assert scene["vector"]["label"] == ("B" if shield else "F")
        np.testing.assert_allclose(
            scene["scalar"]["values"],
            np.hypot(scene["vector"]["u"], scene["vector"]["v"]),
            rtol=2e-15,
        )
        assert scene["metadata"]["manual_seed_count"] == 1
        assert sum(line.get("seed_index") == 0 for line in scene["lines"]) == 1
        assert scene["metadata"]["manual_termination_counts"] == {"domain_exit": 1}
        assert (
            client.post(
                "/api/scene", json={**request, "sources": [{"x": 0, "y": 0, "kind": "positive"}]}
            ).status_code
            == 422
        )
        if shield:
            assert scene["regions"] == [
                {
                    "kind": preset,
                    "x": 0.0,
                    "y": 0.0,
                    "radius": 1.0,
                    "inner_radius": 0.7,
                    "relative_permeability": 100.0,
                    "unit": "m",
                }
            ]
        else:
            assert scene["regions"] == []


def test_null_and_equilibrium_manual_seeds_have_different_meanings() -> None:
    for preset in ("laplace_saddle", "laplace_multipole"):
        scene = build_scene(
            SceneRequest(preset=preset, density=6, resolution=32, seeds=[SeedInput(x=0, y=0)])
        )
        assert scene.metadata.manual_termination_counts == {"null_field": 1}
        assert not any(line.seed_index is not None for line in scene.lines)
    scene = build_scene(
        SceneRequest(preset="slope_logistic", density=6, resolution=32, seeds=[SeedInput(x=0, y=0)])
    )
    manual = next(line for line in scene.lines if line.seed_index == 0)
    assert manual.termination == manual.start_termination == "domain_exit"
    np.testing.assert_array_equal(np.asarray(manual.points)[:, 1], 0.0)


@pytest.mark.parametrize("radius", [0.7, 1.0])
def test_manual_seed_on_either_shell_surface_is_retained(radius: float) -> None:
    scene = build_scene(
        SceneRequest(
            preset="magnetic_shield_cylinder",
            density=6,
            resolution=32,
            seeds=[SeedInput(x=radius, y=0)],
        )
    )
    manual = next(line for line in scene.lines if line.seed_index == 0)
    assert (radius, 0.0) in manual.points
    assert manual.termination == manual.start_termination == "domain_exit"
    xy = np.asarray(manual.points)
    np.testing.assert_allclose(xy[:, 1], 0, atol=1e-12)
    for boundary in (-1.0, -0.7, 0.7, 1.0):
        assert np.min(np.abs(xy[:, 0] - boundary)) < 1e-7


@pytest.mark.parametrize(
    "change",
    [
        {"inner_radius": None},
        {"inner_radius": 0},
        {"inner_radius": 1},
        {"relative_permeability": None},
        {"relative_permeability": 0.5},
        {"relative_permittivity": 2},
        {"kind": "conducting_sphere"},
    ],
)
def test_annular_region_rejects_incomplete_or_mixed_material_contract(change: dict) -> None:
    payload = {
        "kind": "magnetic_shield_cylinder",
        "x": 0,
        "y": 0,
        "radius": 1,
        "inner_radius": 0.7,
        "relative_permeability": 100,
        "unit": "m",
    }
    with pytest.raises(ValidationError):
        RegionPayload.model_validate({**payload, **change})
