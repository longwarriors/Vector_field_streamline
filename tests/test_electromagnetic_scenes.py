"""Frozen retarded-field slices, SI contracts and independent planar oracles."""

from copy import deepcopy

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scipy.constants import epsilon_0

from vectorviz.web import scene as scene_module
from vectorviz.web.app import create_app
from vectorviz.web.scene import _build_model, _ElectromagneticSlice, build_scene
from vectorviz.web.schemas import SceneRequest, SceneResponse


def request(**changes):
    return SceneRequest.model_validate({
        "preset": "dipole_radiation", "density": 6, "resolution": 33, **changes,
    })


@pytest.fixture(scope="module")
def scenes():
    return {name: build_scene(request(observable=name))
            for name in ("electric", "magnetic", "poynting")}


def spherical_oracle(x, z, observation):
    """Component equations from the dipole's retarded scalar/vector potentials."""
    c, omega, p0 = 299792458.0, 2 * np.pi * 1e8, 1e-9
    r = np.hypot(x, z)
    cosine, sine = z / r, x / r
    phase = omega * (observation - r / c)
    p, dp, ddp = p0 * np.cos(phase), -p0 * omega * np.sin(phase), -p0 * omega**2 * np.cos(phase)
    factor = 1 / (4 * np.pi * epsilon_0)
    er = 2 * factor * cosine * (p / r**3 + dp / (c * r**2))
    et = factor * sine * (p / r**3 + dp / (c * r**2) + ddp / (c**2 * r))
    bphi = factor / c**2 * sine * (dp / r**2 + ddp / (c * r))
    return np.array((er * sine + et * cosine, er * cosine - et * sine)), bphi


@pytest.mark.parametrize("observable,symbol,unit,slice_name,vertical", [
    ("electric", "E", "V/m", "xz", "z"),
    ("magnetic", "B", "T", "xy", "y"),
    ("poynting", "S", "W/m²", "xz", "z"),
])
def test_true_slices_share_retarded_model_and_si_metadata(
    scenes, observable, symbol, unit, slice_name, vertical
):
    scene = scenes[observable]
    em = scene.electromagnetic
    assert em.observation_time == 1.25e-9
    assert em.time_unit == "s"
    assert (em.slice, em.horizontal_axis, em.vertical_axis) == (slice_name, "x", vertical)
    assert em.frequency_hz * em.period == pytest.approx(1)
    assert em.wavelength == pytest.approx(2.99792458)
    assert scene.domain.x == scene.domain.y == (-6, 6)
    assert scene.domain.unit == "m"
    assert scene.scalar.label == f"|{symbol}|"
    assert scene.vector.label == symbol
    assert scene.scalar.unit == scene.vector.unit == unit
    assert scene.flow is None and scene.sources == scene.regions == []
    assert scene.metadata.seed_mode == "coverage"
    assert sum(scene.metadata.termination_counts.values()) == 6
    assert scene.metadata.suppressed_count == 0
    assert scene.scalar.mask[16 * 33 + 16]
    assert scene.scalar.values[16 * 33 + 16] is None
    assert scene.vector.u[16 * 33 + 16] is None
    assert scene.vector.v[16 * 33 + 16] is None
    # Off-axis sample lies exactly on this grid: x=1.5, vertical=0.75.
    index = 14 * 33 + 20
    x, y = 1.5, 0.75
    if observable == "magnetic":
        rho = np.hypot(x, y)
        _, bphi = spherical_oracle(rho, 0, em.observation_time)
        expected = bphi * np.array((-y / rho, x / rho))
    else:
        electric, bphi = spherical_oracle(x, y, em.observation_time)
        expected = electric if observable == "electric" else (
            epsilon_0 * 299792458.0**2 * bphi * np.array((-electric[1], electric[0]))
        )
    np.testing.assert_allclose([scene.vector.u[index], scene.vector.v[index]], expected, rtol=5e-14)
    assert scene.scalar.values[index] == pytest.approx(np.linalg.norm(expected), rel=5e-14)


def test_magnetic_slice_is_real_circular_field_and_terminates_after_one_loop(scenes):
    scene = scenes["magnetic"]
    assert scene.metadata.termination_counts == {"closed_loop": 6}
    for line, radius in zip(scene.lines, np.linspace(0.2, 5.7, 6), strict=True):
        points = np.asarray(line.points)
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), radius, atol=2e-6)
        assert np.linalg.norm(points[-1] - points[0]) < 0.003
        assert line.start_termination is None


def test_instantaneous_energy_keeps_near_zone_inward_direction_and_zero_axis():
    # At t=T/8 and r=0.2 m the source is absorbing stored near-field energy.
    field = _ElectromagneticSlice(1.25e-9, "poynting")
    assert field.evaluate([0.2, 0])[0] < 0
    np.testing.assert_array_equal(field.evaluate([0, 1]), [0, 0])
    scene = build_scene(request(observable="poynting", seeds=[{"x": 0, "y": 1}]))
    assert scene.metadata.manual_termination_counts == {"null_field": 1}
    assert scene.metadata.manual_start_termination_counts == {"null_field": 1}
    assert not any(line.seed_index is not None for line in scene.lines)


def test_magnetic_nodal_circle_terminates_without_inventing_a_direction():
    # B_phi is proportional to sin(phi)+kr*cos(phi); choose phi=-atan(kr).
    omega = 2 * np.pi * 1e8
    kr = omega * 0.2 / 299792458.0
    observation = (kr - np.arctan(kr)) / omega
    scene = build_scene(request(observable="magnetic", em_time={"observation": observation}))
    assert scene.metadata.termination_counts == {"null_field": 1, "closed_loop": 5}
    assert len(scene.lines) == 5


def test_manual_source_exclusion_and_full_slice_boundary_are_reproducible():
    scene = build_scene(request(observable="electric", seeds=[
        {"x": 0, "y": 0}, {"x": 6, "y": 0},
    ]))
    assert scene.metadata.manual_seed_count == 2
    assert scene.metadata.manual_termination_counts["exclusion_hit"] == 1
    assert scene.metadata.manual_start_termination_counts["exclusion_hit"] == 1
    assert all(line.seed_index != 0 for line in scene.lines)
    for line in scene.lines:
        points = np.asarray(line.points)
        # Closure localization uses the scene integrator's spatial tolerances.
        assert np.max(np.abs(points)) <= 6 + 3e-6
        assert np.min(np.linalg.norm(points, axis=1)) >= 0.12 - 1e-8


def test_observable_and_observation_participate_in_both_curve_cache_keys(monkeypatch):
    scene_module._TRACE_CACHE.clear()
    counts = [0, 0]
    original = scene_module._trace_lines
    original_manual = scene_module._trace_manual_lines

    def count_lines(model):
        counts[0] += 1
        return original(model)

    def count_manual(model, seeds):
        counts[1] += 1
        return original_manual(model, seeds)

    monkeypatch.setattr(scene_module, "_trace_lines", count_lines)
    monkeypatch.setattr(scene_module, "_trace_manual_lines", count_manual)
    seeds = [{"x": 0.3, "y": 0.4}]
    first = build_scene(request(seeds=seeds))
    reference = deepcopy(first.lines)
    first.lines[0].points.clear()
    again = build_scene(request(resolution=32, seeds=seeds))
    assert again.lines == reference
    assert counts == [1, 1]
    build_scene(request(observable="magnetic", seeds=seeds))
    assert counts == [2, 2]
    build_scene(request(em_time={"observation": 2.5e-9}, seeds=seeds))
    assert counts == [3, 3]


@pytest.mark.parametrize("change", [
    {"observable": "average"}, {"observable": None},
    {"em_time": {"observation": True}}, {"em_time": {"observation": "1e-9"}},
    {"em_time": {"observation": float("nan")}},
    {"em_time": {"observation": float("inf")}},
    {"em_time": {"observation": -1e-20}}, {"em_time": {"observation": 1.00001e-8}},
    {"em_time": {"initial": 0}}, {"time": {}},
    {"sources": [{"x": 0, "y": 0, "kind": "positive"}]},
    {"seeds": [{"x": 6.00001, "y": 0}]},
    {"seeds": [{"x": float("nan"), "y": 0}]},
    {"preset": "uniform", "em_time": {}},
    {"preset": "uniform", "em_time": None},
    {"preset": "uniform", "observable": "electric"},
    {"preset": "flow_steady", "observable": "poynting"},
    {"preset": "uniform", "seeds": [{"x": 3.01, "y": 0}]},
])
def test_invalid_request_boundaries_and_cross_preset_parameters(change):
    with pytest.raises(ValidationError):
        request(**change)


@pytest.mark.parametrize("mutation", [
    lambda s: s["electromagnetic"].update(slice="xy"),
    lambda s: s["electromagnetic"].update(vertical_axis="y"),
    lambda s: s["electromagnetic"].update(observation_time=True),
    lambda s: s["electromagnetic"].update(observation_time=1.1e-8),
    lambda s: s["electromagnetic"].update(time_unit="ns"),
    lambda s: s["electromagnetic"].update(period=1e-9),
    lambda s: s["electromagnetic"].update(wavelength=3),
    lambda s: s["electromagnetic"].update(moment_amplitude=2e-9),
    lambda s: s["electromagnetic"].update(source_position=[1, 0, 0]),
    lambda s: s["electromagnetic"].update(exclusion_radius=0.1),
    lambda s: s["scalar"].update(unit="T"),
    lambda s: s["scalar"].update(label="|B|"),
    lambda s: s["vector"].update(label="B"),
    lambda s: s.update(vector=None),
    lambda s: s["domain"].update(y=[-3, 3]),
    lambda s: s["domain"].update(unit="1"),
    lambda s: s["sources"].append({
        "x": 0, "y": 0, "kind": "positive", "strength": 1, "strength_unit": "nC",
    }),
])
def test_response_rejects_inconsistent_slice_units_time_and_source_geometry(scenes, mutation):
    data = scenes["electric"].model_dump()
    mutation(data)
    with pytest.raises(ValidationError):
        SceneResponse.model_validate(data)


def test_http_capability_time_boundaries_and_static_compatibility():
    with TestClient(create_app()) as client:
        preset = client.get("/api/presets").json()[-1]
        assert preset["id"] == "dipole_radiation"
        assert preset["electromagnetic"] == {
            "observables": ["electric", "magnetic", "poynting"],
            "time_min": 0, "time_max": 1e-8, "time_default": 1.25e-9, "time_unit": "s",
        }
        for observation in (0, 1e-8):
            response = client.post("/api/scene", json={
                "preset": "dipole_radiation", "density": 6, "resolution": 32,
                "observable": "magnetic", "em_time": {"observation": observation},
            })
            assert response.status_code == 200, response.text
            assert response.json()["electromagnetic"]["observation_time"] == observation
        assert client.post("/api/scene", json={
            "preset": "dipole_radiation", "em_time": {"observation": 6},
        }).status_code == 422
        static = client.post("/api/scene", json={"preset": "uniform", "density": 6, "resolution": 32})
        assert static.status_code == 200
        assert "electromagnetic" not in static.json()
        assert "flow" not in static.json()


def test_slice_adapter_checks_shape_and_seeding_uses_every_budget_slot():
    with pytest.raises(ValueError, match="shape"):
        _ElectromagneticSlice(0, "electric").evaluate([1, 2, 3])
    for observable in ("electric", "magnetic", "poynting"):
        for density in (6, 7, 8, 40):
            model = _build_model(request(observable=observable, density=density))
            assert len(model.trace_jobs) == density
            assert np.all(model.domain.contains(model.seeds))
            assert np.all(model.exclusions[0].margin(model.seeds) > 0)
