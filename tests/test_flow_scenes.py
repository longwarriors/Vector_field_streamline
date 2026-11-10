"""Physical-time scene contracts, independent analytic oracles and cache isolation."""

from copy import deepcopy

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from vectorviz.web.app import create_app
from vectorviz.web.scene import _trace_flow_cached, build_scene
from vectorviz.web.schemas import FlowTime, SceneRequest, SceneResponse


def request(**changes):
    return SceneRequest.model_validate({
        "preset": "flow_oscillating", "density": 7, "resolution": 32, **changes,
    })


@pytest.mark.parametrize("preset", ["flow_oscillating", "flow_steady"])
def test_three_curve_families_match_independent_analytic_solutions(preset):
    scene = build_scene(request(preset=preset, seeds=[{"x": -1, "y": -0.5}]))
    flow = scene.flow
    assert flow.time == FlowTime(observation=4)
    assert flow.time_unit == "s"
    assert scene.domain.unit == "m"
    assert scene.scalar.unit == scene.vector.unit == "m/s"
    assert len(scene.lines) == 2
    assert scene.metadata.termination_counts == {"domain_exit": 1}
    assert scene.metadata.manual_seed_count == 1
    assert scene.metadata.manual_termination_counts == {"domain_exit": 1}
    assert flow.release_sites == [(-2, 0), (-1, -0.5)]
    oscillating = preset == "flow_oscillating"
    velocity_y = 0.6 * np.sin(4) if oscillating else 0.4
    np.testing.assert_allclose(scene.vector.u, 1)
    np.testing.assert_allclose(scene.vector.v, velocity_y)
    np.testing.assert_allclose(scene.scalar.values, np.hypot(1, velocity_y))
    for index, (x, y) in enumerate(flow.release_sites):
        path = flow.pathlines[index]
        times = np.asarray(path.times)
        points = np.asarray(path.points)
        assert path.termination == "time_end"
        np.testing.assert_allclose(points[:, 0], x + times, atol=2e-12)
        np.testing.assert_allclose(points[:, 1], y + (
            0.6 * (1 - np.cos(times)) if oscillating else 0.4 * times
        ), atol=2e-10)
        streak = flow.streaklines[index]
        releases = np.asarray(streak.release_times)
        np.testing.assert_allclose(releases, np.linspace(0, 4, 7))
        assert streak.arrival_times == [4] * 7
        assert streak.terminations == ["time_end"] * 7
        points = np.asarray(streak.points)
        np.testing.assert_allclose(points[:, 0], x + 4 - releases, atol=2e-12)
        np.testing.assert_allclose(points[:, 1], y + (
            0.6 * (np.cos(releases) - np.cos(4)) if oscillating else 0.4 * (4 - releases)
        ), atol=2e-10)
        line = scene.lines[index]
        xy = np.asarray(line.points)
        np.testing.assert_allclose(xy[:, 1] - y, velocity_y * (xy[:, 0] - x), atol=2e-10)
    if oscillating:
        # At x=0 all three default curves have clearly distinct y values.
        y_path = 0.6 * (1 - np.cos(2))
        y_streak = 0.6 * (np.cos(2) - np.cos(4))
        y_stream = 2 * velocity_y
        assert min(abs(y_path-y_streak), abs(y_path-y_stream), abs(y_streak-y_stream)) > 0.7


def test_zero_duration_and_exited_particles_have_reproducible_time_semantics():
    instant = build_scene(request(time={"observation": 0})).flow
    assert instant.pathlines[0].points == [(-2, 0)]
    assert instant.pathlines[0].times == [0]
    assert instant.streaklines[0].points == [(-2, 0)]
    assert instant.streaklines[0].release_times == [0]
    late = build_scene(request(time={"observation": 6}, density=7)).flow
    assert late.pathlines[0].termination == "domain_exit"
    assert late.pathlines[0].times[-1] == pytest.approx(5, abs=2e-12)
    assert late.streaklines[0].points[0] is None
    assert late.streaklines[0].arrival_times[0] == pytest.approx(5, abs=2e-12)
    assert late.streaklines[0].terminations[0] == "domain_exit"
    assert late.streaklines[0].points[-1] == (-2, 0)
    assert late.streaklines[0].arrival_times[-1] == 6


def test_narrow_release_window_coalesces_only_unrepresentable_duplicate_times():
    start, end = 1.0, float(np.nextafter(1.0, 2.0))
    flow = build_scene(request(time={
        "observation": 2, "release_start": start, "release_end": end,
    })).flow
    streak = flow.streaklines[0]
    assert streak.release_times == [start, end]
    assert streak.arrival_times == [2, 2]
    assert streak.terminations == ["time_end", "time_end"]
    np.testing.assert_allclose(
        streak.points, np.column_stack((
            -np.array([start, end]),
            0.6 * (np.cos([start, end]) - np.cos(2)),
        )), atol=2e-12,
    )


def test_each_time_coordinate_and_release_density_participates_in_cache_key():
    _trace_flow_cached.cache_clear()
    first = build_scene(request()).flow
    base_info = _trace_flow_cached.cache_info()
    assert base_info.misses == 1
    # Copies protect every nested field, even site lists and the time model.
    first.time.observation = 1
    first.pathlines[0].points[0] = (9, 9)
    first.streaklines[0].points.clear()
    second = build_scene(request(resolution=40)).flow
    assert _trace_flow_cached.cache_info().hits == 1
    assert second.time.observation == 4
    assert second.pathlines[0].points[0] == (-2, 0)
    assert len(second.streaklines[0].points) == 7
    for change in (
        {"time": {"observation": 3}}, {"time": {"initial": 1}},
        {"time": {"release_start": 1}}, {"time": {"release_end": 3}},
        {"density": 8}, {"seeds": [{"x": 0, "y": 0}]}, {"preset": "flow_steady"},
    ):
        before = _trace_flow_cached.cache_info().misses
        build_scene(request(**change))
        assert _trace_flow_cached.cache_info().misses == before + 1


@pytest.mark.parametrize("time", [
    {"observation": -0.01}, {"observation": 6.01}, {"observation": float("nan")},
    {"initial": float("inf")}, {"initial": 5}, {"release_start": -1},
    {"release_start": 3, "release_end": 2}, {"release_end": 5},
    {"release_end": None}, {"observation": 2, "initial": 3}, {"unknown": 1},
])
def test_invalid_time_requests_are_rejected(time):
    with pytest.raises(ValidationError):
        request(time=time)


@pytest.mark.parametrize("time", [
    {"observation": True}, {"initial": "1"}, {"release_start": False}, {"release_end": "2"},
])
def test_time_contract_accepts_json_numbers_without_string_or_boolean_coercion(time):
    with pytest.raises(ValidationError):
        request(time=time)


def test_http_time_defaults_manual_boundary_and_static_contract():
    with TestClient(create_app()) as client:
        response = client.post("/api/scene", json={
            "preset": "flow_oscillating", "density": 6, "resolution": 32,
            "time": {"observation": 2}, "seeds": [{"x": 3, "y": 0}],
        })
        assert response.status_code == 200, response.text
        scene = response.json()
        assert scene["flow"]["time"]["release_end"] == 2
        path = scene["flow"]["pathlines"][1]
        assert path["termination"] == "domain_exit"
        assert path["points"] == [[3, 0]]
        assert path["times"] == [0]
        assert client.post("/api/scene", json={"preset": "uniform", "time": {}}).status_code == 422
        assert client.post("/api/scene", json={
            "preset": "flow_steady", "sources": [{"x": 0, "y": 0, "kind": "positive"}],
        }).status_code == 422
        assert client.post("/api/scene", json={
            "preset": "flow_steady", "time": {"observation": 1, "release_end": 2},
        }).status_code == 422
        static = client.post("/api/scene", json={"preset": "uniform", "density": 6, "resolution": 32})
        assert static.status_code == 200
        assert "flow" not in static.json()


@pytest.mark.parametrize("mutation", [
    lambda s: s["flow"].update(time_unit="ms"),
    lambda s: s["scalar"].update(unit="T"),
    lambda s: s["flow"]["pathlines"][0]["times"].pop(),
    lambda s: s["flow"]["pathlines"][0]["times"].__setitem__(1, 0),
    lambda s: s["flow"]["pathlines"][0]["times"].__setitem__(-1, 3.99),
    lambda s: s["flow"]["pathlines"][0].update(site_index=1),
    lambda s: s["flow"]["pathlines"][0]["points"].__setitem__(0, [0, 0]),
    lambda s: s["flow"]["pathlines"][0]["points"].__setitem__(1, [9, 0]),
    lambda s: s["flow"]["streaklines"][0]["arrival_times"].__setitem__(0, 3),
    lambda s: s["flow"]["streaklines"][0]["arrival_times"].__setitem__(0, 7),
    lambda s: s["flow"]["streaklines"][0]["release_times"].__setitem__(0, -1),
    lambda s: s["flow"]["streaklines"][0]["release_times"].__setitem__(1, 0),
    lambda s: s["flow"]["streaklines"][0]["points"].__setitem__(0, None),
    lambda s: s["flow"]["streaklines"][0]["points"].__setitem__(-1, [0, 1]),
    lambda s: s["flow"]["pathlines"][0]["times"].__setitem__(0, False),
    lambda s: s["flow"]["streaklines"][0]["release_times"].__setitem__(0, "0"),
    lambda s: s["flow"]["streaklines"][0]["arrival_times"].__setitem__(0, "4"),
    lambda s: s["flow"]["streaklines"][0]["terminations"].pop(),
    lambda s: s["flow"]["release_sites"].append([0, 0]),
])
def test_response_rejects_inconsistent_particle_time_and_visibility_contract(mutation):
    payload = deepcopy(build_scene(request()).model_dump())
    mutation(payload)
    with pytest.raises(ValidationError):
        SceneResponse.model_validate(payload)
