"""Contracts for serializable scenes and the browser-facing FastAPI app."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vectorviz import __version__
from vectorviz.web import app as web_app
from vectorviz.web.app import STATIC_DIR, create_app
from vectorviz.web.scene import build_scene
from vectorviz.web.schemas import SceneRequest, SourceInput


@pytest.mark.parametrize(
    ("preset", "source_kinds", "scalar_label", "scalar_unit"),
    [
        ("electric_dipole", {"positive", "negative"}, "|E|", "V/m"),
        ("magnetic_dipole", {"dipole"}, "|B|", "T"),
        # A uniform field has no localized source marker.
        ("uniform", set(), "|E|", "V/m"),
    ],
)
def test_scene_presets_are_finite_serializable_and_trace_requested_lines(
    preset: str,
    source_kinds: set[str],
    scalar_label: str,
    scalar_unit: str,
) -> None:
    density = 6
    resolution = 32
    scene = build_scene(
        SceneRequest(preset=preset, density=density, resolution=resolution)  # type: ignore[arg-type]
    )

    assert scene.domain.x == (-3.0, 3.0)
    assert scene.domain.y == (-3.0, 3.0)
    assert scene.scalar.nx == resolution
    assert scene.scalar.ny == resolution
    assert len(scene.scalar.values) == resolution**2
    assert len(scene.scalar.mask) == resolution**2
    assert all(math.isfinite(value) for value in scene.scalar.values)
    assert math.isfinite(scene.scalar.vmin)
    assert math.isfinite(scene.scalar.vmax)
    assert scene.scalar.vmin <= scene.scalar.vmax
    assert scene.scalar.label == scalar_label
    assert scene.scalar.unit == scalar_unit
    assert {source.kind for source in scene.sources} == source_kinds

    # Every requested seed should produce a drawable line and a counted reason.
    assert len(scene.lines) == density
    assert sum(scene.metadata.termination_counts.values()) == density
    assert all(len(line.points) >= 2 for line in scene.lines)
    assert all(
        math.isfinite(coordinate)
        for line in scene.lines
        for point in line.points
        for coordinate in point
    )

    # Reject NaN/Infinity just as a strict browser JSON parser would.
    json.dumps(scene.model_dump(mode="json"), allow_nan=False)


def test_electric_scene_honors_source_override() -> None:
    sources = [
        SourceInput(x=-1.25, y=0.35, kind="positive", strength=2.0),
        SourceInput(x=0.75, y=-0.2, kind="negative", strength=-0.5),
    ]
    scene = build_scene(
        SceneRequest(
            preset="electric_dipole",
            density=6,
            resolution=32,
            sources=sources,
        )
    )

    assert [source.model_dump() for source in scene.sources] == [
        source.model_dump() for source in sources
    ]
    assert len(scene.lines) == 6
    assert sum(scene.metadata.termination_counts.values()) == 6


def test_scene_request_validates_incompatible_source_overrides() -> None:
    with pytest.raises(ValueError, match="positive and negative"):
        SceneRequest(
            preset="electric_dipole",
            sources=[SourceInput(x=0.0, y=0.0, kind="dipole")],
        )
    with pytest.raises(ValueError, match="does not accept"):
        SceneRequest(
            preset="uniform",
            sources=[SourceInput(x=0.0, y=0.0, kind="uniform")],
        )


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_and_preset_endpoints(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "version": __version__}

    presets = client.get("/api/presets")
    assert presets.status_code == 200
    payload = presets.json()
    assert {preset["id"] for preset in payload} == {
        "electric_dipole",
        "magnetic_dipole",
        "uniform",
    }
    assert all(preset["label"] and preset["description"] for preset in payload)


def test_scene_endpoint_returns_browser_contract(client: TestClient) -> None:
    response = client.post(
        "/api/scene",
        json={"preset": "uniform", "density": 6, "resolution": 32},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scalar"]["nx"] == 32
    assert len(payload["lines"]) == 6
    assert payload["metadata"]["termination_counts"] == {"domain_exit": 6}


def test_static_index_and_assets_are_served(client: TestClient) -> None:
    if not Path(STATIC_DIR).is_dir():
        pytest.skip("frontend assets are not part of this installation")

    index = client.get("/")
    assert index.status_code == 200
    assert "text/html" in index.headers["content-type"]
    assert "VectorViz" in index.text
    assert "field-canvas" in index.text

    for asset in ("/app.js", "/styles.css"):
        response = client.get(asset)
        assert response.status_code == 200
        assert response.text.strip()


@pytest.mark.parametrize(
    ("argv", "expected_host", "expected_port", "expected_reload"),
    [
        ([], "127.0.0.1", 8000, False),
        (["--host", "0.0.0.0", "--port", "9000", "--reload"], "0.0.0.0", 9000, True),
    ],
)
def test_cli_passes_server_options_to_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected_host: str,
    expected_port: int,
    expected_reload: bool,
) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    def record_run(target: object, **options: object) -> None:
        calls.append((target, options))

    monkeypatch.setattr(web_app.uvicorn, "run", record_run)

    web_app.main(argv)

    assert len(calls) == 1
    target, options = calls[0]
    if expected_reload:
        assert target == "vectorviz.web.app:create_app"
    else:
        assert getattr(target, "title", None) == "VectorViz API"
    assert options == {
        "host": expected_host,
        "port": expected_port,
        "reload": expected_reload,
        "factory": expected_reload,
    }
