"""Contracts for serializable scenes and the browser-facing FastAPI app."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vectorviz import CircularLoopField, MagneticDipoleField, __version__
from vectorviz.web import app as web_app
from vectorviz.web.app import STATIC_DIR, create_app
from vectorviz.web.scene import (
    SOURCE_RADIUS,
    _allocate_seed_counts,
    _build_model,
    _PlanarCircularLoopField,
    build_scene,
)
from vectorviz.web.schemas import SceneRequest, SourceInput, SourcePayload


@pytest.mark.parametrize(
    (
        "preset",
        "source_kinds",
        "source_strength_units",
        "scalar_label",
        "scalar_unit",
        "density",
    ),
    [
        ("electric_dipole", {"positive", "negative"}, {"nC"}, "|E|", "V/m", 6),
        ("magnetic_dipole", {"dipole"}, {"A·m²"}, "|B|", "T", 6),
        ("halbach_array", {"dipole"}, {"A·m²"}, "|B|", "T", 8),
        ("current_loop", {"wire_out", "wire_into"}, {"A"}, "|B|", "T", 6),
        # A uniform field has no localized source marker.
        ("uniform", set(), set(), "|E|", "V/m", 6),
    ],
)
def test_scene_presets_are_finite_serializable_and_trace_requested_lines(
    preset: str,
    source_kinds: set[str],
    source_strength_units: set[str],
    scalar_label: str,
    scalar_unit: str,
    density: int,
) -> None:
    resolution = 32
    scene = build_scene(
        SceneRequest(preset=preset, density=density, resolution=resolution)  # type: ignore[arg-type]
    )

    assert scene.domain.x == (-3.0, 3.0)
    assert scene.domain.y == (-3.0, 3.0)
    assert scene.domain.coordinate_system == "cartesian"
    assert scene.domain.unit == "m"
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
    assert {source.strength_unit for source in scene.sources} == source_strength_units

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

    assert [source.model_dump(exclude_none=True) for source in scene.sources] == [
        {**source.model_dump(exclude_none=True), "strength_unit": "nC"} for source in sources
    ]
    assert len(scene.lines) == 6
    assert sum(scene.metadata.termination_counts.values()) == 6


def test_current_loop_uses_response_only_markers_and_closed_loop_tracing() -> None:
    request = SceneRequest(preset="current_loop", density=6, resolution=32)

    model = _build_model(request)
    scene = build_scene(request)

    assert [source.model_dump(exclude_none=True) for source in scene.sources] == [
        {
            "x": -1.0,
            "y": 0.0,
            "kind": "wire_out",
            "strength": 1.0,
            "strength_unit": "A",
        },
        {
            "x": 1.0,
            "y": 0.0,
            "kind": "wire_into",
            "strength": 1.0,
            "strength_unit": "A",
        },
    ]
    assert model.seeds.shape == (request.density, 2)
    assert np.all(model.exclusions[0].margin(model.seeds) > 0.0)
    sorted_seed_x = np.sort(model.seeds[:, 0])
    np.testing.assert_allclose(sorted_seed_x, -sorted_seed_x[::-1], rtol=0.0, atol=0.0)
    assert model.trace_options.closure_tolerance is not None
    assert scene.metadata.termination_counts.get("closed_loop", 0) >= 2
    assert scene.metadata.termination_counts.get("max_arc_length", 0) == 0
    assert sum(scene.metadata.termination_counts.values()) == request.density
    assert len(scene.lines) == request.density
    domain_exit_lines = [line for line in scene.lines if line.termination == "domain_exit"]
    assert domain_exit_lines
    for line in domain_exit_lines:
        points = np.asarray(line.points)
        assert np.min(points[:, 1]) < -0.1
        assert np.max(points[:, 1]) > 0.1
        assert np.min(points[:, 1]) == pytest.approx(-np.max(points[:, 1]))
        assert np.count_nonzero(points[:, 1] == 0.0) == 1
        segments = np.diff(points, axis=0)
        vectors = model.field.evaluate(0.5 * (points[:-1] + points[1:]))
        tangent_cosine = np.einsum("ij,ij->i", segments, vectors) / (
            np.linalg.norm(segments, axis=1) * np.linalg.norm(vectors, axis=1)
        )
        assert np.min(tangent_cosine) > 0.99
    assert scene.metadata.field_model.startswith("三维理想圆形电流线圈")
    assert "子午面" in scene.metadata.projection_note
    assert "等通量" not in scene.metadata.seed_mode


def test_current_loop_planar_adapter_matches_the_invariant_3d_field() -> None:
    loop = CircularLoopField(1.0, 1.0, normal=(0.0, 1.0, 0.0))
    planar = _PlanarCircularLoopField(loop)
    points = np.array(((0.4, 0.3), (-0.4, 0.3), (0.4, -0.3)))
    embedded = np.column_stack((points, np.zeros(points.shape[0])))

    vectors = planar.evaluate(points)

    np.testing.assert_allclose(vectors, loop.evaluate(embedded)[:, :2], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(vectors[1], (-vectors[0, 0], vectors[0, 1]))
    np.testing.assert_allclose(vectors[2], (-vectors[0, 0], vectors[0, 1]))


def test_current_loop_masks_both_finite_radius_conductor_cross_sections() -> None:
    scene = build_scene(SceneRequest(preset="current_loop", density=6, resolution=64))
    mask = np.asarray(scene.scalar.mask).reshape(scene.scalar.ny, scene.scalar.nx)
    x = np.linspace(scene.domain.x[0], scene.domain.x[1], scene.scalar.nx)
    y = np.linspace(scene.domain.y[1], scene.domain.y[0], scene.scalar.ny)

    for wire_x in (-1.0, 1.0):
        x_index = int(np.argmin(np.abs(x - wire_x)))
        y_index = int(np.argmin(np.abs(y)))
        assert mask[y_index, x_index]


def test_current_loop_odd_seed_budget_adds_one_axis_line() -> None:
    request = SceneRequest(preset="current_loop", density=7, resolution=32)

    model = _build_model(request)

    assert model.seeds.shape == (request.density, 2)
    axis_seeds = model.seeds[model.seeds[:, 0] == 0.0]
    np.testing.assert_allclose(axis_seeds, ((0.0, -2.9999),), rtol=0.0, atol=1.0e-15)
    nonaxis_x = np.sort(model.seeds[model.seeds[:, 0] != 0.0, 0])
    np.testing.assert_allclose(nonaxis_x, -nonaxis_x[::-1], rtol=0.0, atol=0.0)
    assert np.all(model.exclusions[0].margin(model.seeds) > 0.0)


def test_dipole_angle_defaults_to_positive_y_and_is_returned_explicitly(
    client: TestClient,
) -> None:
    source = SourceInput(x=0.25, y=-0.4, kind="dipole", strength=-2.0)

    assert source.angle_deg == 90.0
    response = client.post(
        "/api/scene",
        json={
            "preset": "magnetic_dipole",
            "density": 6,
            "resolution": 32,
            "sources": [{"x": source.x, "y": source.y, "kind": source.kind}],
        },
    )

    assert response.status_code == 200
    assert response.json()["sources"] == [
        {
            "x": source.x,
            "y": source.y,
            "kind": "dipole",
            "strength": 1.0,
            "strength_unit": "A·m²",
            "angle_deg": 90.0,
        }
    ]


def test_dipole_angle_is_published_by_request_response_and_openapi_schemas(
    client: TestClient,
) -> None:
    input_schema = SourceInput.model_json_schema()
    payload_schema = SourcePayload.model_json_schema()
    openapi_schemas = client.get("/openapi.json").json()["components"]["schemas"]

    for schema in (input_schema, payload_schema, openapi_schemas["SourceInput"], openapi_schemas["SourcePayload"]):
        angle_schema = schema["properties"]["angle_deg"]
        number_branch = next(
            branch for branch in angle_schema["anyOf"] if branch.get("type") == "number"
        )
        assert number_branch["minimum"] == 0.0
        assert number_branch["exclusiveMaximum"] == 360.0
        assert "counterclockwise" in angle_schema["description"]
        assert "only for dipole sources" in angle_schema["description"]
        assert "cannot use null" in angle_schema["description"]
        assert "non-dipole sources must omit" in angle_schema["description"]
    assert "default" not in input_schema["properties"]["angle_deg"]
    assert "default" not in openapi_schemas["SourceInput"]["properties"]["angle_deg"]


@pytest.mark.parametrize(
    ("preset", "kind", "strength", "angle_deg"),
    [
        ("electric_dipole", "positive", 1.0, 0.0),
        ("electric_dipole", "negative", -1.0, None),
        ("uniform", "uniform", 1.0, 0.0),
    ],
)
def test_non_dipole_sources_reject_angle_deg(
    client: TestClient,
    preset: str,
    kind: str,
    strength: float,
    angle_deg: float | None,
) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": preset,
            "sources": [
                {
                    "x": 0.0,
                    "y": 0.0,
                    "kind": kind,
                    "strength": strength,
                    "angle_deg": angle_deg,
                }
            ],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == (
        "Value error, angle_deg is only valid for dipole sources"
    )


@pytest.mark.parametrize("angle_deg", [-1.0, 360.0, None])
def test_dipole_sources_reject_invalid_or_null_angle(
    client: TestClient, angle_deg: float | None
) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": "magnetic_dipole",
            "sources": [
                {"x": 0.0, "y": 0.0, "kind": "dipole", "angle_deg": angle_deg}
            ],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]


@pytest.mark.parametrize("angle_deg", [float("nan"), float("inf"), float("-inf")])
def test_dipole_sources_reject_nonfinite_angles(angle_deg: float) -> None:
    with pytest.raises(ValueError, match="finite number"):
        SourceInput(x=0.0, y=0.0, kind="dipole", angle_deg=angle_deg)


def test_nonstandard_json_nan_is_reported_as_validation_error() -> None:
    with TestClient(create_app(), raise_server_exceptions=False) as client:
        response = client.post(
            "/api/scene",
            content=(
                '{"preset":"magnetic_dipole","sources":['
                '{"x":0,"y":0,"kind":"dipole","angle_deg":NaN}]}'
            ),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 422
    assert "finite number" in response.text


@pytest.mark.parametrize("angle_deg", [0.0, float(np.nextafter(360.0, 0.0))])
def test_dipole_angle_accepts_both_legal_boundaries(
    client: TestClient, angle_deg: float
) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": "magnetic_dipole",
            "density": 6,
            "resolution": 32,
            "sources": [
                {"x": 0.0, "y": 0.0, "kind": "dipole", "angle_deg": angle_deg}
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["sources"][0]["angle_deg"] == angle_deg


def test_magnetic_seed_axes_follow_signed_in_plane_moments() -> None:
    angles = np.array((0.0, 35.0, 90.0, 180.0, 245.0, 315.0))
    strengths = np.array((1.0, -2.0, 0.5, -0.75, 3.0, -1.25))
    centers = np.column_stack((np.linspace(-2.4, 2.4, angles.size), np.zeros(angles.size)))
    sources = [
        SourceInput(
            x=float(center[0]),
            y=float(center[1]),
            kind="dipole",
            strength=float(strength),
            angle_deg=float(angle),
        )
        for center, strength, angle in zip(centers, strengths, angles, strict=True)
    ]

    model = _build_model(
        SceneRequest(
            preset="magnetic_dipole", density=len(sources), resolution=32, sources=sources
        )
    )

    base_axes = np.column_stack((np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))))
    actual_axes = np.sign(strengths)[:, np.newaxis] * base_axes
    expected = centers + (SOURCE_RADIUS + 2.0e-3) * actual_axes
    np.testing.assert_allclose(model.seeds, expected, rtol=0.0, atol=2.0e-15)
    radial = model.seeds - centers
    assert np.all(np.einsum("ij,ij->i", model.field.evaluate(model.seeds), radial) > 0.0)


def test_halbach_defaults_form_two_quarter_turn_cycles_and_seed_each_dipole_once() -> None:
    request = SceneRequest(preset="halbach_array", density=8, resolution=32)
    model = _build_model(request)

    centers = np.array([(source.x, source.y) for source in model.sources])
    angles = np.array([source.angle_deg for source in model.sources])
    strengths = np.array([source.strength for source in model.sources])
    assert len(model.sources) == 8
    assert {source.kind for source in model.sources} == {"dipole"}
    assert {source.strength_unit for source in model.sources} == {"A·m²"}
    np.testing.assert_allclose(angles, np.tile((0.0, 90.0, 180.0, 270.0), 2))
    np.testing.assert_allclose(strengths, np.ones(8), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(centers[:, 1], np.zeros(8), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(centers[:, 0], -centers[::-1, 0], rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        np.diff(centers[:, 0]), np.full(7, centers[1, 0] - centers[0, 0]), rtol=0.0, atol=1e-15
    )

    axes = np.column_stack((np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))))
    np.testing.assert_allclose(
        model.seeds, centers + (SOURCE_RADIUS + 2.0e-3) * axes, rtol=0.0, atol=2.0e-15
    )


@pytest.mark.parametrize("density", [6, 7])
def test_halbach_default_rejects_density_below_its_eight_source_minimum(
    client: TestClient, density: int
) -> None:
    response = client.post(
        "/api/scene",
        json={"preset": "halbach_array", "density": density, "resolution": 32},
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "halbach_array 有 8 个磁偶极子参与播种，density 至少为 8"
    }


def test_halbach_accepts_dipole_only_source_overrides(client: TestClient) -> None:
    accepted = client.post(
        "/api/scene",
        json={
            "preset": "halbach_array",
            "density": 6,
            "resolution": 32,
            "sources": [
                {"x": -0.5, "y": 0.0, "kind": "dipole", "angle_deg": 10.0},
                {"x": 0.5, "y": 0.0, "kind": "dipole", "angle_deg": 100.0},
            ],
        },
    )
    rejected = client.post(
        "/api/scene",
        json={
            "preset": "halbach_array",
            "sources": [
                {"x": -0.5, "y": 0.0, "kind": "positive"},
                {"x": 0.5, "y": 0.0, "kind": "negative"},
            ],
        },
    )

    assert accepted.status_code == 200
    assert [source["angle_deg"] for source in accepted.json()["sources"]] == [10.0, 100.0]
    accepted_metadata = accepted.json()["metadata"]
    assert "可编辑" in accepted_metadata["title"]
    assert "Halbach" not in accepted_metadata["field_model"]
    assert "八个" not in accepted_metadata["field_model"]
    assert rejected.status_code == 422
    assert rejected.json()["detail"][0]["msg"] == (
        "Value error, halbach_array accepts dipole sources only"
    )


def test_halbach_planar_adapter_matches_core_and_has_zero_normal_component() -> None:
    model = _build_model(SceneRequest(preset="halbach_array", density=8, resolution=32))
    centers = np.array([(source.x, source.y) for source in model.sources])
    angles = np.deg2rad([source.angle_deg for source in model.sources])
    strengths = np.array([source.strength for source in model.sources])
    moments = strengths[:, np.newaxis] * np.column_stack(
        (np.cos(angles), np.sin(angles), np.zeros(angles.size))
    )
    positions = np.column_stack((centers, np.zeros(centers.shape[0])))
    core = MagneticDipoleField(moments, positions)
    points = np.array(((-1.83, 0.72), (-0.1, -1.15), (1.61, 0.44), (2.7, -2.1)))
    embedded = np.column_stack((points, np.zeros(points.shape[0])))

    core_vectors = core.evaluate(embedded)

    np.testing.assert_allclose(model.field.evaluate(points), core_vectors[:, :2], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(core_vectors[:, 2], np.zeros(points.shape[0]), rtol=0.0, atol=0.0)


def test_halbach_has_a_quantitatively_stronger_positive_y_sampling_band() -> None:
    model = _build_model(SceneRequest(preset="halbach_array", density=8, resolution=32))
    x = np.linspace(-1.8, 1.8, 19)
    y = np.array((0.55, 0.75, 1.0, 1.25, 1.5))
    xx, yy = np.meshgrid(x, y)
    strong_points = np.column_stack((xx.ravel(), yy.ravel()))
    weak_points = strong_points.copy()
    weak_points[:, 1] *= -1.0

    strong_energy = np.mean(np.linalg.norm(model.field.evaluate(strong_points), axis=1) ** 2)
    weak_energy = np.mean(np.linalg.norm(model.field.evaluate(weak_points), axis=1) ** 2)

    # The reference geometry gives about 13.5. Requiring 8 keeps substantial
    # margin for finite-array edge effects while testing a band, not one tuned point.
    assert strong_energy > 8.0 * weak_energy


def test_rotating_dipole_position_moment_and_query_rotates_the_field() -> None:
    rotation_deg = 73.0
    angle_deg = 24.0
    rotation_rad = np.deg2rad(rotation_deg)
    rotation = np.array(
        (
            (np.cos(rotation_rad), -np.sin(rotation_rad)),
            (np.sin(rotation_rad), np.cos(rotation_rad)),
        )
    )
    center = np.array((0.45, -0.35))
    rotated_center = rotation @ center
    points = np.array(((-1.2, 0.7), (0.1, 1.6), (1.8, -0.9)))
    base = _build_model(
        SceneRequest(
            preset="magnetic_dipole",
            sources=[
                SourceInput(
                    x=center[0],
                    y=center[1],
                    kind="dipole",
                    strength=-1.7,
                    angle_deg=angle_deg,
                )
            ],
        )
    )
    rotated = _build_model(
        SceneRequest(
            preset="magnetic_dipole",
            sources=[
                SourceInput(
                    x=rotated_center[0],
                    y=rotated_center[1],
                    kind="dipole",
                    strength=-1.7,
                    angle_deg=angle_deg + rotation_deg,
                )
            ],
        )
    )

    expected = base.field.evaluate(points) @ rotation.T
    actual = rotated.field.evaluate(points @ rotation.T)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=1.0e-21)


def test_halbach_traces_preserve_seed_budget_and_are_tangent_to_the_field() -> None:
    request = SceneRequest(preset="halbach_array", density=8, resolution=32)
    model = _build_model(request)
    scene = build_scene(request)

    assert len(scene.lines) == request.density
    assert sum(scene.metadata.termination_counts.values()) == request.density
    residuals: list[float] = []
    for line in scene.lines:
        points = np.asarray(line.points)
        segments = np.diff(points, axis=0)
        vectors = model.field.evaluate(0.5 * (points[:-1] + points[1:]))
        residuals.extend(
            (
                np.abs(segments[:, 0] * vectors[:, 1] - segments[:, 1] * vectors[:, 0])
                / (np.linalg.norm(segments, axis=1) * np.linalg.norm(vectors, axis=1))
            ).tolist()
        )

    assert residuals
    assert max(residuals) < 0.02


def test_source_input_and_payload_publish_separate_kind_vocabularies(
    client: TestClient,
) -> None:
    input_kinds = set(SourceInput.model_json_schema()["properties"]["kind"]["enum"])
    payload_kinds = set(SourcePayload.model_json_schema()["properties"]["kind"]["enum"])

    assert input_kinds == {"positive", "negative", "dipole", "uniform"}
    assert payload_kinds == input_kinds | {"wire_out", "wire_into"}
    openapi_schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert set(openapi_schemas["SourceInput"]["properties"]["kind"]["enum"]) == input_kinds
    assert set(openapi_schemas["SourcePayload"]["properties"]["kind"]["enum"]) == payload_kinds


@pytest.mark.parametrize("wire_kind", ["wire_out", "wire_into"])
def test_wire_marker_kinds_cannot_be_submitted_as_sources(
    client: TestClient,
    wire_kind: str,
) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": "current_loop",
            "sources": [{"x": -1.0, "y": 0.0, "kind": wire_kind}],
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["loc"] == ["body", "sources", 0, "kind"]
    assert "positive" in detail[0]["msg"]


def test_current_loop_rejects_even_legacy_source_overrides(client: TestClient) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": "current_loop",
            "sources": [{"x": 0.0, "y": 0.0, "kind": "dipole"}],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == (
        "Value error, current_loop preset does not accept source overrides"
    )


def _electric_sources(positive_count: int) -> list[SourceInput]:
    positions = [
        (-2.4, -1.5),
        (-1.2, -1.5),
        (0.0, -1.5),
        (1.2, -1.5),
        (2.4, -1.5),
        (-1.2, 1.5),
        (1.2, 1.5),
    ]
    positives = [
        SourceInput(x=x, y=y, kind="positive", strength=1.0)
        for x, y in positions[:positive_count]
    ]
    return [*positives, SourceInput(x=0.0, y=0.5, kind="negative", strength=-1.0)]


def test_seed_budget_boundary_assigns_one_seed_to_each_seeding_source() -> None:
    sources = _electric_sources(positive_count=6)
    sources = [
        source.model_copy(update={"strength": 10.0 if index == 0 else 0.1})
        if source.kind == "positive"
        else source
        for index, source in enumerate(sources)
    ]
    request = SceneRequest(
        preset="electric_dipole",
        density=6,
        resolution=32,
        sources=sources,
    )

    model = _build_model(request)
    scene = build_scene(request)

    assert model.seeds.shape == (request.density, 2)
    for source in request.sources[:-1]:
        nearby = sum(
            math.dist(seed, (source.x, source.y)) < 0.17
            for seed in model.seeds.tolist()
        )
        assert nearby == 1
    assert len(scene.lines) == request.density
    assert sum(scene.metadata.termination_counts.values()) == request.density


def test_multisource_scene_keeps_density_as_the_total_seed_budget() -> None:
    request = SceneRequest(
        preset="electric_dipole",
        density=10,
        resolution=32,
        sources=_electric_sources(positive_count=3),
    )

    scene = build_scene(request)

    assert len(scene.lines) == request.density
    assert sum(scene.metadata.termination_counts.values()) == request.density


def test_remaining_seed_budget_uses_largest_remainders() -> None:
    counts = _allocate_seed_counts(np.array((3.0, 2.0, 1.0)), total=10)

    np.testing.assert_array_equal(counts, (5, 3, 2))


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


def test_health_reports_runtime_version(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "version": __version__}


def test_openapi_reports_runtime_version(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["version"] == __version__


def test_preset_endpoint_lists_supported_scenes(client: TestClient) -> None:
    presets = client.get("/api/presets")
    assert presets.status_code == 200
    payload = presets.json()
    assert {preset["id"] for preset in payload} == {
        "electric_dipole",
        "magnetic_dipole",
        "halbach_array",
        "current_loop",
        "uniform",
    }
    assert all(preset["label"] and preset["description"] for preset in payload)


def test_scene_endpoint_returns_browser_contract(client: TestClient) -> None:
    response = client.post(
        "/api/scene",
        json={"preset": "electric_dipole", "density": 6, "resolution": 32},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["domain"] == {
        "x": [-3.0, 3.0],
        "y": [-3.0, 3.0],
        "coordinate_system": "cartesian",
        "unit": "m",
    }
    assert {source["strength_unit"] for source in payload["sources"]} == {"nC"}
    assert payload["scalar"]["nx"] == 32
    assert len(payload["lines"]) == 6
    assert sum(payload["metadata"]["termination_counts"].values()) == 6


@pytest.mark.parametrize(
    ("candidate", "companion"),
    [
        (
            {"x": -1.0, "y": 0.0, "kind": "positive", "strength": 2.0},
            {"x": 1.0, "y": 0.0, "kind": "negative", "strength": -1.0},
        ),
        (
            {"x": 1.0, "y": 0.0, "kind": "negative", "strength": -2.0},
            {"x": -1.0, "y": 0.0, "kind": "positive", "strength": 1.0},
        ),
    ],
)
def test_charge_kind_accepts_matching_nonzero_strength_sign(
    client: TestClient,
    candidate: dict[str, object],
    companion: dict[str, object],
) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": "electric_dipole",
            "density": 6,
            "resolution": 32,
            "sources": [candidate, companion],
        },
    )

    assert response.status_code == 200
    returned = response.json()["sources"]
    assert returned[0]["kind"] == candidate["kind"]
    assert returned[0]["strength"] == candidate["strength"]
    assert returned[0]["strength_unit"] == "nC"


@pytest.mark.parametrize(
    ("candidate", "companion", "message"),
    [
        (
            {"x": -1.0, "y": 0.0, "kind": "positive", "strength": -1.0},
            {"x": 1.0, "y": 0.0, "kind": "negative", "strength": -1.0},
            "positive source strength must be greater than 0; "
            "zero and negative values are invalid",
        ),
        (
            {"x": 1.0, "y": 0.0, "kind": "negative", "strength": 1.0},
            {"x": -1.0, "y": 0.0, "kind": "positive", "strength": 1.0},
            "negative source strength must be less than 0; "
            "zero and positive values are invalid",
        ),
        (
            {"x": -1.0, "y": 0.0, "kind": "positive", "strength": 0.0},
            {"x": 1.0, "y": 0.0, "kind": "negative", "strength": -1.0},
            "positive source strength must be greater than 0; "
            "zero and negative values are invalid",
        ),
        (
            {"x": 1.0, "y": 0.0, "kind": "negative", "strength": 0.0},
            {"x": -1.0, "y": 0.0, "kind": "positive", "strength": 1.0},
            "negative source strength must be less than 0; "
            "zero and positive values are invalid",
        ),
    ],
)
def test_charge_kind_rejects_mismatched_or_zero_strength(
    client: TestClient,
    candidate: dict[str, object],
    companion: dict[str, object],
    message: str,
) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": "electric_dipole",
            "density": 6,
            "resolution": 32,
            "sources": [candidate, companion],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == f"Value error, {message}"


def test_charge_kinds_use_conditional_defaults_when_strength_is_omitted(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": "electric_dipole",
            "density": 6,
            "resolution": 32,
            "sources": [
                {"x": -1.0, "y": 0.0, "kind": "positive"},
                {"x": 1.0, "y": 0.0, "kind": "negative"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["sources"][0]["strength"] == 1.0
    assert response.json()["sources"][1] == {
        "x": 1.0,
        "y": 0.0,
        "kind": "negative",
        "strength": -1.0,
        "strength_unit": "nC",
    }


def test_conditional_charge_default_is_not_misrepresented_in_json_schema() -> None:
    source_schema = SourceInput.model_json_schema()
    strength_schema = source_schema["properties"]["strength"]

    assert "strength" not in source_schema["required"]
    assert "default" not in strength_schema
    assert "positive and dipole default to 1; negative defaults to -1" in str(
        strength_schema["description"]
    )


def test_scene_endpoint_rejects_insufficient_seed_budget_with_actionable_detail(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": "electric_dipole",
            "density": 6,
            "resolution": 32,
            "sources": [
                source.model_dump(exclude_none=True)
                for source in _electric_sources(positive_count=7)
            ],
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "electric_dipole 有 7 个正电荷参与播种，density 至少为 7"
    }


def test_magnetic_seed_budget_error_names_its_seeding_sources(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": "magnetic_dipole",
            "density": 6,
            "resolution": 32,
            "sources": [
                {"x": -2.4 + 0.7 * index, "y": 0.0, "kind": "dipole"}
                for index in range(7)
            ],
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "magnetic_dipole 有 7 个磁偶极子参与播种，density 至少为 7"
    }


def test_degenerate_seed_results_are_counted_but_not_rendered(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": "magnetic_dipole",
            "density": 6,
            "resolution": 32,
            "sources": [{"x": 0.0, "y": 0.0, "kind": "dipole", "strength": 0.0}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["lines"] == []
    assert payload["metadata"]["termination_counts"] == {"null_field": 6}


@pytest.mark.parametrize(
    "payload",
    [
        {"preset": "not-a-preset"},
        {"unknown_top_level": True},
        {"density": -1},
        {"density": 41},
        {"resolution": 145},
        {
            "preset": "electric_dipole",
            "sources": [{"x": 0.0, "kind": "positive"}],
        },
        {
            "preset": "electric_dipole",
            "sources": [
                {"x": 0.0, "y": 0.0, "kind": "positive", "strenght": 1.0},
                {"x": 1.0, "y": 0.0, "kind": "negative"},
            ],
        },
        {
            "preset": "electric_dipole",
            "sources": [
                {
                    "x": -1.0,
                    "y": 0.0,
                    "kind": "positive",
                    "strength": 1.0,
                    "strength_unit": "nC",
                },
                {"x": 1.0, "y": 0.0, "kind": "negative"},
            ],
        },
        {"preset": "electric_dipole", "sources": []},
        {
            "preset": "electric_dipole",
            "sources": [
                {"x": float(index) / 10, "y": 0.0, "kind": "positive"}
                for index in range(9)
            ],
        },
        {
            "preset": "electric_dipole",
            "sources": [{"x": 0.0, "y": 0.0, "kind": "positive"}],
        },
        {
            "preset": "uniform",
            "sources": [{"x": 0.0, "y": 0.0, "kind": "uniform"}],
        },
    ],
)
def test_scene_endpoint_rejects_invalid_limits_and_malformed_sources(
    client: TestClient,
    payload: dict[str, object],
) -> None:
    response = client.post("/api/scene", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]


def test_minimal_scene_request_and_every_advertised_preset_are_usable(
    client: TestClient,
) -> None:
    assert client.post("/api/scene", json={}).status_code == 200

    preset_ids = [item["id"] for item in client.get("/api/presets").json()]
    for preset_id in preset_ids:
        density = 8 if preset_id == "halbach_array" else 6
        response = client.post(
            "/api/scene",
            json={"preset": preset_id, "density": density, "resolution": 32},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["metadata"]["projection_note"]
        assert payload["scalar"]["unit"]
        assert len(payload["scalar"]["values"]) == 32 * 32


def test_static_index_and_assets_are_served(client: TestClient) -> None:
    if not Path(STATIC_DIR).is_dir():
        pytest.skip("frontend assets are not part of this installation")

    index = client.get("/")
    assert index.status_code == 200
    assert "text/html" in index.headers["content-type"]
    assert "VectorViz" in index.text
    assert "field-canvas" in index.text
    option_values = re.findall(r'<option value="([^"]+)">', index.text)
    advertised_presets = [item["id"] for item in client.get("/api/presets").json()]
    assert option_values == advertised_presets

    for asset in (
        "/app.js",
        "/source-controls.js",
        "/coordinates.js",
        "/color-scale.js",
        "/styles.css",
    ):
        response = client.get(asset)
        assert response.status_code == 200
        assert response.text.strip()
    assert '<script type="module" src="app.js"></script>' in index.text


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
