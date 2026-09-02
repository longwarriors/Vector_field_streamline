"""Contracts for serializable scenes and the browser-facing FastAPI app."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vectorviz import CircularLoopField, MagneticDipoleField, __version__
from vectorviz.tracing import TerminationReason, TraceBranch, TraceDirection, TraceResult
from vectorviz.web import app as web_app
from vectorviz.web import scene as web_scene
from vectorviz.web import schemas as web_schemas
from vectorviz.web.app import STATIC_DIR, create_app
from vectorviz.web.scene import (
    SOURCE_RADIUS,
    _build_model,
    _PlanarCircularLoopField,
    build_scene,
)
from vectorviz.web.schemas import SceneRequest, SeedMode, SourceInput, SourcePayload
from vectorviz.web.seeding import TraceJob, allocate_seed_counts


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
    assert all(value is None or math.isfinite(value) for value in scene.scalar.values)
    assert all(
        masked is (value is None)
        for value, masked in zip(scene.scalar.values, scene.scalar.mask, strict=True)
    )
    assert math.isfinite(scene.scalar.vmin)
    assert math.isfinite(scene.scalar.vmax)
    assert scene.scalar.vmin <= scene.scalar.vmax
    assert scene.scalar.label == scalar_label
    assert scene.scalar.unit == scalar_unit
    assert {source.kind for source in scene.sources} == source_kinds
    assert {source.strength_unit for source in scene.sources} == source_strength_units

    # Every trace job contributes one primary termination. Rendered lines can be
    # fewer only when a duplicate is suppressed or a trace is degenerate.
    assert sum(scene.metadata.termination_counts.values()) == density
    assert scene.metadata.rendered_line_count == len(scene.lines)
    assert len(scene.lines) + scene.metadata.suppressed_count <= density
    assert scene.metadata.seed_mode in SeedMode
    assert scene.metadata.seed_description
    assert all(count >= 0 for count in scene.metadata.start_termination_counts.values())
    assert all(len(line.points) >= 2 for line in scene.lines)
    assert all(
        math.isfinite(coordinate)
        for line in scene.lines
        for point in line.points
        for coordinate in point
    )

    # Reject NaN/Infinity just as a strict browser JSON parser would.
    json.dumps(scene.model_dump(mode="json"), allow_nan=False)


def test_scene_scalar_preserves_unclipped_finite_values() -> None:
    resolution = 80
    model = _build_model(SceneRequest(preset="electric_dipole", density=6, resolution=resolution))

    scalar = web_scene._sample_scalar(model, resolution)
    x = np.linspace(web_scene.DOMAIN.lower[0], web_scene.DOMAIN.upper[0], resolution)
    y = np.linspace(web_scene.DOMAIN.upper[1], web_scene.DOMAIN.lower[1], resolution)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    expected = np.linalg.norm(model.field.evaluate(np.stack((xx, yy), axis=-1)), axis=-1)
    mask = np.asarray(scalar.mask).reshape(resolution, resolution)
    actual = np.array(
        [np.nan if value is None else value for value in scalar.values], dtype=float
    ).reshape(resolution, resolution)

    assert np.any(expected[~mask] > scalar.vmax)
    assert np.any(expected[~mask] < scalar.vmin)
    np.testing.assert_array_equal(actual[~mask], expected[~mask])


def test_scalar_mask_is_encoded_as_null(client: TestClient) -> None:
    response = client.post(
        "/api/scene",
        json={"preset": "electric_dipole", "density": 6, "resolution": 32},
    )
    assert response.status_code == 200
    scalar = response.json()["scalar"]

    assert any(scalar["mask"])
    assert all(
        masked is (value is None)
        for value, masked in zip(scalar["values"], scalar["mask"], strict=True)
    )
    assert "null" in response.text


def test_response_models_forbid_extra_fields() -> None:
    capability = web_schemas.SourceSeparationCapability(
        exclusive_minimum=0.322,
        unit="m",
    )
    domain = web_schemas.DomainPayload(
        x=(-1.0, 1.0),
        y=(-1.0, 1.0),
        coordinate_system="cartesian",
        unit="m",
    )
    scalar = web_schemas.ScalarPayload(
        nx=1,
        ny=1,
        values=[1.0],
        mask=[False],
        scale="linear",
        label="|F|",
        unit="u",
        vmin=0.5,
        vmax=1.5,
    )
    line = web_schemas.LinePayload(
        points=[(0.0, 0.0), (1.0, 0.0)],
        direction=1,
        termination="future_reason",
        start_termination="future_start_reason",
    )
    source = SourcePayload(
        x=0.0,
        y=0.0,
        kind="dipole",
        strength=1.0,
        strength_unit="A·m²",
        angle_deg=0.0,
    )
    metadata = web_schemas.MetadataPayload(
        title="t",
        projection_note="p",
        field_model="f",
        seed_mode="coverage",
        seed_description="d",
        termination_counts={"future_reason": 0},
        start_termination_counts={"future_start_reason": 0},
        suppressed_count=0,
        rendered_line_count=1,
    )
    scene = web_schemas.SceneResponse(
        domain=domain,
        scalar=scalar,
        lines=[line],
        sources=[source],
        metadata=metadata,
    )
    preset = web_schemas.PresetPayload(
        id="magnetic_dipole",
        label="m",
        description="d",
        source_separation=capability,
    )

    for payload in (capability, domain, scalar, line, source, metadata, scene, preset):
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            type(payload).model_validate({**payload.model_dump(), "unexpected": True})

    with pytest.raises(ValueError, match=r"rendered_line_count must equal len\(lines\)"):
        web_schemas.SceneResponse.model_validate(
            {
                **scene.model_dump(),
                "metadata": {**metadata.model_dump(), "rendered_line_count": 0},
            }
        )


@pytest.mark.parametrize(
    "update",
    [
        {"values": []},
        {"mask": []},
        {"values": [None], "mask": [False]},
        {"values": [1.0], "mask": [True]},
        {"vmin": 1.0, "vmax": 1.0},
        {"vmin": 2.0, "vmax": 1.0},
        {"scale": "log", "vmin": 0.0},
        {"mask": [1]},
        {"values": [float("inf")]},
    ],
)
def test_scalar_payload_rejects_invalid_cross_field_invariants(
    update: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "nx": 1,
        "ny": 1,
        "values": [1.0],
        "mask": [False],
        "scale": "linear",
        "label": "|F|",
        "unit": "u",
        "vmin": 0.5,
        "vmax": 1.5,
    }
    payload.update(update)

    with pytest.raises(ValueError):
        web_schemas.ScalarPayload.model_validate(payload)


@pytest.mark.parametrize(
    "count_field",
    ["termination_counts", "start_termination_counts"],
)
@pytest.mark.parametrize("invalid_count", [-1, True, 1.0, "1"])
def test_metadata_rejects_invalid_termination_counts(
    count_field: str,
    invalid_count: object,
) -> None:
    payload: dict[str, object] = {
        "title": "t",
        "projection_note": "p",
        "field_model": "f",
        "seed_mode": "coverage",
        "seed_description": "d",
        "termination_counts": {"future_reason": 0},
        "start_termination_counts": {"future_reason": 0},
        "suppressed_count": 0,
        "rendered_line_count": 1,
    }
    payload[count_field] = {"future_reason": invalid_count}

    with pytest.raises(ValueError):
        web_schemas.MetadataPayload.model_validate(payload)

    accepted = web_schemas.MetadataPayload(
        title="t",
        projection_note="p",
        field_model="f",
        seed_mode="coverage",
        seed_description="d",
        termination_counts={"future_reason": 0},
        start_termination_counts={"future_reason": 0},
        suppressed_count=0,
        rendered_line_count=1,
    )
    assert accepted.termination_counts == {"future_reason": 0}


@pytest.mark.parametrize("field", ["suppressed_count", "rendered_line_count"])
@pytest.mark.parametrize("invalid_count", [-1, True, 1.0, "1"])
def test_metadata_rejects_invalid_scalar_counts(field: str, invalid_count: object) -> None:
    payload: dict[str, object] = {
        "title": "t",
        "projection_note": "p",
        "field_model": "f",
        "seed_mode": "coverage",
        "seed_description": "d",
        "termination_counts": {},
        "start_termination_counts": {},
        "suppressed_count": 0,
        "rendered_line_count": 0,
    }
    payload[field] = invalid_count

    with pytest.raises(ValueError):
        web_schemas.MetadataPayload.model_validate(payload)


def test_seed_mode_is_closed_and_metadata_requires_a_description() -> None:
    assert {mode.value for mode in SeedMode} == {"coverage", "equal_flux", "feature"}
    payload = {
        "title": "t",
        "projection_note": "p",
        "field_model": "f",
        "seed_mode": "coverage",
        "seed_description": "coverage around active sources",
        "termination_counts": {},
        "start_termination_counts": {},
        "suppressed_count": 0,
        "rendered_line_count": 0,
    }

    assert web_schemas.MetadataPayload.model_validate(payload).seed_mode is SeedMode.COVERAGE
    with pytest.raises(ValueError):
        web_schemas.MetadataPayload.model_validate({**payload, "seed_mode": "future_mode"})
    with pytest.raises(ValueError):
        web_schemas.MetadataPayload.model_validate({**payload, "seed_description": ""})


def test_line_payload_accepts_an_optional_nonempty_start_termination() -> None:
    payload = {
        "points": [(0.0, 0.0), (1.0, 0.0)],
        "direction": 1,
        "termination": "domain_exit",
    }

    assert web_schemas.LinePayload.model_validate(payload).start_termination is None
    assert (
        web_schemas.LinePayload.model_validate(
            {**payload, "start_termination": "future_reason"}
        ).start_termination
        == "future_reason"
    )
    with pytest.raises(ValueError):
        web_schemas.LinePayload.model_validate({**payload, "start_termination": ""})


def test_openapi_types_seed_modes_and_line_accounting_metadata(
    client: TestClient,
) -> None:
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    metadata = schemas["MetadataPayload"]
    line = schemas["LinePayload"]

    assert schemas["SeedMode"]["enum"] == ["coverage", "equal_flux", "feature"]
    assert {
        "seed_mode",
        "seed_description",
        "termination_counts",
        "start_termination_counts",
        "suppressed_count",
        "rendered_line_count",
    } <= set(metadata["required"])
    for field in ("termination_counts", "start_termination_counts"):
        assert metadata["properties"][field]["additionalProperties"] == {
            "minimum": 0,
            "type": "integer",
        }
    for field in ("suppressed_count", "rendered_line_count"):
        assert metadata["properties"][field]["type"] == "integer"
        assert metadata["properties"][field]["minimum"] == 0
    assert "start_termination" not in line["required"]


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
    assert sum(scene.metadata.termination_counts.values()) == 6
    assert scene.metadata.rendered_line_count == len(scene.lines)
    assert len(scene.lines) + scene.metadata.suppressed_count == 6
    assert scene.metadata.seed_mode is SeedMode.COVERAGE
    assert scene.metadata.start_termination_counts == {}


def test_unequal_electric_sources_share_budget_and_preserve_boundary_inflow() -> None:
    request = SceneRequest(
        preset="electric_dipole",
        density=18,
        resolution=32,
        sources=[
            SourceInput(x=-0.85, y=0.0, kind="positive", strength=1.0),
            SourceInput(x=0.85, y=0.0, kind="negative", strength=-5.0),
        ],
    )

    model = _build_model(request)
    scene = build_scene(request)

    assert Counter(job.direction for job in model.trace_jobs) == {
        TraceDirection.FORWARD: 4,
        TraceDirection.BACKWARD: 14,
    }
    assert scene.metadata.termination_counts == {
        "exclusion_hit": 9,
        "domain_exit": 9,
    }
    assert sum(scene.metadata.termination_counts.values()) == request.density
    assert Counter((line.direction, line.termination) for line in scene.lines) == {
        (1, "exclusion_hit"): 4,
        (-1, "domain_exit"): 9,
    }
    assert scene.metadata.suppressed_count == 5
    assert scene.metadata.rendered_line_count == 13


_TraceScript = dict[
    tuple[float, float],
    tuple[list[tuple[float, float]], TerminationReason],
]


def _install_scripted_tracer(
    monkeypatch: pytest.MonkeyPatch,
    script: _TraceScript,
) -> None:
    class ScriptedTracer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def trace(self, seed: object, direction: TraceDirection) -> TraceResult:
            seed_array = np.asarray(seed, dtype=float)
            key = (float(seed_array[0]), float(seed_array[1]))
            raw_points, termination = script[key]
            points = np.asarray(raw_points, dtype=float)
            if points.shape[0] <= 1:
                arc_length = np.zeros(points.shape[0], dtype=float)
            else:
                arc_length = np.concatenate(
                    ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
                )
            branch = TraceBranch(
                direction=direction,
                points=points,
                arc_length=arc_length,
                field_magnitude=np.ones(points.shape[0]),
                termination=termination,
                message="scripted test trace",
                nfev=1,
            )
            return TraceResult(
                seed=seed_array,
                points=points,
                arc_length=arc_length,
                field_magnitude=np.ones(points.shape[0]),
                forward=branch if direction is TraceDirection.FORWARD else None,
                backward=branch if direction is TraceDirection.BACKWARD else None,
            )

    monkeypatch.setattr(web_scene, "FieldLineTracer", ScriptedTracer)


def test_pair_suppression_is_scoped_to_observed_renderable_source_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _build_model(
        SceneRequest(
            preset="electric_dipole",
            density=6,
            resolution=32,
            sources=[
                SourceInput(x=-2.0, y=0.0, kind="positive", strength=1.0),
                SourceInput(x=0.0, y=0.0, kind="negative", strength=-1.0),
                SourceInput(x=2.0, y=0.0, kind="negative", strength=-1.0),
            ],
        )
    )
    jobs = (
        TraceJob((-1.838, 0.0), TraceDirection.FORWARD, 0),
        TraceJob((0.162, 0.0), TraceDirection.BACKWARD, 1),
        TraceJob((2.162, 0.0), TraceDirection.BACKWARD, 2),
        TraceJob((2.0, 0.162), TraceDirection.BACKWARD, 2),
    )
    _install_scripted_tracer(
        monkeypatch,
        {
            jobs[0].seed: ([jobs[0].seed, (0.16, 0.0)], TerminationReason.EXCLUSION_HIT),
            jobs[1].seed: ([jobs[1].seed, (-1.84, 0.0)], TerminationReason.EXCLUSION_HIT),
            jobs[2].seed: ([jobs[2].seed, (-1.84, 0.0)], TerminationReason.EXCLUSION_HIT),
            jobs[3].seed: ([jobs[3].seed, (3.0, 1.0)], TerminationReason.DOMAIN_EXIT),
        },
    )

    traces = web_scene._trace_lines(replace(base, trace_jobs=jobs))

    assert traces.termination_counts == Counter({"exclusion_hit": 3, "domain_exit": 1})
    assert traces.start_termination_counts == Counter()
    assert traces.suppressed_count == 1
    assert Counter((line.direction, line.termination) for line in traces.lines) == {
        (1, "exclusion_hit"): 1,
        (-1, "exclusion_hit"): 1,
        (-1, "domain_exit"): 1,
    }


def test_degenerate_positive_trace_does_not_suppress_a_negative_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _build_model(SceneRequest(preset="electric_dipole", density=6, resolution=32))
    jobs = (
        TraceJob((-0.688, 0.0), TraceDirection.FORWARD, 0),
        TraceJob((0.688, 0.0), TraceDirection.BACKWARD, 1),
    )
    _install_scripted_tracer(
        monkeypatch,
        {
            jobs[0].seed: ([jobs[0].seed], TerminationReason.EXCLUSION_HIT),
            jobs[1].seed: ([jobs[1].seed, (-0.69, 0.0)], TerminationReason.EXCLUSION_HIT),
        },
    )

    traces = web_scene._trace_lines(replace(base, trace_jobs=jobs))

    assert traces.termination_counts == Counter({"exclusion_hit": 2})
    assert traces.suppressed_count == 0
    assert len(traces.lines) == 1
    assert traces.lines[0].direction == -1


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
    assert scene.metadata.seed_mode is SeedMode.EQUAL_FLUX
    assert "相等磁通间隔" in scene.metadata.seed_description
    assert "轴线特征线" in scene.metadata.seed_description
    assert "线数不表示磁感应强度" in scene.metadata.seed_description
    assert scene.metadata.start_termination_counts == {}


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


def test_current_loop_nonaxis_seed_pairs_are_equally_spaced_in_flux() -> None:
    request = SceneRequest(preset="current_loop", density=8, resolution=32)
    model = _build_model(request)
    loop = CircularLoopField(1.0, 1.0, normal=(0.0, 1.0, 0.0))
    positive_seeds = model.seeds[model.seeds[:, 0] > 0.0]
    embedded = np.column_stack((positive_seeds, np.zeros(positive_seeds.shape[0])))
    flux = np.sort(loop.flux_function(embedded))

    assert positive_seeds.shape == (request.density // 2, 2)
    np.testing.assert_allclose(np.diff(flux), np.full(flux.size - 1, np.diff(flux)[0]))
    assert all(job.direction is web_scene.TraceDirection.FORWARD for job in model.trace_jobs)


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

    for schema in (
        input_schema,
        payload_schema,
        openapi_schemas["SourceInput"],
        openapi_schemas["SourcePayload"],
    ):
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
            "sources": [{"x": 0.0, "y": 0.0, "kind": "dipole", "angle_deg": angle_deg}],
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
def test_dipole_angle_accepts_both_legal_boundaries(client: TestClient, angle_deg: float) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": "magnetic_dipole",
            "density": 6,
            "resolution": 32,
            "sources": [{"x": 0.0, "y": 0.0, "kind": "dipole", "angle_deg": angle_deg}],
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
        SceneRequest(preset="magnetic_dipole", density=len(sources), resolution=32, sources=sources)
    )

    base_axes = np.column_stack((np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))))
    actual_axes = np.sign(strengths)[:, np.newaxis] * base_axes
    expected = centers + (SOURCE_RADIUS + 2.0e-3) * actual_axes
    np.testing.assert_allclose(model.seeds, expected, rtol=0.0, atol=2.0e-15)
    radial = model.seeds - centers
    assert np.all(np.einsum("ij,ij->i", model.field.evaluate(model.seeds), radial) > 0.0)
    assert all(job.direction is web_scene.TraceDirection.FORWARD for job in model.trace_jobs)
    assert model.seed_mode is SeedMode.COVERAGE


def test_zero_strength_dipoles_are_markers_but_not_active_sources() -> None:
    zero_position = np.array((-1.0, 0.0))
    active_position = np.array((1.0, 0.0))
    request = SceneRequest(
        preset="magnetic_dipole",
        density=6,
        resolution=32,
        sources=[
            SourceInput(
                x=float(zero_position[0]),
                y=float(zero_position[1]),
                kind="dipole",
                strength=0.0,
                angle_deg=37.0,
            ),
            SourceInput(
                x=float(active_position[0]),
                y=float(active_position[1]),
                kind="dipole",
                strength=2.0,
                angle_deg=0.0,
            ),
        ],
    )

    model = _build_model(request)

    assert [source.strength for source in model.sources] == [0.0, 2.0]
    np.testing.assert_array_equal(model.exclusions[0].centers, active_position[None, :])
    offsets = model.seeds - active_position
    np.testing.assert_allclose(offsets[:, 0], np.zeros(request.density), atol=2.0e-15)
    assert np.all(np.linalg.norm(offsets, axis=1) >= SOURCE_RADIUS + 2.0e-3)
    assert all(job.direction is web_scene.TraceDirection.BOTH for job in model.trace_jobs)
    assert model.seed_mode is SeedMode.COVERAGE
    assert np.all(np.isfinite(model.field.evaluate(zero_position)))


def test_single_dipole_both_lines_count_both_ends_and_follow_positive_field() -> None:
    request = SceneRequest(
        preset="magnetic_dipole",
        density=6,
        resolution=32,
        sources=[
            SourceInput(
                x=0.3,
                y=-0.2,
                kind="dipole",
                strength=1.5,
                angle_deg=37.0,
            )
        ],
    )

    model = _build_model(request)
    scene = build_scene(request)

    assert all(job.direction is TraceDirection.BOTH for job in model.trace_jobs)
    assert sum(scene.metadata.termination_counts.values()) == request.density
    assert sum(scene.metadata.start_termination_counts.values()) == request.density
    assert scene.metadata.rendered_line_count == request.density
    assert scene.metadata.suppressed_count == 0
    for line in scene.lines:
        assert line.direction == 1
        assert line.start_termination is not None
        points = np.asarray(line.points)
        segments = np.diff(points, axis=0)
        vectors = model.field.evaluate(0.5 * (points[:-1] + points[1:]))
        tangent_cosine = np.einsum("ij,ij->i", segments, vectors) / (
            np.linalg.norm(segments, axis=1) * np.linalg.norm(vectors, axis=1)
        )
        assert np.min(tangent_cosine) > 0.99


def test_both_serialization_uses_forward_as_main_and_backward_as_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _build_model(SceneRequest(preset="magnetic_dipole", density=6, resolution=32))
    job = TraceJob((0.0, 1.0), TraceDirection.BOTH, 0)
    seed = np.asarray(job.seed)
    forward = TraceBranch(
        direction=TraceDirection.FORWARD,
        points=np.asarray([seed, (1.0, 1.0)]),
        arc_length=np.asarray([0.0, 1.0]),
        field_magnitude=np.ones(2),
        termination=TerminationReason.DOMAIN_EXIT,
        message="scripted forward",
        nfev=1,
    )
    backward = TraceBranch(
        direction=TraceDirection.BACKWARD,
        points=np.asarray([seed, (-1.0, 1.0)]),
        arc_length=np.asarray([0.0, 1.0]),
        field_magnitude=np.ones(2),
        termination=TerminationReason.EXCLUSION_HIT,
        message="scripted backward",
        nfev=1,
    )
    result = TraceResult(
        seed=seed,
        points=np.asarray([(-1.0, 1.0), seed, (1.0, 1.0)]),
        arc_length=np.asarray([0.0, 1.0, 2.0]),
        field_magnitude=np.ones(3),
        forward=forward,
        backward=backward,
    )

    class BothTracer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def trace(self, actual_seed: object, direction: TraceDirection) -> TraceResult:
            np.testing.assert_array_equal(actual_seed, seed)
            assert direction is TraceDirection.BOTH
            return result

    monkeypatch.setattr(web_scene, "FieldLineTracer", BothTracer)

    traces = web_scene._trace_lines(replace(base, trace_jobs=(job,)))

    assert traces.termination_counts == Counter({"domain_exit": 1})
    assert traces.start_termination_counts == Counter({"exclusion_hit": 1})
    assert len(traces.lines) == 1
    assert traces.lines[0].direction == 1
    assert traces.lines[0].termination == "domain_exit"
    assert traces.lines[0].start_termination == "exclusion_hit"
    assert traces.lines[0].points == [(-1.0, 1.0), (0.0, 1.0), (1.0, 1.0)]


@pytest.mark.parametrize(
    "strength",
    [np.nextafter(0.0, 1.0), np.nextafter(0.0, -1.0)],
)
def test_every_nonzero_dipole_strength_remains_active(strength: float) -> None:
    request = SceneRequest(
        preset="magnetic_dipole",
        density=6,
        resolution=32,
        sources=[
            SourceInput(
                x=0.25,
                y=-0.5,
                kind="dipole",
                strength=float(strength),
                angle_deg=17.0,
            )
        ],
    )

    model = _build_model(request)

    np.testing.assert_array_equal(model.exclusions[0].centers, [[0.25, -0.5]])
    assert model.seeds.shape == (request.density, 2)
    assert np.all(np.isnan(model.field.evaluate((0.25, -0.5))))


def test_halbach_defaults_form_two_quarter_turn_cycles_and_use_two_seed_rails() -> None:
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

    assert set(model.seeds[:, 1]) == {-0.45, 0.45}
    assert np.min(model.seeds[:, 0]) >= -2.1
    assert np.max(model.seeds[:, 0]) <= 2.1
    assert all(job.direction is web_scene.TraceDirection.BOTH for job in model.trace_jobs)
    assert model.seed_mode is SeedMode.COVERAGE


@pytest.mark.parametrize("density", [6, 7])
def test_halbach_default_density_is_not_tied_to_its_eight_source_count(
    client: TestClient, density: int
) -> None:
    response = client.post(
        "/api/scene",
        json={"preset": "halbach_array", "density": density, "resolution": 32},
    )

    assert response.status_code == 200
    payload = response.json()
    assert sum(payload["metadata"]["termination_counts"].values()) == density
    assert sum(payload["metadata"]["start_termination_counts"].values()) == density
    assert payload["metadata"]["seed_mode"] == "coverage"


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
    assert accepted_metadata["seed_mode"] == "coverage"
    assert accepted_metadata["start_termination_counts"] == {}
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

    np.testing.assert_allclose(
        model.field.evaluate(points), core_vectors[:, :2], rtol=0.0, atol=0.0
    )
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
    assert sum(scene.metadata.start_termination_counts.values()) == request.density
    assert scene.metadata.suppressed_count == 0
    assert scene.metadata.rendered_line_count == request.density
    residuals: list[float] = []
    tangent_cosines: list[float] = []
    for line in scene.lines:
        assert line.direction == 1
        assert line.start_termination is not None
        points = np.asarray(line.points)
        segments = np.diff(points, axis=0)
        vectors = model.field.evaluate(0.5 * (points[:-1] + points[1:]))
        residuals.extend(
            (
                np.abs(segments[:, 0] * vectors[:, 1] - segments[:, 1] * vectors[:, 0])
                / (np.linalg.norm(segments, axis=1) * np.linalg.norm(vectors, axis=1))
            ).tolist()
        )
        tangent_cosines.extend(
            (
                np.einsum("ij,ij->i", segments, vectors)
                / (np.linalg.norm(segments, axis=1) * np.linalg.norm(vectors, axis=1))
            ).tolist()
        )

    assert residuals
    assert max(residuals) < 0.02
    assert min(tangent_cosines) > 0.99


def test_halbach_default_output_reaches_above_the_array() -> None:
    scene = build_scene(SceneRequest(preset="halbach_array", density=18, resolution=32))

    upper_reaching_lines = [
        line for line in scene.lines if np.max(np.asarray(line.points)[:, 1]) > 1.0
    ]

    assert len(scene.lines) == 18
    assert len(upper_reaching_lines) >= 2
    assert all(line.direction == 1 and line.start_termination for line in scene.lines)


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
        SourceInput(x=x, y=y, kind="positive", strength=1.0) for x, y in positions[:positive_count]
    ]
    return [*positives, SourceInput(x=0.0, y=0.5, kind="negative", strength=-1.0)]


def test_seed_budget_boundary_assigns_one_seed_to_each_seeding_source() -> None:
    sources = _electric_sources(positive_count=5)
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
    for source in request.sources:
        nearby = sum(math.dist(seed, (source.x, source.y)) < 0.17 for seed in model.seeds.tolist())
        assert nearby == 1
    assert sum(scene.metadata.termination_counts.values()) == request.density
    assert scene.metadata.rendered_line_count + scene.metadata.suppressed_count == request.density


def test_multisource_scene_keeps_density_as_the_total_seed_budget() -> None:
    request = SceneRequest(
        preset="electric_dipole",
        density=10,
        resolution=32,
        sources=_electric_sources(positive_count=3),
    )

    scene = build_scene(request)

    assert sum(scene.metadata.termination_counts.values()) == request.density
    assert scene.metadata.rendered_line_count == len(scene.lines)
    assert scene.metadata.rendered_line_count + scene.metadata.suppressed_count == request.density


def test_remaining_seed_budget_uses_largest_remainders() -> None:
    counts = allocate_seed_counts(np.array((3.0, 2.0, 1.0)), total=10)

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
    by_id = {preset["id"]: preset for preset in payload}
    expected_capability = {
        "exclusive_minimum": web_scene.MIN_SOURCE_SEPARATION,
        "unit": "m",
    }
    for preset_id in ("electric_dipole", "magnetic_dipole", "halbach_array"):
        assert by_id[preset_id]["source_separation"] == expected_capability
    for preset_id in ("current_loop", "uniform"):
        assert "source_separation" not in by_id[preset_id]


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
    assert sum(payload["metadata"]["termination_counts"].values()) == 6
    assert payload["metadata"]["rendered_line_count"] == len(payload["lines"])
    assert payload["metadata"]["rendered_line_count"] + payload["metadata"]["suppressed_count"] == 6


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
            "positive source strength must be greater than 0; zero and negative values are invalid",
        ),
        (
            {"x": 1.0, "y": 0.0, "kind": "negative", "strength": 1.0},
            {"x": -1.0, "y": 0.0, "kind": "positive", "strength": 1.0},
            "negative source strength must be less than 0; zero and positive values are invalid",
        ),
        (
            {"x": -1.0, "y": 0.0, "kind": "positive", "strength": 0.0},
            {"x": 1.0, "y": 0.0, "kind": "negative", "strength": -1.0},
            "positive source strength must be greater than 0; zero and negative values are invalid",
        ),
        (
            {"x": 1.0, "y": 0.0, "kind": "negative", "strength": 0.0},
            {"x": -1.0, "y": 0.0, "kind": "positive", "strength": 1.0},
            "negative source strength must be less than 0; zero and positive values are invalid",
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
    assert response.json() == {"detail": "electric_dipole 有 8 个电荷参与播种，density 至少为 8"}


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
                {"x": -2.4 + 0.7 * index, "y": 0.0, "kind": "dipole"} for index in range(7)
            ],
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "magnetic_dipole 有 7 个磁偶极子参与播种，density 至少为 7"
    }


def test_zero_strength_dipoles_do_not_consume_seed_budget(client: TestClient) -> None:
    sources = [
        {
            "x": -2.1 + 0.7 * index,
            "y": 0.0,
            "kind": "dipole",
            "strength": 0.0 if index == 0 else 1.0,
        }
        for index in range(7)
    ]

    response = client.post(
        "/api/scene",
        json={
            "preset": "magnetic_dipole",
            "density": 6,
            "resolution": 32,
            "sources": sources,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["sources"]) == 7
    assert payload["sources"][0]["strength"] == 0.0
    assert sum(payload["metadata"]["termination_counts"].values()) == 6


@pytest.mark.parametrize("preset", ["magnetic_dipole", "halbach_array"])
def test_all_zero_magnetic_scenes_return_422(
    client: TestClient,
    preset: str,
) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": preset,
            "density": 6,
            "resolution": 32,
            "sources": [
                {"x": -0.5, "y": 0.0, "kind": "dipole", "strength": 0.0},
                {"x": 0.5, "y": 0.0, "kind": "dipole", "strength": -0.0},
            ],
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "至少需要一个非零磁偶极"}


def _separation_sources(preset: str, distance: float) -> list[dict[str, object]]:
    if preset == "electric_dipole":
        return [
            {"x": 0.0, "y": 0.0, "kind": "positive", "strength": 1.0},
            {"x": distance, "y": 0.0, "kind": "negative", "strength": -1.0},
        ]
    return [
        {
            "x": 0.0,
            "y": 0.0,
            "kind": "dipole",
            "strength": 1.0,
            "angle_deg": 0.0,
        },
        {
            "x": distance,
            "y": 0.0,
            "kind": "dipole",
            "strength": 1.0,
            "angle_deg": 0.0,
        },
    ]


@pytest.mark.parametrize(
    ("preset", "distance"),
    [
        (preset, distance)
        for preset in ("electric_dipole", "magnetic_dipole", "halbach_array")
        for distance in (0.321999, 0.322)
    ],
)
def test_active_sources_at_or_below_minimum_separation_return_422(
    client: TestClient,
    preset: str,
    distance: float,
) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": preset,
            "density": 6,
            "resolution": 32,
            "sources": _separation_sources(preset, distance),
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "sources[0] 与 sources[1] 的中心距离必须大于 0.322 m"}


@pytest.mark.parametrize(
    "preset",
    ["electric_dipole", "magnetic_dipole", "halbach_array"],
)
def test_active_sources_strictly_above_minimum_separation_are_accepted(
    client: TestClient,
    preset: str,
) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": preset,
            "density": 6,
            "resolution": 32,
            "sources": _separation_sources(preset, 0.322001),
        },
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    "preset",
    ["electric_dipole", "magnetic_dipole", "halbach_array"],
)
def test_source_separation_has_no_hidden_tolerance(
    client: TestClient,
    preset: str,
) -> None:
    below = float(np.nextafter(web_scene.MIN_SOURCE_SEPARATION, 0.0))
    above = float(np.nextafter(web_scene.MIN_SOURCE_SEPARATION, np.inf))

    rejected = client.post(
        "/api/scene",
        json={
            "preset": preset,
            "density": 6,
            "resolution": 32,
            "sources": _separation_sources(preset, below),
        },
    )
    accepted = client.post(
        "/api/scene",
        json={
            "preset": preset,
            "density": 6,
            "resolution": 32,
            "sources": _separation_sources(preset, above),
        },
    )

    assert rejected.status_code == 422
    assert accepted.status_code == 200


def test_inactive_dipoles_do_not_own_exclusions_or_separation_constraints(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": "magnetic_dipole",
            "density": 6,
            "resolution": 32,
            "sources": [
                {"x": 0.0, "y": 0.0, "kind": "dipole", "strength": 0.0},
                {"x": 0.0, "y": 0.0, "kind": "dipole", "strength": 1.0},
            ],
        },
    )

    assert response.status_code == 200
    assert [source["strength"] for source in response.json()["sources"]] == [0.0, 1.0]


def test_source_separation_error_uses_original_request_indices(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/scene",
        json={
            "preset": "magnetic_dipole",
            "density": 6,
            "resolution": 32,
            "sources": [
                {"x": -2.0, "y": 0.0, "kind": "dipole", "strength": 0.0},
                {"x": 0.0, "y": 0.0, "kind": "dipole", "strength": 1.0},
                {"x": 0.322, "y": 0.0, "kind": "dipole", "strength": 1.0},
            ],
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "sources[1] 与 sources[2] 的中心距离必须大于 0.322 m"}


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
                {"x": float(index) / 10, "y": 0.0, "kind": "positive"} for index in range(9)
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
        response = client.post(
            "/api/scene",
            json={"preset": preset_id, "density": 6, "resolution": 32},
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


@pytest.mark.parametrize(
    "update",
    [
        {"kind": "dipole", "strength_unit": "nC", "angle_deg": 90.0},
        {"kind": "dipole", "strength_unit": "A·m²", "angle_deg": None},
        {"kind": "positive", "angle_deg": 10.0},
        {"kind": "positive", "strength_unit": "A·m²"},
        {"kind": "positive", "strength": -1.0},
        {"kind": "negative", "strength": 1.0},
        {"kind": "wire_out", "strength_unit": "nC"},
        {"kind": "wire_into", "strength_unit": "A", "strength": -1.0},
    ],
)
def test_source_payload_rejects_kind_unit_and_sign_mismatches(update: dict[str, object]) -> None:
    payload: dict[str, object] = {
        "x": 0.0,
        "y": 0.0,
        "kind": "positive",
        "strength": 1.0,
        "strength_unit": "nC",
    }
    payload.update(update)

    with pytest.raises(ValueError):
        SourcePayload.model_validate(payload)


def test_domain_payload_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="strictly less"):
        web_schemas.DomainPayload(
            x=(1.0, -1.0),
            y=(-1.0, 1.0),
            coordinate_system="cartesian",
            unit="m",
        )


def test_source_input_after_validator_guards_attribute_based_inputs() -> None:
    accepted = SourceInput.model_validate(
        SimpleNamespace(x=0.0, y=0.0, kind="positive", strength=1.0, angle_deg=None),
        from_attributes=True,
    )
    assert accepted == SourceInput(x=0.0, y=0.0, kind="positive")

    with pytest.raises(ValueError, match="only valid for dipole"):
        SourceInput.model_validate(
            SimpleNamespace(x=0.0, y=0.0, kind="positive", strength=1.0, angle_deg=5.0),
            from_attributes=True,
        )
