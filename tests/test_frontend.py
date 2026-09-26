"""Browser-executed semantic tests for the build-free ES module frontend."""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from collections.abc import Iterator
from itertools import pairwise

import pytest
import uvicorn
from playwright.sync_api import Browser, Page, Route, expect, sync_playwright

from vectorviz.web.app import create_app


def _browser_scene() -> dict[str, object]:
    return {
        "domain": {
            "x": [-2.0, 4.0],
            "y": [-3.0, 1.0],
            "coordinate_system": "cartesian",
            "unit": "m",
        },
        "scalar": {
            "nx": 3,
            "ny": 3,
            "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "mask": [False] * 9,
            "scale": "linear",
            "label": "|F|",
            "unit": "u",
            "vmin": 1.0,
            "vmax": 9.0,
        },
        "lines": [
            {
                "points": [[-2.0, 1.0], [1.0, -1.0], [4.0, -3.0]],
                "direction": 1,
                "termination": "domain_exit",
            }
        ],
        "sources": [
            {
                "x": 1.0,
                "y": -1.0,
                "kind": "positive",
                "strength": 1.0,
                "strength_unit": "nC",
            }
        ],
        "metadata": {
            "title": "Browser fixture",
            "projection_note": "Browser semantic fixture",
            "field_model": "test",
            "seed_mode": "coverage",
            "seed_description": "Fixture coverage around active sources.",
            "termination_counts": {"domain_exit": 1},
            "start_termination_counts": {},
            "suppressed_count": 0,
            "rendered_line_count": 1,
        },
    }


def _browser_presets() -> list[dict[str, object]]:
    editable = {"exclusive_minimum": 0.322, "unit": "m"}
    return [
        {
            "id": "electric_dipole",
            "label": "电偶极子",
            "description": "fixture",
            "source_separation": editable,
        },
        {
            "id": "electric_quadrupole",
            "label": "电四极子",
            "description": "fixture",
            "source_separation": editable,
        },
        {
            "id": "electric_hexagon",
            "label": "六个正电荷",
            "description": "fixture",
            "source_separation": editable,
        },
        {
            "id": "electric_hexagon_alternating",
            "label": "三对交替电荷",
            "description": "fixture",
            "source_separation": editable,
        },
        {
            "id": "magnetic_dipole",
            "label": "磁偶极子",
            "description": "fixture",
            "source_separation": editable,
        },
        {
            "id": "halbach_array",
            "label": "Halbach 阵列",
            "description": "fixture",
            "source_separation": editable,
        },
        {"id": "current_loop", "label": "圆形电流线圈", "description": "fixture"},
        {"id": "charged_ring", "label": "带电圆环", "description": "fixture"},
        {"id": "dielectric_sphere", "label": "介质球", "description": "fixture"},
        {"id": "conducting_sphere", "label": "导体球", "description": "fixture"},
        {"id": "uniform", "label": "匀强场", "description": "fixture"},
    ]


def _browser_loop_scene() -> dict[str, object]:
    scene = json.loads(json.dumps(_browser_scene()))
    scalar = scene["scalar"]
    metadata = scene["metadata"]
    assert isinstance(scalar, dict)
    assert isinstance(metadata, dict)
    scalar.update({"label": "|B|", "unit": "T"})
    scene["sources"] = [
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
    metadata.update(
        {
            "title": "Current loop fixture",
            "projection_note": "Invariant meridional plane",
            "field_model": "test loop",
            "seed_mode": "equal_flux",
            "seed_description": "Fixture equal-ψ loop seeds.",
        }
    )
    return scene


def _browser_ring_scene() -> dict[str, object]:
    scene = json.loads(json.dumps(_browser_scene()))
    metadata = scene["metadata"]
    assert isinstance(metadata, dict)
    scene["sources"] = [
        {
            "x": -1.0,
            "y": 0.0,
            "kind": "ring_charge",
            "strength": 1.0,
            "strength_unit": "nC",
        },
        {
            "x": 1.0,
            "y": 0.0,
            "kind": "ring_charge",
            "strength": 1.0,
            "strength_unit": "nC",
        },
    ]
    metadata.update(
        {
            "title": "Charged ring fixture",
            "projection_note": "Invariant meridional plane",
            "field_model": "test ring",
            "seed_mode": "equal_flux",
            "seed_description": "Fixture equal-flux ring seeds.",
        }
    )
    return scene


def _browser_sphere_scene() -> dict[str, object]:
    scene = json.loads(json.dumps(_browser_scene()))
    metadata = scene["metadata"]
    assert isinstance(metadata, dict)
    scene["sources"] = []
    scene["regions"] = [
        {
            "kind": "dielectric_sphere",
            "x": 0.5,
            "y": -0.5,
            "radius": 1.0,
            "relative_permittivity": 4.0,
            "unit": "m",
        }
    ]
    metadata.update(
        {
            "title": "Dielectric sphere fixture",
            "projection_note": "Invariant plane through the axis",
            "field_model": "test sphere",
            "seed_mode": "equal_flux",
            "seed_description": "Fixture equal-flux sphere seeds.",
        }
    )
    return scene


def _browser_halbach_scene(count: int = 8) -> dict[str, object]:
    scene = json.loads(json.dumps(_browser_scene()))
    scalar = scene["scalar"]
    metadata = scene["metadata"]
    assert isinstance(scalar, dict)
    assert isinstance(metadata, dict)
    scalar.update({"label": "|B|", "unit": "T"})
    scene["sources"] = [
        {
            "x": -2.1 + 0.6 * index,
            "y": 0.0,
            "kind": "dipole",
            "strength": 1.0,
            "strength_unit": "A·m²",
            "angle_deg": float((index % 4) * 90),
        }
        for index in range(count)
    ]
    metadata.update(
        {
            "title": "Halbach fixture",
            "projection_note": "Eight editable in-plane dipoles",
            "field_model": "test Halbach",
            "seed_mode": "coverage",
            "seed_description": "Fixture two-rail coverage seeds.",
        }
    )
    return scene


# Ports above 1024 that Chromium refuses to load (net::ERR_UNSAFE_PORT).
CHROMIUM_RESTRICTED_PORTS = frozenset(
    {1719, 1720, 1723, 2049, 3659, 4045, 4190, 5060, 5061, 6000, 6566, 6665, 6666, 6667,
     6668, 6669, 6679, 6697, 10080}
)


@pytest.fixture(scope="session")
def frontend_url() -> Iterator[str]:
    while True:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        if listener.getsockname()[1] not in CHROMIUM_RESTRICTED_PORTS:
            break
        listener.close()
    listener.listen(128)
    host, port = listener.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(),
            log_level="error",
            access_log=False,
            lifespan="off",
        )
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5.0
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2.0)
        pytest.fail("the browser-test uvicorn server did not start")

    yield f"http://{host}:{port}"

    server.should_exit = True
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "the browser-test uvicorn server did not stop"


@pytest.fixture(scope="session")
def chromium_browser() -> Iterator[Browser]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def browser_page(chromium_browser: Browser) -> Iterator[tuple[Page, list[str]]]:
    context = chromium_browser.new_context(
        viewport={"width": 1280, "height": 800},
        device_scale_factor=1,
        locale="zh-CN",
    )
    page = context.new_page()
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.route(
        "**/api/presets",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(_browser_presets()),
        ),
    )
    yield page, page_errors
    context.close()


def _route_scene(page: Page, scene: dict[str, object]) -> None:
    page.route(
        "**/api/scene",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(scene),
        ),
    )


def _instrument_canvas(page: Page) -> None:
    page.add_init_script(
        """(() => {
            const calls = {
              putImages: [], drawImages: [], paints: [], texts: [], rotations: [],
            };
            Object.defineProperty(window, '__vectorVizCanvasCalls', {value: calls});
            const prototype = CanvasRenderingContext2D.prototype;
            const paths = new WeakMap();

            const originalPutImageData = prototype.putImageData;
            prototype.putImageData = function (imageData, ...args) {
              calls.putImages.push({
                width: imageData.width,
                height: imageData.height,
                data: Array.from(imageData.data),
              });
              return originalPutImageData.call(this, imageData, ...args);
            };

            const originalDrawImage = prototype.drawImage;
            prototype.drawImage = function (...args) {
              // Record the destination rectangle of 5- and 9-argument calls.
              if (args.length === 5) {
                calls.drawImages.push(args.slice(1).map(Number));
              } else if (args.length === 9) {
                calls.drawImages.push(args.slice(5).map(Number));
              }
              return originalDrawImage.apply(this, args);
            };

            for (const [name, operation] of [
              ['moveTo', 'M'],
              ['lineTo', 'L'],
              ['arc', 'A'],
            ]) {
              const original = prototype[name];
              prototype[name] = function (...args) {
                const path = paths.get(this) || [];
                path.push([operation, ...args.map(Number)]);
                paths.set(this, path);
                return original.apply(this, args);
              };
            }

            const originalBeginPath = prototype.beginPath;
            prototype.beginPath = function (...args) {
              paths.set(this, []);
              return originalBeginPath.apply(this, args);
            };
            const originalClosePath = prototype.closePath;
            prototype.closePath = function (...args) {
              const path = paths.get(this) || [];
              path.push(['Z']);
              paths.set(this, path);
              return originalClosePath.apply(this, args);
            };
            for (const name of ['stroke', 'fill']) {
              const original = prototype[name];
              prototype[name] = function (...args) {
                calls.paints.push({kind: name, path: [...(paths.get(this) || [])]});
                return original.apply(this, args);
              };
            }

            const originalFillText = prototype.fillText;
            prototype.fillText = function (value, x, y, ...args) {
              calls.texts.push({text: String(value), x: Number(x), y: Number(y)});
              return originalFillText.call(this, value, x, y, ...args);
            };

            const originalRotate = prototype.rotate;
            prototype.rotate = function (angle) {
              calls.rotations.push(Number(angle));
              return originalRotate.call(this, angle);
            };
        })();"""
    )


def _open_ready_scene(page: Page, frontend_url: str, scene: dict[str, object]) -> None:
    _route_scene(page, scene)
    page.goto(frontend_url)
    expect(page.locator("#connection-label")).to_have_text("已同步")
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "ready")
    expect(page.locator("#loading-overlay")).to_be_hidden()


@pytest.mark.browser
def test_failed_request_does_not_present_a_stale_scene(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    request_count = 0
    request_bodies: list[dict[str, object]] = []

    def route_scene(route: Route) -> None:
        nonlocal request_count
        request_count += 1
        request_bodies.append(route.request.post_data_json)
        if request_count == 1:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(scene),
            )
        else:
            route.fulfill(
                status=503,
                content_type="application/json",
                body=json.dumps({"detail": "fixture unavailable"}),
            )

    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)
    expect(page.locator("#scene-title")).to_have_text("Browser fixture")
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "ready")

    # Change the pending request without firing the range input's debounce.
    # The failed response must not leave the old density-18 scene on screen as
    # if it represented this new density-20 request.
    page.locator("#density").evaluate("input => { input.value = '20'; }")
    page.locator("#run-button").click()

    expect(page.locator("#error-banner")).to_be_visible()
    expect(page.locator("#error-message")).to_have_text("fixture unavailable")
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "error")
    expect(page.locator("#scene-title")).to_have_text("场景不可用")
    expect(page.locator("#line-count")).to_have_text("—")
    expect(page.locator("#grid-size")).to_have_text("—")
    expect(page.locator("#field-unit")).to_have_text("—")
    expect(page.locator("#colorbar")).to_be_hidden()
    expect(page.locator("#probe")).to_be_hidden()
    assert "1 条场线" not in page.locator("#field-canvas").get_attribute("aria-label")
    assert request_count == 2
    assert request_bodies[0]["density"] == 18
    assert request_bodies[1]["density"] == 20
    assert "sources" not in request_bodies[1]
    assert page_errors == []


@pytest.mark.browser
@pytest.mark.parametrize(
    "invalid_case",
    [
        "missing domain unit",
        "wrong domain unit",
        "missing source unit",
        "mismatched source unit",
        "mismatched source sign",
        "string source strength",
    ],
)
def test_invalid_response_units_or_source_types_invalidate_the_scene(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
    invalid_case: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    domain = scene["domain"]
    sources = scene["sources"]
    assert isinstance(domain, dict)
    assert isinstance(sources, list)
    source = sources[0]
    assert isinstance(source, dict)
    if invalid_case == "missing domain unit":
        domain.pop("unit")
    elif invalid_case == "wrong domain unit":
        domain["unit"] = "cm"
    elif invalid_case == "missing source unit":
        source.pop("strength_unit")
    elif invalid_case == "mismatched source unit":
        source["strength_unit"] = "A·m²"
    elif invalid_case == "mismatched source sign":
        source["strength"] = -1.0
    else:
        source["strength"] = "1"

    _route_scene(page, scene)
    page.goto(frontend_url)

    expect(page.locator("#error-banner")).to_be_visible()
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "error")
    expect(page.locator("#scene-title")).to_have_text("场景不可用")
    assert page_errors == []


@pytest.mark.browser
def test_source_units_remain_visible_without_narrow_viewport_overflow(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    page.set_viewport_size({"width": 320, "height": 900})
    scene = _browser_scene()
    sources = scene["sources"]
    assert isinstance(sources, list)
    source = sources[0]
    assert isinstance(source, dict)
    source.update({"kind": "dipole", "strength_unit": "A·m²"})

    _open_ready_scene(page, frontend_url, scene)

    expect(page.locator(".source-strength")).to_have_text("1 A·m²")
    layout = page.evaluate(
        """() => {
          const row = document.querySelector('.source-editor');
          const rowRect = row.getBoundingClientRect();
          const children = [
            document.querySelector('.source-strength'),
            ...document.querySelectorAll('.coordinate-field input'),
          ];
          return {
            documentOverflow:
              document.documentElement.scrollWidth - document.documentElement.clientWidth,
            rowOverflow: row.scrollWidth - row.clientWidth,
            childrenInside: children.every((child) => {
              const rect = child.getBoundingClientRect();
              return rect.left >= rowRect.left - 1 && rect.right <= rowRect.right + 1;
            }),
          };
        }"""
    )
    assert layout["documentOverflow"] <= 1
    assert layout["rowOverflow"] <= 1
    assert layout["childrenInside"] is True
    assert page_errors == []


@pytest.mark.browser
def test_scalar_lines_arrows_sources_and_probe_share_one_transform(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    _instrument_canvas(page)
    _open_ready_scene(page, frontend_url, scene)

    geometry = page.evaluate(
        """async (scene) => {
            const { calculatePlotRect, createCoordinateTransform, sampleNearest } =
              await import('/coordinates.js');
            const plot = calculatePlotRect(900, 600, scene.domain);
            const transform = createCoordinateTransform(scene.domain, plot);
            const source = transform.worldToCanvas(1, -1);
            const line = transform.projectPoints(scene.lines[0].points);
            const roundTrip = transform.canvasToWorld(...source);
            return {
              plot,
              source,
              lineMiddle: line[1],
              roundTrip,
              rasterTopLeft: transform.worldToCanvas(-2, 1),
              rasterBottomRight: transform.worldToCanvas(4, -3),
              sampled: sampleNearest(scene.scalar, scene.domain, 1, -1),
            };
        }""",
        scene,
    )

    assert geometry["source"] == pytest.approx(geometry["lineMiddle"])
    assert geometry["roundTrip"] == pytest.approx([1.0, -1.0])
    assert geometry["rasterTopLeft"] == pytest.approx(
        [geometry["plot"]["left"], geometry["plot"]["top"]]
    )
    assert geometry["rasterBottomRight"] == pytest.approx(
        [geometry["plot"]["right"], geometry["plot"]["bottom"]]
    )
    assert geometry["sampled"] == 5.0

    production_wiring = page.evaluate(
        """async () => {
            const canvas = document.querySelector('#field-canvas');
            const rect = canvas.getBoundingClientRect();
            const domain = {x: [-2, 4], y: [-3, 1]};
            const {calculatePlotRect, createCoordinateTransform} =
              await import('/coordinates.js');
            const plot = calculatePlotRect(rect.width, rect.height, domain);
            const transform = createCoordinateTransform(domain, plot);
            const topLeft = transform.worldToCanvas(-2, 1);
            const middle = transform.worldToCanvas(1, -1);
            const bottomRight = transform.worldToCanvas(4, -3);
            const calls = window.__vectorVizCanvasCalls;
            const close = (left, right, tolerance = 1e-6) =>
              Math.abs(left - right) <= tolerance;
            const pointMatches = (operation, expected, name) =>
              operation?.[0] === name && close(operation[1], expected[0]) &&
                close(operation[2], expected[1]);
            const onExpectedLine = (operation) => {
              if (operation?.[0] !== 'M') return false;
              const [x, y] = operation.slice(1);
              const dx = bottomRight[0] - topLeft[0];
              const dy = bottomRight[1] - topLeft[1];
              const cross = Math.abs((x - topLeft[0]) * dy - (y - topLeft[1]) * dx);
              const dot = (x - topLeft[0]) * dx + (y - topLeft[1]) * dy;
              return cross <= 1e-5 * Math.hypot(dx, dy) && dot > 0 &&
                dot < dx * dx + dy * dy;
            };
            return {
              heatmap: calls.drawImages.some(([x, y, width, height]) =>
                close(x, plot.left) && close(y, plot.top) &&
                close(width, plot.right - plot.left) &&
                close(height, plot.bottom - plot.top)),
              line: calls.paints.some(({kind, path}) => kind === 'stroke' &&
                pointMatches(path[0], topLeft, 'M') &&
                pointMatches(path[1], middle, 'L') &&
                pointMatches(path[2], bottomRight, 'L')),
              source: calls.paints.some(({kind, path}) => kind === 'fill' &&
                path.some((operation) => operation[0] === 'A' &&
                  close(operation[1], middle[0]) && close(operation[2], middle[1]) &&
                  close(operation[3], 10))),
              arrow: calls.paints.some(({kind, path}) => kind === 'fill' &&
                path.length === 4 && path.at(-1)[0] === 'Z' && onExpectedLine(path[0])),
            };
        }"""
    )
    assert production_wiring == {
        "heatmap": True,
        "line": True,
        "source": True,
        "arrow": True,
    }

    target = page.evaluate(
        """async () => {
            const canvas = document.querySelector('#field-canvas');
            const rect = canvas.getBoundingClientRect();
            const { calculatePlotRect, createCoordinateTransform } = await import('/coordinates.js');
            const domain = {x: [-2, 4], y: [-3, 1]};
            const transform = createCoordinateTransform(
              domain,
              calculatePlotRect(rect.width, rect.height, domain),
            );
            const [x, y] = transform.worldToCanvas(1, -1);
            return {x: rect.left + x, y: rect.top + y};
        }"""
    )
    hit_target = page.evaluate(
        "target => document.elementFromPoint(target.x, target.y)?.id || null",
        target,
    )
    assert hit_target == "field-canvas"
    expect(page.locator(".source-strength")).to_be_visible()
    expect(page.locator(".source-strength")).to_have_text("1 nC")
    expect(page.locator(".coordinate-axis")).to_have_text(["x/m", "y/m"])
    coordinate_inputs = page.locator(".coordinate-field input")
    expect(coordinate_inputs.nth(0)).to_have_attribute("aria-label", "正电荷 1 x 坐标（m）")
    expect(coordinate_inputs.nth(1)).to_have_attribute("aria-label", "正电荷 1 y 坐标（m）")
    page.mouse.move(target["x"], target["y"])
    expect(page.locator("#probe")).to_be_visible()
    expect(page.locator("#probe-position")).to_have_text("x 1 m · y -1 m")
    expect(page.locator("#probe-value")).to_have_text("|F| 5 u")

    page.mouse.down()
    expect(page.locator("#field-canvas")).to_have_attribute("data-dragging", "true")
    page.mouse.up()
    assert page_errors == []


@pytest.mark.browser
def test_layer_toggles_redraw_without_requests_and_keep_readouts_honest(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    scene["scalar"]["mask"][4] = True
    scene["scalar"]["values"][4] = None
    requests: list[dict[str, object]] = []

    def route_scene(route: Route) -> None:
        requests.append(route.request.post_data_json)
        route.fulfill(status=200, content_type="application/json", body=json.dumps(scene))

    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "ready")
    baseline_requests = len(requests)
    toggles = {
        name: page.locator(f'.layer-toggle[data-layer="{name}"]')
        for name in ("heatmap", "lines", "arrows", "sources", "grid")
    }
    for toggle in toggles.values():
        expect(toggle).to_have_attribute("aria-pressed", "true")

    def snapshot() -> str:
        return page.evaluate("() => document.querySelector('#field-canvas').toDataURL()")

    full = snapshot()

    # Heatmap off: colorbar and hatch legend go, the probe still reads the grid.
    toggles["heatmap"].click()
    expect(toggles["heatmap"]).to_have_attribute("aria-pressed", "false")
    expect(page.locator("#colorbar")).to_be_hidden()
    expect(page.locator("#legend-uncolored")).to_be_hidden()
    without_heatmap = snapshot()
    assert without_heatmap != full
    target = page.evaluate(
        """async () => {
            const canvas = document.querySelector('#field-canvas');
            const rect = canvas.getBoundingClientRect();
            const {calculatePlotRect, createCoordinateTransform} = await import('/coordinates.js');
            const domain = {x: [-2, 4], y: [-3, 1]};
            const transform = createCoordinateTransform(
              domain, calculatePlotRect(rect.width, rect.height, domain));
            const [x, y] = transform.worldToCanvas(4, 1);
            const [sx, sy] = transform.worldToCanvas(1, -1);
            return {x: rect.left + x - 2, y: rect.top + y + 2, sx: rect.left + sx, sy: rect.top + sy};
        }"""
    )
    page.mouse.move(target["x"], target["y"])
    expect(page.locator("#probe")).to_be_visible()
    # The probe snaps to the node at (4, 1) although the pointer is 2 px off it.
    expect(page.locator("#probe-position")).to_have_text("x 4 m · y 1 m")
    expect(page.locator("#probe-value")).to_have_text("|F| 3 u")
    expect(page.locator("#probe-node")).to_be_visible()
    toggles["heatmap"].click()
    expect(page.locator("#colorbar")).to_be_visible()
    expect(page.locator("#legend-uncolored")).to_be_visible()
    assert snapshot() == full

    # Sources off: the marker disappears and cannot be dragged.
    toggles["sources"].click()
    expect(page.locator("#field-canvas")).to_have_attribute("data-draggable", "false")
    expect(page.locator('.legend-chip[data-layer="sources"]')).to_be_hidden()
    page.mouse.move(target["sx"], target["sy"])
    page.mouse.down()
    page.mouse.move(target["sx"] + 30, target["sy"] + 20, steps=3)
    assert page.locator("#field-canvas").get_attribute("data-dragging") is None
    page.mouse.up()
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "ready")
    toggles["sources"].click()
    expect(page.locator("#field-canvas")).to_have_attribute("data-draggable", "true")
    expect(page.locator('.legend-chip[data-layer="sources"]')).to_be_visible()

    # Arrows off keeps the line chip; lines off takes the arrow chip with it.
    toggles["arrows"].click()
    expect(page.locator('.legend-chip[data-layer="arrows"]')).to_be_hidden()
    expect(page.locator('.legend-chip[data-layer="lines"]')).to_be_visible()
    assert snapshot() != full
    toggles["arrows"].click()
    toggles["lines"].click()
    expect(page.locator('.legend-chip[data-layer="lines"]')).to_be_hidden()
    expect(page.locator('.legend-chip[data-layer="arrows"]')).to_be_hidden()
    toggles["lines"].click()
    toggles["grid"].click()
    assert snapshot() != full
    toggles["grid"].click()
    assert snapshot() == full
    assert len(requests) == baseline_requests

    # A stale scene stays stale whatever the layers say.
    page.mouse.move(target["sx"], target["sy"])
    page.mouse.down()
    page.mouse.move(target["sx"] + 30, target["sy"] + 20, steps=3)
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "stale")
    # The coordinate inputs follow the pointer while the drag is still open.
    expect(page.locator('#source-editor-list input[data-source-field="x"]')).not_to_have_value("1")
    # Script clicks keep the pointer held down, as a second input device would.
    for _repeat in range(2):
        page.evaluate("() => document.querySelector('.layer-toggle[data-layer=\"heatmap\"]').click()")
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "stale")
    expect(page.locator("#colorbar")).to_be_hidden()
    with page.expect_request("**/api/scene"):
        page.mouse.up()
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "ready")
    expect(page.locator("#colorbar")).to_be_visible()
    assert page_errors == []


@pytest.mark.browser
def test_hatch_legend_follows_the_markers_actually_drawn(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    # A fine grid puts the source's masked node entirely under its marker, so
    # the hatch legend is off while the marker is drawn and on once the
    # source layer hides the marker and exposes the hatching.
    scene = _browser_scene()
    scalar = scene["scalar"]
    assert isinstance(scalar, dict)
    nx = ny = 61
    values: list[float | None] = [1.0 + (index % 7) for index in range(nx * ny)]
    mask = [False] * (nx * ny)
    column = round((1.0 - (-2.0)) / 6.0 * (nx - 1))
    row = round((1.0 - (-1.0)) / 4.0 * (ny - 1))
    mask[row * nx + column] = True
    values[row * nx + column] = None
    scalar.update({"nx": nx, "ny": ny, "values": values, "mask": mask})
    _open_ready_scene(page := browser_page[0], frontend_url, scene)

    expect(page.locator("#legend-uncolored")).to_be_hidden()
    expect(page.locator("#field-canvas")).to_have_attribute(
        "aria-label", "Browser fixture二维可视化，共 1 条场线、1 个可移动场源。"
    )
    page.locator('.layer-toggle[data-layer="sources"]').click()
    expect(page.locator("#legend-uncolored")).to_be_visible()
    expect(page.locator("#field-canvas")).to_have_attribute(
        "aria-label", "Browser fixture二维可视化，共 1 条场线、0 个可移动场源。"
    )
    page.locator('.layer-toggle[data-layer="sources"]').click()
    expect(page.locator("#legend-uncolored")).to_be_hidden()
    assert browser_page[1] == []


@pytest.mark.browser
def test_region_circle_is_drawn_from_the_scene_payload(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    ordinary_scene = _browser_scene()
    sphere_scene = _browser_sphere_scene()
    broken_scene = json.loads(json.dumps(ordinary_scene))
    broken_scene["regions"] = [
        {"kind": "dielectric_sphere", "x": 0.0, "y": 0.0, "radius": 0.0, "unit": "m", "relative_permittivity": 4.0}
    ]
    # The dielectric region served under the conducting preset is a mismatch.
    wrong_kind_scene = json.loads(json.dumps(sphere_scene))
    null_regions_scene = json.loads(json.dumps(ordinary_scene))
    null_regions_scene["regions"] = None
    responses = {
        "dielectric_sphere": sphere_scene,
        "conducting_sphere": wrong_kind_scene,
        "halbach_array": null_regions_scene,
        "magnetic_dipole": broken_scene,
    }

    def route_scene(route: Route) -> None:
        body = route.request.post_data_json
        response_scene = responses.get(body["preset"], ordinary_scene)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(response_scene),
        )

    _instrument_canvas(page)
    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "ready")

    page.locator("#preset").select_option("dielectric_sphere")
    expect(page.locator("#scene-title")).to_have_text("Dielectric sphere fixture")
    expect(page.locator("#source-editor-list .empty-sources")).to_contain_text("虚线圆")
    drawn = page.evaluate(
        """async () => {
            const canvas = document.querySelector('#field-canvas');
            const rect = canvas.getBoundingClientRect();
            const {calculatePlotRect, createCoordinateTransform} =
              await import('/coordinates.js');
            const domain = {x: [-2, 4], y: [-3, 1]};
            const transform = createCoordinateTransform(
              domain,
              calculatePlotRect(rect.width, rect.height, domain),
            );
            const [cx, cy] = transform.worldToCanvas(0.5, -0.5);
            const [ex] = transform.worldToCanvas(1.5, -0.5);
            const arcs = window.__vectorVizCanvasCalls.paints
              .filter(({kind, path}) => kind === 'stroke' && path.length === 1 && path[0][0] === 'A')
              .map(({path}) => path[0].slice(1, 4));
            const texts = window.__vectorVizCanvasCalls.texts.map(({text}) => text);
            return {cx, cy, radius: ex - cx, arcs, texts};
        }"""
    )
    matching = [
        arc
        for arc in drawn["arcs"]
        if abs(arc[0] - drawn["cx"]) < 0.5
        and abs(arc[1] - drawn["cy"]) < 0.5
        and abs(arc[2] - drawn["radius"]) < 0.5
    ]
    assert len(matching) >= 2
    assert "εr = 4" in drawn["texts"]

    # A region whose material does not match the preset, an explicit null
    # list and a region without a usable radius each invalidate the scene.
    for preset in ("conducting_sphere", "halbach_array", "magnetic_dipole"):
        with page.expect_response("**/api/scene"):
            page.locator("#preset").select_option("dielectric_sphere")
        expect(page.locator("#scene-title")).to_have_text("Dielectric sphere fixture")
        with page.expect_response("**/api/scene"):
            page.locator("#preset").select_option(preset)
        expect(page.locator("#error-banner")).to_be_visible()
        expect(page.locator("#error-message")).to_contain_text("区域几何")
        expect(page.locator("#scene-title")).to_have_text("场景不可用")
    assert page_errors == []


@pytest.mark.browser
@pytest.mark.parametrize(
    ("preset", "fixed_scene", "title", "labels", "strengths", "symbols", "note"),
    [
        (
            "current_loop",
            _browser_loop_scene(),
            "Current loop fixture",
            ["电流出屏", "电流入屏"],
            ["1 A", "1 A"],
            ["⊙", "⊗"],
            "电流与位置由预设固定",
        ),
        (
            "charged_ring",
            _browser_ring_scene(),
            "Charged ring fixture",
            ["圆环截面", "圆环截面"],
            ["1 nC", "1 nC"],
            ["+", "+"],
            "电荷与位置由预设固定",
        ),
    ],
)
def test_fixed_preset_markers_are_response_only_sources(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
    preset: str,
    fixed_scene: dict[str, object],
    title: str,
    labels: list[str],
    strengths: list[str],
    symbols: list[str],
    note: str,
) -> None:
    page, page_errors = browser_page
    ordinary_scene = _browser_scene()
    request_bodies: list[dict[str, object]] = []

    def route_scene(route: Route) -> None:
        body = route.request.post_data_json
        request_bodies.append(body)
        response_scene = fixed_scene if body["preset"] == preset else ordinary_scene
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(response_scene),
        )

    _instrument_canvas(page)
    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "ready")

    page.locator("#preset").select_option(preset)
    expect(page.locator("#scene-title")).to_have_text(title)
    expect(page.locator("#field-canvas")).to_have_attribute("data-draggable", "false")
    expect(page.locator("#field-canvas")).to_have_attribute(
        "aria-label", f"{title}二维可视化，共 1 条场线、0 个可移动场源。"
    )
    expect(page.locator(".source-label")).to_have_text(labels)
    expect(page.locator(".source-strength")).to_have_text(strengths)
    expect(page.locator("#source-editor-list input")).to_have_count(0)
    expect(page.locator("#reset-sources")).to_be_disabled()
    expect(page.locator(".fixed-sources-note")).to_contain_text("不可移动")
    expect(page.locator(".fixed-sources-note")).to_contain_text(note)

    rendered_symbols = page.evaluate(
        """(symbols) => window.__vectorVizCanvasCalls.texts
          .map(({text}) => text)
          .filter((text) => symbols.includes(text))""",
        symbols,
    )
    assert rendered_symbols[-2:] == symbols

    with page.expect_request("**/api/scene") as request_info:
        page.locator("#run-button").click()
    request_body = request_info.value.post_data_json
    assert request_body["preset"] == preset
    assert set(request_body) == {"preset", "density", "resolution"}
    expect(page.locator("#loading-overlay")).to_be_hidden()
    expect(page.locator("#connection-label")).to_have_text("已同步")
    expect(page.locator("#scene-title")).to_have_text(title)

    target = page.evaluate(
        """async () => {
            const canvas = document.querySelector('#field-canvas');
            const rect = canvas.getBoundingClientRect();
            const {calculatePlotRect, createCoordinateTransform} =
              await import('/coordinates.js');
            const domain = {x: [-2, 4], y: [-3, 1]};
            const transform = createCoordinateTransform(
              domain,
              calculatePlotRect(rect.width, rect.height, domain),
            );
            const [x, y] = transform.worldToCanvas(-1, 0);
            return {x: rect.left + x, y: rect.top + y};
        }"""
    )
    baseline_requests = len(request_bodies)
    page.mouse.move(target["x"], target["y"])
    page.mouse.down()
    assert page.locator("#field-canvas").get_attribute("data-dragging") is None
    page.mouse.move(target["x"] + 30, target["y"] + 20)
    page.mouse.up()
    page.locator("#field-canvas").focus()
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(550)

    assert len(request_bodies) == baseline_requests
    expect(page.locator(".source-strength")).to_have_text(strengths)
    assert page_errors == []


@pytest.mark.browser
def test_log_scale_and_mask_do_not_create_false_hotspots(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    scalar = scene["scalar"]
    assert isinstance(scalar, dict)
    scalar.update(
        {
            "values": [0.0, -2.0, 1.0, 10.0, None, 2.0, 3.0, 4.0, 5.0],
            "mask": [False, False, False, False, True, False, False, False, False],
            "scale": "log",
            "vmin": 1.0,
            "vmax": 10.0,
        }
    )
    _instrument_canvas(page)
    _open_ready_scene(page, frontend_url, scene)

    raster = page.evaluate(
        """() => window.__vectorVizCanvasCalls.putImages
          .filter((image) => image.width === 3 && image.height === 3)
          .at(-1)"""
    )
    assert raster is not None

    def pixel(index: int) -> list[int]:
        start = index * 4
        return raster["data"][start : start + 4]

    assert pixel(0)[3] == 0
    assert pixel(1)[3] == 0
    assert pixel(4)[3] == 0
    assert pixel(3) == [100, 79, 172, 255]
    expect(page.locator("#colorbar-max")).to_have_text("10")
    # A log colorbar from 1 to 10 marks the decades 10^0 and 10^1.
    expect(page.locator(".colorbar-tick")).to_have_count(2)
    expect(page.locator(".colorbar-tick b")).to_have_text(["100", "101"])

    result = page.evaluate(
        """async () => {
            const { colorForScalar, normalizeScalar, resolveScale } =
              await import('/color-scale.js');
            const scalar = {
              scale: 'log',
              values: [0, -2, 1, 10, null],
              mask: [false, false, false, false, true],
            };
            const scale = resolveScale(scalar);
            const unit = {type: 'linear', minimum: 0, maximum: 1};
            // A uniform field: the server widens vmax by 1e-12.
            const constant = resolveScale({
              scale: 'linear',
              vmin: 1.038,
              vmax: 1.038 * (1 + 1e-12),
              values: [1.038, 1.038, null],
              mask: [false, false, true],
            });
            const micro = resolveScale({
              scale: 'linear', vmin: 1, vmax: 1.0000005, values: [1, 1.0000005],
            });
            const narrow = {type: 'linear', minimum: 1, maximum: 1.000000000001};
            const constantOutside = resolveScale({
              scale: 'linear', vmin: 2, vmax: 3, values: [1, 1],
            });
            const constantAbove = resolveScale({
              scale: 'linear', vmin: 1, vmax: 2, values: [10, 10],
            });
            return {
              scale,
              zero: normalizeScalar(0, scale),
              negative: normalizeScalar(-2, scale),
              masked: colorForScalar(null, true, scale),
              invalid: colorForScalar(0, false, scale),
              validMaximum: colorForScalar(10, false, scale),
              ramp: Array.from({length: 257}, (_, i) => colorForScalar(i / 256, false, unit)),
              constant: colorForScalar(1.038, false, constant),
              middle: colorForScalar(0.5, false, unit),
              micro: [1, 1.0000005].map((value) => normalizeScalar(value, micro)),
              narrow: [0.1, 10].map((value) => normalizeScalar(value, narrow)),
              constantOutside: normalizeScalar(1, constantOutside),
              constantAbove: normalizeScalar(10, constantAbove),
            };
        }"""
    )

    assert result["scale"] == {"type": "log", "minimum": 1, "maximum": 10, "constant": False}
    assert result["zero"] is None
    assert result["negative"] is None
    assert result["masked"][3] == 0
    assert result["invalid"][3] == 0
    assert result["masked"] != result["validMaximum"]
    assert result["invalid"] != result["validMaximum"]
    assert result["validMaximum"] == [100, 79, 172, 255]
    # A stronger field is never a lighter colour, anywhere along the ramp.
    lightness = [_cie_lightness(rgba[:3]) for rgba in result["ramp"]]
    assert all(later <= earlier for earlier, later in pairwise(lightness)), lightness
    assert lightness[0] - lightness[-1] > 50
    # Only a truly constant field inside the limits gets the middle colour:
    # a real tiny variation keeps the full ramp and clipped values their ends.
    assert result["constant"] == result["middle"]
    assert result["micro"] == [0, 1]
    assert result["narrow"] == [0, 1]
    assert result["constantOutside"] == 0
    assert result["constantAbove"] == 1
    assert page_errors == []


@pytest.mark.browser
def test_probe_distinguishes_masked_null_from_zero(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    scalar = scene["scalar"]
    assert isinstance(scalar, dict)
    values = [1.0] * 15
    mask = [False] * 15
    values[6] = None
    mask[6] = True
    values[7] = 0.0
    values[8] = 50.0
    scalar.update(
        {
            "nx": 5,
            "ny": 3,
            "values": values,
            "mask": mask,
            "vmin": 1.0,
            "vmax": 10.0,
        }
    )
    _open_ready_scene(page, frontend_url, scene)

    points = page.evaluate(
        """async () => {
          const canvas = document.querySelector('#field-canvas');
          const rect = canvas.getBoundingClientRect();
          const domain = {x: [-2, 4], y: [-3, 1]};
          const {calculatePlotRect, createCoordinateTransform} = await import('/coordinates.js');
          const transform = createCoordinateTransform(
            domain, calculatePlotRect(rect.width, rect.height, domain),
          );
          return Object.fromEntries([
            ['masked', [-0.5, -1]],
            ['zero', [1, -1]],
            ['raw', [2.5, -1]],
          ].map(([name, point]) => {
            const [x, y] = transform.worldToCanvas(...point);
            return [name, {x: rect.left + x, y: rect.top + y}];
          }));
        }"""
    )

    page.mouse.move(points["masked"]["x"], points["masked"]["y"])
    expect(page.locator("#probe-value")).to_have_text("|F| — u")
    page.mouse.move(points["zero"]["x"], points["zero"]["y"])
    expect(page.locator("#probe-value")).to_have_text("|F| 0 u")
    page.mouse.move(points["raw"]["x"], points["raw"]["y"])
    expect(page.locator("#probe-value")).to_have_text("|F| 50 u")
    expect(page.locator("#colorbar-max")).to_have_text("10")
    assert page_errors == []


@pytest.mark.browser
def test_probe_value_remains_consistent_after_resize(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    _open_ready_scene(page, frontend_url, _browser_scene())

    def probe_target() -> dict[str, float]:
        return page.evaluate(
            """async () => {
                const canvas = document.querySelector('#field-canvas');
                const rect = canvas.getBoundingClientRect();
                const { calculatePlotRect, createCoordinateTransform } =
                  await import('/coordinates.js');
                const domain = {x: [-2, 4], y: [-3, 1]};
                const transform = createCoordinateTransform(
                  domain,
                  calculatePlotRect(rect.width, rect.height, domain),
                );
                const [x, y] = transform.worldToCanvas(1, -1);
                return {x: rect.left + x, y: rect.top + y, width: rect.width};
            }"""
        )

    before = probe_target()
    page.mouse.move(before["x"], before["y"])
    expect(page.locator("#probe-position")).to_have_text("x 1 m · y -1 m")
    expect(page.locator("#probe-value")).to_have_text("|F| 5 u")

    backing_width = page.locator("#field-canvas").evaluate("canvas => canvas.width")
    page.set_viewport_size({"width": 820, "height": 900})
    page.wait_for_function(
        "previous => document.querySelector('#field-canvas').width !== previous",
        arg=backing_width,
    )
    page.locator("#field-canvas").scroll_into_view_if_needed()
    after = probe_target()
    assert not math.isclose(after["width"], before["width"])

    page.mouse.move(after["x"], after["y"])
    expect(page.locator("#probe")).to_be_visible()
    expect(page.locator("#probe-position")).to_have_text("x 1 m · y -1 m")
    expect(page.locator("#probe-value")).to_have_text("|F| 5 u")
    assert page_errors == []


@pytest.mark.browser
def test_heatmap_texel_centers_align_with_scalar_nodes(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    # The server samples an endpoint-inclusive grid, so node i sits at
    # xmin + i * dx; the heatmap must put texel centres on those nodes.
    page, page_errors = browser_page
    scene = _browser_scene()
    scalar = scene["scalar"]
    metadata = scene["metadata"]
    assert isinstance(scalar, dict)
    assert isinstance(metadata, dict)
    scalar.update(
        {
            "nx": 5,
            "ny": 3,
            "values": [1.0, 9.0, 1.0, 9.0, 1.0] * 3,
            "mask": [False] * 15,
            "vmin": 1.0,
            "vmax": 9.0,
        }
    )
    scene["lines"] = []
    scene["sources"] = []
    metadata.update(
        {"termination_counts": {}, "suppressed_count": 0, "rendered_line_count": 0}
    )
    _open_ready_scene(page, frontend_url, scene)

    # Nodes x = -0.5 and x = 2.5 lie between x ticks; y = -1.5 lies between
    # y ticks, and every row is identical, so grid lines cannot pollute them.
    # Skia quantises bilinear weights to 1/16, so only the pixels on one side
    # of a node show its exact colour; sample the pixels either side of it.
    # The pre-fix misalignment put nodes a twentieth of the plot width away
    # from their texel centres, about 20% of a colour step off.
    samples = page.evaluate(
        """async () => {
          const canvas = document.querySelector('#field-canvas');
          const rect = canvas.getBoundingClientRect();
          const domain = {x: [-2, 4], y: [-3, 1]};
          const {calculatePlotRect, createCoordinateTransform} =
            await import('/coordinates.js');
          const transform = createCoordinateTransform(
            domain, calculatePlotRect(rect.width, rect.height, domain),
          );
          const context = canvas.getContext('2d');
          const ratio = canvas.width / rect.width;
          return [-0.5, 2.5].map((x) => {
            const [px, py] = transform.worldToCanvas(x, -1.5);
            const data = context.getImageData(
              Math.floor(px * ratio) - 1, Math.floor(py * ratio), 3, 1,
            ).data;
            return [0, 1, 2].map((index) => Array.from(data.slice(index * 4, index * 4 + 3)));
          });
        }"""
    )
    maximum = [100, 79, 172]

    def closest_to_maximum(pixels: list[list[int]]) -> int:
        return min(
            max(abs(channel - expected) for channel, expected in zip(pixel, maximum, strict=True))
            for pixel in pixels
        )

    for pixels in samples:
        assert closest_to_maximum(pixels) <= 3, pixels

    # The transposed pattern checks the vertical axis: rows alternate and
    # nodes y = -0.5 and -1.5 lie between the integer y ticks.
    transposed = json.loads(json.dumps(scene))
    transposed["scalar"].update(
        {
            "nx": 3,
            "ny": 9,
            "values": [value for row in range(9) for value in [9.0 if row % 2 else 1.0] * 3],
            "mask": [False] * 27,
        }
    )
    page.unroute("**/api/scene")
    _route_scene(page, transposed)
    page.locator("#run-button").click()
    expect(page.locator("#connection-label")).to_have_text("已同步")
    vertical = page.evaluate(
        """async () => {
          const canvas = document.querySelector('#field-canvas');
          const rect = canvas.getBoundingClientRect();
          const domain = {x: [-2, 4], y: [-3, 1]};
          const {calculatePlotRect, createCoordinateTransform} =
            await import('/coordinates.js');
          const transform = createCoordinateTransform(
            domain, calculatePlotRect(rect.width, rect.height, domain),
          );
          const context = canvas.getContext('2d');
          const ratio = canvas.width / rect.width;
          return [-0.5, -1.5].map((y) => {
            const [px, py] = transform.worldToCanvas(-1.5, y);
            const data = context.getImageData(
              Math.floor(px * ratio), Math.floor(py * ratio) - 1, 1, 3,
            ).data;
            return [0, 1, 2].map((index) => Array.from(data.slice(index * 4, index * 4 + 3)));
          });
        }"""
    )
    for pixels in vertical:
        assert closest_to_maximum(pixels) <= 3, pixels
    assert page_errors == []


@pytest.mark.browser
def test_canvas_focus_ring_is_not_clipped(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    _open_ready_scene(page, frontend_url, _browser_scene())

    for _ in range(80):
        page.keyboard.press("Tab")
        if page.evaluate("() => document.activeElement?.id") == "field-canvas":
            break
    else:
        pytest.fail("keyboard Tab never reached the field canvas")

    ring = page.evaluate(
        """() => {
          const canvas = document.querySelector('#field-canvas');
          const style = getComputedStyle(canvas);
          const rect = canvas.getBoundingClientRect();
          const clips = [];
          for (let node = canvas.parentElement; node; node = node.parentElement) {
            const parentStyle = getComputedStyle(node);
            if (parentStyle.overflowX !== 'visible' || parentStyle.overflowY !== 'visible') {
              const box = node.getBoundingClientRect();
              clips.push({left: box.left, top: box.top, right: box.right, bottom: box.bottom});
            }
          }
          return {
            focusVisible: canvas.matches(':focus-visible'),
            outlineStyle: style.outlineStyle,
            outlineWidth: parseFloat(style.outlineWidth),
            outlineOffset: parseFloat(style.outlineOffset),
            rect: {left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom},
            clips,
          };
        }"""
    )
    assert ring["focusVisible"] is True
    assert ring["outlineStyle"] != "none"
    assert ring["outlineWidth"] >= 2
    reach = ring["outlineWidth"] + ring["outlineOffset"]
    outer = {
        "left": ring["rect"]["left"] - reach,
        "top": ring["rect"]["top"] - reach,
        "right": ring["rect"]["right"] + reach,
        "bottom": ring["rect"]["bottom"] + reach,
    }
    assert ring["clips"], "expected the stage to clip its overlays"
    for clip in ring["clips"]:
        assert outer["left"] >= clip["left"] - 0.5
        assert outer["top"] >= clip["top"] - 0.5
        assert outer["right"] <= clip["right"] + 0.5
        assert outer["bottom"] <= clip["bottom"] + 0.5
    assert page_errors == []


# Desktop, laptop, projector windows (1280x800 and 1366x768 minus browser
# chrome), a tablet and a phone.
PLOT_READOUT_VIEWPORTS = [
    (1440, 900),
    (1280, 800),
    (1280, 650),
    (1366, 657),
    (1280, 560),
    (820, 900),
    (390, 844),
]


@pytest.mark.browser
@pytest.mark.parametrize(
    "viewport", PLOT_READOUT_VIEWPORTS, ids=[f"{w}x{h}" for w, h in PLOT_READOUT_VIEWPORTS]
)
def test_plot_readouts_stay_inside_stage_and_off_the_plot(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
    viewport: tuple[int, int],
) -> None:
    page, page_errors = browser_page
    page.set_viewport_size({"width": viewport[0], "height": viewport[1]})
    scene = _browser_scene()
    scalar = scene["scalar"]
    assert isinstance(scalar, dict)
    # Real Halbach limits: the widest colorbar labels the presets produce.
    scalar.update(
        {
            "values": [3.584e-9, 1e-8, 1e-7, 1e-6, 5e-6, 1.437e-5, 2e-6, 3e-7, 4e-8],
            "scale": "log",
            "label": "|B|",
            "unit": "T",
            "vmin": 3.584e-9,
            "vmax": 1.437e-5,
        }
    )
    _instrument_canvas(page)
    _open_ready_scene(page, frontend_url, scene)
    expect(page.locator("#colorbar-max")).to_have_text("1.44e-5")
    expect(page.locator("#colorbar-min")).to_have_text("3.58e-9")

    # The whole plot is on the first screen and nothing opaque covers it.
    first_screen = page.evaluate(
        """async () => {
          const canvas = document.querySelector('#field-canvas');
          const rect = canvas.getBoundingClientRect();
          const domain = {x: [-2, 4], y: [-3, 1]};
          const {calculatePlotRect} = await import('/coordinates.js');
          const plot = calculatePlotRect(rect.width, rect.height, domain);
          const left = rect.left + plot.left;
          const top = rect.top + plot.top;
          const right = rect.left + plot.right;
          const bottom = rect.top + plot.bottom;
          const probes = [
            [(left + right) / 2, (top + bottom) / 2],
            [left + 2, top + 2], [right - 2, top + 2],
            [left + 2, bottom - 2], [right - 2, bottom - 2],
          ];
          return {
            inViewport: left >= 0 && top >= 0 &&
              right <= window.innerWidth && bottom <= window.innerHeight,
            hits: probes.map(([x, y]) => document.elementFromPoint(x, y)?.id ?? null),
            overflowX:
              document.documentElement.scrollWidth - document.documentElement.clientWidth,
          };
        }"""
    )
    assert first_screen["inViewport"] is True
    assert first_screen["hits"] == ["field-canvas"] * 5
    assert first_screen["overflowX"] <= 1

    layout = page.evaluate(
        """async () => {
          const canvas = document.querySelector('#field-canvas');
          const rect = canvas.getBoundingClientRect();
          const domain = {x: [-2, 4], y: [-3, 1]};
          const {calculatePlotRect} = await import('/coordinates.js');
          const plot = calculatePlotRect(rect.width, rect.height, domain);
          const box = (element) => {
            const r = element.getBoundingClientRect();
            return {left: r.left, top: r.top, right: r.right, bottom: r.bottom};
          };
          const colorbar = document.querySelector('#colorbar');
          const titles = window.__vectorVizCanvasCalls.texts
            .filter(({text}) => text === 'x / m' || text === 'y / m');
          return {
            canvas: box(canvas),
            stage: box(document.querySelector('#canvas-stage')),
            plot: {
              left: rect.left + plot.left,
              top: rect.top + plot.top,
              right: rect.left + plot.right,
              bottom: rect.top + plot.bottom,
            },
            colorbar: box(colorbar),
            colorbarOverflow: Math.max(
              colorbar.scrollWidth - colorbar.clientWidth,
              colorbar.scrollHeight - colorbar.clientHeight,
            ),
            titles: titles.map(({text, x, y}) => ({
              text, x: rect.left + x, y: rect.top + y,
            })),
          };
        }"""
    )

    def inside(inner: dict[str, float], outer: dict[str, float]) -> bool:
        return (
            inner["left"] >= outer["left"] - 0.5
            and inner["top"] >= outer["top"] - 0.5
            and inner["right"] <= outer["right"] + 0.5
            and inner["bottom"] <= outer["bottom"] + 0.5
        )

    def overlaps(first: dict[str, float], second: dict[str, float]) -> bool:
        return (
            min(first["right"], second["right"]) - max(first["left"], second["left"]) > 0.5
            and min(first["bottom"], second["bottom"]) - max(first["top"], second["top"]) > 0.5
        )

    plot = layout["plot"]
    assert inside(layout["colorbar"], layout["stage"])
    assert not overlaps(layout["colorbar"], plot)
    assert layout["colorbarOverflow"] <= 1
    # The fixture limits equal the data extremes, so nothing is clipped.
    assert page.locator("#colorbar").evaluate(
        "bar => bar.hasAttribute('data-extend-over') || bar.hasAttribute('data-extend-under')"
    ) is False
    assert {title["text"] for title in layout["titles"]} == {"x / m", "y / m"}
    for title in layout["titles"]:
        assert layout["canvas"]["left"] <= title["x"] <= layout["canvas"]["right"]
        assert layout["canvas"]["top"] <= title["y"] <= layout["canvas"]["bottom"]
        assert not (
            plot["left"] < title["x"] < plot["right"] and plot["top"] < title["y"] < plot["bottom"]
        )

    # The probe reports raw values, so it must stay readable anywhere on the
    # plot and, where the colorbar sits beside the plot, keep off it.
    def probe_box() -> dict[str, float]:
        return page.locator("#probe").evaluate(
            """(element) => {
              const r = element.getBoundingClientRect();
              return {left: r.left, top: r.top, right: r.right, bottom: r.bottom};
            }"""
        )

    beside_plot = page.locator("#canvas-stage").get_attribute("data-plot-layout") == "wide"
    width = plot["right"] - plot["left"]
    middle = (plot["top"] + plot["bottom"]) / 2
    targets = [(plot["right"] - 2, plot["top"] + 2)]
    targets += [(plot["left"] + width * step / 8, middle) for step in range(1, 8)]
    for x, y in targets:
        page.mouse.move(x, y)
        expect(page.locator("#probe")).to_be_visible()
        probe = probe_box()
        assert inside(probe, layout["stage"]), (x, y, probe)
        if beside_plot:
            assert not overlaps(probe, layout["colorbar"]), (x, y, probe)

    expect(page.locator("#projection-note")).to_have_text("Browser semantic fixture")
    expect(page.locator("#colorbar-label")).to_contain_text("T")
    expect(page.get_by_text("不代表场强")).to_be_visible()

    # Scientific details must be reachable with the mouse wheel in every layout.
    seed_mode = page.locator("#seed-mode")
    for _ in range(30):
        if seed_mode.evaluate(
            """(element) => {
              const r = element.getBoundingClientRect();
              return r.top >= 0 && r.bottom <= window.innerHeight;
            }"""
        ):
            break
        column = page.locator(".details-column").bounding_box()
        x = min(max(column["x"] + column["width"] / 2, 1), viewport[0] - 1)
        y = min(max(column["y"] + 40, 1), viewport[1] - 1)
        page.mouse.move(x, y)
        page.mouse.wheel(0, 300)
        page.wait_for_timeout(60)
    expect(seed_mode).to_be_in_viewport()

    # A resize invalidates the probe's placement; it must not linger outside.
    if viewport[0] >= 1024:
        page.mouse.move(plot["right"] - 2, plot["top"] + 2)
        expect(page.locator("#probe")).to_be_visible()
        backing = page.locator("#field-canvas").evaluate("canvas => canvas.width")
        page.set_viewport_size({"width": viewport[0] - 160, "height": viewport[1]})
        page.wait_for_function(
            "previous => document.querySelector('#field-canvas').width !== previous",
            arg=backing,
        )
        page.wait_for_timeout(100)
        stage = page.locator("#canvas-stage").evaluate(
            """(element) => {
              const r = element.getBoundingClientRect();
              return {left: r.left, top: r.top, right: r.right, bottom: r.bottom};
            }"""
        )
        assert page.locator("#probe").is_hidden() or inside(probe_box(), stage)
    assert page_errors == []


def _srgb_luminance(rgb: list[int]) -> float:
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in (channel / 255 for channel in rgb)
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _cie_lightness(rgb: list[int]) -> float:
    luminance = _srgb_luminance(rgb)
    if luminance > 216 / 24389:
        return 116 * luminance ** (1 / 3) - 16
    return 24389 / 27 * luminance


def _colormap_ramp(page: Page) -> list[list[int]]:
    """RGB of the colormap at 257 evenly spaced positions, weakest first."""

    return page.evaluate(
        """async () => {
            const { colorForScalar } = await import('/color-scale.js');
            const unit = {type: 'linear', minimum: 0, maximum: 1};
            return Array.from(
              {length: 257}, (_, i) => colorForScalar(i / 256, false, unit).slice(0, 3),
            );
        }"""
    )


@pytest.mark.browser
def test_colored_cells_keep_their_color_up_to_the_mask_edge(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    # The masked column separates two constant regions, one at each end of
    # the scale. Neither may fade toward the hatch grey (on this colormap
    # that reads as a weaker field beside a source) or toward the other side.
    page, page_errors = browser_page
    scene = _browser_scene()
    scalar = scene["scalar"]
    metadata = scene["metadata"]
    assert isinstance(scalar, dict)
    assert isinstance(metadata, dict)
    row = [1.0, 1.0, None, 10.0, 10.0]
    scalar.update(
        {
            "nx": 5,
            "ny": 3,
            "values": row * 3,
            "mask": [value is None for value in row] * 3,
            "vmin": 1.0,
            "vmax": 10.0,
        }
    )
    scene["lines"] = []
    scene["sources"] = []
    metadata.update(
        {"termination_counts": {}, "suppressed_count": 0, "rendered_line_count": 0}
    )
    _open_ready_scene(page, frontend_url, scene)
    ramp = _colormap_ramp(page)

    # Nodes sit at x = -2, -0.5, 1, 2.5, 4; the masked cell spans 0.25-1.75.
    # These points lie in the coloured cells, off the integer grid lines.
    pixels = page.evaluate(
        """async (xs) => {
          const canvas = document.querySelector('#field-canvas');
          const rect = canvas.getBoundingClientRect();
          const domain = {x: [-2, 4], y: [-3, 1]};
          const {calculatePlotRect, createCoordinateTransform} =
            await import('/coordinates.js');
          const transform = createCoordinateTransform(
            domain, calculatePlotRect(rect.width, rect.height, domain),
          );
          const ratio = canvas.width / rect.width;
          return xs.map((x) => {
            const [px, py] = transform.worldToCanvas(x, -1.5);
            return Array.from(canvas.getContext('2d').getImageData(
              Math.floor(px * ratio), Math.floor(py * ratio), 1, 1,
            ).data.slice(0, 3));
          });
        }""",
        [-0.8, -0.3, 0.1, 1.9, 2.3, 2.8],
    )
    expected = [ramp[0]] * 3 + [ramp[-1]] * 3
    for pixel, colour in zip(pixels, expected, strict=True):
        assert max(abs(a - b) for a, b in zip(pixel, colour, strict=True)) <= 3, pixels
    assert page_errors == []


@pytest.mark.browser
def test_uncolored_cells_and_colorbar_extend_follow_scene_state(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    scalar = scene["scalar"]
    assert isinstance(scalar, dict)
    # Column x = -0.5 is masked; 0.5 and 20 fall outside the 1-10 colour limits.
    row = [0.5, None, 5.0, 20.0, 5.0]
    scalar.update(
        {
            "nx": 5,
            "ny": 3,
            "values": row * 3,
            "mask": [value is None for value in row] * 3,
            "vmin": 1.0,
            "vmax": 10.0,
        }
    )
    _instrument_canvas(page)
    _open_ready_scene(page, frontend_url, scene)

    def pixel_at(x: float, y: float) -> list[int]:
        return page.evaluate(
            """async ([x, y]) => {
              const canvas = document.querySelector('#field-canvas');
              const rect = canvas.getBoundingClientRect();
              const domain = {x: [-2, 4], y: [-3, 1]};
              const {calculatePlotRect, createCoordinateTransform} =
                await import('/coordinates.js');
              const [px, py] = createCoordinateTransform(
                domain, calculatePlotRect(rect.width, rect.height, domain),
              ).worldToCanvas(x, y);
              const ratio = canvas.width / rect.width;
              return Array.from(canvas.getContext('2d').getImageData(
                Math.floor(px * ratio), Math.floor(py * ratio), 1, 1,
              ).data.slice(0, 3));
            }""",
            [x, y],
        )


    # RGB of a 16 CSS px horizontal run centred on a world point.
    def pixel_run(x: float, y: float) -> list[list[int]]:
        return page.evaluate(
            """async ([x, y]) => {
              const canvas = document.querySelector('#field-canvas');
              const rect = canvas.getBoundingClientRect();
              const domain = {x: [-2, 4], y: [-3, 1]};
              const {calculatePlotRect, createCoordinateTransform} =
                await import('/coordinates.js');
              const [px, py] = createCoordinateTransform(
                domain, calculatePlotRect(rect.width, rect.height, domain),
              ).worldToCanvas(x, y);
              const ratio = canvas.width / rect.width;
              const data = canvas.getContext('2d').getImageData(
                Math.floor((px - 8) * ratio), Math.floor(py * ratio), Math.ceil(16 * ratio), 1,
              ).data;
              const run = [];
              for (let i = 0; i < data.length; i += 4) run.push([data[i], data[i + 1], data[i + 2]]);
              return run;
            }""",
            [x, y],
        )

    # (-0.5, -1.5) is the middle of a masked cell, off every grid line; its
    # lightest pixel is the hatch base and its darkest lies on a hatch line.
    ramp = _colormap_ramp(page)
    hatched = pixel_run(-0.5, -1.5)
    base = max(hatched, key=_srgb_luminance)
    line = min(hatched, key=_srgb_luminance)
    assert max(base) - min(base) <= 8 and max(base) < 240, base  # neutral, not paper
    # Neither a weak nor the strongest field: far from every colormap colour
    # and much lighter than its maximum.
    distance = min(max(abs(a - b) for a, b in zip(base, colour, strict=True)) for colour in ramp)
    assert distance >= 24, (base, distance)
    assert _srgb_luminance(base) > 2 * _srgb_luminance(ramp[-1]), base
    # 1 CSS px lines of 60% ink measure about 2.3:1 at pixel ratio 1 (50% ink: 2.0).
    assert (_srgb_luminance(base) + 0.05) / (_srgb_luminance(line) + 0.05) >= 2.1, (base, line)
    # The cell at x = 2.5 holds 20 > vmax: it shows the colormap maximum,
    # smoothed only toward the lighter x = 1 and x = 4 cells, and a stray
    # hatch line would pull a channel below the maximum's.
    for rgb in pixel_run(2.5, -1.5):
        assert all(a >= b - 3 for a, b in zip(rgb, ramp[-1], strict=True)), rgb
    expect(page.locator(".colorbar-extend-over")).to_be_visible()
    expect(page.locator(".colorbar-extend-under")).to_be_visible()
    raster = page.evaluate(
        """() => window.__vectorVizCanvasCalls.putImages
          .filter((image) => image.width === 5 && image.height === 3).at(-1)"""
    )
    assert raster["data"][1 * 4 + 3] == 0
    expect(page.locator("#legend-uncolored")).to_be_visible()
    colorbar = page.locator("#colorbar")
    expect(colorbar).to_have_attribute("data-extend-over", "")
    expect(colorbar).to_have_attribute("data-extend-under", "")
    expect(page.locator("#colorbar-extent")).to_have_text(
        "3 个格点高于上限、3 个格点低于下限，按端色显示。"
    )
    expect(page.locator("#caption-seed")).to_have_text("播种：覆盖播种 · 渲染 1 / 抑制 0")

    # Dragging hides every scene-dependent mark without moving the plot.
    canvas_box = page.locator("#field-canvas").bounding_box()
    target = page.evaluate(
        """async () => {
          const canvas = document.querySelector('#field-canvas');
          const rect = canvas.getBoundingClientRect();
          const domain = {x: [-2, 4], y: [-3, 1]};
          const {calculatePlotRect, createCoordinateTransform} =
            await import('/coordinates.js');
          const [x, y] = createCoordinateTransform(
            domain, calculatePlotRect(rect.width, rect.height, domain),
          ).worldToCanvas(1, -1);
          return {x: rect.left + x, y: rect.top + y};
        }"""
    )
    page.mouse.move(target["x"], target["y"])
    page.mouse.down()
    page.mouse.move(target["x"] + 40, target["y"] + 20)
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "stale")
    assert page.locator("#field-canvas").bounding_box() == canvas_box
    assert min(pixel_at(-0.5, -1.5)) >= 250
    expect(page.locator("#legend-uncolored")).to_be_hidden()
    expect(page.locator("#caption-seed")).to_be_hidden()
    page.mouse.up()
    expect(page.locator("#connection-label")).to_have_text("已同步")

    # Hatching wholly under a marker cannot be seen, so the legend stays away.
    fine = _browser_scene()
    fine_scalar = fine["scalar"]
    assert isinstance(fine_scalar, dict)
    nx, ny = 61, 41
    covered = 20 * nx + 30  # the node at the source, (1, -1)
    fine_scalar.update(
        {
            "nx": nx,
            "ny": ny,
            "values": [None if index == covered else 5.0 for index in range(nx * ny)],
            "mask": [index == covered for index in range(nx * ny)],
        }
    )
    page.unroute("**/api/scene")
    _route_scene(page, fine)
    page.locator("#run-button").click()
    expect(page.locator("#connection-label")).to_have_text("已同步")
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "ready")
    expect(page.locator("#legend-uncolored")).to_be_hidden()

    # The status names the failure: a computation error or an unreachable service.
    page.unroute("**/api/scene")
    page.route(
        "**/api/scene",
        lambda route: route.fulfill(
            status=500,
            content_type="application/json",
            body=json.dumps({"detail": "fixture failure"}),
        ),
    )
    page.locator("#run-button").click()
    expect(page.locator("#connection-label")).to_have_text("计算失败")
    page.unroute("**/api/scene")
    page.route("**/api/scene", lambda route: route.abort())
    page.locator("#run-button").click()
    expect(page.locator("#connection-label")).to_have_text("连接失败")
    assert page_errors == []


@pytest.mark.browser
def test_density_slider_reaches_odd_values(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    _open_ready_scene(page, frontend_url, _browser_scene())

    density = page.locator("#density")
    expect(density).to_have_attribute("step", "1")
    density.evaluate(
        "input => { input.value = '7'; input.dispatchEvent(new Event('input', {bubbles: true})); }"
    )
    expect(density).to_have_value("7")
    expect(page.locator("#density-output")).to_have_text("7")
    assert page_errors == []


@pytest.mark.browser
def test_current_loop_odd_density_is_submitted_from_browser(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    requests: list[dict[str, object]] = []

    def route_scene(route: Route) -> None:
        body = route.request.post_data_json
        requests.append(body)
        scene = _browser_loop_scene() if body["preset"] == "current_loop" else _browser_scene()
        route.fulfill(status=200, content_type="application/json", body=json.dumps(scene))

    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)
    expect(page.locator("#connection-label")).to_have_text("已同步")
    page.locator("#density").evaluate("input => { input.value = '7'; }")
    page.locator("#preset").select_option("current_loop")
    expect(page.locator("#scene-title")).to_have_text("Current loop fixture")

    assert requests[-1]["density"] == 7
    assert "sources" not in requests[-1]
    assert page_errors == []


@pytest.mark.browser
def test_electric_plus_one_minus_five_override_does_not_raise_density(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    scene["sources"] = [
        {"x": -0.8, "y": 0.0, "kind": "positive", "strength": 1.0, "strength_unit": "nC"},
        {"x": 0.8, "y": 0.0, "kind": "negative", "strength": -5.0, "strength_unit": "nC"},
    ]
    requests: list[dict[str, object]] = []

    def route_scene(route: Route) -> None:
        body = route.request.post_data_json
        requests.append(body)
        response_scene = json.loads(json.dumps(scene))
        if isinstance(body.get("sources"), list):
            response_scene["sources"] = [
                {**source, "strength_unit": "nC"} for source in body["sources"]
            ]
        route.fulfill(status=200, content_type="application/json", body=json.dumps(response_scene))

    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)
    expect(page.locator("#connection-label")).to_have_text("已同步")
    page.locator("#density").evaluate(
        "input => { input.value = '6'; input.dispatchEvent(new Event('input', {bubbles: true})); }"
    )
    first_x = page.locator('.coordinate-field input[data-source-field="x"]').first
    first_x.evaluate("input => { input.value = '-0.9'; }")
    with page.expect_request("**/api/scene") as override_request:
        first_x.dispatch_event("change")

    body = override_request.value.post_data_json
    assert body["density"] == 6
    assert [source["strength"] for source in body["sources"]] == [1, -5]
    expect(page.locator("#density")).to_have_value("6")
    assert page_errors == []


@pytest.mark.browser
def test_frontend_loads_presets_before_initial_scene(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    events: list[str] = []
    page.unroute("**/api/presets")

    def route_presets(route: Route) -> None:
        events.append("presets")
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(_browser_presets()),
        )

    def route_scene(route: Route) -> None:
        events.append("scene")
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(_browser_scene()),
        )

    page.route("**/api/presets", route_presets)
    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)
    expect(page.locator("#connection-label")).to_have_text("已同步")

    assert events == ["presets", "scene"]
    assert page_errors == []


@pytest.mark.browser
def test_frontend_rejects_nonpositive_source_separation_capability(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    presets = _browser_presets()
    presets[0]["source_separation"] = {"exclusive_minimum": 0, "unit": "m"}
    scene_requests = 0
    page.unroute("**/api/presets")
    page.route(
        "**/api/presets",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(presets),
        ),
    )

    def route_scene(route: Route) -> None:
        nonlocal scene_requests
        scene_requests += 1
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(_browser_scene()),
        )

    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)

    expect(page.locator("#error-banner")).to_be_visible()
    expect(page.locator("#error-message")).to_contain_text("场源间距能力无效")
    assert scene_requests == 0
    assert page_errors == []


@pytest.mark.browser
def test_frontend_displays_seed_mode_and_termination_counts(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    metadata = scene["metadata"]
    assert isinstance(metadata, dict)
    metadata.update(
        {
            "field_model": "finite-volume fixture model",
            "seed_mode": "equal_flux",
            "seed_description": "Equal-flux fixture description from the server.",
            "termination_counts": {"domain_exit": 3, "future_reason": 2},
            "start_termination_counts": {"exclusion_hit": 4, "future_start": 1},
            "suppressed_count": 2,
            "rendered_line_count": 1,
        }
    )
    _open_ready_scene(page, frontend_url, scene)

    expect(page.locator("#field-model")).to_have_text("finite-volume fixture model")
    expect(page.locator("#seed-mode")).to_have_text("等通量播种")
    expect(page.locator("#seed-description")).to_have_text(
        "Equal-flux fixture description from the server."
    )
    expect(page.locator("#termination-counts")).to_contain_text("离开计算域 3")
    expect(page.locator("#termination-counts")).to_contain_text("future_reason 2")
    expect(page.locator("#start-termination-counts")).to_contain_text("命中排除区 4")
    expect(page.locator("#start-termination-counts")).to_contain_text("future_start 1")
    expect(page.locator("#rendered-line-count")).to_have_text("1")
    expect(page.locator("#suppressed-count")).to_have_text("2")
    assert page_errors == []


@pytest.mark.browser
def test_frontend_unknown_seed_mode_falls_back_to_raw_token(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    metadata = scene["metadata"]
    assert isinstance(metadata, dict)
    metadata.update(
        {
            "seed_mode": "future_flux_v2",
            "seed_description": "A future server-owned seeding description.",
        }
    )

    _open_ready_scene(page, frontend_url, scene)

    expect(page.locator("#seed-mode")).to_have_text("future_flux_v2")
    expect(page.locator("#seed-description")).to_have_text(
        "A future server-owned seeding description."
    )
    assert page_errors == []


@pytest.mark.browser
def test_frontend_accepts_legacy_free_text_seed_mode_without_description(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    metadata = scene["metadata"]
    assert isinstance(metadata, dict)
    metadata["seed_mode"] = "旧缓存：从正电荷排除面覆盖播种。"
    metadata.pop("seed_description")

    _open_ready_scene(page, frontend_url, scene)

    expect(page.locator("#seed-mode")).to_have_text("旧缓存：从正电荷排除面覆盖播种。")
    expect(page.locator("#seed-description")).to_have_text("—")
    assert page_errors == []


@pytest.mark.browser
def test_frontend_ignores_unknown_additive_scene_fields(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    scalar = scene["scalar"]
    domain = scene["domain"]
    metadata = scene["metadata"]
    lines = scene["lines"]
    sources = scene["sources"]
    assert isinstance(scalar, dict)
    assert isinstance(domain, dict)
    assert isinstance(metadata, dict)
    assert isinstance(lines, list)
    assert isinstance(sources, list)
    scalar["values"][4] = None
    scalar["mask"][4] = True
    scene["future_top_level"] = {"revision": 3}
    scalar["future_scalar_field"] = "accepted"
    domain["future_domain_field"] = True
    metadata["future_metadata_field"] = [1, 2, 3]
    lines[0]["future_line_field"] = "accepted"
    lines[0]["start_termination"] = "future_start_reason"
    sources[0]["future_source_field"] = "accepted"

    _open_ready_scene(page, frontend_url, scene)

    expect(page.locator("#scene-title")).to_have_text("Browser fixture")
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "ready")
    assert page_errors == []


@pytest.mark.browser
@pytest.mark.parametrize("mask_value, scalar_value", [(False, None), (True, 5.0)])
def test_frontend_rejects_scalar_mask_value_mismatch(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
    mask_value: bool,
    scalar_value: float | None,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    scalar = scene["scalar"]
    assert isinstance(scalar, dict)
    scalar["mask"][4] = mask_value
    scalar["values"][4] = scalar_value
    _route_scene(page, scene)
    page.goto(frontend_url)

    expect(page.locator("#error-banner")).to_be_visible()
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "error")
    assert page_errors == []


@pytest.mark.browser
@pytest.mark.parametrize(
    "invalid_case",
    [
        "missing vmin",
        "non-increasing bounds",
        "nonpositive log vmin",
        "missing lines",
        "missing sources",
        "missing start termination counts",
        "invalid suppressed count",
        "missing rendered count",
        "rendered count mismatch",
        "known seed mode missing description",
    ],
)
def test_frontend_rejects_missing_or_invalid_required_scene_fields(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
    invalid_case: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    scalar = scene["scalar"]
    metadata = scene["metadata"]
    assert isinstance(scalar, dict)
    assert isinstance(metadata, dict)
    if invalid_case == "missing vmin":
        scalar.pop("vmin")
    elif invalid_case == "non-increasing bounds":
        scalar["vmax"] = scalar["vmin"]
    elif invalid_case == "nonpositive log vmin":
        scalar.update({"scale": "log", "vmin": 0.0})
    elif invalid_case == "missing lines":
        scene.pop("lines")
    elif invalid_case == "missing sources":
        scene.pop("sources")
    elif invalid_case == "missing start termination counts":
        metadata.pop("start_termination_counts")
    elif invalid_case == "invalid suppressed count":
        metadata["suppressed_count"] = -1
    elif invalid_case == "missing rendered count":
        metadata.pop("rendered_line_count")
    elif invalid_case == "rendered count mismatch":
        metadata["rendered_line_count"] = 2
    else:
        metadata.pop("seed_description")
    _route_scene(page, scene)
    page.goto(frontend_url)

    expect(page.locator("#error-banner")).to_be_visible()
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "error")
    assert page_errors == []


@pytest.mark.browser
def test_numeric_source_collision_rolls_back_without_request(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    scene["sources"] = [
        {"x": -0.4, "y": 0.0, "kind": "positive", "strength": 1.0, "strength_unit": "nC"},
        {"x": 0.4, "y": 0.0, "kind": "negative", "strength": -1.0, "strength_unit": "nC"},
    ]
    requests: list[dict[str, object]] = []

    def route_scene(route: Route) -> None:
        requests.append(route.request.post_data_json)
        route.fulfill(status=200, content_type="application/json", body=json.dumps(scene))

    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)
    expect(page.locator("#connection-label")).to_have_text("已同步")
    baseline = len(requests)
    first_x = page.locator('.coordinate-field input[data-source-field="x"]').first
    first_x.evaluate("input => { input.value = '0.4'; }")
    first_x.dispatch_event("change")
    page.wait_for_timeout(500)

    expect(first_x).to_have_value("-0.4")
    expect(page.locator("#source-status")).to_contain_text("大于 0.322 m")
    expect(page.locator("#source-status")).to_contain_text("恢复原坐标")
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "ready")
    assert len(requests) == baseline
    assert page_errors == []


@pytest.mark.browser
def test_drag_snaps_to_advertised_source_separation(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    scene["sources"] = [
        {"x": -0.4, "y": 0.0, "kind": "positive", "strength": 1.0, "strength_unit": "nC"},
        {"x": 0.4, "y": 0.0, "kind": "negative", "strength": -1.0, "strength_unit": "nC"},
    ]
    requests: list[dict[str, object]] = []

    def route_scene(route: Route) -> None:
        body = route.request.post_data_json
        requests.append(body)
        response_scene = json.loads(json.dumps(scene))
        if isinstance(body.get("sources"), list):
            response_scene["sources"] = [
                {**source, "strength_unit": "nC"} for source in body["sources"]
            ]
        route.fulfill(status=200, content_type="application/json", body=json.dumps(response_scene))

    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)
    expect(page.locator("#connection-label")).to_have_text("已同步")
    points = page.evaluate(
        """async () => {
          const canvas = document.querySelector('#field-canvas');
          const rect = canvas.getBoundingClientRect();
          const domain = {x: [-2, 4], y: [-3, 1]};
          const {calculatePlotRect, createCoordinateTransform} = await import('/coordinates.js');
          const transform = createCoordinateTransform(
            domain, calculatePlotRect(rect.width, rect.height, domain),
          );
          const start = transform.worldToCanvas(-0.4, 0);
          const target = transform.worldToCanvas(0.4, 0);
          return {
            start: {x: rect.left + start[0], y: rect.top + start[1]},
            target: {x: rect.left + target[0], y: rect.top + target[1]},
          };
        }"""
    )

    page.mouse.move(points["start"]["x"], points["start"]["y"])
    page.mouse.down()
    with page.expect_request("**/api/scene") as final_request:
        page.mouse.move(points["target"]["x"], points["target"]["y"])
        expect(page.locator("#source-status")).to_contain_text("间距吸附")
        page.mouse.up()
    submitted = final_request.value.post_data_json["sources"]
    assert (
        math.hypot(
            submitted[0]["x"] - submitted[1]["x"],
            submitted[0]["y"] - submitted[1]["y"],
        )
        > 0.322
    )
    assert page_errors == []


@pytest.mark.browser
def test_multi_source_snap_falls_back_to_last_legal_position(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    _open_ready_scene(page, frontend_url, _browser_scene())

    result = page.evaluate(
        """async () => {
          const controls = await import('/source-controls.js');
          const sources = [
            {x: -1, y: 0, kind: 'positive', strength: 1},
            {x: 0, y: 0, kind: 'negative', strength: -1},
            {x: 0.5, y: 0, kind: 'positive', strength: 1},
          ];
          const snapped = controls.snapSourcePosition(
            'electric_dipole', sources, 0, {x: 0.25, y: 0}, 0.322,
            {xmin: -2, xmax: 2, ymin: -2, ymax: 2},
          );
          return {
            snapped,
            conflict: controls.sourceSeparationConflict(
              'electric_dipole', sources, 0, snapped, 0.322,
            ),
            partialCandidateConflict: controls.sourceSeparationConflict(
              'electric_dipole',
              [
                {x: -1, y: 1, kind: 'positive', strength: 1},
                {x: 0, y: 1, kind: 'negative', strength: -1},
              ],
              0,
              {x: 0},
              0.322,
            ),
          };
        }"""
    )

    assert result == {
        "snapped": {"x": -1, "y": 0, "snapped": False},
        "conflict": None,
        "partialCandidateConflict": {"index": 1, "distance": 0},
    }
    assert page_errors == []


@pytest.mark.browser
def test_dragged_source_marks_rendered_field_stale_until_response(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    _instrument_canvas(page)
    _open_ready_scene(page, frontend_url, _browser_scene())
    target = page.evaluate(
        """async () => {
          const canvas = document.querySelector('#field-canvas');
          const rect = canvas.getBoundingClientRect();
          const {calculatePlotRect, createCoordinateTransform} =
            await import('/coordinates.js');
          const transform = createCoordinateTransform(
            {x: [-2, 4], y: [-3, 1]},
            calculatePlotRect(rect.width, rect.height, {x: [-2, 4], y: [-3, 1]}),
          );
          const [x, y] = transform.worldToCanvas(1, -1);
          return {x: rect.left + x, y: rect.top + y};
        }"""
    )
    page.mouse.move(target["x"], target["y"])
    page.mouse.down()
    raster_count = page.evaluate("() => window.__vectorVizCanvasCalls.drawImages.length")
    page.mouse.move(target["x"] + 60, target["y"] + 20)

    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "stale")
    expect(page.locator("#scale-badge")).to_have_text("场待重算")
    expect(page.locator("#colorbar")).to_be_hidden()
    assert page.evaluate("() => window.__vectorVizCanvasCalls.drawImages.length") == raster_count
    page.mouse.up()
    expect(page.locator("#connection-label")).to_have_text("已同步")
    assert page_errors == []


@pytest.mark.browser
def test_drag_issues_only_final_scene_request(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    requests: list[dict[str, object]] = []

    def route_scene(route: Route) -> None:
        body = route.request.post_data_json
        requests.append(body)
        response_scene = json.loads(json.dumps(scene))
        if isinstance(body.get("sources"), list):
            response_scene["sources"] = [
                {**source, "strength_unit": "nC"} for source in body["sources"]
            ]
        route.fulfill(status=200, content_type="application/json", body=json.dumps(response_scene))

    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)
    expect(page.locator("#connection-label")).to_have_text("已同步")
    target = page.evaluate(
        """async () => {
          const canvas = document.querySelector('#field-canvas');
          const rect = canvas.getBoundingClientRect();
          const domain = {x: [-2, 4], y: [-3, 1]};
          const {calculatePlotRect, createCoordinateTransform} = await import('/coordinates.js');
          const [x, y] = createCoordinateTransform(
            domain, calculatePlotRect(rect.width, rect.height, domain),
          ).worldToCanvas(1, -1);
          return {x: rect.left + x, y: rect.top + y};
        }"""
    )
    baseline = len(requests)

    page.mouse.move(target["x"], target["y"])
    page.mouse.down()
    for offset in (10, 20, 30, 40):
        page.mouse.move(target["x"] + offset, target["y"])
    page.wait_for_timeout(500)
    assert len(requests) == baseline
    page.mouse.up()
    expect(page.locator("#connection-label")).to_have_text("已同步")
    assert len(requests) == baseline + 1
    assert "sources" in requests[-1]
    assert page_errors == []


@pytest.mark.browser
def test_drag_cancels_pending_edit_and_submits_one_final_request(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    requests: list[dict[str, object]] = []

    def route_scene(route: Route) -> None:
        body = route.request.post_data_json
        requests.append(body)
        response_scene = json.loads(json.dumps(scene))
        if isinstance(body.get("sources"), list):
            response_scene["sources"] = [
                {**source, "strength_unit": "nC"} for source in body["sources"]
            ]
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(response_scene),
        )

    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)
    expect(page.locator("#connection-label")).to_have_text("已同步")
    points = page.evaluate(
        """async () => {
          const canvas = document.querySelector('#field-canvas');
          const rect = canvas.getBoundingClientRect();
          const domain = {x: [-2, 4], y: [-3, 1]};
          const {calculatePlotRect, createCoordinateTransform} = await import('/coordinates.js');
          const transform = createCoordinateTransform(
            domain, calculatePlotRect(rect.width, rect.height, domain),
          );
          const initial = transform.worldToCanvas(1, -1);
          const keyboardMoved = transform.worldToCanvas(1.06, -1);
          return {
            initial: {x: rect.left + initial[0], y: rect.top + initial[1]},
            moved: {x: rect.left + keyboardMoved[0], y: rect.top + keyboardMoved[1]},
          };
        }"""
    )
    baseline = len(requests)

    page.mouse.click(points["initial"]["x"], points["initial"]["y"])
    page.locator("#field-canvas").focus()
    page.keyboard.press("ArrowRight")
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "stale")
    page.mouse.move(points["moved"]["x"], points["moved"]["y"])
    page.mouse.down()
    page.mouse.move(points["moved"]["x"] + 30, points["moved"]["y"])
    page.wait_for_timeout(500)
    assert len(requests) == baseline

    page.mouse.up()
    expect(page.locator("#connection-label")).to_have_text("已同步")
    assert len(requests) == baseline + 1
    assert requests[-1]["sources"][0]["x"] > 1.06
    assert page_errors == []


@pytest.mark.browser
def test_source_control_module_encodes_angle_budget_and_request_contract(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    _open_ready_scene(page, frontend_url, _browser_scene())

    result = page.evaluate(
        """async () => {
          const controls = await import('/source-controls.js');
          const dipoles = Array.from({length: 8}, (_, index) => ({
            x: index / 10,
            y: 0,
            kind: 'dipole',
            strength: 1,
            angle_deg: index * 90,
          }));
          const replacements = dipoles.map((source) => ({...source}));
          replacements.splice(1, 1);
          replacements.push(controls.createSource('dipole', replacements));
          replacements.splice(1, 1);
          replacements.push(controls.createSource('dipole', replacements));
          const obliqueSources = [
            {x: -1, y: 1, kind: 'dipole', strength: 1},
            {
              x: -0.2928932188134524,
              y: 1.7071067811865475,
              kind: 'dipole',
              strength: 1,
            },
          ];
          const obliqueSnapped = controls.snapSourcePosition(
            'magnetic_dipole',
            obliqueSources,
            1,
            {x: -0.9292893218813453, y: 1.0707106781186548},
            0.322,
          );
          return {
            wrappedNegative: controls.normalizeAngleDeg(-90),
            wrappedLarge: controls.normalizeAngleDeg(450),
            defaultAngle: controls.serializeSource({
              x: 0,
              y: 0,
              kind: 'dipole',
              strength: 1,
            }),
            charge: controls.serializeSource({
              x: 9,
              y: -9,
              kind: 'positive',
              strength: 1,
              angle_deg: 30,
            }),
            reversedMoment: controls.effectiveDipoleAngleDeg({
              kind: 'dipole',
              strength: -2,
              angle_deg: 30,
            }),
            seedCount: controls.seedingSourceCount('halbach_array', dipoles),
            electricSeedCount: controls.seedingSourceCount('electric_dipole', [
              {kind: 'positive'},
              {kind: 'negative'},
              {kind: 'positive'},
            ]),
            signedStrengthSeedCount: controls.seedingSourceCount('electric_dipole', [
              {kind: 'positive', strength: 1},
              {kind: 'negative', strength: -5},
            ]),
            defaultElectricSeedCount: controls.seedingSourceCount(
              'electric_dipole', null,
            ),
            defaultMagneticSeedCount: controls.seedingSourceCount(
              'magnetic_dipole', null,
            ),
            defaultHalbachSeedCount: controls.seedingSourceCount(
              'halbach_array', null,
            ),
            defaultHexagonSeedCount: controls.seedingSourceCount(
              'electric_hexagon', null,
            ),
            quadrupoleSeedCount: controls.seedingSourceCount('electric_quadrupole', [
              {kind: 'positive', strength: 1},
              {kind: 'negative', strength: -1},
              {kind: 'negative', strength: -1},
            ]),
            electricFamily: controls.ELECTRIC_PRESETS.filter(controls.isElectricPreset),
            canRemoveToOneCharge: controls.canRemoveSource(
              'electric_hexagon',
              [{kind: 'positive', strength: 1}, {kind: 'positive', strength: 1}],
              0,
            ),
            cannotRemoveOnlyDipolePolarity: controls.canRemoveSource(
              'electric_dipole',
              [
                {kind: 'positive', strength: 1},
                {kind: 'positive', strength: 1},
                {kind: 'negative', strength: -1},
              ],
              2,
            ),
            nonzeroDipoleSeedCount: controls.seedingSourceCount('magnetic_dipole', [
              {kind: 'dipole', strength: 0},
              {kind: 'dipole', strength: -0},
              {kind: 'dipole', strength: -2},
            ]),
            zeroDipoleIsActive: controls.sourceIsActive(
              'magnetic_dipole', {kind: 'dipole', strength: -0},
            ),
            zeroChargeIsActive: controls.sourceIsActive(
              'electric_dipole', {kind: 'positive', strength: 0},
            ),
            canRemoveLastActiveDipole: controls.canRemoveSource(
              'magnetic_dipole',
              [
                {kind: 'dipole', strength: 1},
                {kind: 'dipole', strength: 0},
              ],
              0,
            ),
            boundaryConflicts: Boolean(controls.sourceSeparationConflict(
              'electric_dipole',
              [
                {x: 0, y: 0, kind: 'positive', strength: 1},
                {x: 1, y: 0, kind: 'negative', strength: -1},
              ],
              1,
              {x: 0.322, y: 0},
              0.322,
            )),
            snapped: controls.snapSourcePosition(
              'magnetic_dipole',
              [
                {x: 0, y: 0, kind: 'dipole', strength: 1},
                {x: 1, y: 0, kind: 'dipole', strength: 1},
              ],
              1,
              {x: 0.1, y: 0},
              0.731,
            ),
            snappedAboveMinimum: controls.snapSourcePosition(
              'magnetic_dipole',
              [
                {x: 0, y: 0, kind: 'dipole', strength: 1},
                {x: 1, y: 0, kind: 'dipole', strength: 1},
              ],
              1,
              {x: 0.1, y: 0},
              0.731,
            ).x > 0.731,
            snappedConflicts: Boolean(controls.sourceSeparationConflict(
              'magnetic_dipole',
              [
                {x: 0, y: 0, kind: 'dipole', strength: 1},
                {x: 1, y: 0, kind: 'dipole', strength: 1},
              ],
              1,
              controls.snapSourcePosition(
                'magnetic_dipole',
                [
                  {x: 0, y: 0, kind: 'dipole', strength: 1},
                  {x: 1, y: 0, kind: 'dipole', strength: 1},
                ],
                1,
                {x: 0.1, y: 0},
                0.731,
              ),
              0.731,
            )),
            obliqueSnapped: obliqueSnapped.snapped,
            obliqueDistanceAboveMinimum:
              Math.hypot(obliqueSnapped.x + 1, obliqueSnapped.y - 1) > 0.322,
            obliqueConflicts: Boolean(controls.sourceSeparationConflict(
              'magnetic_dipole',
              obliqueSources,
              1,
              obliqueSnapped,
              0.322,
            )),
            density: controls.densityForSeedBudget(6, 8, {
              min: 6,
              max: 40,
              step: 2,
            }),
            uniqueReplacementPositions: new Set(
              replacements.map(({x, y}) => `${x},${y}`),
            ).size,
          };
        }"""
    )

    assert result == {
        "wrappedNegative": 270,
        "wrappedLarge": 90,
        "defaultAngle": {
            "x": 0,
            "y": 0,
            "kind": "dipole",
            "strength": 1,
            "angle_deg": 90,
        },
        "charge": {"x": 2.8, "y": -2.8, "kind": "positive", "strength": 1},
        "reversedMoment": 210,
        "seedCount": 8,
        "electricSeedCount": 3,
        "signedStrengthSeedCount": 2,
        "defaultElectricSeedCount": 2,
        "defaultMagneticSeedCount": 1,
        "defaultHalbachSeedCount": 0,
        "defaultHexagonSeedCount": 6,
        "quadrupoleSeedCount": 3,
        "electricFamily": [
            "electric_dipole",
            "electric_quadrupole",
            "electric_hexagon",
            "electric_hexagon_alternating",
        ],
        "canRemoveToOneCharge": True,
        "cannotRemoveOnlyDipolePolarity": False,
        "nonzeroDipoleSeedCount": 1,
        "zeroDipoleIsActive": False,
        "zeroChargeIsActive": True,
        "canRemoveLastActiveDipole": False,
        "boundaryConflicts": True,
        "snapped": {
            "x": pytest.approx(0.731, abs=1e-12),
            "y": 0,
            "snapped": True,
        },
        "snappedAboveMinimum": True,
        "snappedConflicts": False,
        "obliqueSnapped": True,
        "obliqueDistanceAboveMinimum": True,
        "obliqueConflicts": False,
        "density": 8,
        "uniqueReplacementPositions": 8,
    }
    assert page_errors == []


@pytest.mark.browser
def test_halbach_sources_are_editable_but_cannot_bypass_api_contracts(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    ordinary_scene = _browser_scene()
    requests: list[dict[str, object]] = []

    def route_scene(route: Route) -> None:
        body = route.request.post_data_json
        requests.append(body)
        if body["preset"] != "halbach_array":
            response_scene = ordinary_scene
        else:
            source_requests = body.get("sources")
            response_scene = _browser_halbach_scene(
                len(source_requests) if isinstance(source_requests, list) else 8
            )
            if isinstance(source_requests, list):
                response_scene["sources"] = [
                    {**source, "strength_unit": "A·m²"} for source in source_requests
                ]
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(response_scene),
        )

    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)
    expect(page.locator("#connection-label")).to_have_text("已同步")

    page.locator("#density").evaluate(
        "input => { input.value = '6'; input.dispatchEvent(new Event('input', {bubbles: true})); }"
    )
    page.locator("#preset").select_option("halbach_array")
    expect(page.locator("#scene-title")).to_have_text("Halbach fixture")
    assert requests[-1]["density"] == 6
    assert "sources" not in requests[-1]
    expect(page.locator("#density")).to_have_value("6")
    expect(page.locator("#density-output")).to_have_text("6")
    expect(page.locator("#source-status")).to_have_text("")
    expect(page.locator(".source-editor")).to_have_count(8)
    expect(page.locator(".source-label")).to_have_text(
        [f"磁偶极子 {index}" for index in range(1, 9)]
    )
    expect(page.locator(".angle-field input")).to_have_count(8)
    expect(page.locator("#add-dipole-source")).to_be_disabled()
    expect(page.locator("#field-canvas")).to_have_attribute("data-draggable", "true")

    with page.expect_request("**/api/scene") as unchanged_request:
        page.locator("#run-button").click()
    assert "sources" not in unchanged_request.value.post_data_json
    assert unchanged_request.value.post_data_json["density"] == 6
    expect(page.locator("#scene-title")).to_have_text("Halbach fixture")

    angle_input = page.locator(".angle-field input").first
    angle_input.evaluate("input => { input.value = '450'; }")
    with page.expect_request("**/api/scene") as angle_request:
        angle_input.dispatch_event("change")
    assert angle_request.value.post_data_json["sources"][0]["angle_deg"] == 90
    assert angle_request.value.post_data_json["density"] == 8
    expect(angle_input).to_have_value("90")
    expect(page.locator("#density")).to_have_value("8")
    expect(page.locator("#density-output")).to_have_text("8")
    expect(page.locator("#source-status")).to_contain_text("8 个播种源")

    first_remove = page.locator(".remove-source").first
    with page.expect_request("**/api/scene") as delete_request:
        first_remove.click()
    deleted_body = delete_request.value.post_data_json
    assert len(deleted_body["sources"]) == 7
    expect(page.locator(".source-editor")).to_have_count(7)
    expect(page.locator("#add-dipole-source")).to_be_enabled()

    with page.expect_request("**/api/scene") as add_request:
        page.locator("#add-dipole-source").click()
    added_body = add_request.value.post_data_json
    assert len(added_body["sources"]) == 8
    assert added_body["sources"][-1]["kind"] == "dipole"
    assert added_body["sources"][-1]["angle_deg"] == 90
    expect(page.locator("#add-dipole-source")).to_be_disabled()

    x_input = page.locator('.coordinate-field input[data-source-field="x"]').first
    x_input.evaluate("input => { input.value = '9'; }")
    with page.expect_request("**/api/scene") as coordinate_request:
        x_input.dispatch_event("change")
    assert coordinate_request.value.post_data_json["sources"][0]["x"] == 2.8
    expect(x_input).to_have_value("2.8")

    page.locator("#preset").select_option("electric_dipole")
    expect(page.locator("#scene-title")).to_have_text("Browser fixture")
    expect(page.locator("#source-status")).to_have_text("")
    assert page_errors == []


@pytest.mark.browser
def test_source_removal_preserves_required_electric_polarities(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    scene["sources"] = [
        {
            "x": -0.8,
            "y": 0.0,
            "kind": "positive",
            "strength": 1.0,
            "strength_unit": "nC",
        },
        {
            "x": 0.8,
            "y": 0.0,
            "kind": "negative",
            "strength": -1.0,
            "strength_unit": "nC",
        },
    ]
    requests: list[dict[str, object]] = []

    def route_scene(route: Route) -> None:
        body = route.request.post_data_json
        requests.append(body)
        response_scene = json.loads(json.dumps(scene))
        if isinstance(body.get("sources"), list):
            response_scene["sources"] = [
                {**source, "strength_unit": "nC"} for source in body["sources"]
            ]
        route.fulfill(status=200, content_type="application/json", body=json.dumps(response_scene))

    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)
    expect(page.locator(".source-editor")).to_have_count(2)
    remove_buttons = page.locator(".remove-source")
    expect(remove_buttons).to_have_count(2)
    expect(remove_buttons.nth(0)).to_be_disabled()
    expect(remove_buttons.nth(1)).to_be_disabled()

    with page.expect_request("**/api/scene"):
        page.locator("#add-positive-source").click()
    expect(page.locator(".source-editor")).to_have_count(3)
    expect(page.locator(".remove-source").nth(0)).to_be_enabled()
    expect(page.locator(".remove-source").nth(1)).to_be_disabled()

    with page.expect_request("**/api/scene") as remove_request:
        page.locator(".remove-source").nth(0).click()
    kinds = [source["kind"] for source in remove_request.value.post_data_json["sources"]]
    assert kinds.count("positive") == 1
    assert kinds.count("negative") == 1
    expect(page.locator(".remove-source").nth(0)).to_be_disabled()
    expect(page.locator(".remove-source").nth(1)).to_be_disabled()
    assert page_errors == []


@pytest.mark.browser
def test_dipole_canvas_arrow_uses_angle_and_reverses_negative_strength(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_halbach_scene(2)
    sources = scene["sources"]
    assert isinstance(sources, list)
    sources[0].update({"angle_deg": 30.0, "strength": 1.0})
    sources[1].update({"angle_deg": 30.0, "strength": -1.0})

    _instrument_canvas(page)
    page.route(
        "**/api/scene",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(scene),
        ),
    )
    page.goto(frontend_url)
    page.locator("#preset").select_option("halbach_array")
    expect(page.locator("#scene-title")).to_have_text("Halbach fixture")

    canvas_calls = page.evaluate(
        """() => ({
          rotations: window.__vectorVizCanvasCalls.rotations.slice(-4),
          arrows: window.__vectorVizCanvasCalls.texts
            .map(({text}) => text)
            .filter((text) => text === '→')
            .slice(-2),
        })"""
    )
    assert canvas_calls["arrows"] == ["→", "→"]
    assert canvas_calls["rotations"] == pytest.approx(
        [-math.radians(30), math.radians(30), -math.radians(210), math.radians(210)]
    )
    assert page_errors == []


@pytest.mark.browser
def test_drag_and_keyboard_requests_never_exceed_source_coordinate_contract(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    scene = _browser_scene()
    request_bodies: list[dict[str, object]] = []

    def route_scene(route: Route) -> None:
        body = route.request.post_data_json
        request_bodies.append(body)
        response_scene = json.loads(json.dumps(scene))
        if isinstance(body.get("sources"), list):
            response_scene["sources"] = [
                {**source, "strength_unit": "nC"} for source in body["sources"]
            ]
        route.fulfill(status=200, content_type="application/json", body=json.dumps(response_scene))

    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)
    expect(page.locator("#connection-label")).to_have_text("已同步")
    points = page.evaluate(
        """async () => {
          const canvas = document.querySelector('#field-canvas');
          const rect = canvas.getBoundingClientRect();
          const {calculatePlotRect, createCoordinateTransform} =
            await import('/coordinates.js');
          const domain = {x: [-2, 4], y: [-3, 1]};
          const transform = createCoordinateTransform(
            domain,
            calculatePlotRect(rect.width, rect.height, domain),
          );
          const source = transform.worldToCanvas(1, -1);
          const outsideApiBounds = transform.worldToCanvas(4, -3);
          return {
            source: {x: rect.left + source[0], y: rect.top + source[1]},
            target: {
              x: rect.left + outsideApiBounds[0],
              y: rect.top + outsideApiBounds[1],
            },
          };
        }"""
    )

    page.mouse.move(points["source"]["x"], points["source"]["y"])
    page.mouse.down()
    with page.expect_request("**/api/scene") as drag_request:
        page.mouse.move(points["target"]["x"], points["target"]["y"])
        page.mouse.up()
    dragged = drag_request.value.post_data_json["sources"][0]
    assert dragged["x"] == pytest.approx(2.8)
    assert dragged["y"] == pytest.approx(-2.8)

    coordinate_input = page.locator('.coordinate-field input[data-source-field="x"]').first
    coordinate_input.focus()
    page.locator("#field-canvas").focus()
    page.keyboard.press("ArrowRight")
    with page.expect_request("**/api/scene") as keyboard_request:
        page.keyboard.press("ArrowUp")
    keyboard_source = keyboard_request.value.post_data_json["sources"][0]
    assert keyboard_source["x"] == pytest.approx(2.8)
    assert keyboard_source["y"] > -2.8
    assert all(
        -2.8 <= source[axis] <= 2.8
        for body in request_bodies
        for source in body.get("sources", [])
        for axis in ("x", "y")
    )
    assert page_errors == []
