"""Browser-executed semantic tests for the build-free ES module frontend."""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from collections.abc import Iterator

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
            "seed_mode": "test",
            "termination_counts": {"domain_exit": 1},
        },
    }


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
            "seed_mode": "test coverage",
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
            "seed_mode": "one seed group per dipole",
        }
    )
    return scene


@pytest.fixture(scope="session")
def frontend_url() -> Iterator[str]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
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
              if (args.length === 5) {
                calls.drawImages.push(args.slice(1).map(Number));
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
    expect(coordinate_inputs.nth(0)).to_have_attribute(
        "aria-label", "正电荷 1 x 坐标（m）"
    )
    expect(coordinate_inputs.nth(1)).to_have_attribute(
        "aria-label", "正电荷 1 y 坐标（m）"
    )
    page.mouse.move(target["x"], target["y"])
    expect(page.locator("#probe")).to_be_visible()
    expect(page.locator("#probe-position")).to_have_text("x 1 m · y -1 m")
    expect(page.locator("#probe-value")).to_have_text("|F| 5 u")

    page.mouse.down()
    expect(page.locator("#field-canvas")).to_have_attribute("data-dragging", "true")
    page.mouse.up()
    assert page_errors == []


@pytest.mark.browser
def test_current_loop_markers_are_fixed_response_only_sources(
    browser_page: tuple[Page, list[str]],
    frontend_url: str,
) -> None:
    page, page_errors = browser_page
    ordinary_scene = _browser_scene()
    loop_scene = _browser_loop_scene()
    request_bodies: list[dict[str, object]] = []

    def route_scene(route: Route) -> None:
        body = route.request.post_data_json
        request_bodies.append(body)
        response_scene = loop_scene if body["preset"] == "current_loop" else ordinary_scene
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(response_scene),
        )

    _instrument_canvas(page)
    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "ready")

    page.locator("#preset").select_option("current_loop")
    expect(page.locator("#scene-title")).to_have_text("Current loop fixture")
    expect(page.locator("#field-canvas")).to_have_attribute("data-draggable", "false")
    expect(page.locator("#field-canvas")).to_have_attribute(
        "aria-label", "Current loop fixture二维可视化，共 1 条场线、0 个可移动场源。"
    )
    expect(page.locator(".source-label")).to_have_text(["电流出屏", "电流入屏"])
    expect(page.locator(".source-strength")).to_have_text(["1 A", "1 A"])
    expect(page.locator("#source-editor-list input")).to_have_count(0)
    expect(page.locator("#reset-sources")).to_be_disabled()
    expect(page.locator(".fixed-sources-note")).to_contain_text("不可移动")

    rendered_symbols = page.evaluate(
        """() => window.__vectorVizCanvasCalls.texts
          .map(({text}) => text)
          .filter((text) => text === '⊙' || text === '⊗')"""
    )
    assert rendered_symbols[-2:] == ["⊙", "⊗"]

    with page.expect_request("**/api/scene") as request_info:
        page.locator("#run-button").click()
    request_body = request_info.value.post_data_json
    assert request_body["preset"] == "current_loop"
    assert set(request_body) == {"preset", "density", "resolution"}
    expect(page.locator("#loading-overlay")).to_be_hidden()
    expect(page.locator("#connection-label")).to_have_text("已同步")
    expect(page.locator("#scene-title")).to_have_text("Current loop fixture")

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
    expect(page.locator(".source-strength")).to_have_text(["1 A", "1 A"])
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
            "values": [0.0, -2.0, 1.0, 10.0, 1.0e300, 2.0, 3.0, 4.0, 5.0],
            "mask": [False, False, False, False, True, False, False, False, False],
            "scale": "log",
        }
    )
    scalar.pop("vmin")
    scalar.pop("vmax")
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
    assert pixel(3) == [253, 231, 37, 255]
    expect(page.locator("#colorbar-max")).to_have_text("10")

    result = page.evaluate(
        """async () => {
            const { colorForScalar, normalizeScalar, resolveScale } =
              await import('/color-scale.js');
            const scalar = {
              scale: 'log',
              values: [0, -2, 1, 10, 1e300],
              mask: [false, false, false, false, true],
            };
            const scale = resolveScale(scalar);
            return {
              scale,
              zero: normalizeScalar(0, scale),
              negative: normalizeScalar(-2, scale),
              masked: colorForScalar(1e300, true, scale),
              invalid: colorForScalar(0, false, scale),
              validMaximum: colorForScalar(10, false, scale),
            };
        }"""
    )

    assert result["scale"] == {"type": "log", "minimum": 1, "maximum": 10}
    assert result["zero"] is None
    assert result["negative"] is None
    assert result["masked"][3] == 0
    assert result["invalid"][3] == 0
    assert result["masked"] != result["validMaximum"]
    assert result["invalid"] != result["validMaximum"]
    assert result["validMaximum"] == [253, 231, 37, 255]
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
        "electricSeedCount": 2,
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
                    {**source, "strength_unit": "A·m²"}
                    for source in source_requests
                ]
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(response_scene),
        )

    page.route("**/api/scene", route_scene)
    page.goto(frontend_url)
    expect(page.locator("#connection-label")).to_have_text("已同步")

    page.locator("#density").evaluate("input => { input.value = '6'; }")
    page.locator("#preset").select_option("halbach_array")
    expect(page.locator("#scene-title")).to_have_text("Halbach fixture")
    assert requests[-1]["density"] == 8
    assert "sources" not in requests[-1]
    expect(page.locator("#density")).to_have_value("8")
    expect(page.locator("#density-output")).to_have_text("8")
    expect(page.locator("#source-status")).to_contain_text("8 个播种源")
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
    expect(page.locator("#scene-title")).to_have_text("Halbach fixture")

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

    angle_input = page.locator(".angle-field input").first
    angle_input.evaluate("input => { input.value = '450'; }")
    with page.expect_request("**/api/scene") as angle_request:
        angle_input.dispatch_event("change")
    assert angle_request.value.post_data_json["sources"][0]["angle_deg"] == 90
    expect(angle_input).to_have_value("90")

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
        page.keyboard.press("ArrowDown")
    keyboard_source = keyboard_request.value.post_data_json["sources"][0]
    assert keyboard_source["x"] == pytest.approx(2.8)
    assert keyboard_source["y"] == pytest.approx(-2.8)
    assert all(
        -2.8 <= source[axis] <= 2.8
        for body in request_bodies
        for source in body.get("sources", [])
        for axis in ("x", "y")
    )
    assert page_errors == []
