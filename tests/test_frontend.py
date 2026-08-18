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
        "domain": {"x": [-2.0, 4.0], "y": [-3.0, 1.0]},
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
            {"x": 1.0, "y": -1.0, "kind": "positive", "strength": 1.0}
        ],
        "metadata": {
            "title": "Browser fixture",
            "projection_note": "Browser semantic fixture",
            "field_model": "test",
            "seed_mode": "test",
            "termination_counts": {"domain_exit": 1},
        },
    }


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
            const calls = {putImages: [], drawImages: [], paints: []};
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
    page.mouse.move(target["x"], target["y"])
    expect(page.locator("#probe")).to_be_visible()
    expect(page.locator("#probe-position")).to_have_text("x 1 · y -1")
    expect(page.locator("#probe-value")).to_have_text("|F| 5 u")

    page.mouse.down()
    expect(page.locator("#field-canvas")).to_have_attribute("data-dragging", "true")
    page.mouse.up()
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
    expect(page.locator("#probe-position")).to_have_text("x 1 · y -1")
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
    expect(page.locator("#probe-position")).to_have_text("x 1 · y -1")
    expect(page.locator("#probe-value")).to_have_text("|F| 5 u")
    assert page_errors == []
