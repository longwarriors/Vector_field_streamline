"""Critical time controls, particle drawing and live flow API browser contracts."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect


def _open_flow(page: Page, url: str, preset: str = "flow_oscillating") -> dict:
    page.unroute("**/api/presets")
    page.goto(url)
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "ready", timeout=30_000)
    page.evaluate("""() => {
      document.querySelector('#density').value = '6';
      document.querySelector('#resolution').value = '32';
    }""")
    with page.expect_response("**/api/scene", timeout=30_000) as response:
        page.locator("#preset").select_option(preset)
    assert response.value.status == 200
    expect(page.locator("#field-canvas")).to_have_attribute("data-scene-state", "ready", timeout=30_000)
    return response.value.json()


@pytest.mark.browser
def test_flow_time_controls_layers_manual_sites_and_static_restore(
    browser_page: tuple[Page, list[str]], frontend_url: str,
) -> None:
    page, errors = browser_page
    requests = []
    page.on("request", lambda request: requests.append(request.post_data_json)
            if request.url.endswith("/api/scene") else None)
    scene = _open_flow(page, frontend_url)
    canvas = page.locator("#field-canvas")
    assert requests[-1]["time"] == {"observation": 4, "initial": 0, "release_start": 0, "release_end": 4}
    assert scene["flow"]["time"] == requests[-1]["time"]
    expect(page.locator("#flow-controls")).to_be_visible()
    expect(page.locator("#caption-flow")).to_contain_text("观察 4 s")
    expect(page.locator("#caption-preview")).to_contain_text("冻结速度场")
    expect(page.locator('.layer-toggle[data-layer="preview"]')).to_be_disabled()
    expect(page.locator("#field-unit")).to_have_text("m/s")
    expect(page.locator("#flow-summary")).to_contain_text("迹线：到达观察时刻 1；脉线粒子：到达观察时刻 6")
    expect(page.locator("#density-label")).to_have_text("每站释放采样数")
    count = len(requests)
    for name in ("pathlines", "streaklines"):
        before = canvas.evaluate("canvas => canvas.toDataURL()")
        page.locator(f'.layer-toggle[data-layer="{name}"]').click()
        assert canvas.evaluate("canvas => canvas.toDataURL()") != before
        page.locator(f'.layer-toggle[data-layer="{name}"]').click()
    assert len(requests) == count

    # A committed invalid window clears the old display without a server request.
    page.locator("#initial-time").fill("5")
    page.locator("#initial-time").press("Tab")
    expect(canvas).to_have_attribute("data-scene-state", "error")
    expect(page.locator("#time-status")).to_contain_text("初始时刻 ≤ 观察时刻")
    assert len(requests) == count
    with page.expect_response("**/api/scene", timeout=30_000) as updated:
        page.locator("#initial-time").fill("1")
        page.locator("#initial-time").press("Tab")
    assert updated.value.json()["flow"]["time"]["initial"] == 1
    expect(canvas).to_have_attribute("data-scene-state", "ready", timeout=30_000)
    expect(page.locator("#caption-flow")).to_contain_text("迹线初始 1 s")

    # Canvas clicks add one common station to all three families through the real API.
    point = page.evaluate("""async (domain) => {
      const canvas = document.querySelector('#field-canvas');
      const rect = canvas.getBoundingClientRect();
      const {calculatePlotRect, createCoordinateTransform} = await import('/coordinates.js');
      const transform = createCoordinateTransform(domain, calculatePlotRect(rect.width, rect.height, domain));
      const [x, y] = transform.worldToCanvas(-1, -1);
      return {x, y};
    }""", scene["domain"])
    with page.expect_response("**/api/scene", timeout=30_000) as seeded:
        canvas.click(position=point)
    assert len(seeded.value.json()["flow"]["release_sites"]) == 2
    expect(canvas).to_have_attribute("data-scene-state", "ready", timeout=30_000)
    expect(page.locator("#flow-summary")).to_contain_text("迹线：到达观察时刻 2")
    with page.expect_response("**/api/scene", timeout=30_000) as steady:
        page.locator("#preset").select_option("flow_steady")
    assert steady.value.status == 200
    expect(canvas).to_have_attribute("data-scene-state", "ready", timeout=30_000)
    assert "seeds" not in requests[-1]
    with page.expect_response("**/api/scene", timeout=30_000) as static:
        page.locator("#preset").select_option("uniform")
    assert "flow" not in static.value.json()
    assert "time" not in requests[-1]
    expect(canvas).to_have_attribute("data-scene-state", "ready", timeout=30_000)
    expect(page.locator("#flow-controls")).to_be_hidden()
    expect(page.locator("#caption-flow")).to_be_hidden()
    expect(page.locator('.layer-toggle[data-layer="pathlines"]')).to_be_hidden()
    expect(page.locator('.layer-toggle[data-layer="preview"]')).to_be_enabled()
    assert errors == []


@pytest.mark.browser
def test_flow_response_rejects_time_particle_and_unit_mismatches(
    browser_page: tuple[Page, list[str]], frontend_url: str,
) -> None:
    page, errors = browser_page
    scene = _open_flow(page, frontend_url)
    rejected = page.evaluate("""async (scene) => {
      const {validateScene} = await import('/scene-validation.js');
      const expected = {seedCount: 0, seeds: [], density: 6, time: {...scene.flow.time}};
      const mutations = [
        s => { delete s.flow; },
        s => { s.flow.time_unit = 'ms'; },
        s => { s.vector.unit = 'm'; },
        s => { s.flow.time.observation = 5; },
        s => { s.flow.time.initial = 7; },
        s => { s.flow.release_sites.push([0, 0]); },
        s => { s.flow.pathlines[0].times.pop(); },
        s => { s.flow.pathlines[0].times[1] = s.flow.pathlines[0].times[0]; },
        s => { s.flow.pathlines[0].site_index = 1; },
        s => { s.flow.pathlines[0].points[0] = [0, 0]; },
        s => { s.flow.pathlines[0].termination = 'null_field'; },
        s => { s.flow.streaklines[0].arrival_times[0] = 0; },
        s => { s.flow.streaklines[0].points[0] = null; },
        s => { s.flow.streaklines[0].terminations[0] = 'domain_exit'; },
        s => { s.flow.streaklines[0].release_times[1] = -1; },
        s => { s.flow.streaklines[0].release_times.pop(); },
        s => { s.flow.streaklines[0].points[0] = [7, 0]; },
        s => { s.flow.streaklines[0].points[s.flow.streaklines[0].points.length - 1] = [0, 0]; },
      ];
      return mutations.map((mutate) => {
        const copy = structuredClone(scene); mutate(copy);
        try { validateScene(copy, 'flow_oscillating', expected); return false; }
        catch { return true; }
      });
    }""", scene)
    assert rejected and all(rejected)
    # A nonzero window can have only two representable release times. The
    # server merges identical floating-point nodes instead of duplicating particles.
    tiny_count = page.evaluate("""async () => {
      const {validateScene} = await import('/scene-validation.js');
      const time = {initial: 0, observation: 2, release_start: 1, release_end: 1 + Number.EPSILON};
      const response = await fetch('/api/scene', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({preset: 'flow_oscillating', density: 6, resolution: 32, time}),
      });
      const scene = validateScene(await response.json(), 'flow_oscillating', {time, density: 6, seedCount: 0, seeds: []});
      return scene.flow.streaklines[0].release_times.length;
    }""")
    assert tiny_count == 2
    assert errors == []


@pytest.mark.browser
def test_particle_renderer_breaks_absent_streak_samples_and_draws_singletons(
    browser_page: tuple[Page, list[str]], frontend_url: str,
) -> None:
    page, errors = browser_page
    _open_flow(page, frontend_url)
    result = page.evaluate("""async () => {
      const {drawFlowCurves} = await import('/flow-renderer.js');
      const {CANVAS_THEME} = await import('/canvas-theme.js');
      const operations = [];
      const context = new Proxy({}, {get: (_, name) => (...args) => operations.push([name, ...args]), set: () => true});
      drawFlowCurves(context, {
        pathlines: [{points: [[7, 8]]}],
        streaklines: [{points: [[1, 2], null, [3, 4], [5, 6]]}],
      }, {worldToCanvas: (...point) => point}, CANVAS_THEME, {pathlines: true, streaklines: true});
      return operations;
    }""")
    assert [item for item in result if item[0] == "moveTo"] == [["moveTo", 7, 8], ["moveTo", 1, 2], ["moveTo", 3, 4]]
    assert [item for item in result if item[0] == "lineTo"] == [["lineTo", 5, 6]]
    assert len([item for item in result if item[0] == "arc"]) == 4
    assert errors == []


@pytest.mark.browser
def test_invalid_time_cancellation_prevents_late_valid_scene_reappearing(
    browser_page: tuple[Page, list[str]], frontend_url: str,
) -> None:
    page, errors = browser_page
    _open_flow(page, frontend_url)
    result = page.evaluate("""async () => {
      const {createSceneLoader} = await import('/scene-loader.js');
      const delivered = [];
      let resolve;
      const loader = createSceneLoader({
        fetchScene: () => new Promise(done => { resolve = done; }),
        onStart: () => {}, onSuccess: value => delivered.push(value),
        onError: () => {}, onSettled: () => {},
      });
      const running = loader.load({time: {observation: 4}});
      loader.cancel();
      resolve('obsolete');
      await running;
      return delivered;
    }""")
    assert result == []
    assert errors == []


@pytest.mark.browser
def test_zero_duration_and_expired_particles_on_narrow_flow_view(
    browser_page: tuple[Page, list[str]], frontend_url: str,
) -> None:
    page, errors = browser_page
    page.set_viewport_size({"width": 390, "height": 800})
    _open_flow(page, frontend_url)
    canvas = page.locator("#field-canvas")
    with page.expect_response("**/api/scene", timeout=30_000) as single:
        page.evaluate("""() => {
          for (const id of ['observation-time', 'initial-time', 'release-start', 'release-end']) {
            document.getElementById(id).value = '6';
          }
          document.querySelector('#observation-time').dispatchEvent(new Event('change'));
        }""")
    expect(canvas).to_have_attribute("data-scene-state", "ready", timeout=30_000)
    flow = single.value.json()["flow"]
    assert len(flow["pathlines"][0]["points"]) == len(flow["streaklines"][0]["points"]) == 1
    expect(page.locator("#flow-summary")).to_contain_text("脉线粒子：到达观察时刻 1")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.locator("#field-canvas").scroll_into_view_if_needed()
    before = canvas.bounding_box()
    page.locator('.layer-toggle[data-layer="streaklines"]').click()
    assert canvas.bounding_box() == before
    page.locator('.layer-toggle[data-layer="streaklines"]').click()
    with page.expect_response("**/api/scene", timeout=30_000) as expired:
        page.evaluate("""() => {
          document.querySelector('#initial-time').value = '0';
          document.querySelector('#release-start').value = '0';
          document.querySelector('#release-start').dispatchEvent(new Event('change'));
        }""")
    expect(canvas).to_have_attribute("data-scene-state", "ready", timeout=30_000)
    flow = expired.value.json()["flow"]
    assert flow["streaklines"][0]["points"][0] is None
    expect(page.locator("#flow-summary")).to_contain_text("离开计算域")
    assert errors == []
