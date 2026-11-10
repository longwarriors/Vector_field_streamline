"""Real API connections for frozen retarded fields and honest slice labels."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect


def _open_dipole(page: Page, url: str) -> dict:
    page.unroute("**/api/presets")
    page.goto(url)
    expect(page.locator("#field-canvas")).to_have_attribute(
        "data-scene-state", "ready", timeout=30_000,
    )
    page.evaluate("""() => {
      document.querySelector('#density').value = '6';
      document.querySelector('#resolution').value = '32';
    }""")
    with page.expect_response("**/api/scene", timeout=60_000) as response:
        page.locator("#preset").select_option("dipole_radiation")
    assert response.value.status == 200
    expect(page.locator("#field-canvas")).to_have_attribute(
        "data-scene-state", "ready", timeout=30_000,
    )
    return response.value.json()


def _canvas_point(page: Page, scene: dict, x: float, y: float) -> dict:
    return page.evaluate("""async ({domain, x, y}) => {
      const rect = document.querySelector('#field-canvas').getBoundingClientRect();
      const {calculatePlotRect, createCoordinateTransform} = await import('/coordinates.js');
      const transform = createCoordinateTransform(domain, calculatePlotRect(rect.width, rect.height, domain));
      const [cx, cy] = transform.worldToCanvas(x, y);
      return {x: cx, y: cy};
    }""", {"domain": scene["domain"], "x": x, "y": y})


@pytest.mark.browser
def test_dipole_observable_time_units_slices_and_manual_seeds(
    browser_page: tuple[Page, list[str]], frontend_url: str,
) -> None:
    page, errors = browser_page
    requests = []
    page.on("request", lambda request: requests.append(request.post_data_json)
            if request.url.endswith("/api/scene") else None)
    scene = _open_dipole(page, frontend_url)
    canvas = page.locator("#field-canvas")
    assert requests[-1]["observable"] == "electric"
    assert requests[-1]["em_time"] == {"observation": 1.25e-9}
    assert "time" not in requests[-1]
    expect(page.locator("#electromagnetic-controls")).to_be_visible()
    expect(page.locator("#flow-controls")).to_be_hidden()
    expect(page.locator("#field-unit")).to_have_text("V/m")
    expect(page.locator("#caption-electromagnetic")).to_contain_text("1.25 ns")
    expect(page.locator("#caption-electromagnetic")).to_contain_text("x–z 子午面")
    expect(page.locator("#line-legend-label")).to_have_text("瞬时电场线")
    expect(page.locator("#electromagnetic-summary")).to_contain_text("永恒简谐稳态")
    expect(page.locator("#caption-preview")).to_contain_text("不是光子或粒子轨迹")
    expect(page.locator('.layer-toggle[data-layer="preview"]')).to_be_disabled()

    point = _canvas_point(page, scene, 2, 1)
    canvas.hover(position=point)
    expect(page.locator("#probe-position")).to_contain_text("· z")
    expect(page.locator("#probe-direction")).to_contain_text("Ez")
    # Check what the canvas actually paints, not only the metadata.
    page.evaluate("""() => {
      window.paintedLabels = [];
      const original = CanvasRenderingContext2D.prototype.fillText;
      CanvasRenderingContext2D.prototype.fillText = function(text, ...args) {
        window.paintedLabels.push(text); return original.call(this, text, ...args);
      };
    }""")
    page.locator('.layer-toggle[data-layer="grid"]').click()
    assert "z / m" in page.evaluate("window.paintedLabels")
    with page.expect_response("**/api/scene", timeout=60_000) as seeded:
        canvas.click(position=point)
    assert seeded.value.json()["metadata"]["manual_seed_count"] == 1
    expect(canvas).to_have_attribute("data-scene-state", "ready", timeout=30_000)
    assert len(requests[-1]["seeds"]) == 1

    with page.expect_response("**/api/scene", timeout=60_000) as magnetic:
        page.locator("#em-observable").select_option("magnetic")
    assert magnetic.value.json()["electromagnetic"]["slice"] == "xy"
    expect(canvas).to_have_attribute("data-scene-state", "ready", timeout=30_000)
    assert "seeds" not in requests[-1]  # New plane means different seed coordinates.
    expect(page.locator("#field-unit")).to_have_text("T")
    expect(page.locator("#line-legend-label")).to_have_text("瞬时磁场线")
    expect(page.locator("#caption-electromagnetic")).to_contain_text("x–y 赤道面")
    assert "y / m" in page.evaluate("window.paintedLabels")
    canvas.hover(position=point)
    expect(page.locator("#probe-direction")).to_contain_text("By")

    count = len(requests)
    # Input follows the thumb without issuing requests; change commits in SI seconds.
    page.locator("#em-time-slider").evaluate("""input => {
      input.value = '2.5'; input.dispatchEvent(new Event('input'));
    }""")
    expect(page.locator("#em-observation-time")).to_have_value("2.5")
    page.wait_for_timeout(150)
    assert len(requests) == count
    with page.expect_response("**/api/scene", timeout=60_000) as timed:
        page.locator("#em-time-slider").dispatch_event("change")
    assert timed.value.json()["electromagnetic"]["observation_time"] == 2.5e-9
    assert requests[-1]["em_time"] == {"observation": 2.5e-9}
    expect(canvas).to_have_attribute("data-scene-state", "ready", timeout=30_000)

    with page.expect_response("**/api/scene", timeout=60_000) as energy:
        page.locator("#em-observable").select_option("poynting")
    assert energy.value.json()["electromagnetic"]["observation_time"] == 2.5e-9
    expect(canvas).to_have_attribute("data-scene-state", "ready", timeout=30_000)
    expect(page.locator("#field-unit")).to_have_text("W/m²")
    expect(page.locator("#line-legend-label")).to_have_text("瞬时能流线")
    expect(page.locator("#caption-preview")).to_contain_text("近区瞬时能流可反向")
    expect(page.locator("#electromagnetic-summary")).to_contain_text("与周期平均能流不同")
    assert errors == []


@pytest.mark.browser
def test_dipole_invalid_time_and_narrow_view_restore_independent_flow_time(
    browser_page: tuple[Page, list[str]], frontend_url: str,
) -> None:
    page, errors = browser_page
    page.set_viewport_size({"width": 390, "height": 800})
    requests = []
    page.on("request", lambda request: requests.append(request.post_data_json)
            if request.url.endswith("/api/scene") else None)
    _open_dipole(page, frontend_url)
    canvas = page.locator("#field-canvas")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    count = len(requests)
    page.locator("#em-observation-time").fill("11")
    page.locator("#em-observation-time").press("Tab")
    expect(canvas).to_have_attribute("data-scene-state", "error")
    expect(page.locator("#em-time-status")).to_contain_text("0–10 ns")
    assert len(requests) == count
    with page.expect_response("**/api/scene", timeout=60_000):
        page.locator("#em-observation-time").fill("10")
        page.locator("#em-observation-time").press("Tab")
    expect(canvas).to_have_attribute("data-scene-state", "ready", timeout=30_000)
    assert requests[-1]["em_time"] == {"observation": 1e-8}
    with page.expect_response("**/api/scene", timeout=60_000) as flow:
        page.locator("#preset").select_option("flow_steady")
    expect(canvas).to_have_attribute("data-scene-state", "ready", timeout=30_000)
    assert "em_time" not in requests[-1] and "observable" not in requests[-1]
    assert flow.value.json()["flow"]["time"]["observation"] == 4
    expect(page.locator("#electromagnetic-controls")).to_be_hidden()
    expect(page.locator("#caption-electromagnetic")).to_be_hidden()
    expect(page.locator("#flow-controls")).to_be_visible()
    with page.expect_response("**/api/scene", timeout=60_000):
        page.locator("#preset").select_option("uniform")
    expect(canvas).to_have_attribute("data-scene-state", "ready", timeout=30_000)
    expect(page.locator('.layer-toggle[data-layer="preview"]')).to_be_enabled()
    expect(page.locator("#line-legend-label")).to_have_text("场线")
    expect(page.locator("#guide-probe")).to_contain_text("+x 朝 +y")
    assert errors == []


@pytest.mark.browser
def test_dipole_response_rejects_observable_slice_time_and_units(
    browser_page: tuple[Page, list[str]], frontend_url: str,
) -> None:
    page, errors = browser_page
    scene = _open_dipole(page, frontend_url)
    rejected = page.evaluate("""async (scene) => {
      const {validateScene} = await import('/scene-validation.js');
      const expected = {observable: 'electric', em_time: {observation: 1.25e-9}};
      const mutations = [
        s => { delete s.electromagnetic; },
        s => { s.electromagnetic.observable = 'magnetic'; },
        s => { s.electromagnetic.observable = 'average_poynting'; },
        s => { s.electromagnetic.observation_time += 1e-10; },
        s => { s.electromagnetic.observation_time = '1.25e-9'; },
        s => { s.electromagnetic.observation_time = true; },
        s => { s.electromagnetic.observation_time = -1e-9; },
        s => { s.electromagnetic.observation_time = 11e-9; },
        s => { s.electromagnetic.time_unit = 'ns'; },
        s => { s.electromagnetic.frequency_hz = 100; },
        s => { s.electromagnetic.period = 10; },
        s => { s.electromagnetic.moment_amplitude = 1; },
        s => { s.electromagnetic.phase = 1; },
        s => { s.electromagnetic.propagation_speed = 1; },
        s => { s.electromagnetic.wavelength = 3; },
        s => { s.electromagnetic.slice = 'xy'; },
        s => { s.electromagnetic.horizontal_axis = 'z'; },
        s => { s.electromagnetic.vertical_axis = 'y'; },
        s => { s.electromagnetic.source_axis = 'x'; },
        s => { s.electromagnetic.source_position = [0, 0]; },
        s => { s.electromagnetic.source_position[0] = 1; },
        s => { s.electromagnetic.exclusion_radius = 0; },
        s => { s.electromagnetic.quantity_description = ''; },
        s => { s.vector.unit = 'T'; },
        s => { s.scalar.unit = 'm/s'; },
        s => { s.vector.label = 'B'; },
        s => { s.scalar.label = '|S|'; },
        s => { s.domain.y = [-3, 3]; },
      ];
      return mutations.map(mutate => {
        const copy = structuredClone(scene); mutate(copy);
        try { validateScene(copy, 'dipole_radiation', expected); return false; }
        catch { return true; }
      });
    }""", scene)
    assert rejected and all(rejected)
    assert errors == []
