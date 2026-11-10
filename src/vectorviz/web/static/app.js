import { CANVAS_THEME } from "./canvas-theme.js";
import {
  PALETTE,
  getScaleType,
  paletteCssGradient,
  resolveScale,
} from "./color-scale.js";
import { formatEditorValue, formatValue } from "./formatting.js";
import {
  NARROW_PLOT_WIDTH,
  calculatePlotRect,
  clamp,
  createCoordinateTransform,
  nearestNode,
} from "./coordinates.js";
import {
  SOURCE_COORDINATE_LIMIT,
  SOURCE_COUNT_LIMIT,
  canRemoveSource,
  createSource,
  densityForSeedBudget,
  isElectricPreset,
  normalizeAngleDeg,
  seedingSourceCount,
  serializeSource,
  snapSourcePosition,
  sourceSeparationConflict,
} from "./source-controls.js";
import { createPreviewGrid, previewBundle } from "./preview.js";
import { DEFAULT_LAYERS, createRenderer, sourceStyle } from "./renderer.js";
import { createSceneLoader } from "./scene-loader.js";
import { FLOW_PRESETS } from "./flow-validation.js";
import { ELECTROMAGNETIC_PRESET, ELECTROMAGNETIC_OBSERVABLES } from "./electromagnetic-validation.js";
import { MATHEMATICAL_PRESETS, REGION_PRESETS, SEED_MODE_LABELS, validateScene } from "./scene-validation.js";

(() => {
  "use strict";

  const API_URL = "/api/scene";
  const PRESETS_URL = "/api/presets";
  // The request contract for manual seeds: at most this many, inside the domain.
  const SEED_COUNT_LIMIT = 16;
  // Seeds of the cursor preview bundle sit this many CSS px apart, across the field.
  const PREVIEW_SPACING_PX = 9;
  // A pointer that travels further than this between down and up is not a click.
  const CLICK_TOLERANCE_PX = 4;
  const EDITABLE_SOURCE_PRESETS = new Set([
    "electric_dipole",
    "electric_quadrupole",
    "electric_hexagon",
    "electric_hexagon_alternating",
    "magnetic_dipole",
    "halbach_array",
  ]);
  const elements = {
    canvas: document.querySelector("#field-canvas"),
    stage: document.querySelector("#canvas-stage"),
    form: document.querySelector("#scene-form"),
    preset: document.querySelector("#preset"),
    density: document.querySelector("#density"),
    densityOutput: document.querySelector("#density-output"),
    densityLabel: document.querySelector("#density-label"),
    densityHelp: document.querySelector("#density-help"),
    flowControls: document.querySelector("#flow-controls"),
    observationTime: document.querySelector("#observation-time"),
    initialTime: document.querySelector("#initial-time"),
    releaseStart: document.querySelector("#release-start"),
    releaseEnd: document.querySelector("#release-end"),
    timeStatus: document.querySelector("#time-status"),
    captionFlow: document.querySelector("#caption-flow"),
    flowSummaryRow: document.querySelector("#flow-summary-row"),
    flowSummary: document.querySelector("#flow-summary"),
    electromagneticControls: document.querySelector("#electromagnetic-controls"),
    emObservable: document.querySelector("#em-observable"),
    emObservationTime: document.querySelector("#em-observation-time"),
    emTimeSlider: document.querySelector("#em-time-slider"),
    emTimeStatus: document.querySelector("#em-time-status"),
    captionElectromagnetic: document.querySelector("#caption-electromagnetic"),
    electromagneticSummaryRow: document.querySelector("#electromagnetic-summary-row"),
    electromagneticSummary: document.querySelector("#electromagnetic-summary"),
    lineLegendLabel: document.querySelector("#line-legend-label"),
    seedHelp: document.querySelector("#seed-help"),
    resolution: document.querySelector("#resolution"),
    resolutionOutput: document.querySelector("#resolution-output"),
    runButton: document.querySelector("#run-button"),
    retryButton: document.querySelector("#retry-button"),
    resetSources: document.querySelector("#reset-sources"),
    sourceActions: document.querySelector("#source-actions"),
    addPositiveSource: document.querySelector("#add-positive-source"),
    addNegativeSource: document.querySelector("#add-negative-source"),
    addDipoleSource: document.querySelector("#add-dipole-source"),
    sourceEditorList: document.querySelector("#source-editor-list"),
    sourceHelp: document.querySelector("#source-help"),
    sourceHeading: document.querySelector("#source-heading"),
    lineCountLabel: document.querySelector("#line-count-label"),
    sourceStatus: document.querySelector("#source-status"),
    interactionHelp: document.querySelector("#interaction-help"),
    sceneTitle: document.querySelector("#scene-title"),
    projectionNote: document.querySelector("#projection-note"),
    scaleBadge: document.querySelector("#scale-badge"),
    loadingOverlay: document.querySelector("#loading-overlay"),
    errorBanner: document.querySelector("#error-banner"),
    errorMessage: document.querySelector("#error-message"),
    connectionState: document.querySelector("#connection-state"),
    connectionLabel: document.querySelector("#connection-label"),
    colorbar: document.querySelector("#colorbar"),
    colorbarMax: document.querySelector("#colorbar-max"),
    colorbarMin: document.querySelector("#colorbar-min"),
    colorbarLabel: document.querySelector("#colorbar-label"),
    colorbarExtent: document.querySelector("#colorbar-extent"),
    colorbarTicks: document.querySelector("#colorbar-ticks"),
    probeNode: document.querySelector("#probe-node"),
    probeDirection: document.querySelector("#probe-direction"),
    captionPreview: document.querySelector("#caption-preview"),
    clearSeeds: document.querySelector("#clear-seeds"),
    seedStatus: document.querySelector("#seed-status"),
    manualSeedSummary: document.querySelector("#manual-seed-summary"),
    legendChips: Array.from(document.querySelectorAll(".legend-chip[data-layer]")),
    layerToggles: Array.from(document.querySelectorAll(".layer-toggle")),
    figure: document.querySelector(".field-figure"),
    captionSeed: document.querySelector("#caption-seed"),
    lineCount: document.querySelector("#line-count"),
    gridSize: document.querySelector("#grid-size"),
    fieldUnit: document.querySelector("#field-unit"),
    fieldModel: document.querySelector("#field-model"),
    seedMode: document.querySelector("#seed-mode"),
    seedDescription: document.querySelector("#seed-description"),
    terminationCounts: document.querySelector("#termination-counts"),
    startTerminationCounts: document.querySelector("#start-termination-counts"),
    renderedLineCount: document.querySelector("#rendered-line-count"),
    suppressedCount: document.querySelector("#suppressed-count"),
    probe: document.querySelector("#probe"),
    probePosition: document.querySelector("#probe-position"),
    probeValue: document.querySelector("#probe-value"),
    liveStatus: document.querySelector("#live-status"),
  };

  const renderer = createRenderer(elements.canvas, CANVAS_THEME);
  const state = {
    scene: null,
    sourceOverrides: null,
    layers: { ...DEFAULT_LAYERS },
    // Manual seeds are request state, like source overrides: they persist
    // across recomputes of the same preset and clear with a preset change.
    seeds: [],
    preview: null,
    previewGrid: null,
    previewFrame: null,
    pendingSeed: null,
    resizeFrame: null,
    drag: null,
    selectedSource: null,
    plotRect: null,
    transform: null,
    scale: null,
    presentationStatus: "idle",
    presetCapabilities: new Map(),
    pixelRatio: 1,
  };

  function setConnectionStatus(status, label) {
    elements.connectionState.dataset.state = status;
    elements.connectionLabel.textContent = label;
  }

  function setLoading(isLoading) {
    elements.loadingOverlay.hidden = !isLoading;
    elements.stage.setAttribute("aria-busy", String(isLoading));
    elements.runButton.disabled = isLoading;
    if (isLoading) {
      setConnectionStatus("loading", "计算中");
    }
  }

  function setError(message, { unreachable = false } = {}) {
    elements.errorMessage.textContent = message;
    elements.errorBanner.hidden = false;
    setConnectionStatus("error", unreachable ? "连接失败" : "计算失败");
    elements.liveStatus.textContent = `场景计算失败：${message}`;
  }

  function clearError() {
    elements.errorBanner.hidden = true;
  }

  function updateRange(input, output, formatter) {
    const value = Number(input.value);
    const progress = ((value - Number(input.min)) / (Number(input.max) - Number(input.min))) * 100;
    input.style.setProperty("--range-progress", `${progress}%`);
    output.textContent = formatter(value);
  }

  function sourcesAreEditable() {
    return EDITABLE_SOURCE_PRESETS.has(elements.preset.value);
  }

  // Hidden markers cannot be dragged or nudged; the editors stay usable.
  function sourcesDraggable() {
    return sourcesAreEditable() && state.layers.sources;
  }

  function selectedSourceSeparation() {
    return state.presetCapabilities.get(elements.preset.value)?.exclusive_minimum ?? null;
  }

  function sourcePositionBounds() {
    const domain = state.scene?.domain;
    return {
      xmin: Math.max(domain?.x?.[0] ?? -SOURCE_COORDINATE_LIMIT, -SOURCE_COORDINATE_LIMIT),
      xmax: Math.min(domain?.x?.[1] ?? SOURCE_COORDINATE_LIMIT, SOURCE_COORDINATE_LIMIT),
      ymin: Math.max(domain?.y?.[0] ?? -SOURCE_COORDINATE_LIMIT, -SOURCE_COORDINATE_LIMIT),
      ymax: Math.min(domain?.y?.[1] ?? SOURCE_COORDINATE_LIMIT, SOURCE_COORDINATE_LIMIT),
    };
  }

  function validatePresetCapabilities(payload) {
    if (!Array.isArray(payload)) throw new Error("预设能力响应不是数组");
    const capabilities = new Map();
    for (const preset of payload) {
      if (!preset || typeof preset !== "object" || typeof preset.id !== "string") {
        throw new Error("预设能力缺少有效 id");
      }
      const separation = preset.source_separation;
      if (separation === undefined || separation === null) continue;
      if (
        typeof separation !== "object" ||
        typeof separation.exclusive_minimum !== "number" ||
        !Number.isFinite(separation.exclusive_minimum) ||
        separation.exclusive_minimum <= 0 ||
        separation.unit !== "m"
      ) {
        throw new Error(`${preset.id} 的场源间距能力无效`);
      }
      capabilities.set(preset.id, {
        exclusive_minimum: separation.exclusive_minimum,
        unit: separation.unit,
      });
    }
    for (const preset of EDITABLE_SOURCE_PRESETS) {
      if (!capabilities.has(preset)) {
        throw new Error(`${preset} 未声明场源间距能力`);
      }
    }
    return capabilities;
  }

  async function loadPresetCapabilities(signal) {
    const response = await fetch(PRESETS_URL, {headers: {Accept: "application/json"}, signal});
    if (!response.ok) throw new Error(await extractError(response));
    const capabilities = validatePresetCapabilities(await response.json());
    signal.throwIfAborted();
    state.presetCapabilities = capabilities;
  }

  function isFlow() { return FLOW_PRESETS.has(elements.preset.value); }
  function isElectromagnetic() { return elements.preset.value === ELECTROMAGNETIC_PRESET; }

  function updateFlowControls() {
    const flow = isFlow();
    const electromagnetic = isElectromagnetic();
    const emQuantity = ELECTROMAGNETIC_OBSERVABLES[elements.emObservable.value];
    const lineLabel = flow ? "流线" : electromagnetic ? emQuantity.label : "场线";
    elements.flowControls.hidden = !flow;
    elements.electromagneticControls.hidden = !electromagnetic;
    elements.densityLabel.textContent = flow ? "每站释放采样数" : "播种预算";
    elements.densityHelp.textContent = flow
      ? "每个固定释放站在释放窗内等间隔采样；零时长窗只采 1 个粒子。" : "每次计算的总种子预算，不代表场强。";
    elements.seedHelp.textContent = flow
      ? "点击添加固定释放站：同一点分别计算流线、迹线、脉线；最多添加 16 个站。"
      : "点击图中位置，由服务端从该点双向积分一条精确场线；最多 16 个。";
    elements.lineLegendLabel.textContent = lineLabel;
    elements.lineCountLabel.textContent = lineLabel;
    document.getElementById("termination-label").textContent = flow ? "流线末端终止" : "末端终止";
    document.getElementById("start-termination-label").textContent = flow ? "流线起点端终止" : "起点端终止";
    document.getElementById("manual-summary-label").textContent = flow ? "手动流线" : "手动种子";
    document.getElementById("trace-outcomes-label").textContent = flow ? "流线尝试结果" : "尝试结果";
    elements.sourceHeading.textContent = flow ? "释放站位置" : "场源位置";
    const flowGuide = {
      "guide-arrows": "箭头仅表示冻结流线的速度方向；红虚线按物理时间排列，末端圆点为粒子最终位置。脉线不画方向箭头，其切线通常不是速度方向。",
      "guide-density": "蓝点按释放时刻等间隔采样：每个点是同一观察时刻的一个粒子，点间虚线仅连接相邻存活样本。采样点密度不能直接当成速度或流量。",
      "guide-preview": "含时流禁用浏览器网格预览；热图、斜率短线和探针只读观察时刻的冻结速度场。点击添加一个固定释放站，由服务端分别计算三类曲线。",
    };
    const emGuide = {
      "guide-arrows": "箭头表示选中观察量在冻结时刻的方向；近区瞬时能流可能指向源，场线不是光子轨迹或质点迹线。",
      "guide-density": "三种观察量使用覆盖播种；场线密度不能当成场强、辐射功率或能量密度。切换观察量也可能切换真实截面。",
      "guide-preview": "迟滞电磁场禁用浏览器网格预览；冻结网格不表达电磁波传播或真实能量运动。点击后由服务端计算选定时刻的场线。",
    };
    Object.entries(flowGuide).forEach(([id, text]) => {
      const item = document.getElementById(id);
      item.dataset.staticText ??= item.textContent;
      item.textContent = flow ? text : electromagnetic ? emGuide[id] : item.dataset.staticText;
    });
    const probeGuide = document.getElementById("guide-probe");
    probeGuide.dataset.staticText ??= probeGuide.textContent;
    probeGuide.textContent = electromagnetic
      ? `探针读取最近网格点的原值和面内分量，未按色标截断、不插值。方向角从 +x 朝 +${emQuantity.vertical} 逆时针；坐标为 (x, ${emQuantity.vertical}) m。`
      : probeGuide.dataset.staticText;
    elements.layerToggles.forEach((button) => {
      if (["pathlines", "streaklines"].includes(button.dataset.layer)) button.hidden = !flow;
      if (["sources", "preview"].includes(button.dataset.layer)) button.hidden = flow;
      if (button.dataset.layer === "lines") {
        button.querySelector("span").textContent = electromagnetic ? "瞬时线" : lineLabel;
        button.title = flow ? "流线：观察时刻冻结的速度场切线" : lineLabel;
      }
    });
  }

  function readTimeControls() {
    const inputs = [elements.observationTime, elements.initialTime, elements.releaseStart, elements.releaseEnd];
    const [observation, initial, release_start, release_end] = inputs.map((input) => input.valueAsNumber);
    if (inputs.some((input) => !input.validity.valid) || initial > observation ||
        release_start > release_end || release_end > observation) {
      throw new Error("时间须为 0–6 s；初始时刻 ≤ 观察时刻，释放窗起点 ≤ 终点 ≤ 观察时刻。");
    }
    return { observation, initial, release_start, release_end };
  }

  function currentRequestBody() {
    const serializedSources =
      sourcesAreEditable() && state.sourceOverrides?.length
        ? state.sourceOverrides.map(serializeSource)
        : null;
    const requiredDensity = seedingSourceCount(
      elements.preset.value,
      serializedSources,
    );
    const requestedDensity = Number(elements.density.value);
    const density = densityForSeedBudget(requestedDensity, requiredDensity, {
      min: Number(elements.density.min),
      max: Number(elements.density.max),
      step: Number(elements.density.step),
    });
    if (density !== requestedDensity) {
      elements.density.value = String(density);
      updateRange(elements.density, elements.densityOutput, (value) => String(value));
      elements.sourceStatus.textContent = `${requiredDensity} 个播种源每个至少需要 1 个种子，播种预算已自动调整为 ${density}。`;
    }
    const body = {
      preset: elements.preset.value,
      density,
      resolution: Number(elements.resolution.value),
    };
    if (isFlow()) body.time = readTimeControls();
    if (isElectromagnetic()) {
      const time = elements.emObservationTime.valueAsNumber;
      if (!elements.emObservationTime.validity.valid || !Number.isFinite(time) || time < 0 || time > 10) {
        throw new Error("电磁场观察时刻须为 0–10 ns 的有限数值；1 ns = 10⁻⁹ s。");
      }
      body.observable = elements.emObservable.value;
      body.em_time = { observation: time / 1e9 };
    }
    if (serializedSources) {
      body.sources = serializedSources;
    }
    if (state.seeds.length) {
      body.seeds = state.seeds.map(({ x, y }) => ({ x, y }));
    }
    return body;
  }

  async function extractError(response) {
    const fallback = `服务返回 ${response.status} ${response.statusText}`.trim();
    try {
      const data = await response.json();
      if (typeof data.detail === "string") return data.detail;
      if (Array.isArray(data.detail)) {
        return data.detail.map((item) => item.msg || String(item)).join("；");
      }
      if (typeof data.message === "string") return data.message;
    } catch (_error) {
      return fallback;
    }
    return fallback;
  }

  function invalidateSceneView(status) {
    state.scene = null;
    state.selectedSource = null;
    state.drag = null;
    state.pendingSeed = null;
    state.preview = null;
    state.previewGrid = null;
    state.plotRect = null;
    state.transform = null;
    state.scale = null;
    state.presentationStatus = status;
    delete elements.canvas.dataset.dragging;
    elements.canvas.dataset.draggable = "false";
    elements.canvas.setAttribute(
      "aria-label",
      status === "error" ? "场景计算失败，当前没有可显示结果。" : "场景正在重新计算。",
    );
    elements.sceneTitle.textContent = status === "error" ? "场景不可用" : "正在计算场景…";
    elements.projectionNote.textContent =
      status === "error"
        ? "上一份数值结果已清除；修正参数或连接后可重试。"
        : "新结果返回前不显示上一份数值场景。";
    elements.scaleBadge.textContent = status === "error" ? "无有效数据" : "等待数据";
    elements.lineCount.textContent = "—";
    elements.gridSize.textContent = "—";
    elements.fieldUnit.textContent = "—";
    elements.fieldModel.textContent = "—";
    elements.seedMode.textContent = "—";
    elements.seedDescription.textContent = "—";
    elements.terminationCounts.textContent = "—";
    elements.startTerminationCounts.textContent = "—";
    elements.manualSeedSummary.textContent = "—";
    elements.captionFlow.hidden = true;
    elements.captionElectromagnetic.hidden = true;
    elements.electromagneticSummary.textContent = "—";
    elements.electromagneticSummaryRow.hidden = !isElectromagnetic();
    elements.flowSummary.textContent = "—";
    elements.flowSummaryRow.hidden = !isFlow();
    elements.renderedLineCount.textContent = "—";
    elements.suppressedCount.textContent = "—";
    elements.colorbar.hidden = true;
    elements.probe.hidden = true;
    elements.probeNode.hidden = true;
    renderSourceEditors();
    renderSeedStatus();
    render();
  }

  function markSceneStale(message = "场源位置已改变，场待重算。") {
    if (!state.scene) return;
    state.presentationStatus = "stale";
    state.scale = null;
    state.preview = null;
    state.pendingSeed = null;
    elements.colorbar.hidden = true;
    elements.probe.hidden = true;
    elements.probeNode.hidden = true;
    elements.scaleBadge.textContent = "场待重算";
    elements.lineCount.textContent = "—";
    elements.terminationCounts.textContent = "待重算";
    elements.startTerminationCounts.textContent = "待重算";
    elements.manualSeedSummary.textContent = state.seeds.length ? "待重算" : "无";
    elements.renderedLineCount.textContent = "—";
    elements.suppressedCount.textContent = "—";
    elements.canvas.setAttribute(
      "aria-label",
      "场源位置已改变；旧数值场已隐藏，等待重新计算。",
    );
    setConnectionStatus("loading", "待重算");
    elements.liveStatus.textContent = message;
    render();
  }

  async function fetchScene(requestBody, signal) {
    // Initialization, retries and preset changes share one cancellable path:
    // a scene cannot become editable before its capabilities are available.
    if (!state.presetCapabilities.size) await loadPresetCapabilities(signal);
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
      signal,
    });
    if (!response.ok) throw new Error(await extractError(response));
    return validateScene(await response.json(), requestBody.preset, {
      seedCount: requestBody.seeds?.length ?? 0,
      seeds: requestBody.seeds || [],
      time: requestBody.time,
      em_time: requestBody.em_time,
      observable: requestBody.observable,
      density: requestBody.density,
    });
  }

  const loader = createSceneLoader({
    fetchScene,
    onStart() {
      setLoading(true);
      clearError();
      invalidateSceneView("loading");
    },
    onSuccess(scene, requestBody) {
      state.scene = scene;
      state.presentationStatus = "ready";
      state.sourceOverrides = sourcesAreEditable() && Array.isArray(requestBody.sources)
        ? scene.sources.map(serializeSource)
        : null;
      state.selectedSource = null;
      state.previewGrid = scene.vector && !scene.flow && !scene.electromagnetic ? createPreviewGrid(scene) : null;
      state.preview = null;
      renderSourceEditors();
      updateSceneDetails();
      render();
      setConnectionStatus("ready", "已同步");
      const manualNote =
        scene.metadata.manual_seed_count > 0
          ? `，其中 ${scene.lines.filter((line) => Number.isInteger(line.seed_index)).length} 条来自 ${scene.metadata.manual_seed_count} 个手动种子`
          : "";
      elements.liveStatus.textContent = scene.flow
        ? `${scene.metadata.title}已加载：${scene.metadata.rendered_line_count} 条流线、${scene.flow.pathlines.length} 条迹线、${scene.flow.streaklines.length} 组脉线。`
        : `${scene.metadata.title || "场景"}已加载，渲染 ${scene.metadata.rendered_line_count} 条场线${manualNote}，抑制 ${scene.metadata.suppressed_count} 次尝试。`;
    },
    onError(error) {
      invalidateSceneView("error");
      setError(error.message || "无法连接场景计算服务", {
        unreachable: error instanceof TypeError,
      });
    },
    onSettled() {
      setLoading(false);
    },
  });

  async function loadScene({ preserveSources = true, preserveSeeds = true } = {}) {
    if (!preserveSources) {
      state.sourceOverrides = null;
      elements.sourceStatus.textContent = "";
    }
    if (!preserveSeeds) {
      state.seeds = [];
      renderSeedStatus();
    }
    updateFlowControls();
    let body;
    try {
      body = currentRequestBody();
      elements.timeStatus.textContent = "";
      elements.emTimeStatus.textContent = "";
    } catch (error) {
      loader.cancel();
      setLoading(false);
      invalidateSceneView("error");
      (isElectromagnetic() ? elements.emTimeStatus : elements.timeStatus).textContent = error.message;
      setError(error.message);
      return;
    }
    await loader.load(body);
  }

  function scheduleLoad(delay = 260) {
    loader.schedule(() => loadScene(), delay);
  }

  const TERMINATION_LABELS = Object.freeze({
    time_end: "到达观察时刻",
    domain_exit: "离开计算域",
    null_field: "零场",
    nonfinite_field: "非有限场",
    max_arc_length: "达到弧长上限",
    solver_failure: "求解失败",
    seed_outside_domain: "种子超出计算域",
    exclusion_hit: "命中排除区",
    closed_loop: "闭合回路",
  });

  function formatTerminationCounts(counts) {
    const entries = Object.entries(counts);
    if (!entries.length) return "无";
    return entries
      .map(([reason, count]) => `${TERMINATION_LABELS[reason] || reason} ${count}`)
      .join(" · ");
  }

  function updateSceneDetails() {
    const { scene } = state;
    const scalar = scene.scalar;
    const title = scene.metadata.title || presetLabel(elements.preset.value);
    const scaleType = getScaleType(scalar.scale);
    const unit = scalar.unit || "—";
    elements.sceneTitle.textContent = title;
    elements.projectionNote.textContent =
      scene.metadata.projection_note || "曲线沿局部场方向积分；颜色表示场强大小。";
    elements.scaleBadge.textContent = `${scaleType === "log" ? "对数" : "线性"}色标 · ${scalar.label || "场强"}`;
    elements.lineCount.textContent = scene.metadata.rendered_line_count.toLocaleString("zh-CN");
    elements.gridSize.textContent = `${scalar.nx}²`.replace(
      `${scalar.nx}²`,
      scalar.nx === scalar.ny ? `${scalar.nx}²` : `${scalar.nx}×${scalar.ny}`,
    );
    elements.fieldUnit.textContent = unit === "1" ? "无量纲" : unit;
    elements.fieldModel.textContent = scene.metadata.field_model;
    elements.seedMode.textContent =
      SEED_MODE_LABELS[scene.metadata.seed_mode] || scene.metadata.seed_mode;
    elements.seedDescription.textContent = scene.metadata.seed_description ?? "—";
    elements.terminationCounts.textContent = formatTerminationCounts(
      scene.metadata.termination_counts,
    );
    elements.startTerminationCounts.textContent = formatTerminationCounts(
      scene.metadata.start_termination_counts,
    );
    elements.manualSeedSummary.textContent = formatManualSeedSummary(scene);
    elements.renderedLineCount.textContent =
      scene.metadata.rendered_line_count.toLocaleString("zh-CN");
    elements.suppressedCount.textContent =
      scene.metadata.suppressed_count.toLocaleString("zh-CN");
    elements.captionFlow.hidden = !scene.flow;
    elements.flowSummaryRow.hidden = !scene.flow;
    elements.captionElectromagnetic.hidden = !scene.electromagnetic;
    elements.electromagneticSummaryRow.hidden = !scene.electromagnetic;
    if (scene.electromagnetic) {
      const em = scene.electromagnetic;
      const quantity = ELECTROMAGNETIC_OBSERVABLES[em.observable];
      const slice = em.slice === "xz" ? "x–z 子午面（y=0）" : "x–y 赤道面（z=0）";
      const timeNs = em.observation_time * 1e9;
      elements.captionElectromagnetic.textContent = `观察 ${formatValue(timeNs)} ns · ${quantity.label} · ${slice} · ${quantity.unit}。所有曲线冻结同一观察时刻。`;
      elements.electromagneticSummary.textContent = `p(t) = 10⁻⁹ cos(2π × 10⁸ t) ẑ C·m，t 以 s 计；迟滞时间 t − r/c，c = 299792458 m/s，波长 ${formatValue(em.wavelength)} m。源点位于原点，半径 ${em.exclusion_radius} m 的圆内排除求值与追踪。近场、感应场、辐射场同时保留。${em.quantity_description} 永恒简谐稳态，不含突然开启后的完整因果瞬态；S = E × B / μ₀ 为瞬时能流，与周期平均能流不同。`;
    }
    if (scene.flow) {
      const { time, pathlines, streaklines } = scene.flow;
      elements.captionFlow.textContent = `观察 ${formatValue(time.observation)} s · 迹线初始 ${formatValue(time.initial)} s · 释放窗 [${formatValue(time.release_start)}, ${formatValue(time.release_end)}] s。黑实线：流线；红虚线：迹线；蓝点线：脉线。`;
      const counts = (reasons) => reasons.reduce((acc, reason) => ({ ...acc, [reason]: (acc[reason] || 0) + 1 }), {});
      elements.flowSummary.textContent = `迹线：${formatTerminationCounts(counts(pathlines.map((line) => line.termination)))}；脉线粒子：${formatTerminationCounts(counts(streaklines.flatMap((line) => line.terminations)))}。出域粒子不画在脉线上；相邻存活采样点之间的虚线仅作连接。`;
    }

    const scale = resolveScale(scalar);
    elements.colorbar.hidden = false;
    elements.colorbarMax.textContent = formatValue(scale.maximum);
    elements.colorbarMin.textContent = formatValue(scale.minimum);
    renderColorbarTicks(scale);
    // The limits are display choices: flag coloured cells drawn in an end colour.
    let over = 0;
    let under = 0;
    scalar.values.forEach((raw, index) => {
      if (scalar.mask[index]) return;
      const value = Number(raw);
      if (!Number.isFinite(value) || (scale.type === "log" && value <= 0)) return;
      if (value > scale.maximum) over += 1;
      else if (value < scale.minimum) under += 1;
    });
    elements.colorbar.toggleAttribute("data-extend-over", over > 0);
    elements.colorbar.toggleAttribute("data-extend-under", under > 0);
    elements.colorbarExtent.textContent =
      `${over} 个格点高于上限、${under} 个格点低于下限，按端色显示。`;
    elements.captionSeed.textContent = `${scene.flow ? "流线" : ""}播种：${
      SEED_MODE_LABELS[scene.metadata.seed_mode] || scene.metadata.seed_mode
    } · 渲染 ${scene.metadata.rendered_line_count} / 抑制 ${scene.metadata.suppressed_count}`;
    elements.colorbarLabel.textContent = quantityWithUnit(scalar.label || "场强", scalar.unit);
    const draggableSourceCount = sourcesDraggable() ? scene.sources.length : 0;
    elements.canvas.dataset.draggable = String(draggableSourceCount > 0);
    elements.canvas.setAttribute(
      "aria-label",
      `${title}二维可视化，共 ${scene.metadata.rendered_line_count} 条场线、${draggableSourceCount} 个可移动场源。`,
    );
    // Layers that read the vector grid are unavailable without it. The
    // caption line about the preview exists only with a vector grid; the
    // layer toggle then only changes its visibility, so the figure never
    // reflows (and the plot never moves) when a layer is switched.
    const hasVector = Boolean(scene.vector);
    elements.captionPreview.hidden = !hasVector;
    elements.captionPreview.textContent = scene.flow
      ? `热图、向量探针与斜率短线仅表示观察时刻 ${formatValue(scene.flow.time.observation)} s 的冻结速度场；含时流已禁用网格预览。迹线与脉线由服务端按物理时间积分。`
      : scene.electromagnetic
        ? "热图、向量探针与斜率短线只表示当前观察时刻的冻结场；已禁用网格预览，冻结网格不表示电磁波传播或真实能量运动。能流线不是光子或粒子轨迹，近区瞬时能流可反向。"
      : "光标周围的蓝色预览束是浏览器对服务端向量网格做双线性插值再用 RK4 积分的结果，不是服务端积分；点击图中位置可把该种子交给服务端得到精确场线。";
    elements.layerToggles.forEach((button) => {
      if (button.dataset.layer === "slopes" || button.dataset.layer === "preview") {
        const frozenPreview = button.dataset.layer === "preview" && Boolean(scene.flow || scene.electromagnetic);
        button.disabled = !hasVector || frozenPreview;
        button.title = frozenPreview ? "含时场禁用网格预览；热图与向量仅表示观察时刻的冻结场" : hasVector ? button.dataset.title : "此响应没有向量分量，无法显示";
      }
    });
    renderSeedStatus();
  }

  // The manual-seed line of the science details: how many seeds the server
  // traced, how many lines they produced, and both ends' terminations.
  function formatManualSeedSummary(scene) {
    const count = scene.metadata.manual_seed_count;
    if (!count) return "无";
    const drawn = scene.lines.filter((line) => Number.isInteger(line.seed_index)).length;
    return (
      `${count} 个种子 → ${drawn} 条线 · 末端：${formatTerminationCounts(scene.metadata.manual_termination_counts)}` +
      ` · 起点端：${formatTerminationCounts(scene.metadata.manual_start_termination_counts)}`
    );
  }

  function renderSeedStatus() {
    const count = state.seeds.length;
    elements.clearSeeds.disabled = count === 0;
    elements.seedStatus.textContent =
      count === 0
        ? "尚无手动种子。"
        : `${count} 个手动种子（上限 ${SEED_COUNT_LIMIT}）：${state.seeds
            .map(({ x, y }) => `(${formatEditorValue(x)}, ${formatEditorValue(y)})`)
            .join("、")}。`;
  }

  // A log colorbar marks every power of ten inside its limits; a linear one
  // has no natural decades and shows none.
  function renderColorbarTicks(scale) {
    elements.colorbarTicks.replaceChildren();
    if (scale.type !== "log") return;
    const low = Math.log10(scale.minimum);
    const high = Math.log10(scale.maximum);
    for (let exponent = Math.ceil(low); exponent <= Math.floor(high); exponent += 1) {
      const tick = document.createElement("span");
      tick.className = "colorbar-tick";
      tick.style.setProperty("--tick-position", `${((exponent - low) / (high - low)) * 100}%`);
      const mark = document.createElement("i");
      const label = document.createElement("b");
      label.append("10");
      const power = document.createElement("sup");
      power.textContent = String(exponent).replace(/^-/, "\u2212");
      label.append(power);
      tick.append(mark, label);
      elements.colorbarTicks.append(tick);
    }
  }

  // "quantity / unit", bracketing compound units: |E| / (V/m), |B| / T.
  function quantityWithUnit(quantity, unit) {
    const trimmed = String(unit || "").trim();
    if (!trimmed) return quantity;
    if (trimmed === "1") return `${quantity}（无量纲）`;
    return /[\s/·]/.test(trimmed) ? `${quantity} / (${trimmed})` : `${quantity} / ${trimmed}`;
  }

  function presetLabel(value) {
    return {
      electric_dipole: "电偶极子场",
      electric_quadrupole: "电四极子场",
      electric_hexagon: "六个正电荷的电场",
      electric_hexagon_alternating: "三对交替电荷的电场",
      magnetic_dipole: "磁偶极子场",
      halbach_array: "Halbach 阵列磁场",
      current_loop: "圆形电流线圈磁场",
      straight_wire: "长直导线磁场",
      parallel_wires: "同向双导线磁场",
      antiparallel_wires: "反向双导线磁场",
      solenoid: "有限螺线管磁场",
      helmholtz_coils: "Helmholtz 线圈磁场",
      anti_helmholtz_coils: "反 Helmholtz 线圈磁场",
      charged_ring: "带电圆环电场",
      dielectric_sphere: "介质球电场",
      conducting_sphere: "导体球电场",
      uniform: "匀强场",
      laplace_saddle: "Laplace 鞍点场",
      laplace_multipole: "Laplace 三阶谐波场",
      slope_linear: "线性斜率场",
      slope_logistic: "Logistic 斜率场",
      magnetic_shield_cylinder: "理想磁屏蔽筒",
      flow_oscillating: "非定常横向振荡流",
      flow_steady: "定常流对照",
      dipole_radiation: "振荡电偶极子 · 迟滞场",
    }[value] || "物理场";
  }

  function worldToCanvas(x, y) {
    return state.transform.worldToCanvas(x, y);
  }

  function render() {
    const size = renderer.resize();
    state.pixelRatio = size.ratio;
    elements.canvas.dataset.sceneState = state.presentationStatus;
    if (!state.scene) {
      renderer.render({ scene: null, size });
      return;
    }
    state.plotRect = calculatePlotRect(size.width, size.height, state.scene.domain);
    state.transform = createCoordinateTransform(state.scene.domain, state.plotRect);
    publishPlotRect(size.width);
    if (state.presentationStatus !== "stale") state.scale = resolveScale(state.scene.scalar);
    const outcome = renderer.render({
      size,
      scene: state.scene,
      status: state.presentationStatus,
      plotRect: state.plotRect,
      transform: state.transform,
      scale: state.scale,
      selectedSource: state.selectedSource,
      pixelRatio: state.pixelRatio,
      layers: state.layers,
      preview: state.preview,
      seeds: state.seeds,
    });
    if (state.presentationStatus === "stale") return;
    syncLayerChrome(outcome);
  }

  // Readouts follow the layers: a hidden heatmap takes its colorbar and
  // hatch legend with it, hidden lines take the line, arrow and seed chips,
  // hidden markers cannot be dragged, and the slope and preview chips need
  // the vector grid. The probe keeps reading the raw grid.
  function syncLayerChrome(outcome) {
    const { layers, scene } = state;
    const hasVector = Boolean(scene.vector);
    elements.colorbar.hidden = !layers.heatmap;
    // A chip the scene cannot need is removed (`hidden`); a chip whose layer
    // is merely switched off keeps its box (`data-layer-off`), so toggling a
    // layer never re-wraps the legend and never moves the plot.
    const hasManualLines = scene.lines.some((line) => Number.isInteger(line.seed_index));
    const chips = {
      heatmap: [outcome.hatchCells > 0, outcome.hatchVisible],
      lines: [true, layers.lines],
      arrows: [true, layers.lines && layers.arrows],
      sources: [scene.sources.length > 0 || Boolean(scene.electromagnetic), layers.sources],
      slopes: [hasVector, layers.slopes],
      preview: [hasVector && !scene.flow && !scene.electromagnetic, layers.preview],
      pathlines: [Boolean(scene.flow), layers.pathlines],
      streaklines: [Boolean(scene.flow), layers.streaklines],
      seeds: [hasManualLines, layers.lines],
    };
    elements.legendChips.forEach((chip) => {
      const [applicable, shown] = chips[chip.dataset.layer] || [true, true];
      chip.hidden = !applicable;
      chip.toggleAttribute("data-layer-off", applicable && !shown);
    });
    elements.captionPreview.toggleAttribute("data-layer-off", !scene.flow && !scene.electromagnetic && !layers.preview);
    const draggableSourceCount = sourcesDraggable() ? state.scene.sources.length : 0;
    elements.canvas.dataset.draggable = String(draggableSourceCount > 0);
    const title = state.scene.metadata.title || presetLabel(elements.preset.value);
    elements.canvas.setAttribute(
      "aria-label",
      scene.flow
        ? `${title}二维可视化：流线冻结观察时刻，迹线沿物理时间，脉线表示同时刻粒子位置。${scene.flow.release_sites.length} 个固定释放站，坐标 m、速度 m/s、时间 s。`
        : scene.electromagnetic
          ? `${title}二维可视化：${elements.captionElectromagnetic.textContent} 能流线不是光子或粒子轨迹。`
        : `${title}二维可视化，共 ${state.scene.metadata.rendered_line_count} 条场线、${draggableSourceCount} 个可移动场源。`,
    );
  }

  const LAYER_LABELS = Object.freeze({
    heatmap: "热图",
    slopes: "斜率短线",
    lines: "场线",
    pathlines: "迹线",
    streaklines: "脉线",
    arrows: "方向箭头",
    preview: "光标预览束",
    sources: "场源",
    grid: "网格",
  });

  // Layers are display state: switching one never sends a request, and a
  // stale scene stays stale until the server answers.
  function toggleLayer(name) {
    if (!(name in state.layers)) return;
    state.layers[name] = !state.layers[name];
    if (name === "preview" && !state.layers.preview) state.preview = null;
    elements.layerToggles.forEach((button) => {
      if (button.dataset.layer === name) {
        button.setAttribute("aria-pressed", String(state.layers[name]));
      }
    });
    render();
    elements.liveStatus.textContent = `${LAYER_LABELS[name]}图层已${state.layers[name] ? "显示" : "隐藏"}。`;
  }

  function canvasToWorld(canvasX, canvasY) {
    return state.transform.canvasToWorld(canvasX, canvasY);
  }

  // Overlays such as the colorbar are laid out against the plot, not the stage.
  function publishPlotRect(width) {
    const { left, top, right, bottom } = state.plotRect;
    const style = elements.stage.style;
    style.setProperty("--plot-left", `${left}px`);
    style.setProperty("--plot-top", `${top}px`);
    elements.figure.style.setProperty("--plot-top", `${top}px`);
    style.setProperty("--plot-right", `${right}px`);
    style.setProperty("--plot-bottom", `${bottom}px`);
    style.setProperty("--plot-width", `${right - left}px`);
    style.setProperty("--plot-height", `${bottom - top}px`);
    const { x, y } = state.scene.domain;
    style.setProperty("--domain-aspect", String((x[1] - x[0]) / (y[1] - y[0])));
    elements.stage.dataset.plotLayout = width < NARROW_PLOT_WIDTH ? "narrow" : "wide";
  }

  function createSourceName(source, index) {
    const name = document.createElement("span");
    name.className = "source-name";
    const swatch = document.createElement("i");
    const style = sourceStyle(source);
    swatch.className = "source-swatch";
    swatch.setAttribute("aria-hidden", "true");
    swatch.style.background = style.fill;
    const glyph = document.createElement("span");
    glyph.textContent = style.symbol;
    if (Number.isFinite(style.rotation)) glyph.style.transform = `rotate(${style.rotation}rad)`;
    swatch.append(glyph);
    const sourceLabel = document.createElement("span");
    sourceLabel.className = "source-label";
    sourceLabel.textContent = readableSourceName(source, index);
    const sourceStrength = document.createElement("span");
    sourceStrength.className = "source-strength";
    sourceStrength.textContent = `${formatValue(source.strength)} ${source.strength_unit}`;
    name.append(swatch, sourceLabel, sourceStrength);
    return name;
  }

  function renderSourceEditors() {
    elements.sourceEditorList.replaceChildren();
    const sources = state.scene?.sources || [];
    const editable = sourcesAreEditable();
    const preset = elements.preset.value;
    const sourceLimitReached = sources.length >= SOURCE_COUNT_LIMIT;
    elements.resetSources.disabled = sources.length === 0 || !editable;
    elements.sourceActions.hidden = !editable;
    const electric = isElectricPreset(preset);
    elements.addPositiveSource.hidden = !electric;
    elements.addNegativeSource.hidden = !electric;
    elements.addDipoleSource.hidden =
      preset !== "magnetic_dipole" && preset !== "halbach_array";
    // Sources can only be added to a loaded scene.
    const cannotAdd = sourceLimitReached || !state.scene;
    elements.addPositiveSource.disabled = cannotAdd;
    elements.addNegativeSource.disabled = cannotAdd;
    elements.addDipoleSource.disabled = cannotAdd;
    elements.sourceHelp.textContent = editable
      ? electric
        ? "可增删电荷；拖动标记或输入坐标。"
        : "可增删磁偶极子并编辑面内方向。"
      : isFlow() ? "释放站在整个释放时间窗内保持固定。"
        : isElectromagnetic() ? "理想点电偶极子固定在原点，沿 z 轴振荡。" : "固定预设只显示只读几何标记。";
    elements.interactionHelp.textContent = editable
      ? "拖动场源改变位置；移动指针可探测坐标与场强。"
      : isFlow()
        ? "流线冻结观察时刻；迹线跟随同一粒子；脉线是不同释放时刻的粒子在观察时刻的位置集合。三类图层可分别开关，定常流中重合。"
        : isElectromagnetic()
          ? "切换 E、B、S 观察不同冻结场；B 使用赤道面显示真实环向磁场线。改变观察时刻重算迟滞场，点击画布添加服务端种子。"
        : "移动指针可探测坐标与场强；固定标记不可移动。";
    if (!editable && sources.length) {
      const fixed = document.createElement("p");
      fixed.className = "fixed-sources-note";
      const fixedNotes = {
        charged_ring: "两个标记是同一带电圆环的截面位置，电荷与位置由预设固定、不可移动。",
        current_loop: "两个标记是同一圆环的截面位置，电流与位置由预设固定、不可移动。",
        straight_wire: "标记是无限长直导线的横截面；电流与位置由预设固定、不可移动。",
        parallel_wires: "两个标记分别是两根无限长直导线的横截面；电流同向，电流与位置由预设固定、不可移动。",
        antiparallel_wires: "两个标记分别是两根无限长直导线的横截面；电流反向，电流与位置由预设固定、不可移动。",
        solenoid: "每一对左右标记表示一个圆环的截面；有限螺线管采用离散圆环绕组近似，电流与位置由预设固定、不可移动。",
        helmholtz_coils: "每一对左右标记表示一个圆环的截面；两个圆环电流同向，间距等于半径，电流与位置由预设固定、不可移动。",
        anti_helmholtz_coils: "每一对左右标记表示一个圆环的截面；两个圆环电流反向，电流与位置由预设固定、不可移动。",
      };
      fixed.textContent = fixedNotes[preset] || "场源几何与强度由预设固定、不可移动。";
      elements.sourceEditorList.append(fixed);
      sources.forEach((source, index) => {
        const row = document.createElement("div");
        row.className = "fixed-source";
        row.dataset.sourceKind = source.kind;
        row.append(createSourceName(source, index));
        elements.sourceEditorList.append(row);
      });
      return;
    }
    if (!editable) {
      const fixed = document.createElement("p");
      fixed.className = "empty-sources";
      fixed.textContent =
        elements.preset.value === "uniform"
          ? "匀强场的方向与强度由预设固定。"
          : isFlow()
            ? "固定释放站为 (−2, 0) m；蓝点表示释放位置。点击画布可添加站位，同一站位分别用于三类曲线。速度单位 m/s，时间单位 s。"
          : isElectromagnetic()
            ? "原点标记表示局域点偶极源；虚线圆是半径 0.12 m 的数值排除边界，不代表有限尺寸天线。偶极矩幅值 10⁻⁹ C·m，频率 100 MHz；源几何和参数固定。"
          : elements.preset.value === "magnetic_shield_cylinder"
            ? "无限长圆柱壳的内外半径、恒定磁导率与横向外场固定；两条虚线圆是材料界面。"
          : MATHEMATICAL_PRESETS.has(elements.preset.value)
            ? "数学场的方程由预设固定，坐标与向量均无量纲；可点击画布添加初值。"
          : REGION_PRESETS.has(elements.preset.value)
            ? "球的半径、材料与外场由预设固定；虚线圆是球面在该平面上的截线。"
          : state.presentationStatus === "error"
            ? "场景加载失败，固定标记暂不可显示。"
            : "固定几何标记将在场景加载后显示。";
      elements.sourceEditorList.append(fixed);
      return;
    }
    if (!sources.length) {
      const empty = document.createElement("p");
      empty.className = "empty-sources";
      empty.textContent = {
        error: "场景加载失败，场源暂不可编辑。",
        loading: "场景加载后可调整场源。",
      }[state.presentationStatus] || "此预设没有可移动场源。";
      elements.sourceEditorList.append(empty);
      return;
    }

    sources.forEach((source, index) => {
      const row = document.createElement("div");
      row.className = "source-editor";
      row.dataset.sourceKind = source.kind;
      row.append(createSourceName(source, index));
      row.append(coordinateInput(index, "x"), coordinateInput(index, "y"));
      if (source.kind === "dipole") row.append(angleInput(index));
      row.append(removeSourceButton(index));
      elements.sourceEditorList.append(row);
    });
  }

  function readableSourceName(source, index) {
    if (source.kind === "wire_out") return "电流出屏";
    if (source.kind === "wire_into") return "电流入屏";
    if (source.kind === "ring_charge") return "圆环截面";
    if (source.kind === "dipole") return `磁偶极子 ${index + 1}`;
    if (source.kind === "positive") return `正电荷 ${index + 1}`;
    return `负电荷 ${index + 1}`;
  }

  function coordinateInput(index, axis) {
    const label = document.createElement("label");
    label.className = "coordinate-field";
    const coordinateUnit = String(state.scene.domain.unit || "").trim();
    const axisLabel = document.createElement("span");
    axisLabel.className = "coordinate-axis";
    axisLabel.textContent = coordinateUnit ? `${axis}/${coordinateUnit}` : axis;
    label.append(axisLabel);
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.min = String(-SOURCE_COORDINATE_LIMIT);
    input.max = String(SOURCE_COORDINATE_LIMIT);
    input.dataset.sourceField = axis;
    input.value = formatEditorValue(state.scene.sources[index][axis]);
    input.setAttribute(
      "aria-label",
      `${readableSourceName(state.scene.sources[index], index)} ${axis} 坐标${coordinateUnit ? `（${coordinateUnit}）` : ""}`,
    );
    input.addEventListener("focus", () => {
      state.selectedSource = index;
      render();
    });
    input.addEventListener("change", () => {
      const value = Number(input.value);
      if (!Number.isFinite(value)) {
        input.value = formatEditorValue(state.scene.sources[index][axis]);
        return;
      }
      const source = state.scene.sources[index];
      const previous = source[axis];
      const bounded = clamp(value, -SOURCE_COORDINATE_LIMIT, SOURCE_COORDINATE_LIMIT);
      const candidate = {x: source.x, y: source.y, [axis]: bounded};
      const minimum = selectedSourceSeparation();
      const conflict = sourceSeparationConflict(
        elements.preset.value,
        state.scene.sources,
        index,
        candidate,
        minimum,
      );
      if (conflict) {
        input.value = formatEditorValue(previous);
        elements.sourceStatus.textContent = `场源间距必须大于 ${formatEditorValue(minimum)} m；已恢复原坐标。`;
        elements.liveStatus.textContent = "坐标与另一有效场源冲突，未发送计算请求。";
        return;
      }
      input.value = formatEditorValue(bounded);
      updateSource(index, axis, bounded);
      scheduleLoad(360);
    });
    label.append(input);
    return label;
  }

  function angleInput(index) {
    const label = document.createElement("label");
    label.className = "angle-field";
    const angleLabel = document.createElement("span");
    angleLabel.textContent = "θ/°";
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.min = "0";
    input.max = "359.999999";
    input.dataset.sourceField = "angle_deg";
    input.value = formatEditorValue(normalizeAngleDeg(state.scene.sources[index].angle_deg));
    input.setAttribute(
      "aria-label",
      `${readableSourceName(state.scene.sources[index], index)} 方向角（度，从 +x 朝 +y 逆时针）`,
    );
    input.addEventListener("focus", () => {
      state.selectedSource = index;
      render();
    });
    input.addEventListener("change", () => {
      const angle = normalizeAngleDeg(input.value);
      input.value = formatEditorValue(angle);
      updateSource(index, "angle_deg", angle);
      scheduleLoad(360);
    });
    label.append(angleLabel, input);
    return label;
  }

  function removeSourceButton(index) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "remove-source";
    button.textContent = "删除";
    button.disabled = !canRemoveSource(
      elements.preset.value,
      state.scene.sources,
      index,
    );
    button.setAttribute(
      "aria-label",
      `删除${readableSourceName(state.scene.sources[index], index)}`,
    );
    button.addEventListener("click", () => removeSource(index));
    return button;
  }

  // The moving source's coordinate inputs follow the pointer; the row is
  // rebuilt only when the drag ends.
  function syncSourceEditorInputs(index) {
    const row = elements.sourceEditorList.querySelectorAll(".source-editor")[index];
    if (!row) return;
    const source = state.scene.sources[index];
    row.querySelectorAll("input[data-source-field]").forEach((input) => {
      const field = input.dataset.sourceField;
      if (field === "x" || field === "y") input.value = formatEditorValue(source[field]);
    });
  }

  function syncSourceOverrides() {
    state.sourceOverrides = state.scene.sources.map(serializeSource);
  }

  function updateSource(index, axis, value) {
    if (!sourcesAreEditable()) return;
    state.scene.sources[index][axis] = value;
    syncSourceOverrides();
    markSceneStale();
  }

  function addSource(kind) {
    if (!state.scene || !sourcesAreEditable()) return;
    if (state.scene.sources.length >= SOURCE_COUNT_LIMIT) return;
    state.scene.sources.push(
      createSource(
        kind,
        state.scene.sources,
        elements.preset.value,
        selectedSourceSeparation(),
      ),
    );
    syncSourceOverrides();
    elements.sourceStatus.textContent = `已添加场源，当前 ${state.sourceOverrides.length} 个。`;
    loadScene();
  }

  function removeSource(index) {
    if (!state.scene || !sourcesAreEditable()) return;
    if (!canRemoveSource(elements.preset.value, state.scene.sources, index)) return;
    state.scene.sources.splice(index, 1);
    syncSourceOverrides();
    elements.sourceStatus.textContent = `已删除场源，当前 ${state.sourceOverrides.length} 个。`;
    loadScene();
  }

  function pointerPosition(event) {
    const rect = elements.canvas.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }

  function sourceAt(canvasX, canvasY) {
    if (!state.scene || !sourcesDraggable()) return null;
    let closest = null;
    let closestDistance = 18;
    state.scene.sources.forEach((source, index) => {
      const [sourceX, sourceY] = worldToCanvas(source.x, source.y);
      const distance = Math.hypot(canvasX - sourceX, canvasY - sourceY);
      if (distance <= closestDistance) {
        closest = index;
        closestDistance = distance;
      }
    });
    return closest;
  }

  function insidePlot(canvasX, canvasY) {
    if (!state.plotRect) return false;
    const { left, right, top, bottom } = state.plotRect;
    return canvasX >= left && canvasX <= right && canvasY >= top && canvasY <= bottom;
  }

  function handlePointerDown(event) {
    if (!state.scene) return;
    const [canvasX, canvasY] = pointerPosition(event);
    const sourceIndex = sourceAt(canvasX, canvasY);
    if (sourceIndex === null) {
      // A click on the field (down and up in the same place) pins a seed.
      state.pendingSeed =
        state.presentationStatus === "ready" && insidePlot(canvasX, canvasY)
          ? { pointerId: event.pointerId, canvasX, canvasY }
          : null;
      return;
    }
    const needsRequest = state.presentationStatus === "stale";
    loader.cancelScheduled();
    state.selectedSource = sourceIndex;
    state.drag = {pointerId: event.pointerId, sourceIndex, moved: false, needsRequest};
    elements.canvas.setPointerCapture(event.pointerId);
    elements.canvas.dataset.dragging = "true";
    elements.probe.hidden = true;
    elements.probeNode.hidden = true;
    render();
    event.preventDefault();
  }

  function handlePointerMove(event) {
    if (!state.scene) return;
    const [canvasX, canvasY] = pointerPosition(event);
    if (state.drag?.pointerId === event.pointerId) {
      const [worldX, worldY] = canvasToWorld(canvasX, canvasY);
      const source = state.scene.sources[state.drag.sourceIndex];
      const position = snapSourcePosition(
        elements.preset.value,
        state.scene.sources,
        state.drag.sourceIndex,
        {x: worldX, y: worldY},
        selectedSourceSeparation(),
        sourcePositionBounds(),
      );
      if (!position) return;
      const moved = position.x !== source.x || position.y !== source.y;
      if (!moved) return;
      state.drag.moved = true;
      source.x = position.x;
      source.y = position.y;
      syncSourceOverrides();
      syncSourceEditorInputs(state.drag.sourceIndex);
      if (position.snapped) {
        elements.sourceStatus.textContent = `已按大于 ${formatEditorValue(selectedSourceSeparation())} m 的场源间距吸附。`;
      }
      markSceneStale();
      return;
    }
    updateProbe(canvasX, canvasY);
    updatePreview(canvasX, canvasY);
  }

  // The cursor bundle is recomputed from the vector grid on every move and
  // drawn on the next frame; it is display state and never sends a request.
  function updatePreview(canvasX, canvasY) {
    const available =
      state.layers.preview &&
      state.previewGrid &&
      state.presentationStatus === "ready" &&
      insidePlot(canvasX, canvasY);
    if (!available) {
      clearPreview();
      return;
    }
    const [worldX, worldY] = canvasToWorld(canvasX, canvasY);
    const { domain } = state.scene;
    const spacing =
      (PREVIEW_SPACING_PX * (domain.x[1] - domain.x[0])) / (state.plotRect.right - state.plotRect.left);
    state.preview = previewBundle(state.previewGrid, [worldX, worldY], { spacing });
    schedulePreviewFrame();
  }

  function clearPreview() {
    if (!state.preview) return;
    state.preview = null;
    schedulePreviewFrame();
  }

  function schedulePreviewFrame() {
    window.cancelAnimationFrame(state.previewFrame);
    state.previewFrame = window.requestAnimationFrame(() => {
      state.previewFrame = null;
      render();
    });
  }

  function pinSeed(canvasX, canvasY) {
    if (state.presentationStatus !== "ready" || !insidePlot(canvasX, canvasY)) return;
    if (state.seeds.length >= SEED_COUNT_LIMIT) {
      elements.seedStatus.textContent = `最多 ${SEED_COUNT_LIMIT} 个手动种子；请先清除再添加。`;
      elements.liveStatus.textContent = elements.seedStatus.textContent;
      return;
    }
    const [worldX, worldY] = canvasToWorld(canvasX, canvasY);
    const { domain } = state.scene;
    // Pointer positions carry about a hundredth of a metre; keep four decimals.
    const round = (value) => Math.round(value * 1e4) / 1e4;
    const seed = {
      x: round(clamp(worldX, domain.x[0], domain.x[1])),
      y: round(clamp(worldY, domain.y[0], domain.y[1])),
    };
    state.seeds.push(seed);
    renderSeedStatus();
    elements.liveStatus.textContent = `已在 x ${formatEditorValue(seed.x)}，y ${formatEditorValue(seed.y)} 固定手动种子，正在请求服务端精确场线。`;
    loadScene();
  }

  function clearSeeds() {
    if (!state.seeds.length) return;
    state.seeds = [];
    renderSeedStatus();
    elements.liveStatus.textContent = "已清除手动种子，正在重新计算。";
    loadScene();
  }

  function finishPointerDrag(event) {
    if (state.pendingSeed?.pointerId === event.pointerId) {
      const pending = state.pendingSeed;
      state.pendingSeed = null;
      if (event.type === "pointerup") {
        const [canvasX, canvasY] = pointerPosition(event);
        const travelled = Math.hypot(canvasX - pending.canvasX, canvasY - pending.canvasY);
        if (travelled <= CLICK_TOLERANCE_PX) pinSeed(canvasX, canvasY);
      }
      return;
    }
    if (!state.drag || state.drag.pointerId !== event.pointerId) return;
    const index = state.drag.sourceIndex;
    const moved = state.drag.moved;
    const needsRequest = state.drag.needsRequest;
    state.drag = null;
    delete elements.canvas.dataset.dragging;
    if (elements.canvas.hasPointerCapture(event.pointerId)) {
      elements.canvas.releasePointerCapture(event.pointerId);
    }
    renderSourceEditors();
    if (!moved) {
      if (needsRequest) {
        loadScene();
        return;
      }
      render();
      return;
    }
    const source = state.scene.sources[index];
    elements.liveStatus.textContent = `${readableSourceName(source, index)}移动到 x ${formatEditorValue(source.x)}，y ${formatEditorValue(source.y)}，正在重新计算。`;
    loadScene();
  }

  function updateProbe(canvasX, canvasY) {
    if (state.presentationStatus !== "ready") {
      elements.probe.hidden = true;
    elements.probeNode.hidden = true;
      return;
    }
    const { left, right, top, bottom } = state.plotRect;
    if (canvasX < left || canvasX > right || canvasY < top || canvasY > bottom) {
      elements.probe.hidden = true;
    elements.probeNode.hidden = true;
      return;
    }
    const [worldX, worldY] = canvasToWorld(canvasX, canvasY);
    const node = nearestNode(state.scene.scalar, state.scene.domain, worldX, worldY);
    const coordinateUnit = String(state.scene.domain.unit || "").trim();
    const unitSuffix = coordinateUnit === "1" ? "" : coordinateUnit ? ` ${coordinateUnit}` : "";
    const verticalAxis = state.scene.electromagnetic?.vertical_axis || "y";
    elements.probePosition.textContent = `x ${formatEditorValue(node.x)}${unitSuffix} · ${verticalAxis} ${formatEditorValue(node.y)}${unitSuffix}`;
    elements.probeValue.textContent = `${state.scene.scalar.label || "场强"} ${formatValue(node.value)} ${state.scene.scalar.unit === "1" ? "（无量纲）" : state.scene.scalar.unit || ""}`.trim();
    elements.probeDirection.textContent = formatProbeDirection(node);
    elements.probeDirection.hidden = !elements.probeDirection.textContent;
    const [nodeX, nodeY] = worldToCanvas(node.x, node.y);
    elements.probeNode.style.left = `${nodeX}px`;
    elements.probeNode.style.top = `${nodeY}px`;
    elements.probeNode.hidden = false;
    elements.probe.hidden = false;
    placeProbe(canvasX, canvasY);
  }

  // The field direction at the probed node, as the angle from +x toward +y
  // (the convention of dipole angle_deg) and the raw components; empty
  // without a vector grid, and explicit about a true zero vector.
  function formatProbeDirection(node) {
    const { vector, scalar } = state.scene;
    if (!vector || scalar.mask[node.index]) return "";
    const u = Number(vector.u[node.index]);
    const v = Number(vector.v[node.index]);
    if (!(Math.hypot(u, v) > 0)) return `${vector.label} 为零向量，无方向`;
    const degrees = ((Math.atan2(v, u) * 180) / Math.PI + 360) % 360;
    const verticalAxis = state.scene.electromagnetic?.vertical_axis || "y";
    return `方向 ${degrees.toFixed(1)}° · ${vector.label}x ${formatValue(u)} · ${vector.label}${verticalAxis} ${formatValue(v)}`;
  }

  // Up and to the right of the pointer by default; to the left when that
  // would cross the plot's right edge (and the colorbar beside it) or the
  // stage, below when there is no room above, and clamped inside the stage
  // when neither side fits, so the raw reading is never clipped.
  function placeProbe(canvasX, canvasY) {
    const gap = 12;
    const inset = 4;
    const width = elements.probe.offsetWidth;
    const height = elements.probe.offsetHeight;
    const stageWidth = elements.stage.clientWidth;
    const stageHeight = elements.stage.clientHeight;
    const rightLimit =
      elements.stage.dataset.plotLayout === "wide" ? state.plotRect.right : stageWidth - inset;
    let x = canvasX + gap;
    if (x + width > rightLimit) x = canvasX - gap - width;
    x = clamp(x, inset, Math.max(inset, stageWidth - width - inset));
    let y = canvasY - gap - height;
    if (y < inset) y = canvasY + gap;
    y = clamp(y, inset, Math.max(inset, stageHeight - height - inset));
    elements.probe.style.left = `${x}px`;
    elements.probe.style.top = `${y}px`;
  }

  function handleCanvasKeydown(event) {
    if (
      !sourcesDraggable() ||
      state.selectedSource === null ||
      !state.scene?.sources[state.selectedSource]
    ) {
      return;
    }
    const deltas = {
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
      ArrowUp: [0, 1],
      ArrowDown: [0, -1],
    };
    if (!(event.key in deltas)) return;
    event.preventDefault();
    const [xDirection, yDirection] = deltas[event.key];
    const source = state.scene.sources[state.selectedSource];
    const fraction = event.shiftKey ? 0.05 : 0.01;
    const xStep = (state.scene.domain.x[1] - state.scene.domain.x[0]) * fraction;
    const yStep = (state.scene.domain.y[1] - state.scene.domain.y[0]) * fraction;
    const position = snapSourcePosition(
      elements.preset.value,
      state.scene.sources,
      state.selectedSource,
      {
        x: source.x + xDirection * xStep,
        y: source.y + yDirection * yStep,
      },
      selectedSourceSeparation(),
      sourcePositionBounds(),
    );
    if (!position || (position.x === source.x && position.y === source.y)) return;
    source.x = position.x;
    source.y = position.y;
    syncSourceOverrides();
    if (position.snapped) {
      elements.sourceStatus.textContent = `已按大于 ${formatEditorValue(selectedSourceSeparation())} m 的场源间距吸附。`;
    }
    markSceneStale();
    renderSourceEditors();
    scheduleLoad(400);
  }

  elements.form.addEventListener("submit", (event) => event.preventDefault());
  elements.preset.addEventListener("change", () =>
    loadScene({ preserveSources: false, preserveSeeds: false }),
  );
  // The sliders commit on release: `input` only follows the thumb, `change`
  // (once per release, or per keyboard step) requests the scene after a short
  // debounce that merges key repeats; the loader aborts superseded requests.
  const SLIDER_COMMIT_DELAY = 90;
  elements.density.addEventListener("input", () => {
    updateRange(elements.density, elements.densityOutput, (value) => String(value));
  });
  elements.density.addEventListener("change", () => {
    updateRange(elements.density, elements.densityOutput, (value) => String(value));
    scheduleLoad(SLIDER_COMMIT_DELAY);
  });
  elements.resolution.addEventListener("input", () => {
    updateRange(elements.resolution, elements.resolutionOutput, (value) => `${value} × ${value}`);
  });
  elements.resolution.addEventListener("change", () => {
    updateRange(elements.resolution, elements.resolutionOutput, (value) => `${value} × ${value}`);
    scheduleLoad(SLIDER_COMMIT_DELAY);
  });
  [elements.observationTime, elements.initialTime, elements.releaseStart, elements.releaseEnd].forEach((input) => {
    input.addEventListener("change", () => { if (isFlow()) loadScene(); });
  });
  elements.emObservable.addEventListener("change", () => {
    // A different plane changes the meaning of the second seed coordinate.
    if (isElectromagnetic()) loadScene({ preserveSeeds: false });
  });
  elements.emObservationTime.addEventListener("change", () => {
    if (elements.emObservationTime.validity.valid) elements.emTimeSlider.value = elements.emObservationTime.value;
    if (isElectromagnetic()) loadScene();
  });
  elements.emTimeSlider.addEventListener("input", () => {
    elements.emObservationTime.value = elements.emTimeSlider.value;
  });
  elements.emTimeSlider.addEventListener("change", () => {
    elements.emObservationTime.value = elements.emTimeSlider.value;
    if (isElectromagnetic()) scheduleLoad(SLIDER_COMMIT_DELAY);
  });
  elements.runButton.addEventListener("click", () => loadScene());
  elements.retryButton.addEventListener("click", () => loadScene());
  elements.resetSources.addEventListener("click", () => loadScene({ preserveSources: false }));
  elements.clearSeeds.addEventListener("click", clearSeeds);
  elements.addPositiveSource.addEventListener("click", () => addSource("positive"));
  elements.addNegativeSource.addEventListener("click", () => addSource("negative"));
  elements.addDipoleSource.addEventListener("click", () => addSource("dipole"));
  elements.canvas.addEventListener("pointerdown", handlePointerDown);
  elements.canvas.addEventListener("pointermove", handlePointerMove);
  elements.canvas.addEventListener("pointerup", finishPointerDrag);
  elements.canvas.addEventListener("pointercancel", finishPointerDrag);
  elements.canvas.addEventListener("pointerleave", () => {
    if (!state.drag) elements.probe.hidden = true;
    elements.probeNode.hidden = true;
    clearPreview();
  });
  elements.canvas.addEventListener("keydown", handleCanvasKeydown);
  elements.layerToggles.forEach((button) => {
    button.dataset.title = button.title;
    button.addEventListener("click", () => toggleLayer(button.dataset.layer));
  });

  const resizeObserver = new ResizeObserver(() => {
    // The probe was placed for the old geometry; the next pointer move shows it again.
    elements.probe.hidden = true;
    elements.probeNode.hidden = true;
    window.cancelAnimationFrame(state.resizeFrame);
    state.resizeFrame = window.requestAnimationFrame(render);
  });
  resizeObserver.observe(elements.stage);

  // Redraw at the new backing resolution when only the pixel ratio changes,
  // for example after moving the window to a display with another scale.
  function watchPixelRatio() {
    window
      .matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`)
      .addEventListener(
        "change",
        () => {
          render();
          watchPixelRatio();
        },
        { once: true },
      );
  }
  watchPixelRatio();

  document.querySelectorAll(".brand-pole").forEach((pole) => {
    pole.style.fill = CANVAS_THEME.marker.fill[pole.dataset.pole];
  });
  const [bottomColor, topColor] = [PALETTE[0][1], PALETTE.at(-1)[1]];
  elements.colorbar.style.setProperty("--colormap-bottom", `rgb(${bottomColor.join(", ")})`);
  elements.colorbar.style.setProperty("--colormap-top", `rgb(${topColor.join(", ")})`);
  elements.colorbar.style.setProperty("--colormap-vertical", paletteCssGradient("to top"));
  elements.colorbar.style.setProperty("--colormap-horizontal", paletteCssGradient("to right"));
  updateRange(elements.density, elements.densityOutput, (value) => String(value));
  updateRange(elements.resolution, elements.resolutionOutput, (value) => `${value} × ${value}`);
  loadScene({preserveSources: false});
})();
