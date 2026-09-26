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
  sampleNearest as sampleScalarNearest,
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
import { createRenderer, sourceStyle } from "./renderer.js";
import { createSceneLoader } from "./scene-loader.js";
import { REGION_PRESETS, SEED_MODE_LABELS, validateScene } from "./scene-validation.js";

(() => {
  "use strict";

  const API_URL = "/api/scene";
  const PRESETS_URL = "/api/presets";
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
    legendUncolored: document.querySelector("#legend-uncolored"),
    legendSources: document.querySelector('.legend-chip[data-layer="sources"]'),
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

  async function loadPresetCapabilities() {
    const response = await fetch(PRESETS_URL, {headers: {Accept: "application/json"}});
    if (!response.ok) throw new Error(await extractError(response));
    state.presetCapabilities = validatePresetCapabilities(await response.json());
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
    if (serializedSources) {
      body.sources = serializedSources;
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
    elements.renderedLineCount.textContent = "—";
    elements.suppressedCount.textContent = "—";
    elements.colorbar.hidden = true;
    elements.probe.hidden = true;
    renderSourceEditors();
    render();
  }

  function markSceneStale(message = "场源位置已改变，场待重算。") {
    if (!state.scene) return;
    state.presentationStatus = "stale";
    state.scale = null;
    elements.colorbar.hidden = true;
    elements.probe.hidden = true;
    elements.scaleBadge.textContent = "场待重算";
    elements.lineCount.textContent = "—";
    elements.terminationCounts.textContent = "待重算";
    elements.startTerminationCounts.textContent = "待重算";
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
    return validateScene(await response.json(), requestBody.preset);
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
      renderSourceEditors();
      updateSceneDetails();
      render();
      setConnectionStatus("ready", "已同步");
      elements.liveStatus.textContent = `${scene.metadata.title || "场景"}已加载，渲染 ${scene.metadata.rendered_line_count} 条场线，抑制 ${scene.metadata.suppressed_count} 次尝试。`;
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

  async function loadScene({ preserveSources = true } = {}) {
    if (!preserveSources) {
      state.sourceOverrides = null;
      elements.sourceStatus.textContent = "";
    }
    await loader.load(currentRequestBody());
  }

  function scheduleLoad(delay = 260) {
    loader.schedule(() => loadScene(), delay);
  }

  const TERMINATION_LABELS = Object.freeze({
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
    elements.fieldUnit.textContent = unit;
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
    elements.renderedLineCount.textContent =
      scene.metadata.rendered_line_count.toLocaleString("zh-CN");
    elements.suppressedCount.textContent =
      scene.metadata.suppressed_count.toLocaleString("zh-CN");

    const scale = resolveScale(scalar);
    elements.colorbar.hidden = false;
    elements.colorbarMax.textContent = formatValue(scale.maximum);
    elements.colorbarMin.textContent = formatValue(scale.minimum);
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
    elements.captionSeed.textContent = `播种：${
      SEED_MODE_LABELS[scene.metadata.seed_mode] || scene.metadata.seed_mode
    } · 渲染 ${scene.metadata.rendered_line_count} / 抑制 ${scene.metadata.suppressed_count}`;
    elements.colorbarLabel.textContent = quantityWithUnit(scalar.label || "场强", scalar.unit);
    const draggableSourceCount = sourcesAreEditable() ? scene.sources.length : 0;
    elements.canvas.dataset.draggable = String(draggableSourceCount > 0);
    elements.canvas.setAttribute(
      "aria-label",
      `${title}二维可视化，共 ${scene.metadata.rendered_line_count} 条场线、${draggableSourceCount} 个可移动场源。`,
    );
  }

  // "quantity / unit", bracketing compound units: |E| / (V/m), |B| / T.
  function quantityWithUnit(quantity, unit) {
    const trimmed = String(unit || "").trim();
    if (!trimmed) return quantity;
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
      charged_ring: "带电圆环电场",
      dielectric_sphere: "介质球电场",
      conducting_sphere: "导体球电场",
      uniform: "匀强场",
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
    });
    if (state.presentationStatus === "stale") return;
    elements.legendUncolored.hidden = !outcome.hatchVisible;
    elements.legendSources.hidden = state.scene.sources.length === 0;
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
      : "固定预设只显示只读几何标记。";
    elements.interactionHelp.textContent = editable
      ? "拖动场源改变位置；移动指针可探测坐标与场强。"
      : "移动指针可探测坐标与场强；固定标记不可移动。";
    if (!editable && sources.length) {
      const fixed = document.createElement("p");
      fixed.className = "fixed-sources-note";
      fixed.textContent =
        preset === "charged_ring"
          ? "两个标记是同一带电圆环的截面位置，电荷与位置由预设固定、不可移动。"
          : "两个标记是同一圆环的截面位置，电流与位置由预设固定、不可移动。";
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
    if (!state.scene || !sourcesAreEditable()) return null;
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

  function handlePointerDown(event) {
    if (!state.scene || !state.scene.sources.length || !sourcesAreEditable()) return;
    const [canvasX, canvasY] = pointerPosition(event);
    const sourceIndex = sourceAt(canvasX, canvasY);
    if (sourceIndex === null) return;
    const needsRequest = state.presentationStatus === "stale";
    loader.cancelScheduled();
    state.selectedSource = sourceIndex;
    state.drag = {pointerId: event.pointerId, sourceIndex, moved: false, needsRequest};
    elements.canvas.setPointerCapture(event.pointerId);
    elements.canvas.dataset.dragging = "true";
    elements.probe.hidden = true;
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
      if (position.snapped) {
        elements.sourceStatus.textContent = `已按大于 ${formatEditorValue(selectedSourceSeparation())} m 的场源间距吸附。`;
      }
      markSceneStale();
      return;
    }
    updateProbe(canvasX, canvasY);
  }

  function finishPointerDrag(event) {
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
      return;
    }
    const { left, right, top, bottom } = state.plotRect;
    if (canvasX < left || canvasX > right || canvasY < top || canvasY > bottom) {
      elements.probe.hidden = true;
      return;
    }
    const [worldX, worldY] = canvasToWorld(canvasX, canvasY);
    const value = sampleNearest(worldX, worldY);
    const coordinateUnit = String(state.scene.domain.unit || "").trim();
    const unitSuffix = coordinateUnit ? ` ${coordinateUnit}` : "";
    elements.probePosition.textContent = `x ${formatEditorValue(worldX)}${unitSuffix} · y ${formatEditorValue(worldY)}${unitSuffix}`;
    elements.probeValue.textContent = `${state.scene.scalar.label || "场强"} ${formatValue(value)} ${state.scene.scalar.unit || ""}`.trim();
    elements.probe.hidden = false;
    placeProbe(canvasX, canvasY);
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

  function sampleNearest(x, y) {
    const { scalar, domain } = state.scene;
    return sampleScalarNearest(scalar, domain, x, y);
  }

  function handleCanvasKeydown(event) {
    if (
      !sourcesAreEditable() ||
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

  async function bootstrap() {
    setLoading(true);
    clearError();
    try {
      await loadPresetCapabilities();
      await loadScene({preserveSources: false});
    } catch (error) {
      invalidateSceneView("error");
      setError(error.message || "无法读取预设能力", {
        unreachable: error instanceof TypeError,
      });
      setLoading(false);
    }
  }

  elements.form.addEventListener("submit", (event) => event.preventDefault());
  elements.preset.addEventListener("change", () => loadScene({ preserveSources: false }));
  elements.density.addEventListener("input", () => {
    updateRange(elements.density, elements.densityOutput, (value) => String(value));
    scheduleLoad();
  });
  elements.resolution.addEventListener("input", () => {
    updateRange(elements.resolution, elements.resolutionOutput, (value) => `${value} × ${value}`);
    scheduleLoad();
  });
  elements.runButton.addEventListener("click", () => loadScene());
  elements.retryButton.addEventListener("click", () => loadScene());
  elements.resetSources.addEventListener("click", () => loadScene({ preserveSources: false }));
  elements.addPositiveSource.addEventListener("click", () => addSource("positive"));
  elements.addNegativeSource.addEventListener("click", () => addSource("negative"));
  elements.addDipoleSource.addEventListener("click", () => addSource("dipole"));
  elements.canvas.addEventListener("pointerdown", handlePointerDown);
  elements.canvas.addEventListener("pointermove", handlePointerMove);
  elements.canvas.addEventListener("pointerup", finishPointerDrag);
  elements.canvas.addEventListener("pointercancel", finishPointerDrag);
  elements.canvas.addEventListener("pointerleave", () => {
    if (!state.drag) elements.probe.hidden = true;
  });
  elements.canvas.addEventListener("keydown", handleCanvasKeydown);

  const resizeObserver = new ResizeObserver(() => {
    // The probe was placed for the old geometry; the next pointer move shows it again.
    elements.probe.hidden = true;
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
  bootstrap();
})();
