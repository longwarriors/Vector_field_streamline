import { colorForScalar, getScaleType, resolveScale } from "./color-scale.js";
import {
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
  effectiveDipoleAngleDeg,
  normalizeAngleDeg,
  seedingSourceCount,
  serializeSource,
  snapSourcePosition,
  sourceSeparationConflict,
} from "./source-controls.js";

(() => {
  "use strict";

  const API_URL = "/api/scene";
  const PRESETS_URL = "/api/presets";
  const SOURCE_STRENGTH_UNITS = Object.freeze({
    positive: "nC",
    negative: "nC",
    dipole: "A·m²",
    wire_out: "A",
    wire_into: "A",
  });
  const EDITABLE_SOURCE_PRESETS = new Set([
    "electric_dipole",
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
    lineCount: document.querySelector("#line-count"),
    gridSize: document.querySelector("#grid-size"),
    fieldUnit: document.querySelector("#field-unit"),
    fieldModel: document.querySelector("#field-model"),
    seedMode: document.querySelector("#seed-mode"),
    terminationCounts: document.querySelector("#termination-counts"),
    probe: document.querySelector("#probe"),
    probePosition: document.querySelector("#probe-position"),
    probeValue: document.querySelector("#probe-value"),
    liveStatus: document.querySelector("#live-status"),
  };

  const context = elements.canvas.getContext("2d", { alpha: false });
  const state = {
    scene: null,
    sourceOverrides: null,
    requestController: null,
    requestSequence: 0,
    debounceTimer: null,
    resizeFrame: null,
    drag: null,
    selectedSource: null,
    plotRect: null,
    transform: null,
    scale: null,
    presentationStatus: "idle",
    presetCapabilities: new Map(),
  };

  function finiteNumber(value, fallback = 0) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

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

  function setError(message) {
    elements.errorMessage.textContent = message;
    elements.errorBanner.hidden = false;
    setConnectionStatus("error", "连接失败");
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

  function validateScene(scene) {
    if (!scene || typeof scene !== "object") throw new Error("响应不是有效的场景对象");
    const xDomain = scene.domain?.x;
    const yDomain = scene.domain?.y;
    const coordinateSystem = scene.domain?.coordinate_system;
    const coordinateUnit = scene.domain?.unit;
    if (
      !Array.isArray(xDomain) ||
      !Array.isArray(yDomain) ||
      xDomain.length !== 2 ||
      yDomain.length !== 2 ||
      !xDomain.every(Number.isFinite) ||
      !yDomain.every(Number.isFinite) ||
      xDomain[0] >= xDomain[1] ||
      yDomain[0] >= yDomain[1] ||
      coordinateSystem !== "cartesian" ||
      coordinateUnit !== "m"
    ) {
      throw new Error("场景缺少有效的笛卡尔 domain.x / domain.y / unit");
    }

    const nx = Number(scene.scalar?.nx);
    const ny = Number(scene.scalar?.ny);
    const values = scene.scalar?.values;
    const mask = scene.scalar?.mask;
    if (!Number.isInteger(nx) || !Number.isInteger(ny) || nx < 2 || ny < 2) {
      throw new Error("标量网格尺寸无效");
    }
    if (!Array.isArray(values) || values.length !== nx * ny) {
      throw new Error(`标量网格应包含 ${nx * ny} 个值`);
    }
    if (!Array.isArray(mask) || mask.length !== nx * ny) {
      throw new Error(`标量遮罩应包含 ${nx * ny} 个布尔值`);
    }
    const invalidScalarIndex = values.findIndex((value, index) => {
      if (typeof mask[index] !== "boolean") return true;
      return mask[index]
        ? value !== null
        : typeof value !== "number" || !Number.isFinite(value);
    });
    if (invalidScalarIndex !== -1) {
      throw new Error(`标量值与遮罩在索引 ${invalidScalarIndex} 未严格配对`);
    }
    if (
      (scene.scalar.scale !== "linear" && scene.scalar.scale !== "log") ||
      typeof scene.scalar.label !== "string" ||
      typeof scene.scalar.unit !== "string" ||
      !Number.isFinite(scene.scalar.vmin) ||
      !Number.isFinite(scene.scalar.vmax) ||
      scene.scalar.vmax <= scene.scalar.vmin ||
      (scene.scalar.scale === "log" && scene.scalar.vmin <= 0)
    ) {
      throw new Error("标量色标元数据无效");
    }

    if (!Array.isArray(scene.lines) || !Array.isArray(scene.sources)) {
      throw new Error("场景 lines 与 sources 必须为数组");
    }
    const validSources = scene.sources.every((source) => {
      if (
        !source ||
        !Number.isFinite(source.x) ||
        !Number.isFinite(source.y) ||
        !Number.isFinite(source.strength) ||
        SOURCE_STRENGTH_UNITS[source.kind] !== source.strength_unit
      ) {
        return false;
      }
      if (source.kind === "dipole") {
        if (source.angle_deg === undefined || source.angle_deg === null) {
          source.angle_deg = 90;
        }
        return (
          Number.isFinite(source.angle_deg) &&
          source.angle_deg >= 0 &&
          source.angle_deg < 360
        );
      }
      if (source.angle_deg !== undefined && source.angle_deg !== null) return false;
      if (source.kind === "positive") return source.strength > 0;
      if (source.kind === "negative") return source.strength < 0;
      return (
        (source.kind === "wire_out" || source.kind === "wire_into") &&
        source.strength >= 0
      );
    });
    const wireSources = scene.sources.filter(
      ({ kind }) => kind === "wire_out" || kind === "wire_into",
    );
    const expectsLoopMarkers = elements.preset.value === "current_loop";
    const validLoopMarkers = expectsLoopMarkers
      ? wireSources.length === 2 &&
        scene.sources.length === 2 &&
        wireSources.some(({ kind }) => kind === "wire_out") &&
        wireSources.some(({ kind }) => kind === "wire_into") &&
        wireSources[0].strength === wireSources[1].strength
      : wireSources.length === 0;
    if (!validSources || !validLoopMarkers) {
      throw new Error("场源缺少有效坐标、强度或 strength_unit");
    }
    const metadata = scene.metadata;
    const terminationCounts = metadata?.termination_counts;
    if (
      !metadata ||
      typeof metadata !== "object" ||
      Array.isArray(metadata) ||
      typeof metadata.title !== "string" ||
      typeof metadata.projection_note !== "string" ||
      typeof metadata.field_model !== "string" ||
      typeof metadata.seed_mode !== "string" ||
      !terminationCounts ||
      typeof terminationCounts !== "object" ||
      Array.isArray(terminationCounts) ||
      !Object.values(terminationCounts).every(
        (count) => Number.isInteger(count) && count >= 0,
      )
    ) {
      throw new Error("场景 metadata 缺少有效的模型、播种或终止统计");
    }
    return scene;
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
    elements.terminationCounts.textContent = "—";
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
    elements.canvas.setAttribute(
      "aria-label",
      "场源位置已改变；旧数值场已隐藏，等待重新计算。",
    );
    setConnectionStatus("loading", "待重算");
    elements.liveStatus.textContent = message;
    render();
  }

  async function loadScene({ preserveSources = true } = {}) {
    window.clearTimeout(state.debounceTimer);
    state.debounceTimer = null;
    if (!preserveSources) {
      state.sourceOverrides = null;
      elements.sourceStatus.textContent = "";
    }
    state.requestController?.abort();
    const controller = new AbortController();
    const sequence = ++state.requestSequence;
    state.requestController = controller;
    const requestBody = currentRequestBody();

    setLoading(true);
    clearError();
    invalidateSceneView("loading");
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(await extractError(response));
      const scene = validateScene(await response.json());
      if (sequence !== state.requestSequence) return;

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
      elements.liveStatus.textContent = `${scene.metadata.title || "场景"}已加载，共 ${scene.lines.length} 条场线。`;
    } catch (error) {
      if (error.name !== "AbortError" && sequence === state.requestSequence) {
        invalidateSceneView("error");
        setError(error.message || "无法连接场景计算服务");
      }
    } finally {
      if (sequence === state.requestSequence) {
        setLoading(false);
        state.requestController = null;
      }
    }
  }

  function scheduleLoad(delay = 260) {
    window.clearTimeout(state.debounceTimer);
    state.debounceTimer = window.setTimeout(() => {
      state.debounceTimer = null;
      loadScene();
    }, delay);
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
    elements.lineCount.textContent = scene.lines.length.toLocaleString("zh-CN");
    elements.gridSize.textContent = `${scalar.nx}²`.replace(
      `${scalar.nx}²`,
      scalar.nx === scalar.ny ? `${scalar.nx}²` : `${scalar.nx}×${scalar.ny}`,
    );
    elements.fieldUnit.textContent = unit;
    elements.fieldModel.textContent = scene.metadata.field_model;
    elements.seedMode.textContent = scene.metadata.seed_mode;
    elements.terminationCounts.textContent = formatTerminationCounts(
      scene.metadata.termination_counts,
    );

    const scale = resolveScale(scalar);
    elements.colorbar.hidden = false;
    elements.colorbarMax.textContent = formatValue(scale.maximum);
    elements.colorbarMin.textContent = formatValue(scale.minimum);
    elements.colorbarLabel.textContent = [scalar.label || "场强", unit].filter(Boolean).join(" · ");
    const draggableSourceCount = sourcesAreEditable() ? scene.sources.length : 0;
    elements.canvas.dataset.draggable = String(draggableSourceCount > 0);
    elements.canvas.setAttribute(
      "aria-label",
      `${title}二维可视化，共 ${scene.lines.length} 条场线、${draggableSourceCount} 个可移动场源。`,
    );
  }

  function presetLabel(value) {
    return {
      electric_dipole: "电偶极子场",
      magnetic_dipole: "磁偶极子场",
      halbach_array: "Halbach 阵列磁场",
      current_loop: "圆形电流线圈磁场",
      uniform: "匀强场",
    }[value] || "物理场";
  }

  function resizeCanvas() {
    const rect = elements.canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2.5);
    const width = Math.max(1, Math.round(rect.width * ratio));
    const height = Math.max(1, Math.round(rect.height * ratio));
    if (elements.canvas.width !== width || elements.canvas.height !== height) {
      elements.canvas.width = width;
      elements.canvas.height = height;
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { width: rect.width, height: rect.height };
  }

  function worldToCanvas(x, y) {
    return state.transform.worldToCanvas(x, y);
  }

  function canvasToWorld(canvasX, canvasY) {
    return state.transform.canvasToWorld(canvasX, canvasY);
  }

  function render() {
    const size = resizeCanvas();
    context.clearRect(0, 0, size.width, size.height);
    context.fillStyle = "#07111a";
    context.fillRect(0, 0, size.width, size.height);
    elements.canvas.dataset.sceneState = state.presentationStatus;
    if (!state.scene) return;

    state.plotRect = calculatePlotRect(size.width, size.height, state.scene.domain);
    state.transform = createCoordinateTransform(state.scene.domain, state.plotRect);
    if (state.presentationStatus === "stale") {
      drawGridAndAxes();
      drawSources();
      return;
    }
    state.scale = resolveScale(state.scene.scalar);
    drawHeatmap();
    drawGridAndAxes();
    drawStreamlines();
    drawSources();
  }

  function drawHeatmap() {
    const { scalar } = state.scene;
    const offscreen = document.createElement("canvas");
    offscreen.width = scalar.nx;
    offscreen.height = scalar.ny;
    const offscreenContext = offscreen.getContext("2d");
    const image = offscreenContext.createImageData(scalar.nx, scalar.ny);

    // The API uses row-major values, with y descending from ymax to ymin.
    for (let index = 0; index < scalar.values.length; index += 1) {
      const color = colorForScalar(scalar.values[index], Boolean(scalar.mask?.[index]), state.scale);
      const offset = index * 4;
      image.data[offset] = color[0];
      image.data[offset + 1] = color[1];
      image.data[offset + 2] = color[2];
      image.data[offset + 3] = color[3];
    }
    offscreenContext.putImageData(image, 0, 0);

    const { left, top, right, bottom } = state.plotRect;
    context.save();
    context.imageSmoothingEnabled = true;
    context.drawImage(offscreen, left, top, right - left, bottom - top);
    context.restore();
  }

  function niceTicks(minimum, maximum, count = 5) {
    const span = maximum - minimum;
    if (!Number.isFinite(span) || span <= 0) return [minimum];
    const rough = span / count;
    const magnitude = 10 ** Math.floor(Math.log10(rough));
    const normalized = rough / magnitude;
    const step = (normalized < 1.5 ? 1 : normalized < 3 ? 2 : normalized < 7 ? 5 : 10) * magnitude;
    const ticks = [];
    for (let value = Math.ceil(minimum / step) * step; value <= maximum + step * 1e-9; value += step) {
      ticks.push(Math.abs(value) < step * 1e-9 ? 0 : value);
    }
    return ticks;
  }

  function drawGridAndAxes() {
    const { left, top, right, bottom } = state.plotRect;
    const [xmin, xmax] = state.scene.domain.x;
    const [ymin, ymax] = state.scene.domain.y;
    context.save();
    context.lineWidth = 1;
    context.font = "10px ui-sans-serif, system-ui, sans-serif";
    context.fillStyle = "rgba(222, 238, 241, 0.65)";
    context.strokeStyle = "rgba(228, 248, 247, 0.12)";

    niceTicks(xmin, xmax).forEach((tick) => {
      const [x] = worldToCanvas(tick, ymin);
      context.beginPath();
      context.moveTo(x, top);
      context.lineTo(x, bottom);
      context.stroke();
      context.textAlign = "center";
      context.textBaseline = "top";
      context.fillText(formatAxisValue(tick), x, bottom + 8);
    });

    niceTicks(ymin, ymax).forEach((tick) => {
      const [, y] = worldToCanvas(xmin, tick);
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(right, y);
      context.stroke();
      context.textAlign = "right";
      context.textBaseline = "middle";
      context.fillText(formatAxisValue(tick), left - 7, y);
    });

    context.strokeStyle = "rgba(235, 250, 250, 0.3)";
    context.strokeRect(left, top, right - left, bottom - top);
    context.restore();
  }

  function validPoints(line) {
    if (!Array.isArray(line?.points)) return [];
    return line.points
      .filter((point) => Array.isArray(point) && point.length >= 2)
      .map((point) => [Number(point[0]), Number(point[1])])
      .filter((point) => point.every(Number.isFinite));
  }

  function drawStreamlines() {
    const lines = state.scene.lines;
    context.save();
    context.lineJoin = "round";
    context.lineCap = "round";

    for (const line of lines) {
      const points = state.transform.projectPoints(validPoints(line));
      if (points.length < 2) continue;
      tracePath(points);
      context.strokeStyle = "rgba(0, 8, 12, 0.5)";
      context.lineWidth = 3.4;
      context.stroke();
      tracePath(points);
      context.strokeStyle = "rgba(248, 255, 253, 0.88)";
      context.lineWidth = 1.15;
      context.stroke();
      drawDirectionArrows(points, line.direction);
    }
    context.restore();
  }

  function tracePath(points) {
    context.beginPath();
    context.moveTo(points[0][0], points[0][1]);
    for (let index = 1; index < points.length; index += 1) {
      context.lineTo(points[index][0], points[index][1]);
    }
  }

  function directionSign(direction) {
    if (typeof direction === "number") return direction < 0 ? -1 : 1;
    const normalized = String(direction || "forward").toLowerCase();
    return normalized.includes("back") || normalized === "-" ? -1 : 1;
  }

  function drawDirectionArrows(points, direction) {
    const segments = [];
    let total = 0;
    for (let index = 1; index < points.length; index += 1) {
      const dx = points[index][0] - points[index - 1][0];
      const dy = points[index][1] - points[index - 1][1];
      const length = Math.hypot(dx, dy);
      if (length > 0) {
        segments.push({ from: points[index - 1], to: points[index], length, start: total });
        total += length;
      }
    }
    if (total < 28) return;

    const arrowCount = clamp(Math.floor(total / 130), 1, 3);
    const sign = directionSign(direction);
    context.fillStyle = "rgba(255, 255, 255, 0.94)";
    context.strokeStyle = "rgba(0, 8, 12, 0.58)";
    context.lineWidth = 2.6;
    for (let arrowIndex = 1; arrowIndex <= arrowCount; arrowIndex += 1) {
      const target = (total * arrowIndex) / (arrowCount + 1);
      const segment = segments.find((candidate) => candidate.start + candidate.length >= target);
      if (!segment) continue;
      const fraction = (target - segment.start) / segment.length;
      const x = segment.from[0] + (segment.to[0] - segment.from[0]) * fraction;
      const y = segment.from[1] + (segment.to[1] - segment.from[1]) * fraction;
      const ux = ((segment.to[0] - segment.from[0]) / segment.length) * sign;
      const uy = ((segment.to[1] - segment.from[1]) / segment.length) * sign;
      drawArrowhead(x, y, ux, uy);
    }
  }

  function drawArrowhead(x, y, ux, uy) {
    const length = 8;
    const halfWidth = 3.6;
    const baseX = x - ux * length;
    const baseY = y - uy * length;
    const px = -uy;
    const py = ux;
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(baseX + px * halfWidth, baseY + py * halfWidth);
    context.lineTo(baseX - px * halfWidth, baseY - py * halfWidth);
    context.closePath();
    context.stroke();
    context.fill();
  }

  function sourceStyle(source) {
    const kind = String(source.kind || "").toLowerCase();
    if (kind === "wire_out") {
      return { fill: "#ffb45f", symbol: "⊙", className: "wire" };
    }
    if (kind === "wire_into") {
      return { fill: "#ffb45f", symbol: "⊗", className: "wire" };
    }
    if (kind.includes("dipole") || kind.includes("magnet")) {
      return {
        fill: "#ffe08a",
        symbol: "→",
        className: "neutral",
        rotation: (-effectiveDipoleAngleDeg(source) * Math.PI) / 180,
      };
    }
    if (kind.includes("uniform")) {
      return { fill: "#59e1c1", symbol: "→", className: "neutral" };
    }
    if (kind === "positive") return { fill: "#ff725f", symbol: "+", className: "positive" };
    if (kind === "negative") return { fill: "#65b9ff", symbol: "−", className: "negative" };
    if (source.strength > 0) return { fill: "#ff725f", symbol: "+", className: "positive" };
    if (source.strength < 0) return { fill: "#65b9ff", symbol: "−", className: "negative" };
    return { fill: "#ffe08a", symbol: "◆", className: "neutral" };
  }

  function drawSources() {
    state.scene.sources.forEach((source, index) => {
      const [x, y] = worldToCanvas(finiteNumber(source.x), finiteNumber(source.y));
      const style = sourceStyle(source);
      context.save();
      context.shadowColor = "rgba(0, 0, 0, 0.48)";
      context.shadowBlur = 10;
      context.beginPath();
      context.arc(x, y, 10, 0, Math.PI * 2);
      context.fillStyle = style.fill;
      context.fill();
      context.shadowBlur = 0;
      context.strokeStyle = "rgba(255, 255, 255, 0.94)";
      context.lineWidth = 2;
      context.stroke();
      context.fillStyle = "#061018";
      context.font = "800 13px ui-sans-serif, system-ui, sans-serif";
      context.textAlign = "center";
      context.textBaseline = "middle";
      if (Number.isFinite(style.rotation)) {
        context.translate(x, y);
        context.rotate(style.rotation);
        context.fillText(style.symbol, 0, 0.5);
        context.rotate(-style.rotation);
        context.translate(-x, -y);
      } else {
        context.fillText(style.symbol, x, y + 0.5);
      }
      if (index === state.selectedSource) {
        context.beginPath();
        context.arc(x, y, 16, 0, Math.PI * 2);
        context.strokeStyle = "rgba(89, 225, 193, 0.9)";
        context.lineWidth = 2;
        context.setLineDash([3, 3]);
        context.stroke();
      }
      context.restore();
    });
  }

  function createSourceName(source, index) {
    const name = document.createElement("span");
    name.className = "source-name";
    const swatch = document.createElement("i");
    const style = sourceStyle(source);
    swatch.className = `source-swatch ${style.className}`;
    swatch.setAttribute("aria-hidden", "true");
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
    elements.addPositiveSource.hidden = preset !== "electric_dipole";
    elements.addNegativeSource.hidden = preset !== "electric_dipole";
    elements.addDipoleSource.hidden =
      preset !== "magnetic_dipole" && preset !== "halbach_array";
    elements.addPositiveSource.disabled = sourceLimitReached;
    elements.addNegativeSource.disabled = sourceLimitReached;
    elements.addDipoleSource.disabled = sourceLimitReached;
    elements.sourceHelp.textContent = editable
      ? preset === "electric_dipole"
        ? "可增删电荷；拖动标记或输入坐标。"
        : "可增删磁偶极子并编辑面内方向。"
      : "固定预设只显示只读几何标记。";
    elements.interactionHelp.textContent = editable
      ? "拖动场源改变位置；移动指针可探测坐标与场强。"
      : "移动指针可探测坐标与场强；固定标记不可移动。";
    if (!editable && sources.length) {
      const fixed = document.createElement("p");
      fixed.className = "fixed-sources-note";
      fixed.textContent = "两个标记是同一圆环的截面位置，电流与位置由预设固定、不可移动。";
      elements.sourceEditorList.append(fixed);
      sources.forEach((source, index) => {
        const row = document.createElement("div");
        row.className = "fixed-source";
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
          : "固定几何标记将在场景加载后显示。";
      elements.sourceEditorList.append(fixed);
      return;
    }
    if (!sources.length) {
      const empty = document.createElement("p");
      empty.className = "empty-sources";
      empty.textContent = "此预设没有可移动场源。";
      elements.sourceEditorList.append(empty);
      return;
    }

    sources.forEach((source, index) => {
      const row = document.createElement("div");
      row.className = "source-editor";
      row.append(createSourceName(source, index));
      row.append(coordinateInput(index, "x"), coordinateInput(index, "y"));
      if (source.kind === "dipole") row.append(angleInput(index));
      row.append(removeSourceButton(index));
      elements.sourceEditorList.append(row);
    });
  }

  function readableSourceName(source, index) {
    const kind = String(source.kind || "").toLowerCase();
    if (kind === "wire_out") return "电流出屏";
    if (kind === "wire_into") return "电流入屏";
    if (kind.includes("dipole") || kind.includes("magnet")) {
      return `磁偶极子 ${index + 1}`;
    }
    if (kind === "positive") return `正电荷 ${index + 1}`;
    if (kind === "negative") return `负电荷 ${index + 1}`;
    if (source.strength > 0) return `正电荷 ${index + 1}`;
    if (source.strength < 0) return `负电荷 ${index + 1}`;
    return `场源 ${index + 1}`;
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
    window.clearTimeout(state.debounceTimer);
    state.debounceTimer = null;
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
    elements.probe.style.left = `${canvasX}px`;
    elements.probe.style.top = `${canvasY}px`;
    elements.probe.hidden = false;
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

  function formatValue(value) {
    if (!Number.isFinite(value)) return "—";
    const absolute = Math.abs(value);
    if ((absolute > 0 && absolute < 0.001) || absolute >= 10000) return value.toExponential(2);
    return new Intl.NumberFormat("zh-CN", { maximumSignificantDigits: 4 }).format(value);
  }

  function formatAxisValue(value) {
    if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.01)) {
      return value.toExponential(1);
    }
    return Number(value.toPrecision(3)).toString();
  }

  function formatEditorValue(value) {
    const number = finiteNumber(value);
    return Math.abs(number) >= 1e4 || (Math.abs(number) > 0 && Math.abs(number) < 1e-4)
      ? number.toExponential(4)
      : Number(number.toPrecision(6)).toString();
  }

  async function bootstrap() {
    setLoading(true);
    clearError();
    try {
      await loadPresetCapabilities();
      await loadScene({preserveSources: false});
    } catch (error) {
      invalidateSceneView("error");
      setError(error.message || "无法读取预设能力");
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
    window.cancelAnimationFrame(state.resizeFrame);
    state.resizeFrame = window.requestAnimationFrame(render);
  });
  resizeObserver.observe(elements.stage);

  updateRange(elements.density, elements.densityOutput, (value) => String(value));
  updateRange(elements.resolution, elements.resolutionOutput, (value) => `${value} × ${value}`);
  bootstrap();
})();
