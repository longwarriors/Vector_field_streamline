import { colorForScalar, getScaleType, resolveScale } from "./color-scale.js";
import {
  calculatePlotRect,
  clamp,
  createCoordinateTransform,
  sampleNearest as sampleScalarNearest,
} from "./coordinates.js";

(() => {
  "use strict";

  const API_URL = "/api/scene";

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
    sourceEditorList: document.querySelector("#source-editor-list"),
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

  function currentRequestBody() {
    const body = {
      preset: elements.preset.value,
      density: Number(elements.density.value),
      resolution: Number(elements.resolution.value),
    };
    if (elements.preset.value !== "uniform" && state.sourceOverrides?.length) {
      body.sources = state.sourceOverrides.map(({ x, y, kind, strength }) => ({
        x: finiteNumber(x),
        y: finiteNumber(y),
        kind: String(kind ?? "source"),
        strength: finiteNumber(strength, 1),
      }));
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
    if (
      !Array.isArray(xDomain) ||
      !Array.isArray(yDomain) ||
      xDomain.length !== 2 ||
      yDomain.length !== 2 ||
      !xDomain.every(Number.isFinite) ||
      !yDomain.every(Number.isFinite) ||
      xDomain[0] >= xDomain[1] ||
      yDomain[0] >= yDomain[1]
    ) {
      throw new Error("场景缺少有效的 domain.x / domain.y");
    }

    const nx = Number(scene.scalar?.nx);
    const ny = Number(scene.scalar?.ny);
    const values = scene.scalar?.values;
    if (!Number.isInteger(nx) || !Number.isInteger(ny) || nx < 2 || ny < 2) {
      throw new Error("标量网格尺寸无效");
    }
    if (!Array.isArray(values) || values.length !== nx * ny) {
      throw new Error(`标量网格应包含 ${nx * ny} 个值`);
    }
    if (
      scene.scalar.mask !== undefined &&
      (!Array.isArray(scene.scalar.mask) || scene.scalar.mask.length !== nx * ny)
    ) {
      throw new Error(`标量遮罩应包含 ${nx * ny} 个布尔值`);
    }

    scene.lines = Array.isArray(scene.lines) ? scene.lines : [];
    scene.sources = Array.isArray(scene.sources) ? scene.sources : [];
    scene.metadata = scene.metadata && typeof scene.metadata === "object" ? scene.metadata : {};
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
    elements.colorbar.hidden = true;
    elements.probe.hidden = true;
    renderSourceEditors();
    render();
  }

  async function loadScene({ preserveSources = true } = {}) {
    window.clearTimeout(state.debounceTimer);
    if (!preserveSources) state.sourceOverrides = null;
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
      state.sourceOverrides =
        elements.preset.value === "uniform"
          ? null
          : scene.sources.map((source) => ({
              x: finiteNumber(source.x),
              y: finiteNumber(source.y),
              kind: String(source.kind ?? "source"),
              strength: finiteNumber(source.strength, 1),
            }));
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
    state.debounceTimer = window.setTimeout(() => loadScene(), delay);
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

    const scale = resolveScale(scalar);
    elements.colorbar.hidden = false;
    elements.colorbarMax.textContent = formatValue(scale.maximum);
    elements.colorbarMin.textContent = formatValue(scale.minimum);
    elements.colorbarLabel.textContent = [scalar.label || "场强", unit].filter(Boolean).join(" · ");
    const draggableSourceCount = elements.preset.value === "uniform" ? 0 : scene.sources.length;
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
    if (kind.includes("dipole") || kind.includes("magnet")) {
      return { fill: "#ffe08a", symbol: "↕", className: "neutral" };
    }
    if (kind.includes("uniform")) {
      return { fill: "#59e1c1", symbol: "→", className: "neutral" };
    }
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
      context.fillText(style.symbol, x, y + 0.5);
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

  function renderSourceEditors() {
    elements.sourceEditorList.replaceChildren();
    const sources = state.scene?.sources || [];
    const sourcesAreFixed = elements.preset.value === "uniform";
    elements.resetSources.disabled = sources.length === 0 || sourcesAreFixed;
    if (sourcesAreFixed) {
      const fixed = document.createElement("p");
      fixed.className = "empty-sources";
      fixed.textContent = "匀强场的方向与强度由预设固定。";
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
      const name = document.createElement("span");
      name.className = "source-name";
      const swatch = document.createElement("i");
      const style = sourceStyle(source);
      swatch.className = `source-swatch ${style.className}`;
      swatch.setAttribute("aria-hidden", "true");
      const sourceLabel = document.createElement("span");
      sourceLabel.textContent = readableSourceName(source, index);
      name.append(swatch, sourceLabel);
      row.append(name, coordinateInput(index, "x"), coordinateInput(index, "y"));
      elements.sourceEditorList.append(row);
    });
  }

  function readableSourceName(source, index) {
    const kind = String(source.kind || "").toLowerCase();
    if (kind.includes("dipole") || kind.includes("magnet")) return "磁偶极子";
    if (source.strength > 0) return `正电荷 ${index + 1}`;
    if (source.strength < 0) return `负电荷 ${index + 1}`;
    return `场源 ${index + 1}`;
  }

  function coordinateInput(index, axis) {
    const label = document.createElement("label");
    label.className = "coordinate-field";
    label.append(document.createTextNode(axis));
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.value = formatEditorValue(state.scene.sources[index][axis]);
    input.setAttribute("aria-label", `${readableSourceName(state.scene.sources[index], index)} ${axis} 坐标`);
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
      const domain = state.scene.domain[axis];
      const bounded = clamp(value, domain[0], domain[1]);
      input.value = formatEditorValue(bounded);
      updateSource(index, axis, bounded);
      scheduleLoad(360);
    });
    label.append(input);
    return label;
  }

  function updateSource(index, axis, value) {
    state.scene.sources[index][axis] = value;
    state.sourceOverrides = state.scene.sources.map((source) => ({
      x: finiteNumber(source.x),
      y: finiteNumber(source.y),
      kind: String(source.kind ?? "source"),
      strength: finiteNumber(source.strength, 1),
    }));
    render();
  }

  function pointerPosition(event) {
    const rect = elements.canvas.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }

  function sourceAt(canvasX, canvasY) {
    if (!state.scene || elements.preset.value === "uniform") return null;
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
    if (!state.scene || !state.scene.sources.length || elements.preset.value === "uniform") return;
    const [canvasX, canvasY] = pointerPosition(event);
    const sourceIndex = sourceAt(canvasX, canvasY);
    if (sourceIndex === null) return;
    state.selectedSource = sourceIndex;
    state.drag = { pointerId: event.pointerId, sourceIndex };
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
      const [xmin, xmax] = state.scene.domain.x;
      const [ymin, ymax] = state.scene.domain.y;
      const [worldX, worldY] = canvasToWorld(canvasX, canvasY);
      const source = state.scene.sources[state.drag.sourceIndex];
      source.x = clamp(worldX, xmin, xmax);
      source.y = clamp(worldY, ymin, ymax);
      state.sourceOverrides = state.scene.sources.map((item) => ({ ...item }));
      render();
      return;
    }
    updateProbe(canvasX, canvasY);
  }

  function finishPointerDrag(event) {
    if (!state.drag || state.drag.pointerId !== event.pointerId) return;
    const index = state.drag.sourceIndex;
    state.drag = null;
    delete elements.canvas.dataset.dragging;
    if (elements.canvas.hasPointerCapture(event.pointerId)) {
      elements.canvas.releasePointerCapture(event.pointerId);
    }
    renderSourceEditors();
    const source = state.scene.sources[index];
    elements.liveStatus.textContent = `${readableSourceName(source, index)}移动到 x ${formatEditorValue(source.x)}，y ${formatEditorValue(source.y)}，正在重新计算。`;
    loadScene();
  }

  function updateProbe(canvasX, canvasY) {
    const { left, right, top, bottom } = state.plotRect;
    if (canvasX < left || canvasX > right || canvasY < top || canvasY > bottom) {
      elements.probe.hidden = true;
      return;
    }
    const [worldX, worldY] = canvasToWorld(canvasX, canvasY);
    const value = sampleNearest(worldX, worldY);
    elements.probePosition.textContent = `x ${formatEditorValue(worldX)} · y ${formatEditorValue(worldY)}`;
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
    if (state.selectedSource === null || !state.scene?.sources[state.selectedSource]) return;
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
    source.x = clamp(source.x + xDirection * xStep, ...state.scene.domain.x);
    source.y = clamp(source.y + yDirection * yStep, ...state.scene.domain.y);
    state.sourceOverrides = state.scene.sources.map((item) => ({ ...item }));
    render();
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
  loadScene({ preserveSources: false });
})();
