/** Pure Canvas 2D renderer for a scene view; it owns the context and reads no page state. */

import { CANVAS_THEME } from "./canvas-theme.js";
import { colorForScalar } from "./color-scale.js";
import { clamp } from "./coordinates.js";
import { finiteNumber, formatAxisValue, formatValue } from "./formatting.js";
import { effectiveDipoleAngleDeg } from "./source-controls.js";

export const SOURCE_GLYPHS = Object.freeze({
  positive: "+",
  negative: "−",
  dipole: "→",
  wire_out: "⊙",
  wire_into: "⊗",
  ring_charge: "+",
});

// validateScene admits only the kinds listed in SOURCE_STRENGTH_UNITS.
export function sourceStyle(source, theme = CANVAS_THEME) {
  const { kind } = source;
  return {
    kind,
    fill: theme.marker.fill[kind],
    symbol: SOURCE_GLYPHS[kind],
    rotation:
      kind === "dipole" ? (-effectiveDipoleAngleDeg(source) * Math.PI) / 180 : undefined,
  };
}

/**
 * Create a renderer bound to one canvas. `render(view)` paints the paper,
 * then, when the view has a scene, the heatmap, grid and axes, field lines,
 * material regions and source markers in that order; a stale view keeps
 * only the paper grid and the markers. `view.layers` switches the heatmap,
 * lines, arrows, sources and grid lines off individually; the axes, frame
 * and material outlines always draw. It returns whether hatching shows
 * beside the markers so the page can decide about the legend.
 */
export const ALL_LAYERS = Object.freeze({
  heatmap: true,
  lines: true,
  arrows: true,
  sources: true,
  grid: true,
});

export function createRenderer(canvas, theme = CANVAS_THEME) {
  const context = canvas.getContext("2d", { alpha: false });
  let view = null;

  function worldToCanvas(x, y) {
    return view.transform.worldToCanvas(x, y);
  }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2.5);
    const width = Math.max(1, Math.round(rect.width * ratio));
    const height = Math.max(1, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { width: rect.width, height: rect.height, ratio };
  }

  function render(nextView) {
    view = nextView;
    const { size } = view;
    const layers = { ...ALL_LAYERS, ...(view.layers || {}) };
    context.clearRect(0, 0, size.width, size.height);
    context.fillStyle = theme.paper;
    context.fillRect(0, 0, size.width, size.height);
    if (!view.scene) return { hatchVisible: false };
    if (view.status === "stale") {
      drawGridAndAxes({ onField: false, grid: layers.grid });
      if (layers.sources) drawSources();
      return { hatchVisible: false };
    }
    const uncoloredCells = layers.heatmap ? drawHeatmap() : [];
    drawGridAndAxes({ onField: layers.heatmap, grid: layers.grid });
    if (layers.lines) drawStreamlines({ arrows: layers.arrows });
    drawRegions();
    if (layers.sources) drawSources();
    return {
      hatchVisible: layers.heatmap && hatchShowsBesideMarkers(uncoloredCells, layers.sources),
    };
  }

  // The offscreen raster depends only on the scalar grid, the colour scale
  // and the sampling factor; toggling layers or redrawing markers reuses it.
  let raster = null;

  function heatmapRaster(scalar, scale, factor) {
    if (
      raster &&
      raster.scalar === scalar &&
      raster.factor === factor &&
      raster.scale.type === scale.type &&
      raster.scale.minimum === scale.minimum &&
      raster.scale.maximum === scale.maximum &&
      raster.scale.constant === scale.constant
    ) {
      return raster;
    }
    const offscreen = document.createElement("canvas");
    offscreen.width = scalar.nx;
    offscreen.height = scalar.ny;
    const offscreenContext = offscreen.getContext("2d");
    const image = offscreenContext.createImageData(scalar.nx, scalar.ny);

    // The API uses row-major values, with y descending from ymax to ymin.
    const uncolored = [];
    for (let index = 0; index < scalar.values.length; index += 1) {
      const color = colorForScalar(scalar.values[index], Boolean(scalar.mask?.[index]), scale);
      if (color[3] === 0) uncolored.push(index);
      const offset = index * 4;
      image.data[offset] = color[0];
      image.data[offset + 1] = color[1];
      image.data[offset + 2] = color[2];
      image.data[offset + 3] = color[3];
    }
    offscreenContext.putImageData(image, 0, 0);
    const source = uncolored.length
      ? interpolateOverColoredNodes(image.data, scalar.nx, scalar.ny, factor)
      : offscreen;
    raster = { scalar, scale: { ...scale }, factor, source, uncolored };
    return raster;
  }

  function drawHeatmap() {
    const { scalar } = view.scene;
    const { source, uncolored } = heatmapRaster(scalar, view.scale, samplesPerCell(scalar));

    // Scalar nodes include both domain endpoints, so texel centres (i + 0.5)
    // must land on the plot edges: sample the source from the first to the
    // last texel centre instead of stretching whole texels across the plot.
    const { left, top, right, bottom } = view.plotRect;
    context.save();
    context.imageSmoothingEnabled = true;
    context.drawImage(
      source,
      0.5,
      0.5,
      source.width - 1,
      source.height - 1,
      left,
      top,
      right - left,
      bottom - top,
    );
    context.restore();
    const cells = uncolored.map(cellRect);
    if (cells.length) drawHatch(cells);
    return cells;
  }

  // Samples per cell for the masked path: about one per 2 device px, at most
  // 16 per cell and about 1024 across the plot (so a large, dense display
  // stays near 20 ms), and never fewer than 4, which keeps the smoothing
  // between samples inside the uncoloured cells.
  function samplesPerCell(scalar) {
    const { left, top, right, bottom } = view.plotRect;
    const cells = Math.max(scalar.nx, scalar.ny) - 1;
    const cell = Math.max((right - left) / (scalar.nx - 1), (bottom - top) / (scalar.ny - 1));
    const wanted = Math.ceil((cell * view.pixelRatio) / 2);
    return Math.max(4, Math.min(16, wanted, Math.floor(1024 / cells)));
  }

  // Smoothing the raw raster would blend each coloured cell toward its
  // uncoloured neighbours, and on this colormap any colour there reads as a
  // field value. Interpolate bilinearly over the coloured nodes only: each
  // sample is the weight-normalised mean of the coloured corners of its cell,
  // so no colour crosses into, out of or across an uncoloured node, and a
  // sample whose weighted corners are all uncoloured stays transparent (it
  // lies in a hatched cell). Where all four corners are coloured this is the
  // ordinary bilinear value; drawImage then smooths between the samples.
  function interpolateOverColoredNodes(data, nx, ny, factor) {
    const width = (nx - 1) * factor + 1;
    const height = (ny - 1) * factor + 1;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const target = canvas.getContext("2d");
    const image = target.createImageData(width, height);
    const out = image.data;
    let weight = 0;
    let red = 0;
    let green = 0;
    let blue = 0;
    const add = (offset, corner) => {
      if (corner === 0 || data[offset + 3] === 0) return;
      weight += corner;
      red += corner * data[offset];
      green += corner * data[offset + 1];
      blue += corner * data[offset + 2];
    };
    for (let row = 0; row < height; row += 1) {
      const j = Math.min(Math.floor(row / factor), ny - 2);
      const fy = row / factor - j;
      for (let column = 0; column < width; column += 1) {
        const i = Math.min(Math.floor(column / factor), nx - 2);
        const fx = column / factor - i;
        const corner = (j * nx + i) * 4;
        weight = 0;
        red = 0;
        green = 0;
        blue = 0;
        add(corner, (1 - fx) * (1 - fy));
        add(corner + 4, fx * (1 - fy));
        add(corner + nx * 4, (1 - fx) * fy);
        add(corner + nx * 4 + 4, fx * fy);
        if (weight === 0) continue;
        const offset = (row * width + column) * 4;
        out[offset] = Math.round(red / weight);
        out[offset + 1] = Math.round(green / weight);
        out[offset + 2] = Math.round(blue / weight);
        out[offset + 3] = 255;
      }
    }
    target.putImageData(image, 0, 0);
    return canvas;
  }

  // The nearest-node cell of a scalar sample, clipped to the plot.
  function cellRect(index) {
    const { nx, ny } = view.scene.scalar;
    const { left, top, right, bottom } = view.plotRect;
    const dx = (right - left) / (nx - 1);
    const dy = (bottom - top) / (ny - 1);
    const x = left + (index % nx) * dx;
    const y = top + Math.floor(index / nx) * dy;
    return {
      left: Math.max(left, x - dx / 2),
      top: Math.max(top, y - dy / 2),
      right: Math.min(right, x + dx / 2),
      bottom: Math.min(bottom, y + dy / 2),
    };
  }

  function drawHatch(cells) {
    const { left, top, right, bottom } = view.plotRect;
    const height = bottom - top;
    context.save();
    context.beginPath();
    cells.forEach((cell) => {
      context.rect(cell.left, cell.top, cell.right - cell.left, cell.bottom - cell.top);
    });
    // One path, so neighbouring cells join without antialiased seams.
    context.fillStyle = theme.hatch.base;
    context.fill();
    context.clip();
    // Each 45 deg line runs through device pixel centres, so its darkest
    // pixels are covered the same whatever the plot geometry or pixel ratio.
    const ratio = view.pixelRatio;
    const centre = (value) => (Math.round(value * ratio - 0.5) + 0.5) / ratio;
    const run = Math.round(height * ratio) / ratio;
    const startY = centre(bottom);
    context.beginPath();
    for (let offset = -height; offset < right - left; offset += theme.hatch.spacing) {
      const startX = centre(left + offset);
      context.moveTo(startX, startY);
      context.lineTo(startX + run, startY - run);
    }
    context.strokeStyle = theme.hatch.line;
    // One CSS px at every pixel ratio, so the hatching never thins out.
    context.lineWidth = 1;
    context.stroke();
    context.restore();
  }

  // Hatching counts as visible only when at least one hatch spacing of it
  // shows outside a marker's outer ring, which ends at r = 13.
  const MARKER_COVER_RADIUS =
    theme.marker.outerRingRadius +
    theme.marker.outerRingWidth / 2 +
    theme.hatch.spacing;

  // The legend only names hatching that can be seen beside the markers; with
  // the marker layer off nothing covers the hatching, so any cell counts.
  function hatchShowsBesideMarkers(cells, markersDrawn = true) {
    const centres = markersDrawn
      ? view.scene.sources.map((source) =>
          worldToCanvas(finiteNumber(source.x), finiteNumber(source.y)),
        )
      : [];
    return cells.some((cell) => {
      const corners = [
        [cell.left, cell.top],
        [cell.right, cell.top],
        [cell.left, cell.bottom],
        [cell.right, cell.bottom],
      ];
      return !centres.some(([cx, cy]) =>
        corners.every(([x, y]) => Math.hypot(x - cx, y - cy) <= MARKER_COVER_RADIUS),
      );
    });
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

  function drawGridAndAxes({ onField, grid = true }) {
    const { left, top, right, bottom } = view.plotRect;
    const [xmin, xmax] = view.scene.domain.x;
    const [ymin, ymax] = view.scene.domain.y;
    context.save();
    context.lineWidth = 1;
    context.font = theme.axes.font;
    context.fillStyle = theme.axes.tick;
    context.strokeStyle = onField ? theme.axes.gridOnField : theme.axes.gridOnPaper;

    niceTicks(xmin, xmax).forEach((tick) => {
      const [x] = worldToCanvas(tick, ymin);
      context.beginPath();
      context.moveTo(x, top);
      context.lineTo(x, bottom);
      if (grid) context.stroke();
      context.textAlign = "center";
      context.textBaseline = "top";
      context.fillText(formatAxisValue(tick), x, bottom + 8);
    });

    niceTicks(ymin, ymax).forEach((tick) => {
      const [, y] = worldToCanvas(xmin, tick);
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(right, y);
      if (grid) context.stroke();
      context.textAlign = "right";
      context.textBaseline = "middle";
      context.fillText(formatAxisValue(tick), left - 7, y);
    });

    // One device pixel, centred on a pixel row, so the frame stays crisp.
    const ratio = view.pixelRatio;
    const snap = (value) => (Math.round(value * ratio) + 0.5) / ratio;
    context.strokeStyle = theme.axes.frame;
    context.lineWidth = 1 / ratio;
    context.strokeRect(snap(left), snap(top), snap(right) - snap(left), snap(bottom) - snap(top));

    const unit = String(view.scene.domain.unit || "").trim();
    const axisTitle = (axis) => (unit ? `${axis} / ${unit}` : axis);
    context.fillStyle = theme.axes.title;
    context.font = theme.axes.titleFont;
    context.textAlign = "left";
    context.textBaseline = "bottom";
    context.fillText(axisTitle("y"), left, top - 8);
    context.textAlign = "center";
    context.textBaseline = "top";
    context.fillText(axisTitle("x"), (left + right) / 2, bottom + 28);
    context.restore();
  }

  // A material sphere is drawn as its dashed outline in the plane, with the
  // material named above it; the field inside is real and stays coloured.
  function drawRegions() {
    const regionTheme = theme.region;
    for (const region of view.scene.regions) {
      const [x, y] = worldToCanvas(region.x, region.y);
      const [edgeX] = worldToCanvas(region.x + region.radius, region.y);
      const radius = Math.abs(edgeX - x);
      context.save();
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.strokeStyle = regionTheme.halo;
      context.lineWidth = regionTheme.haloWidth;
      context.stroke();
      context.setLineDash(regionTheme.dash);
      context.strokeStyle = regionTheme.outline;
      context.lineWidth = regionTheme.outlineWidth;
      context.stroke();
      context.setLineDash([]);
      const label =
        region.kind === "conducting_sphere"
          ? "导体"
          : `εr = ${formatValue(region.relative_permittivity)}`;
      context.font = regionTheme.labelFont;
      context.textAlign = "center";
      context.textBaseline = "bottom";
      context.lineJoin = "round";
      context.strokeStyle = regionTheme.labelHalo;
      context.lineWidth = 4;
      context.strokeText(label, x, y - radius - 6);
      context.fillStyle = regionTheme.label;
      context.fillText(label, x, y - radius - 6);
      context.restore();
    }
  }

  function validPoints(line) {
    if (!Array.isArray(line?.points)) return [];
    return line.points
      .filter((point) => Array.isArray(point) && point.length >= 2)
      .map((point) => [Number(point[0]), Number(point[1])])
      .filter((point) => point.every(Number.isFinite));
  }

  function drawStreamlines({ arrows = true } = {}) {
    const lines = view.scene.lines;
    context.save();
    context.lineJoin = "round";
    context.lineCap = "round";

    for (const line of lines) {
      const points = view.transform.projectPoints(validPoints(line));
      if (points.length < 2) continue;
      tracePath(points);
      context.strokeStyle = theme.line.halo;
      context.lineWidth = theme.line.haloWidth;
      context.stroke();
      tracePath(points);
      context.strokeStyle = theme.line.core;
      context.lineWidth = theme.line.coreWidth;
      context.stroke();
      if (arrows) drawDirectionArrows(points, line.direction);
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
    context.fillStyle = theme.arrow.fill;
    context.strokeStyle = theme.arrow.stroke;
    context.lineWidth = theme.arrow.strokeWidth;
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

  function drawSources() {
    const { marker } = theme;
    view.scene.sources.forEach((source, index) => {
      const [x, y] = worldToCanvas(finiteNumber(source.x), finiteNumber(source.y));
      const style = sourceStyle(source, theme);
      context.save();
      context.beginPath();
      context.arc(x, y, 10, 0, Math.PI * 2);
      context.fillStyle = style.fill;
      context.fill();
      context.strokeStyle = marker.ring;
      context.lineWidth = marker.ringWidth;
      context.stroke();
      context.beginPath();
      context.arc(x, y, marker.outerRingRadius, 0, Math.PI * 2);
      context.strokeStyle = marker.outerRing;
      context.lineWidth = marker.outerRingWidth;
      context.stroke();
      context.fillStyle = marker.glyph;
      context.font = marker.glyphFont;
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
      if (index === view.selectedSource) {
        // A white under-ring keeps the blue dashes visible on the dark colormap end.
        context.beginPath();
        context.arc(x, y, 16, 0, Math.PI * 2);
        context.strokeStyle = marker.selectionUnder;
        context.lineWidth = 4;
        context.stroke();
        context.strokeStyle = marker.selection;
        context.lineWidth = 2;
        context.setLineDash([4, 3]);
        context.stroke();
      }
      context.restore();
    });
  }

  return { context, resize, render };
}
