/** Pure scalar-scale and pixel-color functions used by the heatmap renderer. */

const PALETTE = [
  [0.0, [68, 1, 84]],
  [0.25, [59, 82, 139]],
  [0.52, [33, 145, 140]],
  [0.76, [94, 201, 98]],
  [1.0, [253, 231, 37]],
];
const INVALID_PIXEL = [7, 17, 26, 0];

export function getScaleType(scale) {
  const value = typeof scale === "string" ? scale : scale?.type;
  return String(value || "linear").toLowerCase().includes("log") ? "log" : "linear";
}

export function resolveScale(scalar) {
  const type = getScaleType(scalar.scale);
  let minimum = Number(scalar.vmin);
  let maximum = Number(scalar.vmax);
  let dataMinimum = Infinity;
  let dataMaximum = -Infinity;
  scalar.values.forEach((rawValue, index) => {
    if (scalar.mask?.[index]) return;
    const value = Number(rawValue);
    if (!Number.isFinite(value) || (type === "log" && value <= 0)) return;
    dataMinimum = Math.min(dataMinimum, value);
    dataMaximum = Math.max(dataMaximum, value);
  });

  if (!Number.isFinite(minimum) || (type === "log" && minimum <= 0)) {
    minimum = Number.isFinite(dataMinimum) ? dataMinimum : type === "log" ? 1 : 0;
  }
  if (!Number.isFinite(maximum) || (type === "log" && maximum <= 0)) {
    maximum = Number.isFinite(dataMaximum) ? dataMaximum : type === "log" ? 10 : 1;
  }
  if (!(maximum > minimum)) {
    maximum = minimum + Math.max(Math.abs(minimum) * 1e-6, 1e-12);
  }
  return { type, minimum, maximum };
}

export function normalizeScalar(value, scale) {
  if (!Number.isFinite(value)) return null;
  if (scale.type === "log") {
    if (value <= 0) return null;
    const minimum = Math.log10(scale.minimum);
    const maximum = Math.log10(scale.maximum);
    return Math.max(0, Math.min(1, (Math.log10(value) - minimum) / (maximum - minimum)));
  }
  return Math.max(0, Math.min(1, (value - scale.minimum) / (scale.maximum - scale.minimum)));
}

function paletteColor(normalized) {
  for (let index = 1; index < PALETTE.length; index += 1) {
    const [rightStop, rightColor] = PALETTE[index];
    const [leftStop, leftColor] = PALETTE[index - 1];
    if (normalized <= rightStop) {
      const fraction = (normalized - leftStop) / (rightStop - leftStop);
      return leftColor.map((component, componentIndex) =>
        Math.round(component + (rightColor[componentIndex] - component) * fraction),
      );
    }
  }
  return [...PALETTE.at(-1)[1]];
}

export function colorForScalar(value, masked, scale) {
  if (masked) return [...INVALID_PIXEL];
  const normalized = normalizeScalar(Number(value), scale);
  if (normalized === null) return [...INVALID_PIXEL];
  return [...paletteColor(normalized), 255];
}
