/** Pure scalar-scale and pixel-color functions used by the heatmap renderer. */

// Lavender nodes on one CIE LCh hue (304 deg): L* falls 97, 84.5, 71, 55.5,
// 40 and chroma rises, so a stronger field is always a darker colour, also
// between nodes where the renderer interpolates in sRGB. The heatmap and the
// colorbar gradient both read this table.
export const PALETTE = Object.freeze(
  [
    [0.0, [248, 245, 254]],
    [0.25, [217, 206, 237]],
    [0.5, [183, 165, 222]],
    [0.75, [142, 120, 199]],
    [1.0, [100, 79, 172]],
  ].map(([stop, color]) => Object.freeze([stop, Object.freeze(color)])),
);
// Masked and uncolorable cells stay transparent; the renderer hatches them.
const INVALID_PIXEL = [7, 17, 26, 0];

export function paletteCssGradient(direction = "to top") {
  const stops = PALETTE.map(
    ([stop, [red, green, blue]]) => `rgb(${red}, ${green}, ${blue}) ${stop * 100}%`,
  );
  return `linear-gradient(${direction}, ${stops.join(", ")})`;
}

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

// A range this narrow relative to its values (the server widens a constant
// field's range by 1e-12) holds a single value up to rounding: show it in the
// middle colour instead of spreading noise over the ramp or painting the pale
// floor, which would look like an empty plot.
const FLAT_RELATIVE_SPAN = 1e-6;

export function normalizeScalar(value, scale) {
  if (!Number.isFinite(value)) return null;
  if (scale.type === "log" && value <= 0) return null;
  const span = scale.maximum - scale.minimum;
  if (span <= FLAT_RELATIVE_SPAN * Math.max(Math.abs(scale.minimum), Math.abs(scale.maximum))) {
    return 0.5;
  }
  if (scale.type === "log") {
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
