/** Number formatting shared by the canvas renderer and the page readouts. */

export function formatValue(value) {
  if (!Number.isFinite(value)) return "—";
  const absolute = Math.abs(value);
  if ((absolute > 0 && absolute < 0.001) || absolute >= 10000) return value.toExponential(2);
  return new Intl.NumberFormat("zh-CN", { maximumSignificantDigits: 4 }).format(value);
}

// Tick labels use the typographic minus sign (U+2212).
export function formatAxisValue(value) {
  const text =
    Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.01)
      ? value.toExponential(1)
      : Number(value.toPrecision(3)).toString();
  return text.replace(/^-/, "\u2212");
}

export function formatEditorValue(value) {
  const number = finiteNumber(value);
  return Math.abs(number) >= 1e4 || (Math.abs(number) > 0 && Math.abs(number) < 1e-4)
    ? number.toExponential(4)
    : Number(number.toPrecision(6)).toString();
}

export function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}
