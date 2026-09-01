export const SOURCE_COORDINATE_LIMIT = 2.8;
export const SOURCE_COUNT_LIMIT = 8;

const DEFAULT_SEEDING_SOURCE_COUNTS = Object.freeze({
  electric_dipole: 1,
  magnetic_dipole: 1,
  halbach_array: 8,
});
const SOURCE_PLACEMENT_CANDIDATES = Object.freeze([
  [-1.8, 0.55],
  [-0.6, 0.55],
  [0.6, 0.55],
  [1.8, 0.55],
  [-1.8, -0.55],
  [-0.6, -0.55],
  [0.6, -0.55],
  [1.8, -0.55],
]);

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

export function normalizeAngleDeg(value) {
  const angle = finiteNumber(value, 90);
  return ((angle % 360) + 360) % 360;
}

export function clampSourceCoordinate(value) {
  return Math.min(
    SOURCE_COORDINATE_LIMIT,
    Math.max(-SOURCE_COORDINATE_LIMIT, finiteNumber(value)),
  );
}

export function serializeSource(source) {
  const serialized = {
    x: clampSourceCoordinate(source.x),
    y: clampSourceCoordinate(source.y),
    kind: String(source.kind ?? "source"),
    strength: finiteNumber(source.strength, 1),
  };
  if (serialized.kind === "dipole") {
    serialized.angle_deg = normalizeAngleDeg(source.angle_deg);
  }
  return serialized;
}

export function effectiveDipoleAngleDeg(source) {
  const reversal = finiteNumber(source.strength, 1) < 0 ? 180 : 0;
  return normalizeAngleDeg(normalizeAngleDeg(source.angle_deg) + reversal);
}

export function seedingSourceCount(preset, sources) {
  if (!Array.isArray(sources)) return DEFAULT_SEEDING_SOURCE_COUNTS[preset] ?? 0;
  if (preset === "electric_dipole") {
    return sources.filter(({ kind }) => kind === "positive").length;
  }
  if (preset === "magnetic_dipole" || preset === "halbach_array") {
    return sources.filter(({ kind }) => kind === "dipole").length;
  }
  return 0;
}

export function densityForSeedBudget(density, required, range) {
  const minimum = finiteNumber(range.min, 6);
  const maximum = finiteNumber(range.max, 40);
  const step = Math.max(1, finiteNumber(range.step, 1));
  const requested = Math.max(minimum, finiteNumber(density, minimum));
  const target = Math.max(requested, finiteNumber(required));
  const legal = minimum + Math.ceil((target - minimum) / step) * step;
  return Math.min(maximum, legal);
}

export function canRemoveSource(preset, sources, index) {
  if (!Array.isArray(sources) || !sources[index] || sources.length <= 1) return false;
  const remaining = sources.filter((_source, sourceIndex) => sourceIndex !== index);
  if (preset === "electric_dipole") {
    return (
      remaining.some(({ kind }) => kind === "positive") &&
      remaining.some(({ kind }) => kind === "negative")
    );
  }
  if (preset === "magnetic_dipole" || preset === "halbach_array") {
    return remaining.some(({ kind }) => kind === "dipole");
  }
  return false;
}

export function createSource(kind, sources = []) {
  const index = sources.length;
  const occupied = sources.map(({ x, y }) => [
    clampSourceCoordinate(x),
    clampSourceCoordinate(y),
  ]);
  const [x, y] =
    SOURCE_PLACEMENT_CANDIDATES.find(([candidateX, candidateY]) =>
      occupied.every(
        ([sourceX, sourceY]) =>
          Math.hypot(candidateX - sourceX, candidateY - sourceY) > 1e-9,
      ),
    ) ?? SOURCE_PLACEMENT_CANDIDATES[index % SOURCE_PLACEMENT_CANDIDATES.length];
  if (kind === "positive") {
    return { x, y, kind, strength: 1 };
  }
  if (kind === "negative") {
    return { x, y, kind, strength: -1 };
  }
  return { x, y, kind: "dipole", strength: 1, angle_deg: 90 };
}
