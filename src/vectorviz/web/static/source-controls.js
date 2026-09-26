export const SOURCE_COORDINATE_LIMIT = 2.8;
export const SOURCE_COUNT_LIMIT = 8;

// Presets that share the point-charge contract with electric_dipole.
export const ELECTRIC_PRESETS = Object.freeze([
  "electric_dipole",
  "electric_quadrupole",
  "electric_hexagon",
  "electric_hexagon_alternating",
]);
const DEFAULT_SEEDING_SOURCE_COUNTS = Object.freeze({
  electric_dipole: 2,
  electric_quadrupole: 4,
  electric_hexagon: 6,
  electric_hexagon_alternating: 6,
  magnetic_dipole: 1,
  halbach_array: 0,
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

export function isElectricPreset(preset) {
  return ELECTRIC_PRESETS.includes(preset);
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
  if (isElectricPreset(preset)) {
    return sources.filter(({ kind }) => kind === "positive" || kind === "negative").length;
  }
  if (preset === "magnetic_dipole" || preset === "halbach_array") {
    return sources.filter(
      ({ kind, strength }) => kind === "dipole" && Number(strength) !== 0,
    ).length;
  }
  return 0;
}

export function sourceIsActive(preset, source) {
  if (!source || typeof source !== "object") return false;
  if (isElectricPreset(preset)) return true;
  if (preset === "magnetic_dipole" || preset === "halbach_array") {
    return source.kind === "dipole" && Number(source.strength) !== 0;
  }
  return false;
}

function legalExclusiveMinimum(value) {
  return Number.isFinite(value) && value > 0 ? value : null;
}

function nextRepresentable(value, towardPositive) {
  if (value === 0) return towardPositive ? Number.MIN_VALUE : -Number.MIN_VALUE;
  const buffer = new ArrayBuffer(8);
  const floats = new Float64Array(buffer);
  const bits = new BigUint64Array(buffer);
  floats[0] = value;
  if ((value > 0) === towardPositive) bits[0] += 1n;
  else bits[0] -= 1n;
  return floats[0];
}

function nudgeOutsideBoundary(position, obstacle, minimum, direction, bounds) {
  const axes = Math.abs(direction.x) >= Math.abs(direction.y) ? ["x", "y"] : ["y", "x"];
  let adjusted = {...position};
  for (const axis of axes) {
    const obstacleCoordinate = finiteNumber(obstacle[axis]);
    const towardPositive = direction[axis] >= 0;
    while (
      Math.hypot(
        adjusted.x - finiteNumber(obstacle.x),
        adjusted.y - finiteNumber(obstacle.y),
      ) <= minimum
    ) {
      const next = nextRepresentable(adjusted[axis], towardPositive);
      if (
        !Number.isFinite(next) ||
        next === adjusted[axis] ||
        next < bounds[`${axis}min`] ||
        next > bounds[`${axis}max`]
      ) {
        break;
      }
      adjusted[axis] = next;
    }
    if (
      Math.hypot(
        adjusted.x - finiteNumber(obstacle.x),
        adjusted.y - finiteNumber(obstacle.y),
      ) > minimum
    ) {
      return adjusted;
    }
  }
  return position;
}

export function sourceSeparationConflict(
  preset,
  sources,
  sourceIndex,
  candidate,
  exclusiveMinimum,
) {
  const minimum = legalExclusiveMinimum(exclusiveMinimum);
  if (minimum === null || !Array.isArray(sources) || !sources[sourceIndex]) return null;
  const moving = {...sources[sourceIndex], ...candidate};
  if (!sourceIsActive(preset, moving)) return null;
  for (let index = 0; index < sources.length; index += 1) {
    if (index === sourceIndex || !sourceIsActive(preset, sources[index])) continue;
    const distance = Math.hypot(
      finiteNumber(moving.x) - finiteNumber(sources[index].x),
      finiteNumber(moving.y) - finiteNumber(sources[index].y),
    );
    if (distance <= minimum) return {index, distance};
  }
  return null;
}

export function snapSourcePosition(
  preset,
  sources,
  sourceIndex,
  candidate,
  exclusiveMinimum,
  bounds = {},
) {
  const source = sources?.[sourceIndex];
  if (!source) return null;
  const minimum = legalExclusiveMinimum(exclusiveMinimum);
  const xmin = finiteNumber(bounds.xmin, -SOURCE_COORDINATE_LIMIT);
  const xmax = finiteNumber(bounds.xmax, SOURCE_COORDINATE_LIMIT);
  const ymin = finiteNumber(bounds.ymin, -SOURCE_COORDINATE_LIMIT);
  const ymax = finiteNumber(bounds.ymax, SOURCE_COORDINATE_LIMIT);
  let position = {
    x: Math.min(xmax, Math.max(xmin, finiteNumber(candidate.x, source.x))),
    y: Math.min(ymax, Math.max(ymin, finiteNumber(candidate.y, source.y))),
  };
  if (minimum === null || !sourceIsActive(preset, {...source, ...position})) {
    return {...position, snapped: false};
  }

  let snapped = false;
  const clearance = nextRepresentable(minimum, true);
  for (let attempt = 0; attempt < Math.max(4, sources.length * 2); attempt += 1) {
    const conflict = sourceSeparationConflict(
      preset,
      sources,
      sourceIndex,
      position,
      minimum,
    );
    if (!conflict) return {...position, snapped};
    const obstacle = sources[conflict.index];
    let dx = position.x - finiteNumber(obstacle.x);
    let dy = position.y - finiteNumber(obstacle.y);
    if (Math.hypot(dx, dy) === 0) {
      dx = finiteNumber(source.x) - finiteNumber(obstacle.x);
      dy = finiteNumber(source.y) - finiteNumber(obstacle.y);
    }
    if (Math.hypot(dx, dy) === 0) {
      const angle = ((sourceIndex + conflict.index + 1) * Math.PI) / 4;
      dx = Math.cos(angle);
      dy = Math.sin(angle);
    }
    const length = Math.hypot(dx, dy);
    position = {
      x: Math.min(
        xmax,
        Math.max(xmin, finiteNumber(obstacle.x) + (dx / length) * clearance),
      ),
      y: Math.min(
        ymax,
        Math.max(ymin, finiteNumber(obstacle.y) + (dy / length) * clearance),
      ),
    };
    position = nudgeOutsideBoundary(
      position,
      obstacle,
      minimum,
      {x: dx, y: dy},
      {xmin, xmax, ymin, ymax},
    );
    snapped = true;
  }

  if (sourceSeparationConflict(preset, sources, sourceIndex, position, minimum)) {
    return {x: finiteNumber(source.x), y: finiteNumber(source.y), snapped: false};
  }
  return {...position, snapped};
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
  if (isElectricPreset(preset)) {
    // Other charge arrangements only need one charge to keep a field.
    return remaining.some(({ kind }) => kind === "positive" || kind === "negative");
  }
  if (preset === "magnetic_dipole" || preset === "halbach_array") {
    return remaining.some((source) => sourceIsActive(preset, source));
  }
  return false;
}

export function createSource(
  kind,
  sources = [],
  preset = kind === "dipole" ? "magnetic_dipole" : "electric_dipole",
  exclusiveMinimum = null,
) {
  const index = sources.length;
  const template =
    kind === "positive"
      ? {kind, strength: 1}
      : kind === "negative"
        ? {kind, strength: -1}
        : {kind: "dipole", strength: 1, angle_deg: 90};
  const candidates = SOURCE_PLACEMENT_CANDIDATES.map(([x, y]) => ({x, y, ...template}));
  const expanded = [...sources, candidates[0]];
  const selected =
    candidates.find((candidate) => {
      expanded[index] = candidate;
      if (legalExclusiveMinimum(exclusiveMinimum) === null) {
        return sources.every(
          (source) => Math.hypot(candidate.x - source.x, candidate.y - source.y) > 0,
        );
      }
      return !sourceSeparationConflict(
        preset,
        expanded,
        index,
        candidate,
        exclusiveMinimum,
      );
    }) ?? candidates[index % candidates.length];
  return selected;
}
