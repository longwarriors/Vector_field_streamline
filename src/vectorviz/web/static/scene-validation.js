/** Structural validation of a scene response against the browser contract. */

export const SOURCE_STRENGTH_UNITS = Object.freeze({
  positive: "nC",
  negative: "nC",
  dipole: "A·m²",
  wire_out: "A",
  wire_into: "A",
  ring_charge: "nC",
});

// Read-only marker kinds, and the exact marker set each fixed preset returns.
export const MARKER_KINDS = new Set(["wire_out", "wire_into", "ring_charge"]);

export const FIXED_MARKER_KINDS = Object.freeze({
  current_loop: ["wire_into", "wire_out"],
  charged_ring: ["ring_charge", "ring_charge"],
});

export const REGION_KINDS = new Set(["dielectric_sphere", "conducting_sphere"]);

export const REGION_PRESETS = new Set(REGION_KINDS);

export const SEED_MODE_LABELS = Object.freeze({
  coverage: "覆盖播种",
  equal_flux: "等通量播种",
  feature: "特征播种",
});

// Validates the response for the preset that requested it: fixed presets
// must return exactly their read-only markers or their own material region.
export function validateScene(scene, preset) {
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
  if (
    !scene.lines.every(
      (line) =>
        line &&
        typeof line === "object" &&
        (line.start_termination === undefined ||
          line.start_termination === null ||
          (typeof line.start_termination === "string" && line.start_termination.length > 0)),
    )
  ) {
    throw new Error("场线 start_termination 必须是非空文本或 null");
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
    if (source.kind === "ring_charge") return source.strength !== 0;
    return (
      (source.kind === "wire_out" || source.kind === "wire_into") &&
      source.strength >= 0
    );
  });
  const markerSources = scene.sources.filter(({ kind }) => MARKER_KINDS.has(kind));
  const expectedMarkers = FIXED_MARKER_KINDS[preset];
  // A fixed preset returns exactly its two markers with one shared strength;
  // every other preset returns no read-only marker at all.
  const validFixedMarkers = expectedMarkers
    ? markerSources.length === expectedMarkers.length &&
      scene.sources.length === expectedMarkers.length &&
      markerSources
        .map(({ kind }) => kind)
        .sort()
        .every((kind, index) => kind === expectedMarkers[index]) &&
      markerSources.every(({ strength }) => strength === markerSources[0].strength)
    : markerSources.length === 0;
  if (!validSources || !validFixedMarkers) {
    throw new Error("场源缺少有效坐标、强度或 strength_unit");
  }
  // Regions are additive: an older server omits the field entirely; once
  // present it must be a list that describes every material sphere.
  if (scene.regions === undefined) {
    scene.regions = [];
  }
  const validRegions =
    Array.isArray(scene.regions) &&
    scene.regions.every((region) => {
      if (!region || typeof region !== "object" || !REGION_KINDS.has(region.kind)) {
        return false;
      }
      if (
        !Number.isFinite(region.x) ||
        !Number.isFinite(region.y) ||
        !Number.isFinite(region.radius) ||
        region.radius <= 0 ||
        region.unit !== "m"
      ) {
        return false;
      }
      const permittivity = region.relative_permittivity;
      if (region.kind === "dielectric_sphere") {
        return Number.isFinite(permittivity) && permittivity >= 1;
      }
      return permittivity === undefined || permittivity === null;
    });
  // A sphere preset returns exactly one region of its own material.
  const expectedRegionKind = REGION_PRESETS.has(preset)
    ? preset
    : null;
  const regionsMatchPreset =
    validRegions &&
    (expectedRegionKind
      ? scene.regions.length === 1 && scene.regions[0].kind === expectedRegionKind
      : scene.regions.length === 0);
  if (!regionsMatchPreset) {
    throw new Error("区域几何缺少有效的种类、圆心、半径、单位或介电常数，或与预设不符");
  }
  const metadata = scene.metadata;
  const terminationCounts = metadata?.termination_counts;
  const startTerminationCounts = metadata?.start_termination_counts;
  const validCountMap = (counts) =>
    counts &&
    typeof counts === "object" &&
    !Array.isArray(counts) &&
    Object.values(counts).every((count) => Number.isInteger(count) && count >= 0);
  const hasSeedDescription = Object.hasOwn(metadata || {}, "seed_description");
  const knownSeedMode = Object.hasOwn(SEED_MODE_LABELS, metadata?.seed_mode);
  const validSeedDescription = hasSeedDescription
    ? typeof metadata.seed_description === "string" && metadata.seed_description.length > 0
    : !knownSeedMode;
  if (
    !metadata ||
    typeof metadata !== "object" ||
    Array.isArray(metadata) ||
    typeof metadata.title !== "string" ||
    typeof metadata.projection_note !== "string" ||
    typeof metadata.field_model !== "string" ||
    typeof metadata.seed_mode !== "string" ||
    metadata.seed_mode.length === 0 ||
    !validSeedDescription ||
    !validCountMap(terminationCounts) ||
    !validCountMap(startTerminationCounts) ||
    !Number.isInteger(metadata.suppressed_count) ||
    metadata.suppressed_count < 0 ||
    !Number.isInteger(metadata.rendered_line_count) ||
    metadata.rendered_line_count < 0 ||
    metadata.rendered_line_count !== scene.lines.length
  ) {
    throw new Error("场景 metadata 缺少有效的模型、播种或终止统计");
  }
  return scene;
}
