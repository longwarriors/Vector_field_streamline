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
  straight_wire: ["wire_out"],
  parallel_wires: ["wire_out", "wire_out"],
  antiparallel_wires: ["wire_into", "wire_out"],
  solenoid: [...Array(8).fill("wire_into"), ...Array(8).fill("wire_out")],
  helmholtz_coils: ["wire_into", "wire_into", "wire_out", "wire_out"],
  anti_helmholtz_coils: ["wire_into", "wire_into", "wire_out", "wire_out"],
  charged_ring: ["ring_charge", "ring_charge"],
});

export const REGION_KINDS = new Set(["dielectric_sphere", "conducting_sphere", "magnetic_shield_cylinder"]);

export const MATHEMATICAL_PRESETS = new Set([
  "laplace_saddle", "laplace_multipole", "slope_linear", "slope_logistic",
]);

export const REGION_PRESETS = new Set(REGION_KINDS);

export const SEED_MODE_LABELS = Object.freeze({
  coverage: "覆盖播种",
  equal_flux: "等通量播种",
  feature: "特征播种",
});

// Validates the response for the preset that requested it: fixed presets
// must return exactly their read-only markers or their own material region.
// `expected.seedCount` is how many manual seeds the request carried; the
// response must account for exactly that many.
export function validateScene(scene, preset, expected = {}) {
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
    coordinateUnit !== (MATHEMATICAL_PRESETS.has(preset) ? "1" : "m")
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

  // The vector grid is additive and nullable in the schema: absent or null
  // means an older server; once present it is the same grid as the scalar,
  // with both components null exactly where the scalar mask is true.
  if (scene.vector === null) delete scene.vector;
  if (scene.vector !== undefined) {
    const vector = scene.vector;
    const validVector =
      vector &&
      typeof vector === "object" &&
      !Array.isArray(vector) &&
      vector.nx === nx &&
      vector.ny === ny &&
      Array.isArray(vector.u) &&
      Array.isArray(vector.v) &&
      vector.u.length === nx * ny &&
      vector.v.length === nx * ny &&
      typeof vector.label === "string" &&
      typeof vector.unit === "string" &&
      vector.u.every((u, index) =>
        mask[index]
          ? u === null && vector.v[index] === null
          : typeof u === "number" &&
            Number.isFinite(u) &&
            typeof vector.v[index] === "number" &&
            Number.isFinite(vector.v[index]),
      );
    if (!validVector) {
      throw new Error("向量网格与标量网格的尺寸或遮罩不一致");
    }
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
  // Each fixed preset returns its own marker set with one shared magnitude;
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
  // present it must be a list describing the material geometry.
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
      const noPermittivity = permittivity === undefined || permittivity === null;
      if (region.kind === "magnetic_shield_cylinder") {
        return noPermittivity && Number.isFinite(region.inner_radius) &&
          region.inner_radius > 0 && region.inner_radius < region.radius &&
          Number.isFinite(region.relative_permeability) && region.relative_permeability >= 1;
      }
      if (region.inner_radius != null || region.relative_permeability != null) return false;
      if (region.kind === "dielectric_sphere") {
        return Number.isFinite(permittivity) && permittivity >= 1;
      }
      return noPermittivity;
    });
  // A material preset returns exactly one region of its own kind.
  const expectedRegionKind = REGION_PRESETS.has(preset)
    ? preset
    : null;
  const regionsMatchPreset =
    validRegions &&
    (expectedRegionKind
      ? scene.regions.length === 1 && scene.regions[0].kind === expectedRegionKind
      : scene.regions.length === 0);
  if (!regionsMatchPreset) {
    throw new Error("区域几何缺少有效的种类、圆心、半径、单位或材料参数，或与预设不符");
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
  // Manual seeds are accounted apart from the budget. An older server has
  // none; a current one must report exactly the seeds the request carried,
  // one end-side and one start-side reason each, and flag at most one line
  // per seed.
  // Only a field that is absent altogether means an older server; an explicit
  // null is a malformed response, as the server schema also says.
  if (metadata.manual_seed_count === undefined) metadata.manual_seed_count = 0;
  if (metadata.manual_termination_counts === undefined) metadata.manual_termination_counts = {};
  if (metadata.manual_start_termination_counts === undefined) {
    metadata.manual_start_termination_counts = {};
  }
  const sum = (counts) => Object.values(counts).reduce((total, count) => total + count, 0);
  const seedIndices = scene.lines
    .map((line) => line.seed_index)
    .filter((index) => index !== undefined && index !== null);
  if (
    !Number.isInteger(metadata.manual_seed_count) ||
    metadata.manual_seed_count < 0 ||
    !validCountMap(metadata.manual_termination_counts) ||
    !validCountMap(metadata.manual_start_termination_counts) ||
    sum(metadata.manual_termination_counts) !== metadata.manual_seed_count ||
    sum(metadata.manual_start_termination_counts) !== metadata.manual_seed_count ||
    !seedIndices.every(
      (index) => Number.isInteger(index) && index >= 0 && index < metadata.manual_seed_count,
    ) ||
    new Set(seedIndices).size !== seedIndices.length
  ) {
    throw new Error("手动种子的统计或 seed_index 无效");
  }
  if (expected.seedCount !== undefined && metadata.manual_seed_count !== expected.seedCount) {
    throw new Error(
      `服务端记录了 ${metadata.manual_seed_count} 个手动种子，请求发送的是 ${expected.seedCount} 个`,
    );
  }
  return scene;
}
