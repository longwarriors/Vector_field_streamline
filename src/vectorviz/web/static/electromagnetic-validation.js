/** One frozen observable of the same three-dimensional retarded dipole field. */
export const ELECTROMAGNETIC_PRESET = "dipole_radiation";
export const ELECTROMAGNETIC_OBSERVABLES = Object.freeze({
  electric: { label: "瞬时电场线", symbol: "E", unit: "V/m", slice: "xz", vertical: "z" },
  magnetic: { label: "瞬时磁场线", symbol: "B", unit: "T", slice: "xy", vertical: "y" },
  poynting: { label: "瞬时能流线", symbol: "S", unit: "W/m²", slice: "xz", vertical: "z" },
});

const close = (a, b) => Number.isFinite(a) && Number.isFinite(b) &&
  Math.abs(a - b) <= 16 * Number.EPSILON * Math.max(Math.abs(a), Math.abs(b), Number.MIN_VALUE);

export function validateElectromagnetic(scene, preset, expected) {
  if (preset !== ELECTROMAGNETIC_PRESET) {
    if (scene.electromagnetic != null) throw new Error("此预设不能带迟滞电磁场数据");
    return;
  }
  const em = scene.electromagnetic;
  const quantity = ELECTROMAGNETIC_OBSERVABLES[em?.observable];
  const fail = () => { throw new Error("迟滞电磁场的观察量、时间、切片或单位契约无效"); };
  if (!em || !quantity || scene.flow != null || scene.sources.length || scene.regions.length ||
      em.time_unit !== "s" || !Number.isFinite(em.observation_time) ||
      em.observation_time < 0 || em.observation_time > 1e-8 ||
      em.frequency_hz !== 1e8 || em.period !== 1e-8 || em.moment_amplitude !== 1e-9 ||
      em.phase !== 0 || em.propagation_speed !== 299792458 || !close(em.wavelength, 2.99792458) ||
      em.slice !== quantity.slice || em.horizontal_axis !== "x" || em.vertical_axis !== quantity.vertical ||
      em.exclusion_radius !== 0.12 || em.source_axis !== "z" ||
      !Array.isArray(em.source_position) || em.source_position.length !== 3 ||
      !em.source_position.every((coordinate) => coordinate === 0) ||
      typeof em.quantity_description !== "string" || !em.quantity_description.trim() ||
      scene.domain.unit !== "m" || scene.domain.x[0] !== -6 || scene.domain.x[1] !== 6 ||
      scene.domain.y[0] !== -6 || scene.domain.y[1] !== 6 ||
      scene.scalar.unit !== quantity.unit || scene.vector?.unit !== quantity.unit ||
      scene.scalar.label !== `|${quantity.symbol}|` || scene.vector?.label !== quantity.symbol ||
      (expected.observable !== undefined && em.observable !== expected.observable) ||
      (expected.em_time && !close(em.observation_time, expected.em_time.observation))) fail();
}
