/** Time and particle contracts, separate from the frozen vector-grid contract. */
export const FLOW_PRESETS = new Set(["flow_oscillating", "flow_steady"]);

const TERMINATIONS = new Set([
  "time_end", "domain_exit", "seed_outside_domain", "nonfinite_field", "solver_failure",
]);
const TIME_KEYS = ["observation", "initial", "release_start", "release_end"];
const point = (value) => Array.isArray(value) && value.length === 2 && value.every(Number.isFinite);
const ordered = (values) => values.every((value, index) => index === 0 || value > values[index - 1]);
const close = (a, b) => Math.abs(a - b) <= 1e-10;

export function validateFlow(scene, preset, expected) {
  if (!FLOW_PRESETS.has(preset)) {
    if (scene.flow != null) throw new Error("静态场景不能带含时流数据");
    return;
  }
  const flow = scene.flow;
  const fail = () => { throw new Error("含时流的时间、粒子采样或单位契约无效"); };
  if (!flow || flow.time_unit !== "s" || scene.scalar.unit !== "m/s" ||
      scene.vector?.unit !== "m/s" || scene.sources.length || scene.regions.length) fail();
  const time = flow.time;
  if (!time || !TIME_KEYS.every((key) => Number.isFinite(time[key]) && time[key] >= 0 && time[key] <= 6) ||
      time.initial > time.observation || time.release_start > time.release_end ||
      time.release_end > time.observation) fail();
  if (expected.time && TIME_KEYS.some((key) => !close(expected.time[key], time[key]))) fail();
  const sites = flow.release_sites;
  const count = scene.metadata.manual_seed_count + 1;
  if (!Array.isArray(sites) || sites.length !== count || !sites.every(point) ||
      sites[0][0] !== -2 || sites[0][1] !== 0 ||
      !Array.isArray(flow.pathlines) || flow.pathlines.length !== count ||
      !Array.isArray(flow.streaklines) || flow.streaklines.length !== count) fail();
  const inDomain = ([x, y]) => x >= scene.domain.x[0] - 1e-8 && x <= scene.domain.x[1] + 1e-8 &&
    y >= scene.domain.y[0] - 1e-8 && y <= scene.domain.y[1] + 1e-8;
  if (!sites.every(inDomain)) fail();
  if (expected.seeds && expected.seeds.some((seed, index) =>
    !close(seed.x, sites[index + 1]?.[0]) || !close(seed.y, sites[index + 1]?.[1]))) fail();
  flow.pathlines.forEach((line, index) => {
    if (!line || line.site_index !== index || !Array.isArray(line.points) || !line.points.length ||
        !line.points.every((value) => point(value) && inDomain(value)) ||
        !Array.isArray(line.times) || line.times.length !== line.points.length ||
        !line.times.every((value) => Number.isFinite(value) && value >= time.initial && value <= time.observation) ||
        !ordered(line.times) || !close(line.times[0], time.initial) ||
        !line.points[0].every((value, axis) => close(value, sites[index][axis])) ||
        !TERMINATIONS.has(line.termination) ||
        (line.termination === "time_end" && !close(line.times.at(-1), time.observation))) fail();
  });
  flow.streaklines.forEach((line, index) => {
    if (!line || line.site_index !== index || !Array.isArray(line.points) || !line.points.length ||
        !Array.isArray(line.release_times) || !Array.isArray(line.arrival_times) ||
        !Array.isArray(line.terminations) ||
        [line.release_times, line.arrival_times, line.terminations].some((values) => values.length !== line.points.length) ||
        !line.release_times.every((value) => Number.isFinite(value) && value >= time.release_start && value <= time.release_end) ||
        !ordered(line.release_times) || !close(line.release_times[0], time.release_start) ||
        !close(line.release_times.at(-1), time.release_end) ||
        (time.release_start === time.release_end && line.points.length !== 1) ||
        (time.release_start !== time.release_end && (line.points.length < 2 ||
          (expected.density !== undefined && line.points.length > expected.density)))) fail();
    line.points.forEach((value, particle) => {
      const reason = line.terminations[particle];
      const arrived = line.arrival_times[particle];
      if (!TERMINATIONS.has(reason) || !Number.isFinite(arrived) ||
          arrived < line.release_times[particle] || arrived > time.observation ||
          (reason === "time_end"
            ? !point(value) || !inDomain(value) || !close(arrived, time.observation)
            : value !== null)) fail();
      if (line.release_times[particle] === time.observation &&
          (reason !== "time_end" || !value.every((coordinate, axis) => coordinate === sites[index][axis]))) fail();
    });
  });
}
