/**
 * Browser-side preview of field lines around the cursor.
 *
 * The preview interpolates the server's sampled vector grid bilinearly and
 * marches along the unit tangent with fixed-step RK4. It is a display aid
 * that depends on the grid resolution; it is not the server's adaptive
 * integration of the analytic field, which a pinned seed requests instead.
 */

// The grid a preview reads: the vector components, the scalar mask that says
// where they are missing, the domain the grid spans, and the conductors,
// whose interior is a true zero field that interpolation across the surface
// must not blur.
export function createPreviewGrid(scene) {
  const { vector, scalar, domain, regions = [] } = scene;
  return {
    nx: vector.nx,
    ny: vector.ny,
    u: vector.u,
    v: vector.v,
    mask: scalar.mask,
    domain: { x: [...domain.x], y: [...domain.y] },
    conductors: regions
      .filter((region) => region.kind === "conducting_sphere")
      .map(({ x, y, radius }) => ({ x, y, radius })),
  };
}

function insideConductor(grid, x, y) {
  return (grid.conductors || []).some(
    (conductor) => Math.hypot(x - conductor.x, y - conductor.y) <= conductor.radius,
  );
}

// Below this fraction of the largest corner magnitude an interpolated vector
// is rounding residue of cancelling corners (about 1e-16 relative at a null
// point), not a direction.
const ZERO_RELATIVE = 1e-9;

// Bilinear interpolation of (u, v) at a world position, or null outside the
// domain or when any corner of the cell is masked, so the preview never
// invents a field inside an exclusion disc or a non-finite hole. Inside a
// conductor, and where the corners cancel to rounding, the vector is zero.
export function interpolateVector(grid, x, y) {
  const { nx, ny, u, v, mask, domain } = grid;
  const [xmin, xmax] = domain.x;
  const [ymin, ymax] = domain.y;
  if (!(x >= xmin && x <= xmax && y >= ymin && y <= ymax)) return null;
  if (insideConductor(grid, x, y)) return [0, 0];
  const s = ((x - xmin) / (xmax - xmin)) * (nx - 1);
  const t = ((ymax - y) / (ymax - ymin)) * (ny - 1);
  const i = Math.min(Math.floor(s), nx - 2);
  const j = Math.min(Math.floor(t), ny - 2);
  const fx = s - i;
  const fy = t - j;
  const corners = [j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1];
  if (corners.some((index) => mask[index])) return null;
  const weights = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy];
  let ux = 0;
  let uy = 0;
  let scale = 0;
  corners.forEach((index, corner) => {
    ux += weights[corner] * u[index];
    uy += weights[corner] * v[index];
    scale = Math.max(scale, Math.hypot(u[index], v[index]));
  });
  if (!(Math.hypot(ux, uy) > ZERO_RELATIVE * scale)) return [0, 0];
  return [ux, uy];
}

function clipToDomain(from, to, domain) {
  const [xmin, xmax] = domain.x;
  const [ymin, ymax] = domain.y;
  let fraction = 1;
  if (to[0] < xmin) fraction = Math.min(fraction, (xmin - from[0]) / (to[0] - from[0]));
  if (to[0] > xmax) fraction = Math.min(fraction, (xmax - from[0]) / (to[0] - from[0]));
  if (to[1] < ymin) fraction = Math.min(fraction, (ymin - from[1]) / (to[1] - from[1]));
  if (to[1] > ymax) fraction = Math.min(fraction, (ymax - from[1]) / (to[1] - from[1]));
  return [from[0] + fraction * (to[0] - from[0]), from[1] + fraction * (to[1] - from[1])];
}

// Unit-speed RK4 from `seed` along (`sign` > 0) or against the field. The
// step is in world units. Stops at the domain edge (clipped onto it), at a
// masked cell, at a zero vector, after `maxSteps`, or on returning to the
// seed. Returns the points after the seed and why the march stopped.
function march(grid, seed, sign, step, maxSteps) {
  const { domain } = grid;
  const points = [];
  let [x, y] = seed;
  let termination = "max_steps";

  const tangent = (px, py) => {
    const vector = interpolateVector(grid, px, py);
    if (!vector) return null;
    const length = Math.hypot(vector[0], vector[1]);
    if (!(length > 0)) return null;
    return [(sign * vector[0]) / length, (sign * vector[1]) / length];
  };
  const stopReason = (px, py) =>
    px < domain.x[0] || px > domain.x[1] || py < domain.y[0] || py > domain.y[1]
      ? "domain_exit"
      : interpolateVector(grid, px, py) === null
        ? "masked"
        : "null_field";

  for (let count = 0; count < maxSteps; count += 1) {
    const k1 = tangent(x, y);
    if (!k1) {
      termination = stopReason(x, y);
      break;
    }
    const k2 = tangent(x + 0.5 * step * k1[0], y + 0.5 * step * k1[1]);
    const k3 = k2 && tangent(x + 0.5 * step * k2[0], y + 0.5 * step * k2[1]);
    const k4 = k3 && tangent(x + step * k3[0], y + step * k3[1]);
    if (!k4) {
      const probe = k3
        ? [x + step * k3[0], y + step * k3[1]]
        : k2
          ? [x + 0.5 * step * k2[0], y + 0.5 * step * k2[1]]
          : [x + 0.5 * step * k1[0], y + 0.5 * step * k1[1]];
      termination = stopReason(probe[0], probe[1]);
      if (termination === "domain_exit") points.push(clipToDomain([x, y], probe, domain));
      break;
    }
    const nextX = x + (step / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
    const nextY = y + (step / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
    if (
      nextX < domain.x[0] ||
      nextX > domain.x[1] ||
      nextY < domain.y[0] ||
      nextY > domain.y[1]
    ) {
      points.push(clipToDomain([x, y], [nextX, nextY], domain));
      termination = "domain_exit";
      break;
    }
    x = nextX;
    y = nextY;
    points.push([x, y]);
    if (count > 8 && Math.hypot(x - seed[0], y - seed[1]) < 0.55 * step) {
      termination = "closed_loop";
      break;
    }
  }
  return { points, termination };
}

// The default step is half a grid cell; the default budget lets each side
// run about two domain widths, enough to cross the plot and to close a loop.
export function previewOptions(grid, overrides = {}) {
  const cell = Math.max(
    (grid.domain.x[1] - grid.domain.x[0]) / (grid.nx - 1),
    (grid.domain.y[1] - grid.domain.y[0]) / (grid.ny - 1),
  );
  return {
    step: overrides.step ?? 0.5 * cell,
    maxSteps: overrides.maxSteps ?? 4 * Math.max(grid.nx, grid.ny),
    ...overrides,
  };
}

// Both directions from one seed, ordered from the backward end through the
// seed to the forward end like the server's bidirectional lines. A forward
// march that closes on the seed already is the whole curve, so the backward
// march is skipped rather than drawn over it.
export function integratePreview(grid, seed, overrides = {}) {
  const { step, maxSteps } = previewOptions(grid, overrides);
  const forward = march(grid, seed, 1, step, maxSteps);
  const backward =
    forward.termination === "closed_loop"
      ? { points: [], termination: "closed_loop" }
      : march(grid, seed, -1, step, maxSteps);
  return {
    points: [...backward.points.reverse(), [seed[0], seed[1]], ...forward.points],
    termination: forward.termination,
    startTermination: backward.termination,
  };
}

// A bundle of preview curves through `centre` and through seeds spaced along
// the normal to the local field direction, so neighbouring lines fan out
// with the field. Empty when the centre has no usable direction.
export function previewBundle(grid, centre, overrides = {}) {
  const options = previewOptions(grid, overrides);
  const count = options.count ?? 5;
  const spacing = options.spacing ?? options.step;
  const vector = interpolateVector(grid, centre[0], centre[1]);
  if (!vector) return [];
  const length = Math.hypot(vector[0], vector[1]);
  if (!(length > 0)) return [];
  const normal = [-vector[1] / length, vector[0] / length];
  const curves = [];
  for (let k = -(count - 1) / 2; k <= (count - 1) / 2; k += 1) {
    const seed = [centre[0] + k * spacing * normal[0], centre[1] + k * spacing * normal[1]];
    if (interpolateVector(grid, seed[0], seed[1]) === null) continue;
    const curve = integratePreview(grid, seed, options);
    if (curve.points.length < 2) continue;
    curves.push({ ...curve, seed, central: k === 0 });
  }
  return curves;
}
