/** Pure coordinate transforms shared by every Canvas scene layer. */

export function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

// Below this canvas width the colorbar moves under the plot.
export const NARROW_PLOT_WIDTH = 520;

// Margins leave room for tick labels, the axis unit titles and the colorbar:
// an 88 px column 14 px right of the plot when wide, a strip under it when
// narrow. The wide left margin also holds the 46 px layer toolbar, which on
// narrow frames sits above the plot instead.
export function plotMargins(width) {
  return width < NARROW_PLOT_WIDTH
    ? { left: 40, right: 16, top: 30, bottom: 108 }
    : { left: 94, right: 112, top: 30, bottom: 52 };
}

export function calculatePlotRect(width, height, domain) {
  const margins = plotMargins(width);
  const leftMargin = margins.left;
  const topMargin = margins.top;
  const availableWidth = Math.max(1, width - margins.left - margins.right);
  const availableHeight = Math.max(1, height - margins.top - margins.bottom);
  const domainWidth = domain.x[1] - domain.x[0];
  const domainHeight = domain.y[1] - domain.y[0];
  const domainAspect = domainWidth / domainHeight;

  let plotWidth = availableWidth;
  let plotHeight = plotWidth / domainAspect;
  if (plotHeight > availableHeight) {
    plotHeight = availableHeight;
    plotWidth = plotHeight * domainAspect;
  }
  const horizontalInset = (availableWidth - plotWidth) / 2;
  const verticalInset = (availableHeight - plotHeight) / 2;
  return {
    left: leftMargin + horizontalInset,
    top: topMargin + verticalInset,
    right: leftMargin + horizontalInset + plotWidth,
    bottom: topMargin + verticalInset + plotHeight,
  };
}

export function createCoordinateTransform(domain, plotRect) {
  const [xmin, xmax] = domain.x;
  const [ymin, ymax] = domain.y;
  const plotWidth = plotRect.right - plotRect.left;
  const plotHeight = plotRect.bottom - plotRect.top;

  function worldToCanvas(x, y) {
    return [
      plotRect.left + ((x - xmin) / (xmax - xmin)) * plotWidth,
      plotRect.top + ((ymax - y) / (ymax - ymin)) * plotHeight,
    ];
  }

  function canvasToWorld(canvasX, canvasY) {
    return [
      xmin + ((canvasX - plotRect.left) / plotWidth) * (xmax - xmin),
      ymax - ((canvasY - plotRect.top) / plotHeight) * (ymax - ymin),
    ];
  }

  function projectPoints(points) {
    return points.map(([x, y]) => worldToCanvas(x, y));
  }

  return {
    plotRect,
    worldToCanvas,
    canvasToWorld,
    projectPoints,
  };
}

// The grid node nearest to a world position: its indices, coordinates and
// raw value. The probe reports this node, not the pointer position.
export function nearestNode(scalar, domain, x, y) {
  const column = clamp(
    Math.round(((x - domain.x[0]) / (domain.x[1] - domain.x[0])) * (scalar.nx - 1)),
    0,
    scalar.nx - 1,
  );
  // Row zero corresponds to ymax in the HTTP scalar contract.
  const row = clamp(
    Math.round(((domain.y[1] - y) / (domain.y[1] - domain.y[0])) * (scalar.ny - 1)),
    0,
    scalar.ny - 1,
  );
  const index = row * scalar.nx + column;
  return {
    index,
    column,
    row,
    x: domain.x[0] + (column / (scalar.nx - 1)) * (domain.x[1] - domain.x[0]),
    y: domain.y[1] - (row / (scalar.ny - 1)) * (domain.y[1] - domain.y[0]),
    value: scalar.mask?.[index] ? null : Number(scalar.values[index]),
  };
}

export function sampleNearest(scalar, domain, x, y) {
  const { index } = nearestNode(scalar, domain, x, y);
  return scalar.mask?.[index] ? Number.NaN : Number(scalar.values[index]);
}
