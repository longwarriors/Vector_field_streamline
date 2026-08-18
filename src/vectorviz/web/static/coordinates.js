/** Pure coordinate transforms shared by every Canvas scene layer. */

export function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

export function calculatePlotRect(width, height, domain) {
  const leftMargin = width < 520 ? 36 : 48;
  const rightMargin = width < 520 ? 55 : 68;
  const topMargin = 20;
  const bottomMargin = 36;
  const availableWidth = Math.max(1, width - leftMargin - rightMargin);
  const availableHeight = Math.max(1, height - topMargin - bottomMargin);
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

export function sampleNearest(scalar, domain, x, y) {
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
  return scalar.mask?.[index] ? Number.NaN : Number(scalar.values[index]);
}
