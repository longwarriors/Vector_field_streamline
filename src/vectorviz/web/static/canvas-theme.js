/** Colors, line widths and fonts used when painting the field canvas. */

function deepFreeze(value) {
  Object.values(value).forEach((child) => {
    if (child && typeof child === "object") deepFreeze(child);
  });
  return Object.freeze(value);
}

const FONT_STACK =
  '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei UI", ' +
  '"Microsoft YaHei", system-ui, sans-serif';

export const CANVAS_THEME = deepFreeze({
  // White paper; the plot frame and marker rings carry the edges instead.
  paper: "#ffffff",
  axes: {
    font: `14px ${FONT_STACK}`,
    tick: "#1c1c1e",
    // Grid lines over the heatmap, and over bare paper while the field is stale.
    gridOnField: "rgba(255, 255, 255, 0.16)",
    gridOnPaper: "#e0e2e8",
    // Steel: 5.0:1 on paper, so the plot edge stays visible where viridis is pale.
    frame: "#6b6f7e",
    title: "#1c1c1e",
    titleFont: `600 14px ${FONT_STACK}`,
  },
  line: {
    halo: "rgba(0, 8, 12, 0.5)",
    haloWidth: 3.4,
    core: "rgba(248, 255, 253, 0.88)",
    coreWidth: 1.15,
  },
  arrow: {
    fill: "rgba(255, 255, 255, 0.94)",
    stroke: "rgba(0, 8, 12, 0.58)",
    strokeWidth: 2.6,
  },
  marker: {
    fill: {
      positive: "#ff725f",
      negative: "#65b9ff",
      dipole: "#ffe08a",
      wire_out: "#ffb45f",
      wire_into: "#ffb45f",
    },
    // A white inner ring and an ink outer ring: on any background one of the
    // two keeps at least 4:1, so markers read on viridis and on bare paper.
    ring: "#ffffff",
    ringWidth: 2,
    outerRing: "#1c1c1e",
    outerRingRadius: 12,
    outerRingWidth: 2,
    glyph: "#1c1c1e",
    glyphFont: `600 13px ${FONT_STACK}`,
    selection: "#4262ff",
    selectionUnder: "#ffffff",
  },
  // Cells that cannot be coloured: a grey darker than the colormap maximum,
  // so a hole is never read as a stronger field, crossed by ink hatching.
  hatch: {
    base: "#c7cad5",
    line: "rgba(28, 28, 30, 0.5)",
    spacing: 4,
  },
});
