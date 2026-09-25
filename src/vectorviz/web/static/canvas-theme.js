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
    gridOnField: "rgba(28, 28, 30, 0.07)",
    gridOnPaper: "#e0e2e8",
    // Steel: 5.0:1 on paper, so the plot edge stays visible where the field is weak.
    frame: "#6b6f7e",
    title: "#1c1c1e",
    titleFont: `600 14px ${FONT_STACK}`,
  },
  // Ink lines on a white halo: the ink reads on the pale end of the colormap
  // and the halo keeps 6.4:1 against its darkest colour.
  line: {
    halo: "rgba(255, 255, 255, 0.75)",
    haloWidth: 3.2,
    core: "#1c1c1e",
    coreWidth: 1.35,
  },
  arrow: {
    fill: "#1c1c1e",
    stroke: "rgba(255, 255, 255, 0.9)",
    strokeWidth: 1.6,
  },
  marker: {
    // Category colours; none is lavender, so a source never blends into
    // the colormap.
    fill: {
      positive: "#e45f95",
      negative: "#5fa5ff",
      dipole: "#fff09a",
      wire_out: "#7fa052",
      wire_into: "#7fa052",
    },
    // A white inner ring and an ink outer ring: on any background one of the
    // two keeps at least 4:1, so markers read on the colormap and on bare paper.
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
  // Cells that cannot be coloured: a neutral grey, unlike any lavender, much
  // lighter than the colormap maximum so a hole is never read as the
  // strongest field, crossed by ink hatching.
  hatch: {
    base: "#d9dade",
    line: "rgba(28, 28, 30, 0.5)",
    spacing: 4,
  },
});
