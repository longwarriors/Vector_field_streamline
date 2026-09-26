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
  // Ink lines on a white halo: the ink reads on the pale end of the colormap,
  // and the halo, composited over the darkest colour, is 4.4:1 against it.
  line: {
    halo: "rgba(255, 255, 255, 0.75)",
    haloWidth: 4,
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
      ring_charge: "#e45f95",
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
  // Cells that cannot be coloured: a mid neutral grey, at least 18 CIEDE2000
  // from every colormap colour (the ramp has chroma 30-46 at this lightness)
  // and 3.2 times as luminous as its maximum, so a hole reads as neither a
  // weak nor the strongest field. Hatch lines are 1 CSS px of 60% ink: 3.1:1
  // against the base where a line covers a whole pixel; antialiased 45 deg
  // lines at pixel ratio 1 show about 2.3:1.
  hatch: {
    base: "#a3a3a3",
    line: "rgba(28, 28, 30, 0.6)",
    spacing: 4,
  },
});
