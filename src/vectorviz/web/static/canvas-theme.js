/** Colors, line widths and fonts used when painting the field canvas. */

function deepFreeze(value) {
  Object.values(value).forEach((child) => {
    if (child && typeof child === "object") deepFreeze(child);
  });
  return Object.freeze(value);
}

export const CANVAS_THEME = deepFreeze({
  paper: "#07111a",
  axes: {
    font: "10px ui-sans-serif, system-ui, sans-serif",
    tick: "rgba(222, 238, 241, 0.65)",
    grid: "rgba(228, 248, 247, 0.12)",
    frame: "rgba(235, 250, 250, 0.3)",
    title: "rgba(222, 238, 241, 0.8)",
    titleFont: "600 11px ui-sans-serif, system-ui, sans-serif",
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
    shadow: "rgba(0, 0, 0, 0.48)",
    shadowBlur: 10,
    ring: "rgba(255, 255, 255, 0.94)",
    ringWidth: 2,
    glyph: "#061018",
    glyphFont: "800 13px ui-sans-serif, system-ui, sans-serif",
    selection: "rgba(89, 225, 193, 0.9)",
  },
});
