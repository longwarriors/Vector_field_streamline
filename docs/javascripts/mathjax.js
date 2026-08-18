window.MathJax = {
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
  },
  svg: {
    fontCache: "global",
  },
};

document$.subscribe(() => {
  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetClear?.();
    window.MathJax.typesetPromise().catch((error) => {
      console.error("MathJax typesetting failed", error);
    });
  }
});
