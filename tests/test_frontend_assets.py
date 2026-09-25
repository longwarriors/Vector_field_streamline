"""File-level guards for the packaged browser frontend (no browser required)."""

from __future__ import annotations

import re
from pathlib import Path

from vectorviz.web.app import STATIC_DIR

STATIC_ROOT = Path(STATIC_DIR)
TEXT_SUFFIXES = {".html", ".css", ".js"}
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
REMOTE_REFERENCE = re.compile(
    r"""https?://"""
    r"""|(?:src|href)\s*=\s*["']//"""
    r"""|url\(\s*["']?//"""
    r"""|@import"""
    r"""|(?:from|import)\s*\(?\s*["']//""",
    re.IGNORECASE,
)
THIRD_PARTY_BRANDS = re.compile(r"miro|roobert", re.IGNORECASE)
BRAND_YELLOWS = ("#ffd02f", "#fcb900")


def _static_text_files() -> list[Path]:
    files = sorted(path for path in STATIC_ROOT.iterdir() if path.suffix in TEXT_SUFFIXES)
    assert files, "the packaged frontend has no HTML, CSS or JavaScript files"
    return files


def test_static_frontend_is_self_contained() -> None:
    offenders: list[str] = []
    for path in _static_text_files():
        text = path.read_text(encoding="utf-8").replace(SVG_NAMESPACE, "")
        for match in REMOTE_REFERENCE.finditer(text):
            offenders.append(f"{path.name}: remote reference {match.group(0)!r}")
        for match in THIRD_PARTY_BRANDS.finditer(text):
            offenders.append(f"{path.name}: third-party brand name {match.group(0)!r}")
    canvas_theme = (STATIC_ROOT / "canvas-theme.js").read_text(encoding="utf-8").lower()
    offenders.extend(
        f"canvas-theme.js: brand yellow {color} inside the plot palette"
        for color in BRAND_YELLOWS
        if color in canvas_theme
    )
    assert offenders == []
