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
    r"""|(?:src|href|action)\s*=\s*["']?//"""
    r"""|url\(\s*["']?//"""
    r"""|@import"""
    r"""|(?:from|import|fetch)\s*\(?\s*["'`]//""",
    re.IGNORECASE,
)
THIRD_PARTY_BRANDS = re.compile(r"miro|roobert", re.IGNORECASE)
# Brand yellow and its deep variant, as hex or rgb().
BRAND_YELLOWS = re.compile(
    r"#ffd02f|#fcb900|rgba?\(\s*255\s*,\s*208\s*,\s*47|rgba?\(\s*252\s*,\s*185\s*,\s*0",
    re.IGNORECASE,
)


CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
ROOT_BLOCK = re.compile(r"^:root \{\n(?P<body>.*?)^\}", re.DOTALL | re.MULTILINE)
CUSTOM_PROPERTY = re.compile(r"--(?P<name>[\w-]+)\s*:\s*(?P<value>[^;]+);")
# Colour literals inside declaration values (never selectors such as #add-...).
DECLARATION_VALUE = re.compile(r"(?<=:)[^;{}]*(?=;)")
COLOR_LITERAL = re.compile(
    r"(?<![\w-])#[0-9a-f]{3,8}(?![\w-])"
    r"|\b(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color-mix|color)\("
    r"|\b(?:white|black|red|green|blue|yellow|orange|purple|pink|gray|grey|silver)\b",
    re.IGNORECASE,
)
NOTE_TOKENS = ("note-positive", "note-negative", "note-dipole", "note-wire", "note-guide")


def _relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(first: str, second: str) -> float:
    lighter, darker = sorted((_relative_luminance(first), _relative_luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _static_text_files() -> list[Path]:
    files = sorted(path for path in STATIC_ROOT.rglob("*") if path.suffix in TEXT_SUFFIXES)
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
    for path in _static_text_files():
        if path.suffix != ".js":
            continue
        for match in BRAND_YELLOWS.finditer(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.name}: brand yellow {match.group(0)!r} in script")
    assert offenders == []


def test_stylesheet_tokens_meet_contrast_floors() -> None:
    css = CSS_COMMENT.sub("", (STATIC_ROOT / "styles.css").read_text(encoding="utf-8"))
    root = ROOT_BLOCK.search(css)
    assert root is not None, "styles.css must open with a :root token block"
    tokens = {
        match.group("name"): match.group("value").strip()
        for match in CUSTOM_PROPERTY.finditer(root.group("body"))
    }
    outside_root = css[: root.start()] + css[root.end() :]
    literals = [
        match.group(0)
        for value in DECLARATION_VALUE.findall(outside_root)
        for match in COLOR_LITERAL.finditer(value)
    ]
    assert literals == [], "colours must be :root tokens"

    def color(name: str) -> str:
        value = tokens[f"vv-{name}"]
        assert re.fullmatch(r"#[0-9a-f]{6}", value, re.IGNORECASE), (name, value)
        return value

    # (foreground, background, floor): 4.5 for text, 3 for control outlines and focus.
    floors = [
        ("text-2", "surface", 4.5),
        ("text-2", "board", 4.5),
        ("text-3", "surface", 4.5),
        ("text-3", "board", 4.5),
        ("link", "surface", 4.5),
        ("danger-bar", "surface", 4.5),
        ("on-ink", "ink", 4.5),
        ("on-ink-muted", "ink", 4.5),
        ("ink", "brand", 4.5),
        ("badge-fg", "badge-bg", 4.5),
        ("ok-fg", "ok-bg", 4.5),
        ("busy-fg", "busy-bg", 4.5),
        ("idle-fg", "idle-bg", 4.5),
        ("danger-fg", "danger-bg", 4.5),
        ("control-border", "surface", 3.0),
        ("control-border", "board", 3.0),
        ("focus", "surface", 3.0),
        ("focus", "board", 3.0),
    ]
    for note in NOTE_TOKENS:
        floors += [("ink", note, 4.5), ("text-2", note, 4.5)]
        floors += [("control-border", note, 3.0), ("focus", note, 3.0)]
    failures = [
        f"{fg} on {bg}: {_contrast(color(fg), color(bg)):.2f} < {floor}"
        for fg, bg, floor in floors
        if _contrast(color(fg), color(bg)) < floor
    ]
    assert failures == []

    # Brand yellow would read as a dipole marker next to the plot: wordmark only.
    brand_rules = re.findall(r"([^{}]+)\{[^{}]*var\(\s*--vv-brand\b", outside_root)
    assert [selector.strip() for selector in brand_rules] == [".brand strong::before"]


def _canvas_theme_value(group: str, key: str) -> str:
    theme = (STATIC_ROOT / "canvas-theme.js").read_text(encoding="utf-8")
    block = re.search(
        rf"^  {group}: \{{\n(?P<body>.*?)^  \}},", theme, re.DOTALL | re.MULTILINE
    )
    assert block is not None, group
    value = re.search(rf'^    {key}: "(?P<value>[^"]+)",', block.group("body"), re.MULTILINE)
    assert value is not None, (group, key)
    return value.group("value")


def test_legend_tokens_mirror_the_canvas_theme() -> None:
    # The legend chips are how a reader learns what lines and hatching mean,
    # so their tokens must equal the colours the canvas actually draws.
    css = CSS_COMMENT.sub("", (STATIC_ROOT / "styles.css").read_text(encoding="utf-8"))
    root = ROOT_BLOCK.search(css)
    assert root is not None
    tokens = {
        match.group("name"): match.group("value").strip()
        for match in CUSTOM_PROPERTY.finditer(root.group("body"))
    }
    mirrored = {
        "vv-legend-core": ("line", "core"),
        "vv-legend-halo": ("line", "halo"),
        "vv-hatch-base": ("hatch", "base"),
        "vv-hatch-line": ("hatch", "line"),
    }
    assert {name: tokens[name] for name in mirrored} == {
        name: _canvas_theme_value(*source) for name, source in mirrored.items()
    }
