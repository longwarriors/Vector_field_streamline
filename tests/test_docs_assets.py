from __future__ import annotations

import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATHJAX_SHA256 = "a4354ff94fd868aea0cc6eaaa79a57fda0588646fc46ee3700a349ee0a11cbe6"


def test_mathjax_runtime_is_local_and_pinned() -> None:
    config = (PROJECT_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    runtime = (
        PROJECT_ROOT / "docs" / "javascripts" / "vendor" / "mathjax" / "tex-svg-full.js"
    )

    assert "javascripts/vendor/mathjax/tex-svg-full.js" in config
    assert "cdn.jsdelivr.net/npm/mathjax" not in config
    assert runtime.is_file()
    assert hashlib.sha256(runtime.read_bytes()).hexdigest() == MATHJAX_SHA256


def test_mathjax_license_is_vendored() -> None:
    license_file = (
        PROJECT_ROOT / "docs" / "javascripts" / "vendor" / "mathjax" / "LICENSE"
    )

    assert license_file.is_file()
    assert "Apache License" in license_file.read_text(encoding="utf-8")
