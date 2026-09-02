"""Dependency direction: the scientific core must never depend on the web layer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_MODULES = ("__init__.py", "core.py", "fields.py", "tracing.py")


def test_core_sources_never_reference_the_web_package() -> None:
    for name in CORE_MODULES:
        source = (ROOT / "src" / "vectorviz" / name).read_text(encoding="utf-8")
        assert "vectorviz.web" not in source, name
        assert "from .web" not in source, name
        assert "from . import web" not in source, name


def test_importing_the_core_does_not_load_web_seeding() -> None:
    script = (
        "import sys\n"
        "import vectorviz\n"
        "from vectorviz import CircularLoopField, FieldLineTracer, TraceOptions\n"
        "loaded = sorted(name for name in sys.modules if name.startswith('vectorviz.web'))\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == ""
