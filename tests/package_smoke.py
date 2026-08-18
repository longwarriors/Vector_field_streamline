"""Smoke-check an installed distribution without importing from the repository root."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def main() -> None:
    """Verify public imports and packaged browser assets from an isolated working directory."""

    original_directory = Path.cwd()
    with tempfile.TemporaryDirectory() as temporary_directory:
        try:
            os.chdir(temporary_directory)

            from vectorviz import FieldLineTracer, UniformField
            from vectorviz.web.app import STATIC_DIR, create_app

            assert FieldLineTracer.__module__ == "vectorviz.tracing"
            assert UniformField.__module__ == "vectorviz.fields"
            assert Path(STATIC_DIR, "index.html").is_file()
            assert create_app().title == "VectorViz API"
        finally:
            os.chdir(original_directory)


if __name__ == "__main__":
    main()
