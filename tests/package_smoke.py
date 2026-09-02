"""Smoke-check an installed distribution without importing from the repository root."""

from __future__ import annotations

import os
import tempfile
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    """Verify public imports and packaged browser assets from an isolated working directory."""

    original_directory = Path.cwd()
    with tempfile.TemporaryDirectory() as temporary_directory:
        try:
            os.chdir(temporary_directory)

            from vectorviz import FieldLineTracer, UniformField, __version__
            from vectorviz.web.app import STATIC_DIR, create_app

            assert version("vector-field-streamline") == __version__
            assert FieldLineTracer.__module__ == "vectorviz.tracing"
            assert UniformField.__module__ == "vectorviz.fields"
            for asset in (
                "index.html",
                "app.js",
                "source-controls.js",
                "coordinates.js",
                "color-scale.js",
                "styles.css",
            ):
                assert Path(STATIC_DIR, asset).is_file()
            app = create_app()
            assert app.title == "VectorViz API"
            assert app.version == __version__
        finally:
            os.chdir(original_directory)


if __name__ == "__main__":
    main()
