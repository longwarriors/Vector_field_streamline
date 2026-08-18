"""FastAPI application serving the scientific API and browser viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from vectorviz import __version__

from .scene import build_scene
from .schemas import PresetPayload, SceneRequest, SceneResponse

STATIC_DIR = Path(__file__).with_name("static")

PRESETS = [
    PresetPayload(
        id="electric_dipole",
        label="电偶极子",
        description="可拖动正负点电荷；显示 z=0 对称平面中的真实电场线。",
    ),
    PresetPayload(
        id="magnetic_dipole",
        label="磁偶极子",
        description="理想点磁偶极子的对称平面磁力线。",
    ),
    PresetPayload(
        id="uniform",
        label="匀强电场",
        description="用于验证积分器的直线基准场。",
    ),
]


def create_app() -> FastAPI:
    """Create an application instance for servers and API tests."""

    application = FastAPI(
        title="VectorViz API",
        summary="Field evaluation and adaptive field-line tracing",
        version=__version__,
    )

    @application.get("/api/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @application.get("/api/presets", response_model=list[PresetPayload], tags=["scenes"])
    def presets() -> list[PresetPayload]:
        return PRESETS

    @application.post("/api/scene", response_model=SceneResponse, tags=["scenes"])
    def scene(request: SceneRequest) -> SceneResponse:
        try:
            return build_scene(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    if not STATIC_DIR.is_dir():
        raise RuntimeError(f"browser assets are missing: {STATIC_DIR}")
    application.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
    return application


def main(argv: list[str] | None = None) -> None:
    """Run the local development server."""

    parser = argparse.ArgumentParser(description="Run the VectorViz browser application.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    arguments = parser.parse_args(argv)
    target: Any = "vectorviz.web.app:create_app" if arguments.reload else create_app()
    uvicorn.run(
        target,
        host=arguments.host,
        port=arguments.port,
        reload=arguments.reload,
        factory=arguments.reload,
    )


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    main()
