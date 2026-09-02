"""FastAPI application serving the scientific API and browser viewer."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from vectorviz import __version__

from .scene import MIN_SOURCE_SEPARATION, build_scene
from .schemas import (
    PresetPayload,
    SceneRequest,
    SceneResponse,
    SourceSeparationCapability,
)

STATIC_DIR = Path(__file__).with_name("static")

SOURCE_SEPARATION_CAPABILITY = SourceSeparationCapability(
    exclusive_minimum=MIN_SOURCE_SEPARATION,
    unit="m",
)

PRESETS = [
    PresetPayload(
        id="electric_dipole",
        label="电偶极子",
        description="可拖动正负点电荷；显示 z=0 对称平面中的真实电场线。",
        source_separation=SOURCE_SEPARATION_CAPABILITY,
    ),
    PresetPayload(
        id="magnetic_dipole",
        label="磁偶极子",
        description="理想点磁偶极子的对称平面磁力线。",
        source_separation=SOURCE_SEPARATION_CAPABILITY,
    ),
    PresetPayload(
        id="halbach_array",
        label="Halbach 阵列",
        description="八个面内磁偶极子依次旋转 90°，形成一侧增强的磁场。",
        source_separation=SOURCE_SEPARATION_CAPABILITY,
    ),
    PresetPayload(
        id="current_loop",
        label="圆形电流线圈",
        description="固定理想细圆环在 z=0 子午面中的真实磁力线。",
    ),
    PresetPayload(
        id="uniform",
        label="匀强电场",
        description="用于验证积分器的直线基准场。",
    ),
]


def _json_safe_float(value: float) -> float | str:
    return value if math.isfinite(value) else repr(value)


def create_app() -> FastAPI:
    """Create an application instance for servers and API tests."""

    application = FastAPI(
        title="VectorViz API",
        summary="Field evaluation and adaptive field-line tracing",
        version=__version__,
    )

    @application.exception_handler(RequestValidationError)
    async def request_validation_error(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        detail = jsonable_encoder(
            error.errors(),
            custom_encoder={float: _json_safe_float},
        )
        return JSONResponse(status_code=422, content={"detail": detail})

    @application.get("/api/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @application.get(
        "/api/presets",
        response_model=list[PresetPayload],
        response_model_exclude_none=True,
        tags=["scenes"],
    )
    def presets() -> list[PresetPayload]:
        return PRESETS

    @application.post(
        "/api/scene",
        response_model=SceneResponse,
        response_model_exclude_none=True,
        tags=["scenes"],
    )
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
