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
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

from vectorviz import __version__

from .scene import MIN_SOURCE_SEPARATION, build_scene
from .schemas import (
    PresetPayload,
    SceneRequest,
    SceneResponse,
    SourceSeparationCapability,
)

STATIC_DIR = Path(__file__).with_name("static")


class _RevalidatingStaticFiles(StaticFiles):
    """Revalidate unversioned HTML and module imports before browser reuse."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        # Include 304 responses: a fresh document must not use a heuristically
        # cached validator from an older release against the current API.
        response.headers["Cache-Control"] = "no-cache"
        return response


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
        id="electric_quadrupole",
        label="电四极子",
        description="正方形顶点上正负交替的四个点电荷；中心是零场鞍点。",
        source_separation=SOURCE_SEPARATION_CAPABILITY,
    ),
    PresetPayload(
        id="electric_hexagon",
        label="六个正电荷",
        description="正六边形顶点上六个等量正电荷；平面内场线汇聚到中心零场点。",
        source_separation=SOURCE_SEPARATION_CAPABILITY,
    ),
    PresetPayload(
        id="electric_hexagon_alternating",
        label="三对交替电荷",
        description="正六边形顶点上正负交替的六个点电荷。",
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
        id="straight_wire",
        label="长直导线",
        description="无限长直导线的横截面磁场；同心闭环方向遵循右手定则。",
    ),
    PresetPayload(
        id="parallel_wires",
        label="同向双导线",
        description="两根无限长平行导线通同向电流；中点为零场点。",
    ),
    PresetPayload(
        id="antiparallel_wires",
        label="反向双导线",
        description="两根无限长平行导线通反向电流；两导线之间的磁场增强。",
    ),
    PresetPayload(
        id="solenoid",
        label="有限螺线管",
        description="有限个同轴理想圆环叠加的离散绕组近似；展示内部轴向场和外部回流。",
    ),
    PresetPayload(
        id="helmholtz_coils",
        label="Helmholtz 线圈",
        description="同向电流的两个同轴圆环，间距等于半径；中心附近磁场较均匀。",
    ),
    PresetPayload(
        id="anti_helmholtz_coils",
        label="反 Helmholtz 线圈",
        description="两个同轴圆环通反向电流；中心为零场点，附近形成磁场梯度。",
    ),
    PresetPayload(
        id="charged_ring",
        label="带电圆环",
        description="固定均匀带电细圆环在 z=0 子午面中的真实电场线；与电流线圈同几何。",
    ),
    PresetPayload(
        id="dielectric_sphere",
        label="介质球",
        description="匀强外场中相对介电常数 4 的介质球；场线在球面折射，球内匀强。",
    ),
    PresetPayload(
        id="conducting_sphere",
        label="导体球",
        description="匀强外场中的导体球；场线垂直终止于球面感应电荷，球内为零。",
    ),
    PresetPayload(
        id="uniform",
        label="匀强电场",
        description="用于验证积分器的直线基准场。",
    ),
    PresetPayload(
        id="laplace_saddle", label="Laplace 鞍点场",
        description="二维谐波势 φ=(x²−y²)/2 的负梯度；无量纲坐标。",
    ),
    PresetPayload(
        id="laplace_multipole", label="Laplace 三阶场",
        description="二维谐波势 φ=(x³−3xy²)/3 的负梯度；无量纲坐标。",
    ),
    PresetPayload(
        id="slope_linear", label="线性斜率场",
        description="dy/dx=x−y 的无向短线与解曲线；无量纲坐标。",
    ),
    PresetPayload(
        id="slope_logistic", label="Logistic 斜率场",
        description="dy/dx=y(1−y)，含 y=0 与 y=1 平衡解；无量纲坐标。",
    ),
    PresetPayload(
        id="magnetic_shield_cylinder", label="理想磁屏蔽筒",
        description="横向匀强磁场中的无限长线性圆柱壳：μr=100；展示筒壁集磁与内腔衰减。",
    ),
    PresetPayload(
        id="flow_oscillating", label="非定常流：三类曲线",
        description="u=(1, 0.6 sin t) m/s；比较瞬时流线、质点迹线与持续释放形成的脉线。",
    ),
    PresetPayload(
        id="flow_steady", label="定常流对照",
        description="u=(1, 0.4) m/s；三类曲线落在同一直线上，时间单位 s。",
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
    application.mount("/", _RevalidatingStaticFiles(directory=STATIC_DIR, html=True), name="frontend")
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
