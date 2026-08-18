# 快速开始

本页给出从干净检出到浏览器中看到场线的最短路径。命令均在仓库根目录执行。

## 环境要求

- [uv](https://docs.astral.sh/uv/)；开发解释器由 `.python-version` 固定为 Python 3.13，兼容范围由 `pyproject.toml` 声明。
- 支持现代 JavaScript、SVG 与 Canvas 的浏览器。
- 不使用 conda、手工 `venv` 或直接 `pip install`。uv 统一管理项目内 `.venv` 与锁文件 `uv.lock`。

## 安装

安装运行与开发依赖并同步锁文件：

```powershell
uv sync
```

`dev` 是 `pyproject.toml` 中的默认 dependency group，因此 `uv sync` 会一并安装 pytest、Ruff 和文档工具。uv 会在仓库内创建或更新 `.venv`，并使用 `uv.lock` 固定完整依赖解析。无需手工激活环境；所有命令都通过 `uv run` 执行。修改 `src/vectorviz` 后无需重新安装项目。

## 启动前端

前端是由 Python 服务直接提供的无构建静态页面，不需要 Node.js 或单独的前端端口：

```powershell
uv run vectorviz
```

开发时使用自动重载：

```powershell
uv run vectorviz --reload
```

服务启动后：

- `/`：交互式场线页面；
- `/api/health`：服务健康状态；
- `/api/presets`：可用场景预设；
- `/api/scene`：根据参数计算并返回场景数据。

默认监听 `http://127.0.0.1:8000`。`vectorviz` 是 `pyproject.toml` 声明的正式项目入口；`--reload` 适合开发调试。

## 检查 API

```powershell
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/presets
```

请求一个电偶极子场景：

```powershell
$body = @{
  preset = "electric_dipole"
  density = 18
  resolution = 96
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/scene `
  -ContentType "application/json" `
  -Body $body
```

请求和响应结构详见 [API 参考](api.md)。

## Python 最小示例

```python
import numpy as np

from vectorviz import Domain, FieldLineTracer, TraceDirection, UniformField

field = UniformField(vector=np.array([1.0, 0.25]))
domain = Domain(lower=np.array([-2.0, -2.0]), upper=np.array([2.0, 2.0]))
tracer = FieldLineTracer(field=field, domain=domain)

result = tracer.trace(
    seed=np.array([0.0, 0.0]),
    direction=TraceDirection.BOTH,
)

print(result.points.shape)
print(result.terminations)
```

场对象支持批量求值：

```python
points = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]])
vectors = field.evaluate(points)
assert vectors.shape == points.shape
```

公共类型和精确语义见 [Python API](api.md#python-api)。

## 打开 Jupyter 笔记本

所有 `.ipynb` 都放在仓库根目录的 `notebooks/`。用 uv 临时提供 JupyterLab：

```powershell
uv run --with jupyterlab jupyter lab notebooks
```

这条命令仍使用项目的 `.venv` 和已安装的 `vectorviz`，不需要 Conda。可复用实现放进 `src/vectorviz/`，笔记本只保留推导、实验参数、图和结果检查。目录约定写在 `notebooks/README.md`。

## 运行测试

```powershell
uv run playwright install chromium
uv run pytest
```

Chromium 只需在第一次运行或 Playwright 升级后安装；测试由 uv 环境中的 Python Playwright 驱动，不需要 Node 项目工具链。

生成 HTML 覆盖率报告：

```powershell
uv run pytest --cov-report=term-missing --cov-report=html
```

测试不应通过目测图片来替代数值断言。验证层次和基准见[开发指南](development.md#testing-strategy)。

## 预览和构建文档

```powershell
uv run mkdocs serve
```

文档预览地址为 `http://127.0.0.1:8001`，与场线前端使用的 8000 端口分离。编辑 `docs/` 后浏览器会自动刷新。发布前执行严格构建：

```powershell
uv run mkdocs build --strict
```

生成的网站位于 `site/`；它是构建产物，不应手工编辑。

!!! note "终端中的 MkDocs 2.0 提示"
    Material for MkDocs 当前会输出一段关于未来 MkDocs 2.0 的上游提示。项目锁定并实际运行的是 MkDocs 1.6；只要终端最后显示 `Serving on http://127.0.0.1:8001/`，文档服务就已正常启动。

## 常见问题

### 场线在源附近非常密或积分失败

理想点源和理想细导线存在真实奇点。应扩大源的排除半径或降低最大积分步长，不要通过给分母加一个很大的 epsilon 来掩盖奇点。

### 二维图上的线为什么与预期三维场线不同

任意切片上只保留平面分量得到的是**投影流线**。只有切片是不变平面时，它才是真实三维场线。参见[二维切片何时包含真实场线](tutorial/04-slices-and-validation.md#true-vs-projected)。

### 改变颜色后为什么不应重新积分

颜色、线宽和相机属于显示状态；场数据和轨迹属于计算状态。它们的缓存键不同，详见[系统架构](architecture.md#cache-boundaries)。
