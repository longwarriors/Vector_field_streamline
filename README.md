# VectorViz

VectorViz 是一个采用 `src` 布局的科学向量场与场线可视化项目。Python 后端负责批量场求值、自适应场线积分、奇点排除和科学元数据；浏览器前端负责热图、场线、方向箭头、场源拖动与探针显示。

当前 MVP 包含：

- 匀强场、点电荷场、理想磁偶极场与复合场；
- 基于 SciPy DOP853/RK 方法的归一化场线追踪；
- 计算域、零场、非有限值和源排除区域终止事件；
- 电偶极子、磁偶极子、匀强场三个浏览器预设；
- FastAPI 场景 API 与无构建 Canvas 前端；
- pytest 数值/契约测试，以及 MkDocs Material 文档站。

## 环境管理

项目只使用 [uv](https://docs.astral.sh/uv/) 管理 Python、项目内 `.venv` 和锁文件 `uv.lock`，不使用 Conda，也不要求手工执行 `pip install`。

```powershell
uv sync
uv run vectorviz
```

打开 <http://127.0.0.1:8000> 即可查看场线。开发模式：

```powershell
uv run vectorviz --reload
```

## 质量检查

```powershell
uv run ruff check .
uv run pytest
uv run mkdocs build --strict
```

文档预览：

```powershell
uv run mkdocs serve
```

文档固定预览在 <http://127.0.0.1:8001>，可与 8000 端口上的场线前端同时运行。

完整说明见：

- [快速开始](docs/getting-started.md)
- [系统架构](docs/architecture.md)
- [物理与数学建模教程](docs/physics-and-numerics.md)
- [参考资料](docs/references.md)
- [API 参考](docs/api.md)
- [开发与测试](docs/development.md)

## 项目结构

```text
src/vectorviz/          科学核心与 Web 应用
tests/                  单元、数值验证和 API 契约测试
docs/                   MkDocs 文档源文件
notebooks/              Jupyter 教程、推导与可复现实验
```

## 图上的线表示什么

场线只表示向量场方向。当前的覆盖播种只负责把画面铺开，线密度不表示场强。三维场在任意切片上丢弃法向分量后得到的是投影流线；只有法向分量处处为零的不变平面，二维曲线才是真实三维场线。前端会直接写明切片类型、物理量单位和奇点 mask。

后续路线包括圆线圈解析场与 Biot–Savart 交叉验证、规则/非结构网格场、FEMM/VTK 导入、真正的三维场线，以及与普通向量场分离的 Schwarzschild/Kerr 测地线后端。
