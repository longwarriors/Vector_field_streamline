# VectorViz

[![CI](https://github.com/longwarriors/Vector_field_streamline/actions/workflows/ci.yml/badge.svg)](https://github.com/longwarriors/Vector_field_streamline/actions/workflows/ci.yml)

VectorViz 是一个采用 `src` 布局的科学向量场与场线可视化项目。Python 后端负责批量场求值、自适应场线积分、奇点排除和科学元数据；浏览器前端负责热图、场线、方向箭头、场源拖动与探针显示。

当前 v0.3.0 包含：

- 匀强场、点电荷场、理想磁偶极场、圆电流环解析场与复合场；
- 基于 SciPy DOP853/RK 方法的归一化场线追踪；
- 计算域、零场、非有限值、源排除区域，以及可选闭环检测与终止判据；
- 电偶极子、磁偶极子、线性 Halbach 阵列、圆形电流线圈、匀强场五个浏览器预设；
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
uv run playwright install chromium
uv run ruff check .
uv run pytest
uv run mkdocs build --strict
```

Chromium 只需在首次运行前端语义测试或 Playwright 版本升级后安装；命令仍由 uv 环境中的 Python Playwright 提供，不需要 Node 项目工具链。默认测试同时执行分支覆盖统计和浏览器语义测试，精确综合覆盖率不得低于 90.00%。CI 在 Ubuntu + Python 3.13 上执行权威覆盖率与浏览器门槛，在 Ubuntu/Windows + Python 3.11/3.14 上验证版本边界，并另以 macOS + Python 3.13 验证非浏览器兼容性。

文档预览：

```powershell
uv run mkdocs serve
```

文档固定预览在 <http://127.0.0.1:8001>，可与 8000 端口上的场线前端同时运行。

完整说明见：

- [在线文档](https://longwarriors.github.io/Vector_field_streamline/)
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

场线只表示向量场方向。几何覆盖播种只负责把画面铺开，线密度不表示场强；圆环预设则用公开磁通函数选择等 $\psi$ 轮廓，并明确标为 `equal_flux`。三维场在任意切片上丢弃法向分量后得到的是投影流线；只有法向分量处处为零的不变平面，二维曲线才是真实三维场线。前端会直接写明切片类型、物理量单位、播种枚举与说明、双端终止统计，以及尝试后抑制和实际渲染的线数。探针读取未裁剪的原始有限场值；`vmin`/`vmax` 只建议色标范围，奇点和其他 masked 样本在 JSON 中是 `null`。

后续路线包括矢量显示图层与缓存、规则/非结构网格场、FEMM/VTK 导入、真正的三维 Web 场线，以及与普通向量场分离的 Schwarzschild/Kerr 测地线后端。

点电荷和面内磁偶极子可在前端增删；磁偶极矩角度可直接编辑。`density` 是 6–40 的整数总预算，浏览器以步长 1 调整它，并在每次提交前自动抬高不足的预算。可编辑预设还从 `/api/presets` 读取严格的源中心最小间距能力：数值编辑发生冲突时回滚，拖动时吸附到合法位置。拖动过程中旧热图和旧场线会隐藏并标记“场待重算”，松开后才提交一次新场景，避免把新源位置与旧场混画。
