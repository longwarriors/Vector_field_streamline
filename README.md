# VectorViz

[![CI](https://github.com/longwarriors/Vector_field_streamline/actions/workflows/ci.yml/badge.svg)](https://github.com/longwarriors/Vector_field_streamline/actions/workflows/ci.yml)

VectorViz 是一个采用 `src` 布局的科学向量场与场线可视化项目。Python 后端负责批量场求值、自适应场线积分、奇点排除和科学元数据；浏览器前端负责热图、场线、方向箭头、场源拖动与探针显示。

当前 v0.3.10 包含：

- 匀强场、点电荷场、理想磁偶极场、无限长直导线、圆电流环、带电圆环、介质球/导体球、无限长磁屏蔽圆柱壳解析场、二维 Laplace 谐波多项式族、斜率场与复合场；
- 基于 SciPy DOP853/RK 方法的归一化场线追踪；
- 计算域、零场、非有限值、源排除区域、材料界面，以及可选闭环检测与终止判据；
- 电偶极子、电四极子、六个正电荷、三对交替电荷、磁偶极子、线性 Halbach 阵列、圆形电流线圈、直导线、同向/反向双导线、有限螺线管、Helmholtz/反 Helmholtz 线圈、带电圆环、介质球、导体球、匀强场，以及 Laplace 鞍点/三阶多极、线性/Logistic 斜率场和理想磁屏蔽筒，以及非定常振荡流与定常流对照，共 24 个浏览器预设；
- FastAPI 场景 API 与无构建 Canvas 前端（浅色白板界面与淡紫热图，含读图说明与未着色、色标截断等诚实标记）；
- 静态预设的热图、斜率短线、场线、方向、场源、网格和光标预览七个独立图层；流场另提供迹线与脉线图层，并禁用冻结网格预览；向量分量探针和最多 16 个交给服务端积分的手动种子；
- pytest 数值/契约测试，以及 MkDocs Material 文档站。

含时流使用独立的物理时间积分：流线冻结观察时刻，迹线跟踪同一质点，脉线收集固定位置持续释放的粒子在同一观察时刻的位置。坐标、速度、时间分别为 m、m/s、s；默认观察 4 s、初始 0 s、释放窗 [0,4] s，四个时间可在合法的 0–6 s 范围内调整。解析公式和可复现对照见[非定常流教程](docs/tutorial/06-unsteady-flow.md)。

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

光标预览束使用采样向量网格的双线性插值和定步长 RK4，随采样分辨率变化，图中明确标为预览。点击选定的种子由服务端直接对场模型做自适应积分；预览与服务端结果都需按各自的误差来源验证。

新增磁静态场景使用固定几何，仍支持全部图层和手动种子。直导线采用无限长理想细导线模型；有限螺线管是多个同轴圆电流环的离散绕组近似；线圈组展示近匀强区和零场/梯度结构。它们采用明确标注的覆盖播种，线密度不代表场强，排除半径不代表实际线径。

v0.3.9 的四个数学预设使用无量纲坐标。Laplace 场展示谐势的负梯度；斜率场展示 $y'=x-y$ 与 $y'=y(1-y)$ 的解曲线，热图是辅助向量 $(1,f)$ 的模长，不是物理场强。理想磁屏蔽筒显示筒壁磁通集中与内腔场减弱，限定为无限长、线性各向同性、恒定相对磁导率的圆柱壳与横向匀强外场；不包含有限长坡莫合金实物的端部、开口、饱和或磁滞。

后续已确认的顺序是振荡电偶极子的迟滞电磁场、自定义表达式与数值数据接入，最后是三维视图与更多动力学。有限长、开口和多层磁屏蔽筒随数值数据接入推进，具体以[开发路线](docs/roadmap.md)为准。

点电荷和面内磁偶极子可在前端增删；磁偶极矩角度可直接编辑。静态场景的 `density` 是 6–40 的整数总预算，浏览器以步长 1 调整它，并在每次提交前自动抬高不足的预算。可编辑预设还从 `/api/presets` 读取严格的源中心最小间距能力：数值编辑发生冲突时回滚，拖动时吸附到合法位置。拖动过程中旧热图和旧场线会隐藏并标记“场待重算”，松开后才提交一次新场景，避免把新源位置与旧场混画。
