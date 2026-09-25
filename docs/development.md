# 开发指南

本项目的完成标准不是“前端能画出曲线”，而是物理、数值、API 和文档均有可重复验证。新功能应从公共契约和可验证基准开始设计。

## 开发环境

```powershell
uv sync
uv run playwright install chromium
```

uv 在项目内维护 `.venv`，`uv.lock` 则固定直接与传递依赖。不要手工修改 `.venv`，也不要在项目工作流中混用 conda 或直接 `pip install`。依赖变更应先改 `pyproject.toml`，再由 uv 更新锁文件。

`.python-version` 固定日常开发使用 Python 3.13；需要验证其他受支持版本时，应在独立 CI 作业中显式选择解释器，不要改动共享虚拟环境。

确认包来自 `src` 布局，而不是仓库根目录的同名文件：

```powershell
uv run python -c "import vectorviz; print(vectorviz.__file__)"
```

路径应指向安装环境映射到的 `src/vectorviz`。

## 日常检查

```powershell
uv run ruff check .
uv run pytest
uv run mkdocs build --strict
```

开发服务器分别用于前端和文档预览：

```powershell
uv run vectorviz --reload
uv run mkdocs serve
```

两项服务可以同时运行：场线前端使用 `127.0.0.1:8000`，文档预览由 `mkdocs.yml` 固定使用 `127.0.0.1:8001`。

数学公式使用仓库内固定版本的 MathJax SVG 运行时，位于 `docs/javascripts/vendor/mathjax/`。不要改回外部 CDN；离线预览和受限网络环境都需要这份本地资源。升级时同时更新运行时文件、许可证和版本说明，并重新检查站内无刷新跳转后的公式。

不要把 `site/`、覆盖率目录、缓存或虚拟环境提交到版本库。

## 持续集成与文档部署

`.github/workflows/ci.yml` 对提交到 `master` 的改动、面向 `master` 的拉取请求和手动运行执行以下门禁：

- Ubuntu + Python 3.13 是权威质量环境，安装 headless Chromium，并运行 Ruff、包含前端语义测试的完整 pytest 覆盖率、严格文档构建以及 wheel/sdist 构件冒烟；
- Ubuntu/Windows + Python 3.11/3.14 只验证版本边界的非浏览器兼容性，不重复下载浏览器或统计覆盖率；
- 独立的 macOS + Python 3.13 作业验证平台兼容性，不与上述矩阵做笛卡尔积；
- 覆盖率配置位于 `pyproject.toml`，使用两位精度，精确综合覆盖率低于 90.00% 时失败；
- CI 固定 uv 与各 GitHub Action 的版本，依赖安装始终从 `uv sync --locked` 开始。
- 同一分支的新 workflow 会取消尚未完成的旧 workflow，避免旧提交在新提交之后反向覆盖 Pages。

只有 `master` 上的权威质量作业和整个兼容矩阵都成功，工作流才会重新严格构建文档并部署到 [GitHub Pages](https://longwarriors.github.io/Vector_field_streamline/)。仓库首次启用时，维护者还需在 GitHub 的 **Settings → Pages → Build and deployment → Source** 中选择 **GitHub Actions**；之后不需要维护 `gh-pages` 分支，也不要手工编辑 `site/`。

## 测试策略 { #testing-strategy }

### 单元测试

目标是小而确定的数学与数据契约：

- `Domain` 的形状检查、包含判断和维数错误；
- 每个场的单点与批量求值形状；
- 浮点 dtype、有限值政策与奇点行为；
- `CompositeField` 的叠加和维数拒绝；
- 追踪选项校验和枚举序列化。

### 数值验证测试

验证“结果在容差内正确”，而不是只验证代码能运行：

- 匀强场得到平行直线；
- 点电荷方向径向且幅值满足 $r^{-2}$；
- 磁偶极子满足轴线、赤道和 $r^{-3}$ 基准；
- 人工旋转场形成闭环；
- 减小积分容差或最大步长时，轨迹按预期收敛；
- 场线切向与局部场的叉积残差足够小。

建议为误差较敏感的测试记录容差选择依据。不要把容差放宽到足以掩盖算法退化。

### 集成测试

- 多个种子共用同一追踪器；
- 正反分支合并时不重复种子点；
- 越界、弱场、奇点和闭环产生正确终止原因；
- 采样解析场再插值追踪，与直接解析追踪在网格细化时收敛一致。

### Web 契约测试

使用测试客户端覆盖：

- `GET /api/health` 返回成功和稳定状态字段；
- `GET /api/presets` 返回可用于场景请求的标识符，以及可编辑点源预设的严格最小中心距能力；
- `POST /api/scene` 的完整与最小合法请求；
- 非法预设、负密度、过大分辨率和畸形源被拒绝；
- `scalar.values`、`mask` 数量与 `nx * ny` 一致，且 masked 项恰为 JSON `null`、未遮罩项是未按色标裁剪的有限原值；
- 所有曲线点都在声明的坐标约定下可解释；
- 响应包含单位、`projection_note`、枚举 `seed_mode`、非空 `seed_description`，以及严格非负整数的双端终止、抑制与渲染统计；
- 电荷的正负向 job 合计恰好消耗总预算，只有已有同源对正向代表的反向线被抑制；
- 默认 Halbach 双轨和单磁偶极赤道线使用 `BOTH`，点序沿正场方向且双端原因分别可审计；
- 圆环非轴种子对应的公开 $\psi$ 值等间距，奇数预算保留唯一轴线 job；
- active 源中心距等于或小于能力下限、以及全零磁偶极场景都返回精确的 422 契约。

### 前端测试

无构建前端仍需要测试。至少覆盖：

- API 请求失败时显示错误，不保留“看似已更新”的旧状态；
- 标量、场线、箭头和源共用同一坐标变换；
- 对数尺度不会对零值或 mask 产生虚假热点；
- 探针显示未裁剪原值，科学详情显示场模型、播种枚举与说明、双端终止统计、抑制数和实际渲染数；
- 未知未来 `seed_mode` 与终止原因原样显示，旧自由文本 mode 缺少独立说明时也能降级展示；
- `density` 步长为 1，奇数预算可提交；默认 Halbach 的双轨不被八个 marker 错误抬高，自定义场景仍满足 active 播种源的最小预算；
- 数值编辑在源间距冲突时回滚，拖动吸附到合法边界；
- 拖动期间旧热图、色标和场线隐藏，只在松开后为最终位置发出一次请求；
- 热图纹素中心落在标量节点上，与探针读数的网格一致；
- 画布焦点环、色标和探针不被舞台裁切，色标不压住图面，探针不遮挡图旁的色标；桌面、投影仪、平板和手机视口的首屏都能看到完整图面，科学详情可以滚动到；
- 无法着色的格点画灰底斜线，只在斜线露出标记外时显示图例；色标超限尖角与播种信息行随场景状态出现，拖动时隐藏且不移动图面；
- 静态资源不引用外部 URL 或第三方品牌名，样式令牌满足文字 4.5:1、控件描边与焦点 3:1 的对比度下限；
- 改变显示图层不再次请求物理场景；
- 窗口缩放后坐标与指针探针仍一致。

截图测试只适合防止布局意外变化，不能替代数值和语义断言。

前端入口和纯函数使用浏览器原生 ES module，由 uv 管理的 Python Playwright 在 Chromium 中直接执行；仓库不使用 `package.json`、Node 包管理器或 bundler。首次运行或 Playwright 升级后安装匹配的浏览器：

```powershell
uv run playwright install chromium
uv run pytest tests/test_frontend.py --no-cov
```

默认 `uv run pytest` 仍包含这些浏览器测试。`browser` marker 仅供兼容作业在没有下载 Chromium 时排除；权威质量作业不得排除它。当前测试已自动验证请求失败不展示旧结果、公共坐标变换、raw/null 标量契约、resize 后探针一致性、科学元数据和未知终止原因显示、拖动旧场隔离、源间距、圆环 wire 标记只读、磁偶极方向显示、点源组成约束、所有请求在提交前满足播种预算，以及热图纹素对齐、焦点环与读数不被裁切、未着色斜线与色标超限标记、静态资源自包含和样式对比度。“改变显示图层不再次请求”留到后续图层开关存在时实现和验收，不提前制造空 UI。

## 测试分组

当前默认命令运行完整测试套件。测试数量增长到需要 marker 时，可引入 `slow`、`validation` 和 `web`；引入 marker 的同一个改动必须在 `pyproject.toml` 注册它们，以保持 pytest 的严格 marker 检查。发布前不得默认排除数值验证。

## 添加新物理场

1. 先写定义、单位、坐标系和奇点集合。
2. 实现批量 `evaluate(points)`，保持前导维度。
3. 添加解析点值与对称性测试。
4. 添加旋转、缩放或远场标度等性质测试。
5. 将模型接入追踪器，验证切向残差。
6. 最后才添加预设与前端控件。

若模型只能在某个对称平面给出，应在类型或元数据中明确表达，不能把它包装成任意三维场。

## 添加新场景预设

预设负责组装已有模型，不应复制公式。每个预设至少声明：

- 稳定标识符和用户标题；
- 默认源与计算域；
- 标量量、单位和默认尺度；
- 播种策略及 `density` 的实际含义；
- 是真实二维场线还是投影；
- 安全的分辨率和源数量上限。

随后更新 `/api/presets` 契约测试和 [API 参考](api.md)。

## 文档要求

- 新公共类型或 HTTP 字段必须同步更新 [API 参考](api.md)；
- 改变科学解释时必须更新[物理与数学建模教程](physics-and-numerics.md)；
- 物理结论和算法说明在对应句子后给脚注，并把完整条目补到[参考资料](references.md)；
- 一段只讲一件事，先定义符号再写公式，不用“赋能”“体系化”“显然可见”等空话；
- 所有内部链接使用相对路径，并通过严格构建检查；
- 示例必须能在当前公共 API 上运行；
- 不在文档中把计划功能写成已完成能力。

预览与检查：

```powershell
uv run mkdocs serve
uv run mkdocs build --strict
```

## 性能与基准

先批量化，再考虑 JIT 或 GPU。性能改动必须保留数值结果并提供前后基准：

- 默认要求结果逐位不变：对改动前后的同一组场景，逐个追踪任务比较终止原因、右端函数调用次数和点数组，而不只比较最终图像；
- 单点快速路径只使用 IEEE 754 正确舍入的 `+ - * /` 与 `sqrt`，以及与批量路径相同的 SciPy ufunc；分数次幂留在数组路径，因为 NumPy 可能以 SIMD 实现数组 `pow`，结果末位可能不同于标量 `pow`；
- Python 浮点除以零会抛出 `ZeroDivisionError`，NumPy 则返回 `inf` 或 `nan`，所以分母可能下溢为零或不是有限值时，单点路径回落到数组路径；
- 记忆化或缓存要保存自己的副本：自定义场可能复用输出数组，调用方也可能修改返回的响应；
- 会改变数值结果的优化（例如跨种子向量化）需单独评估误差，并更新相关基准测试；
- 分离场采样时间、播种时间、积分时间和 JSON 序列化时间；
- 记录点数、源数、种子数和容差；
- 避免一次构造 `N_points × N_sources × D` 的巨大临时数组，必要时按源或点分块；
- 计算可用 `float64`，向浏览器传输时经误差评估后才考虑 `float32`；
- 当前前端拖动点源时隐藏已失效的物理图层，只在停止后请求一次结果；未来若增加低分辨率预览，必须显式区分预览与最终场景。

### v0.3.3 基准 { #v033-benchmark }

下表是经 ASGI 应用端到端请求 `POST /api/scene`（`density=18`，`resolution=80`）的耗时，包含场采样、追踪和 JSON 序列化。v0.3.2 与 v0.3.3 在同一台 Windows 11、Python 3.13 机器上交替运行两轮，每轮 7 次，取两轮中位数的平均，四舍五入到 10 ms。“只改分辨率”指紧接着以 `resolution=96` 请求同一场景。

| 预设 | v0.3.2 | v0.3.3 首次 | v0.3.3 只改分辨率 |
|---|---:|---:|---:|
| `electric_dipole` | 240 ms | 210 ms | 15 ms |
| `magnetic_dipole` | 480 ms | 430 ms | 15 ms |
| `halbach_array` | 230 ms | 200 ms | 15 ms |
| `current_loop` | 1480 ms | 360 ms | 15 ms |
| `uniform` | 230 ms | 190 ms | 15 ms |

v0.3.2 没有曲线缓存，只改分辨率与首次请求耗时相同。首次请求的提速来自三处：去掉单点求值时 NumPy 包装函数的开销、事件函数复用右端函数刚算过的场值，以及圆环一般区域的单点浮点路径。缓存命中后的剩余耗时是标量网格采样和序列化，单次在 6–24 ms 之间。两轮之间同一格的差异约 5%，验收方在同一台机器上的独立复测与此一致。

逐位不变的检查覆盖 64 个场景：五个预设各取 4 种预算和 3 种分辨率，另加不等强电荷、三电荷、旋转磁偶极和含零强度偶极的编辑后 Halbach 阵列。每个场景的完整响应 JSON，以及每个追踪任务的终止原因、右端函数调用次数和点数组，均与 v0.3.2 相同；同一预算的后两种分辨率走缓存命中，因此这也检查了命中后的响应。验收时另用 771 条对抗记录比较了核心接口：随机源布局、倾斜圆环加环面排除区的三维追踪，以及各种输入形状、整数与列表输入、NaN/inf、轴上与导线上的点，Windows 与 WSL Ubuntu 上均与 v0.3.2 逐位相同。这些是抽样验证，不是对所有输入和所有 NumPy 构建的证明；阈值、下溢和复用输出数组等已知边界由 `tests/test_loop_fields.py` 与 `tests/test_tracing.py` 的回归测试覆盖。磁偶极子与点电荷剩余的主要开销是分数次幂和 `einsum`，按上面的规则留在数组路径。

## 代码审查清单

- [ ] 物理量、单位与坐标系明确。
- [ ] 解析式或可信数值基准有测试。
- [ ] 奇点、零点、越界和非有限值政策明确。
- [ ] 批量数组形状和维数错误有测试。
- [ ] 前端没有复制物理与积分逻辑。
- [ ] 新 API 有合法、非法和边界契约测试。
- [ ] 显示参数不会无故使场或曲线缓存失效。
- [ ] 文档和示例已更新，`uv run mkdocs build --strict` 通过。

## 发布前验证

```powershell
uv sync --locked
uv run ruff check .
uv run pytest
uv run mkdocs build --strict
uv run python -c "from vectorviz import UniformField, FieldLineTracer"
uv build --clear
$wheel = (Get-ChildItem dist/*.whl).FullName
$sdist = (Get-ChildItem dist/*.tar.gz).FullName
uv run --isolated --no-project --with $wheel python -I tests/package_smoke.py
uv run --isolated --no-project --with $sdist python -I tests/package_smoke.py
```

`tests/package_smoke.py` 会先切换到临时目录，再从已安装构件导入公共 API，并确认 Web 静态资源被打入包中；它不会借用仓库根目录或 editable 安装来掩盖缺失文件。

还应手工打开前端做一次语义检查：单位可见、投影说明可见、源与曲线对齐、错误提示不会被吞掉。手工检查是自动化测试的补充，不是替代。
