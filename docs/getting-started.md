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

前端提供电偶极子、电四极子、六个正电荷、三对交替电荷、磁偶极子、线性 Halbach 阵列、圆形电流线圈、带电圆环、介质球、导体球和匀强场等预设。四个电荷预设只是默认排布不同，点电荷都可以增删和移动；磁偶极子还可编辑从 $+x$ 轴逆时针量取的面内方向角。Halbach 预设以八个依次旋转 90° 的理想点偶极为初始状态。圆环的 ⊙/⊗ 与带电圆环的两个 + 标记只表示同一个环穿过子午面的两个固定交点，不能拖动；介质球和导体球没有场源，图上的虚线圆是球面在该平面上的截线。

单直导线、同向/反向双导线、有限螺线管、Helmholtz/反 Helmholtz 线圈的电流和几何固定，⊙ 表示电流穿出画面、⊗ 表示穿入；螺线管用 8 个同轴圆环近似有限绕组。可以先在单直导线场景点击距中心不同半径的位置，再对比两种线圈组的中心场；点击零场点或排除区不会生成可绘制场线，科学详情会显示终止原因。

v0.3.9 加入的五个预设可用于数学场与材料边界对照。可以按下面的顺序比较：

1. **Laplace 鞍点与三阶多极**：打开斜率短线，观察原点周围方向；在原点添加手动种子会以 `null_field` 结束。
2. **线性与 Logistic 斜率场**：坐标无量纲，曲线是微分方程的解。Logistic 的 $y=0$ 与 $y=1$ 是水平平衡解，辅助向量仍为 $(1,0)$，不会被误当成零场点。
3. **理想磁屏蔽筒**：比较内腔、筒壁和远处的 B 探针读数；内腔场约为外加场的 7.41%。两个圆是材料界面，筒壁内可以播种，场线会穿越界面继续积分。该模型是无限长线性圆柱壳的解析基准，不是有限长坡莫合金筒的完整模拟。

v0.3.10 再加入两个流场，当前共 24 个预设。选择“非定常流”，默认观察 4 s、初始 0 s、释放窗 [0,4] s；流线冻结在观察时刻，迹线记录同一质点，脉线显示不同释放时刻的粒子在观察时刻的位置。切换“定常流”作几何重合对照。四个时间输入单位都是 s，范围为 0–6，初始时刻与释放窗终点不得晚于观察时刻；增加 `density` 只加密每站的脉线采样。点击图面为三类曲线增加共同的释放站。流场另有迹线与脉线图层开关，冻结网格预览已禁用；热图、探针和斜率短线仍表示观察时刻的速度。详见[非定常流教程](tutorial/06-unsteady-flow.md)。

页面分为图框和几张卡片：宽屏上图在左，参数与场源、读图说明和科学详情依次在右；窄屏上图排在最前，卡片在图下方。图框标题行写明场景和色标类型，图注写明切片含义，以及本次的播种方式和实际渲染、抑制的线数。读图时注意：

- 色标两端的上下限只是显示范围。两端出现尖角，表示有格点超出范围、按端色着色；精确值请用探针，它显示离指针最近的网格点上未截断的原值。
- 图框内的灰底斜线是无法着色的格点：场源排除区（半径是数值设定）、非有限值，或对数色标下为零的值。
- 场源标记的白、黑双环只为在任何底色上都能看清位置，大小不代表源的强度。
- 静态场景中，图框左侧的竖向工具栏是七个图层开关：热图、斜率、场线、方向、场源、网格、预览。切换只改显示，不重新计算；关掉热图时色标和“未着色”图例一起隐藏，关掉场线时方向图例一起隐藏，关掉场源后标记不能拖动（坐标输入框仍可用）。拖动场源后的“场待重算”状态优先于任何开关。窄屏上工具栏横排在图框上方。
- “斜率”图层默认关闭，打开后在采样格点上画沿场方向的等长短线：它只表示方向，不分正反（像微分方程教材里的斜率场），也不表示强弱。
- 在静态场景中，指针移到图上时，周围出现一束蓝色的**预览束**：这是浏览器对服务端返回的向量网格做双线性插值再用 RK4 积分的结果，随采样分辨率变化，图注写明它不是服务端积分。点击图中位置会把该点作为手动种子发给服务端，服务端用同一套追踪器从该点双向积分出精确场线（黑线配蓝色光晕，蓝点标出种子）。手动种子最多 16 个，不占播种预算，“手动种子”卡片里可清除；科学详情单独列出它们得到几条线以及两端的终止原因，落在排除区或零场里的种子得不到线但仍被计数。
- 探针吸附到离指针最近的网格点，显示该点的坐标和原值，并在图上用小圆环标出这个点；有向量分量时另显示该点的场方向角（从 $+x$ 朝 $+y$ 逆时针）和两个分量。对数色标旁的短刻度是十的整数次幂。
- 播种预算和分辨率滑块在松手时才提交一次请求，拖动过程中只更新读数。

科学详情会把播种分类显示为覆盖、等通量或特征播种，并紧接服务端给出的具体说明。电荷场同时从正负电荷出发；默认 Halbach 覆盖上下两条平行轨道；单磁偶极沿旋转赤道线双向追踪；圆环非轴种子对应等间隔的磁通函数值。`density` 在静态场景中仍是 trace-job 总预算，不等于最后画出的线数，也不默认表示场强。

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
