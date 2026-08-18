# API 参考

VectorViz 同时提供 Python 公共 API 与浏览器使用的 HTTP API。本页描述稳定边界；内部帮助函数不属于兼容性承诺。

## Python API

稳定类型从顶层包导入：

```python
from vectorviz import (
    CompositeField,
    Domain,
    FieldLineTracer,
    MagneticDipoleField,
    PointChargeField,
    SphericalExclusion,
    TerminationReason,
    TraceBranch,
    TraceDirection,
    TraceOptions,
    TraceResult,
    UniformField,
    VectorField,
    trace_field_line,
)
```

### `VectorField`

向量场协议。实现提供批量求值：

```python
vectors = field.evaluate(points)
```

`points` 的最后一个轴是空间维数；返回数组形状相同。调用方不应假设具体场是解析式、数值积分还是网格插值。

### `Domain`

`Domain(lower, upper)` 是轴对齐计算区域，保存每一维的下界与上界。它提供 `dimension`、`extent`、`center`、`contains()` 和用于事件定位的 `margin()`。它描述计算域，不等同于源的奇点 mask。

### `SphericalExclusion`

`SphericalExclusion(centers, radii)` 表示二维圆形或三维球形排除区域。`margin()` 在区域外为正、表面为零、内部为负，追踪器用它定位 `EXCLUSION_HIT` 事件。理想点源附近应使用排除几何，而不是修改物理场公式。

### 解析场

| 类型 | 用途 |
|---|---|
| `UniformField` | 匀强向量场与积分器基准 |
| `PointChargeField` | 二维或三维点电荷电场 |
| `MagneticDipoleField` | 磁偶极近似与远场模型 |
| `CompositeField` | 对多个同维场做线性叠加 |

所有物理场内部应采用一致单位。输入源参数的单位与坐标系不得只存在于图标题里。

### `TraceOptions`

构造签名：

```python
TraceOptions(
    max_arc_length=20.0,
    max_step=0.1,
    first_step=None,
    rtol=1e-7,
    atol=1e-9,
    null_threshold=1e-12,
    output_step=None,
    method="DOP853",
    closure_tolerance=None,
    closure_min_arc_length=None,
    closure_tangent_cosine=0.95,
)
```

`null_threshold` 使用场自身单位，是“方向未定义”事件面；它不是加入分母的 epsilon。`output_step` 非空时，结果利用求解器的稠密输出按近似弧长等间隔采样。

闭环检测默认关闭。启用时必须同时给出带坐标单位的 `closure_tolerance` 和 `closure_min_arc_length`，且最小弧长必须大于空间容差的两倍，让轨迹有可分辨的离开过程；只有随后在距种子的局部最近点回到容差内，且当前切向与种子切向的余弦不小于 `closure_tangent_cosine`，才记录 `closed_loop`。该余弦阈值的合法范围是 `[-1, 1)`；上界不取 1，因为普通闭轨的浮点切向点积不会可靠地等于精确的 1。返回检查使用距种子平方沿轨迹的导数定位候选最近点，不会用空间容差暗中改写 `max_step`。双向分支仍分别保留诊断；若两支都闭合，合并后的 `TraceResult.points` 只保留一个方向的一周，不重复绘制同一闭轨。

调用方应通过 `TraceOptions` 配置追踪器，不依赖模块内部常量。

### `FieldLineTracer`

```python
tracer = FieldLineTracer(
    field=field,
    domain=domain,
    options=options,
    exclusions=[SphericalExclusion(centers, radii)],
)
result = tracer.trace(seed, direction=TraceDirection.BOTH)
```

`TraceDirection` 支持正向、反向和双向。双向追踪会在种子处合并曲线，同时在 `TraceBranch` 中保留两个分支各自的终止信息。

### `TraceResult`

`TraceResult` 是渲染与导出的稳定数据边界，字段为 `seed`、`points`、`arc_length`、`field_magnitude`、`forward` 与 `backward`。`terminations` 属性汇总存在分支的终止原因。

每个 `TraceBranch` 包含 `direction`、`points`、`arc_length`、`field_magnitude`、`termination`、`message` 和 `nfev`。消费者不得通过积分步间距推断场强，应读取显式的 `field_magnitude`。

当前 `TerminationReason` 值为 `domain_exit`、`null_field`、`nonfinite_field`、`max_arc_length`、`solver_failure`、`seed_outside_domain`、`exclusion_hit` 和 `closed_loop`。

### `trace_field_line`

一次性便利函数，适合示例和测试。需要追踪多个种子时应复用 `FieldLineTracer`，以便共享配置和后续缓存。

## HTTP API

默认服务地址是 `http://127.0.0.1:8000`，所有机器可读端点位于 `/api` 下。错误响应使用非 2xx 状态码，并返回可读 `detail`；前端必须显示错误，不能继续渲染上一请求的数据却让用户误以为更新成功。

### `GET /api/health`

用于启动检查与自动化探针。

示例响应：

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

此端点只证明 Web 进程可以响应；它不执行昂贵的场积分。

### `GET /api/presets`

返回前端可选择的内置场景。响应是对象数组：

```json
[
  {
    "id": "electric_dipole",
    "label": "电偶极子",
    "description": "两个异号点电荷的二维电场"
  }
]
```

首批预设标识符为：

- `electric_dipole`
- `magnetic_dipole`
- `uniform`

客户端应使用返回值生成选项，不应假设列表永久不变。

### `POST /api/scene`

计算二维场景。请求体：

```json
{
  "preset": "electric_dipole",
  "density": 18,
  "resolution": 96,
  "sources": [
    {"x": -1.0, "y": 0.0, "kind": "positive", "strength": 1.0},
    {"x": 1.0, "y": 0.0, "kind": "negative", "strength": -1.0}
  ]
}
```

字段说明：

| 字段 | 必需 | 语义 |
|---|---:|---|
| `preset` | 否 | `/api/presets` 返回的稳定标识符；默认 `electric_dipole` |
| `density` | 否 | 种子总预算，整数范围 6–40，默认 18，不代表物理场强；每个参与播种的源至少分配 1 个种子 |
| `resolution` | 否 | 两个方向共同使用的标量网格分辨率，整数范围 32–144 |
| `sources` | 否 | 1–8 个自定义源；若省略则使用预设源，显式空列表和未知字段会被拒绝 |

服务端必须为密度、分辨率、源数量和数值范围设置上限，防止一次交互请求耗尽内存或 CPU。

`density` 是一次场景请求的总种子预算，不是“每个源各放多少条”。电偶极预设只从非零正电荷出发，因此这些正电荷参与预算；磁偶极预设的每个偶极子都参与预算。若参与播种的源数超过 `density`，服务端返回 422，`detail` 会给出当前数量和所需最小值，例如 `electric_dipole 有 7 个正电荷参与播种，density 至少为 7`。预算充足时，每个播种源先得到 1 个种子，其余名额再按源强绝对值分配。响应始终满足 `len(lines) <= density` 与 `sum(termination_counts.values()) == density`；当每个种子都得到至少两个有限轨迹点时，前一个不等式取等号。若种子位于零场或非有限场等无法形成曲线的位置，终止计数仍会记录该次追踪，但 `lines` 会排除只有一个点的结果。

成功响应结构如下。为便于阅读，示意片段把网格缩成 $2\times2$，并只展示 18 条轨迹中的 1 条；实际端点接受的 `resolution` 不低于 32，数组会相应更长。

```json
{
  "domain": {
    "x": [-3.0, 3.0],
    "y": [-3.0, 3.0]
  },
  "scalar": {
    "nx": 2,
    "ny": 2,
    "values": [0.15, 0.17, 0.20, 0.18],
    "mask": [false, false, false, false],
    "scale": "log",
    "label": "|E|",
    "unit": "V/m",
    "vmin": 0.01,
    "vmax": 10.0
  },
  "lines": [
    {
      "points": [[-1.8, 0.2], [-1.7, 0.21]],
      "direction": 1,
      "termination": "domain_exit"
    }
  ],
  "sources": [
    {"x": -1.0, "y": 0.0, "kind": "positive", "strength": 1.0},
    {"x": 1.0, "y": 0.0, "kind": "negative", "strength": -1.0}
  ],
  "metadata": {
    "title": "电偶极子的电场线",
    "projection_note": "该平面法向场分量为零，所示曲线是真实场线，不是投影流线。",
    "field_model": "三维点电荷场在 z=0 对称平面上的限制",
    "seed_mode": "从正电荷排除面的覆盖播种；线密度默认不代表场强。",
    "termination_counts": {"exclusion_hit": 11, "domain_exit": 7}
  }
}
```

#### `domain`

给出数值坐标范围。前端使用同一个范围映射标量栅格、曲线和源，不能为每个图层独立自动缩放。

#### `scalar`

- `nx`、`ny` 定义规则网格尺寸；
- `values` 是长度为 `nx * ny` 的 row-major 一维数组：第 0 行对应 `ymax`，每行从 `xmin` 到 `xmax`，随后向 `ymin` 进入下一行；
- `mask` 与 `values` 等长，`true` 表示源排除区或无效采样；
- `scale` 当前只允许 `linear` 或 `log`；
- `label` 与 `unit` 必须一同显示，避免无量纲色图。
- `vmin` 与 `vmax` 是服务端在排除 mask 后给出的建议色标范围。

对数尺度只接受正值。mask、零值和非有限值的编码应与前端协商，不得静默替换成任意小正数。

#### `lines`

每条线是按轨迹顺序排列的坐标点。`direction` 为 `1` 或 `-1`，表示相对场方向；`termination` 记录该分支终止原因。方向元数据不是要求前端把点序颠倒。后续可以增加场强和弧长，但客户端应忽略未知字段以保持向前兼容。

#### `sources`

源图层独立于标量网格。前端据 `kind` 选择符号或形状，据 `strength` 显示数值；不得从颜色像素反推源参数。

#### `metadata`

`projection_note` 不能省略。它说明曲线是二维真实场线、投影流线还是三维曲线切片。定义见[二维切片何时包含真实场线](tutorial/04-slices-and-validation.md#true-vs-projected)。

## API 兼容性

- 顶层 Python 导出与 HTTP 字段属于稳定接口；
- 添加可选 JSON 字段是向后兼容变更；
- 删除字段、改变单位或改变 `values` 排列方式需要版本迁移；
- 预设内部参数可以改进，但应在 `metadata` 中暴露影响科学解释的变化；
- 测试必须覆盖合法请求、边界值、拒绝超限请求和 JSON 形状一致性。

开发与契约测试流程见[开发指南](development.md)。

## 自动生成的 Python 参考

以下签名与成员由当前 `src/vectorviz` 源码生成；手写章节负责解释稳定语义，自动参考负责与实际代码保持同步。

### 核心协议与区域

::: vectorviz.core.VectorField
    options:
      heading_level: 4

::: vectorviz.core.Domain
    options:
      heading_level: 4

::: vectorviz.core.SphericalExclusion
    options:
      heading_level: 4

### 场模型

::: vectorviz.fields.UniformField
    options:
      heading_level: 4

::: vectorviz.fields.PointChargeField
    options:
      heading_level: 4

::: vectorviz.fields.MagneticDipoleField
    options:
      heading_level: 4

::: vectorviz.fields.CompositeField
    options:
      heading_level: 4

### 场线追踪

::: vectorviz.tracing.TraceOptions
    options:
      heading_level: 4

::: vectorviz.tracing.TraceBranch
    options:
      heading_level: 4

::: vectorviz.tracing.TraceResult
    options:
      heading_level: 4

::: vectorviz.tracing.FieldLineTracer
    options:
      heading_level: 4

::: vectorviz.tracing.trace_field_line
    options:
      heading_level: 4
