# API 参考

VectorViz 同时提供 Python 公共 API 与浏览器使用的 HTTP API。本页描述稳定边界；内部帮助函数不属于兼容性承诺。

## Python API

稳定类型从顶层包导入：

```python
from vectorviz import (
    CircularLoopField,
    CompositeField,
    Domain,
    ExclusionRegion,
    FieldLineTracer,
    MagneticDipoleField,
    PointChargeField,
    SphericalExclusion,
    ToroidalExclusion,
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

### `ExclusionRegion` 与 `ToroidalExclusion`

`ExclusionRegion` 是追踪器接受的结构协议：实现只需提供空间 `dimension` 和 signed `margin(points)`。`ToroidalExclusion(center, normal, major_radius, minor_radius)` 是三维圆环导线的有限半径排除管；`normal` 会归一化，并要求 `0 < minor_radius < major_radius`。它的 `margin()` 是到圆形中心线的欧氏距离减去 `minor_radius`。

`minor_radius`（排除管半径）只控制追踪终止与显示 mask，不进入 `CircularLoopField` 的分母，也不把理想细导线改造成有限截面导线模型；`major_radius` 则对应物理圆环半径。

### 解析场

| 类型 | 用途 |
|---|---|
| `UniformField` | 匀强向量场与积分器基准 |
| `PointChargeField` | 二维或三维点电荷电场 |
| `MagneticDipoleField` | 磁偶极近似与远场模型 |
| `CircularLoopField` | 理想细圆电流环的三维磁感应强度 |
| `CompositeField` | 对多个同维场做线性叠加 |

所有物理场内部应采用一致单位。输入源参数的单位与坐标系不得只存在于图标题里。

奇点集合只由源几何决定：`PointChargeField` 与 `MagneticDipoleField` 在源位置、`CircularLoopField` 在导线上都返回显式 `NaN`，即使该源的电荷、磁矩或电流为 0。零强度源在 Web 场景层构建场之前就被剔除；核心不会把源点软化成普通采样点。

#### `CircularLoopField`

```python
loop = CircularLoopField(
    current=1.0,
    radius=0.8,
    center=(0.0, 0.0, 0.0),
    normal=(0.0, 0.0, 1.0),
)
field = loop.evaluate(points)
psi = loop.flux_function(points)
```

`current`、`radius` 和 `center` 分别使用 A、m 和 m；默认 `permeability` 使用 `scipy.constants.mu_0`，因此 `evaluate()` 返回 T。单位法向由构造器归一化，并与正电流按右手定则绑定。模型维数固定为 3；`center` 与 `normal` 是只读数组。

理想细导线圆周是显式 `NaN` 奇点，即使电流为 0 也不把源几何点伪装成普通采样点。`flux_function()` 返回轴对称磁通函数 $\psi=\rho A_\phi$：结果形状是 `points.shape[:-1]`，轴上取 0，细导线上取 `NaN`。测试中的直接 Biot–Savart 求积是独立验证 oracle，不是另一个公共运行时场类。

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

闭环检测默认关闭。启用时必须同时给出带坐标单位的 `closure_tolerance` 和 `closure_min_arc_length`，且最小弧长必须大于空间容差的两倍，让轨迹有可分辨的离开过程；只有随后在距种子的局部最近点回到容差内，且当前切向与种子切向的余弦不小于 `closure_tangent_cosine`，才记录 `closed_loop`。该余弦阈值的合法范围是 `[-1, 1)`；上界不取 1，因为普通闭轨的浮点切向点积不会可靠地等于精确的 1。候选点由 $\tfrac12\lVert\mathbf x-\mathbf x_0\rVert^2$ 沿轨迹的导数定位，不会用空间容差暗中改写 `max_step`。它是 SciPy 的非终止辅助事件：候选本身不会立即停止求解器，当前分段会继续到分段末端或更早的物理终止事件；追踪器随后裁剪返回轨迹并停止后续分段。`nfev` 是各已运行分段的 RHS 求值数之和，因而可能包含闭合点之后的 RHS 求值，但不包含事件函数调用或结果场强重算。双向分支仍分别保留诊断；若两支都闭合，合并后的 `TraceResult.points` 只保留一个方向的一周，不重复绘制同一闭轨。

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
  "version": "0.3.1"
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
    "description": "两个异号点电荷的二维电场",
    "source_separation": {
      "exclusive_minimum": 0.322,
      "unit": "m"
    }
  }
]
```

首批预设标识符为：

- `electric_dipole`
- `magnetic_dipole`
- `halbach_array`
- `current_loop`
- `uniform`

此端点是可用预设及其交互能力的权威目录；当前无构建客户端随版本静态提供同一组选项，并由契约测试防止两边漂移。客户端不应假设列表永久不变。

`electric_dipole`、`magnetic_dipole` 与 `halbach_array` 是可编辑点源预设，因此返回 `source_separation`。`exclusive_minimum: 0.322` 表示任意两个实际拥有排除区域的源中心距离必须**严格大于** 0.322 m；等于该值仍冲突。固定的 `current_loop` 与 `uniform` 没有这项能力，响应省略该字段；兼容客户端也应把缺失或 `null` 都解释为“不提供源间距交互”。

### `POST /api/scene`

计算二维场景。请求体：

```json
{
  "preset": "electric_dipole",
  "density": 18,
  "resolution": 96,
  "sources": [
    {"x": -1.0, "y": 0.0, "kind": "positive", "strength": 1.0},
    {"x": 1.0, "y": 0.0, "kind": "negative", "strength": -5.0}
  ]
}
```

字段说明：

| 字段 | 必需 | 语义 |
|---|---:|---|
| `preset` | 否 | `/api/presets` 返回的稳定标识符；默认 `electric_dipole` |
| `density` | 否 | 种子总预算，整数范围 6–40，默认 18，不代表物理场强；每个参与播种的源至少分配 1 个种子 |
| `resolution` | 否 | 两个方向共同使用的标量网格分辨率，整数范围 32–144 |
| `sources` | 否 | 电偶极预设接受 2–8 个电荷且至少各含一个正、负电荷；磁偶极和 Halbach 预设接受 1–8 个 `dipole`；若省略则使用预设源，显式空列表、未知字段以及固定的 `current_loop`/`uniform` 预设 override 会被拒绝 |

`sources[].x` 与 `sources[].y` 是笛卡尔坐标，单位固定为 m，范围均为 $[-2.8,2.8]$。请求模型 `SourceInput.kind` 只接受 `positive`、`negative`、`dipole`、`uniform`；响应专用的 `wire_out`/`wire_into` 不能提交。电荷源的 `strength` 单位为 nC：`positive` 必须严格大于 0，`negative` 必须严格小于 0，二者都拒绝 0；省略时正电荷默认为 `1`，负电荷按 `kind` 默认为 `-1`。单位由预设决定，请求不得提交 `strength_unit`。

可编辑点源场景无论使用默认源还是 `sources` 覆盖，电荷都拥有圆形排除区域；磁偶极和 Halbach 则只有 `strength != 0` 的 active 偶极拥有排除区域。服务端对这些 active 源逐对检查 `/api/presets` 公布的下限；显式源中心距小于或等于 0.322 m 时返回 422，`detail` 精确指出原请求中的 0-based 下标，例如 `sources[1] 与 sources[2] 的中心距离必须大于 0.322 m`。浏览器的预判与吸附只改善交互，服务端校验仍是权威边界。

磁偶极子的 `strength` 单位为 A·m²，`angle_deg` 是从 $+x$ 朝 $+y$ 逆时针量取的面内角度，必须满足 $0\le\theta<360^\circ$。省略角度时使用 $90^\circ$，以兼容此前正强度指向 $+y$ 的行为；显式 `null`、非有限值和越界角度都非法。实际三维磁矩为

$$
\mathbf m=s\left(\cos\theta,\sin\theta,0\right),
$$

其中 $s$ 是有符号 `strength`，所以负值会把实际方向再翻转 $180^\circ$；单个 $s=0$ 仍是允许且会原样返回的显示 marker，但不参与场、播种、排除区域、源间距或种子预算。`magnetic_dipole` 与 `halbach_array` 场景必须至少有一个非零偶极；全零请求返回 422，`detail` 为 `至少需要一个非零磁偶极`。`angle_deg` 只能出现在 `dipole` 中，电荷即使提交 `null` 也会返回 422。这个参数化存在 $(s,\theta)$ 与 $(-s,\theta+180^\circ)$ 的等价表示，是为兼容 pre-1.0 已有的有符号强度契约而保留。

服务端必须为密度、分辨率、源数量和数值范围设置上限，防止一次交互请求耗尽内存或 CPU。

`density` 是一次场景请求的**整数 trace-job 总预算**，不是“每个源各放多少条”，范围为 6–40；浏览器滑块步长为 1，因此圆环的奇数预算轴线分支可达。电偶极的全部正、负电荷都参与预算：每个源先得到 1 个 job，其余按 $|q|$ 用稳定最大余数法分配；正电荷正向追踪，负电荷反向追踪。自定义磁偶极和 Halbach 场景只让非零 active 偶极参与逐源预算；若参与数量超过 `density`，服务端返回 422，例如 `electric_dipole 有 8 个电荷参与播种，density 至少为 8`。默认 Halbach 使用两条独立轨道，所以最低密度仍是 API 下限 6，不受八个显示偶极数量约束。浏览器在所有 POST 路径上执行相同预判并自动抬高不足的预算。

五个预设的策略分别是：

- 电荷与自定义多磁偶极使用逐源几何覆盖，`seed_mode: "coverage"`；
- 单个 active 磁偶极沿随参数角度旋转的赤道线按距离等距覆盖并双向追踪，`seed_mode: "coverage"`；负强度会反转实际磁矩，但不会改变同一条赤道几何；
- 默认 Halbach 在 $x\in[-2.1,2.1]$、$y=\pm0.45$ m 的两条平行轨道上等距覆盖并双向追踪；上轨分到 $\lceil density/2\rceil$ 个 job，下轨分到 $\lfloor density/2\rfloor$ 个，`seed_mode: "coverage"`；
- 圆环在环内赤道段调用公开的 `CircularLoopField.flux_function()`，以求根方式选择等 $\psi$ 的镜像轮廓；奇数预算再增加一条轴线特征线，整体 `seed_mode: "equal_flux"`；
- 匀强场仍从左边界等距覆盖播种，`seed_mode: "coverage"`。

每个 job 无论是否最终渲染，都恰好给 `metadata.termination_counts` 的一个原因加 1，所以 `sum(termination_counts.values()) == density`。对 `BOTH` job，`termination` 和 `termination_counts` 记录点序末端的正向分支，`start_termination` 与 `start_termination_counts` 另记点序起点的反向分支；非双向场景的起点端统计为空对象。双向曲线的点始终从反向端经过种子排到正向端，因此 `direction` 为 `1`。

电荷场会先完成全部 job，再按实际终止源对去重。若某条可渲染正电荷 $P\to N$ 轨迹已出现，随后终止于同一 $P$ 的负电荷反向 $N\to P$ 轨迹会被抑制；没有正向代表的源对以及负源到计算域边界的轨迹都保留。这里不做浮点几何相似判定。`suppressed_count` 记录被抑制的可渲染轨迹数，`rendered_line_count` 精确等于 `len(lines)`；若还存在不足两个有限点的退化结果，则 `rendered_line_count + suppressed_count <= density`。

固定回归场景 $+1\ \mathrm{nC}$（$x=-0.85$ m）、$-5\ \mathrm{nC}$（$x=0.85$ m）、`density=18` 会分配 4 个正向和 14 个反向 job。实际终止为 9 次 `exclusion_hit`（4 条 $P\to N$ 加 5 条 $N\to P$）与 9 次 `domain_exit`；去重只抑制后 5 条返线，故最终 `suppressed_count=5`、`rendered_line_count=13`。九条从边界进入负源的线全部保留。

成功响应结构如下。为便于阅读，示意片段把网格缩成 $2\times2$，并只展示渲染轨迹中的 1 条；实际端点接受的 `resolution` 不低于 32，数组会相应更长。

```json
{
  "domain": {
    "x": [-3.0, 3.0],
    "y": [-3.0, 3.0],
    "coordinate_system": "cartesian",
    "unit": "m"
  },
  "scalar": {
    "nx": 2,
    "ny": 2,
    "values": [0.15, null, 12.5, 0.18],
    "mask": [false, true, false, false],
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
    {
      "x": -1.0,
      "y": 0.0,
      "kind": "positive",
      "strength": 1.0,
      "strength_unit": "nC"
    },
    {
      "x": 1.0,
      "y": 0.0,
      "kind": "negative",
      "strength": -5.0,
      "strength_unit": "nC"
    }
  ],
  "metadata": {
    "title": "电偶极子的电场线",
    "projection_note": "该平面法向场分量为零，所示曲线是真实场线，不是投影流线。",
    "field_model": "三维点电荷场在 z=0 对称平面上的限制",
    "seed_mode": "coverage",
    "seed_description": "正负电荷按绝对强度共享总预算，并从各自排除面沿场的外向方向覆盖播种；线密度不代表场强。",
    "termination_counts": {"exclusion_hit": 9, "domain_exit": 9},
    "start_termination_counts": {},
    "suppressed_count": 5,
    "rendered_line_count": 13
  }
}
```

#### `domain`

给出数值坐标范围、坐标系和长度单位。当前响应固定为 `coordinate_system: "cartesian"` 与 `unit: "m"`。前端使用同一个范围映射标量栅格、曲线和源，不能为每个图层独立自动缩放；探针与源坐标编辑器必须同时显示该单位。

#### `scalar`

- `nx`、`ny` 定义规则网格尺寸；
- `values` 是长度为 `nx * ny` 的 row-major 一维数组：第 0 行对应 `ymax`，每行从 `xmin` 到 `xmax`，随后向 `ymin` 进入下一行；未遮罩项是未按色标裁剪的原始有限浮点值；
- `mask` 与 `values` 等长并逐项满足 `mask[i] == (values[i] is null)`；`true`/`null` 表示源排除区或非有限采样；
- `scale` 当前只允许 `linear` 或 `log`；
- `label` 与 `unit` 必须一同显示，避免无量纲色图。
- `vmin` 与 `vmax` 是服务端在排除 mask 后给出的建议色标范围，必须满足 `vmax > vmin`，对数尺度还要求 `vmin > 0`。

`vmin`/`vmax` 只控制颜色归一化；探针和数据消费者仍读取 `values` 的原值，即使它低于 `vmin` 或高于 `vmax`。对数尺度不能显示的非正值由显示层标为未着色（浏览器画成灰底斜线），不得改写成任意小正数；它们若本身有限，也仍是未遮罩原值。

#### `lines`

每条线是按轨迹顺序排列的坐标点。`direction` 为 `1` 或 `-1`，表示点序相对场方向；`termination` 记录点序末端采用的主分支终止原因。`start_termination` 是可选的加法字段，只在双向轨迹中记录点序起点端的反向分支原因。方向元数据不是要求前端把点序颠倒。后续可以增加场强和弧长，但客户端应忽略未知字段以保持向前兼容。

#### `sources`

源图层独立于标量网格。前端据 `kind` 选择符号或形状，并把 `strength` 与逐源 `strength_unit` 一同显示；不得从颜色像素反推源参数。电荷响应的单位为 `nC`，磁偶极矩响应的单位为 `A·m²`。`dipole` 响应总是显式返回规范范围内的 `angle_deg`；其他 kind 不返回该字段。前端画的是结合 `strength` 符号后的实际磁矩箭头，而不是只按参数角度画箭头。

`halbach_array` 的默认响应包含 8 个等间距 `dipole`：位置沿 $x\in[-2.1,2.1]$ 排列，`strength=1 A·m²`，角度序列为 $0^\circ,90^\circ,180^\circ,270^\circ$ 并重复两次。它是理想点偶极近似的教学预设，只承诺有限阵列的一侧场增强，不等同于有限尺寸永磁体，也不声称弱侧严格为零。用户一旦增删、移动或旋转源，响应元数据会把场景称为“可编辑面内磁偶极子阵列”，不再把任意排列冒充标准 Halbach 几何。

`current_loop` 固定返回两个只读标记：`wire_out` 画作 ⊙（电流出屏），`wire_into` 画作 ⊗（电流入屏）。两者是**同一个环形导体与 z=0 子午面的两个交点**，不是两根独立导线；二者的 `strength` 完全相同，表示同一个非负回路电流幅值，单位 `A`。电流方向只由 `kind` 承载，不使用一正一负的有符号电流。`strength_unit` 与 `wire_*` 都是只读响应元数据，不得混入后续 `SourceInput` 请求；前端也不得拖动或用键盘移动这两个固定标记。

#### `metadata`

`projection_note` 不能省略。它说明曲线是二维真实场线、投影流线还是三维曲线切片。定义见[二维切片何时包含真实场线](tutorial/04-slices-and-validation.md#true-vs-projected)。

`field_model`、`seed_mode`、`seed_description` 与各项计数都是前端直接展示的科学解释。`seed_mode` 是 `coverage`、`equal_flux`、`feature` 三值枚举；`seed_description` 是不可为空的具体说明。当前没有预设使用 `feature`，它为将来从临界点、分离线等特征位置出发的策略保留。兼容客户端对未来未知 mode 应原样显示，而不是拒绝场景；v0.3.0 前缓存的自由文本 `seed_mode` 在缺少说明时也可原样展示。终止原因使用 `TerminationReason` 的字符串值；客户端可翻译已知值，但遇到未知键必须原样显示。

`termination_counts` 表示所有 job 的末端/主分支原因，和恒等于 `density`；`start_termination_counts` 只汇总 `BOTH` job 的起点端原因，和等于双向 job 数。`suppressed_count` 是已计算但因电荷源对已有正向代表而未画出的可渲染线数，`rendered_line_count` 是实际响应线数。抑制不从尝试预算或终止统计中扣除。

源拖动期间，屏幕上的源位置已经变化而服务端场仍属于上一次请求。浏览器此时只保留坐标网格和新源位置，隐藏旧热图、色标与场线并标记“场待重算”；松开指针后只提交一次最终位置。数值编辑发生源间距冲突时回滚且不发场景请求。这个 UI 状态约束不改变 `POST /api/scene`：服务端仍会独立验证每个请求。

## API 兼容性

!!! warning "pre-1.0 电荷符号契约修正"
    旧实现遇到 `kind` 与 `strength` 符号矛盾时，会在组装电场时用 `abs()` 静默纠正物理符号，却把原始数值返回给客户端。现在 `positive`/`negative` 的不一致符号和 0 都返回 422，不再改写输入。这是 pre-1.0 阶段为消除物理模型与响应自相矛盾而做的契约修正。

!!! warning "v0.2.2 标量编码修正"
    masked 格点从一个伪造的有限填充值改为 JSON `null`，未遮罩格点从色标裁剪值改为原始有限值。依赖旧 `values: list[float]` 假设的 pre-1.0 客户端必须同时读取 `mask`/`null`；`vmin` 与 `vmax` 从此只表示显示建议，不能当作数据截断边界。

!!! warning "v0.3.0 播种元数据修正"
    `metadata.seed_mode` 从自由文本改为封闭枚举，具体说明移入必需的 `seed_description`；同时新增必需的双端终止和渲染/抑制计数。`LinePayload.start_termination` 是可选加法字段。由于项目仍处于 pre-1.0，这次服务端契约修正随版本与 CHANGELOG 一同发布；前端仍保留对旧自由文本说明和未来未知 mode token 的显示回退。

!!! note "pre-1.0 圆环几何迁移预告"
    `wire_out`/`wire_into` 是 v0.2.x–v0.3.x 为固定单圆环提供的响应专用标记。v0.4.x 引入多导体与导入几何时，导体几何将迁移到独立的 `conductors` 集合，并折并这两个临时 kind。`conductors` 目前尚未实现；这里提前记录的是 pre-1.0 契约演进方向，不是现有响应字段。

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

::: vectorviz.core.ToroidalExclusion
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

::: vectorviz.fields.CircularLoopField
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
