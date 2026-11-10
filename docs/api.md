# API 参考

VectorViz 同时提供 Python 公共 API 与浏览器使用的 HTTP API。本页描述稳定边界；内部帮助函数不属于兼容性承诺。

## Python API

稳定类型从顶层包导入：

```python
from vectorviz import (
    ChargedRingField,
    CircularLoopField,
    CompositeField,
    DielectricSphereField,
    Domain,
    ExclusionRegion,
    FieldLineTracer,
    InfiniteWireField,
    LaplaceField,
    MagneticDipoleField,
    MagneticShieldCylinderField,
    PointChargeField,
    SlopeField,
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

符号函数不一定是距离。自定义排除区保留 SciPy 变号事件语义，可能漏掉一步内多次穿越；内置 `SphericalExclusion` 和 `ToroidalExclusion` 另检查稠密数值曲线的步内几何，见[数值追踪](tutorial/03-tracing.md#exclusion-verification)。将自定义区域用作 `interfaces` 时，`abs(margin)` 必须是到界面距离的下界，不能任意倍乘后用于距离限步。

`minor_radius`（排除管半径）只控制追踪终止与显示 mask，不进入 `CircularLoopField` 的分母，也不把理想细导线改造成有限截面导线模型；`major_radius` 则对应物理圆环半径。

### 解析场

| 类型 | 用途 |
|---|---|
| `UniformField` | 匀强向量场与积分器基准 |
| `PointChargeField` | 二维或三维点电荷电场 |
| `MagneticDipoleField` | 磁偶极近似与远场模型 |
| `InfiniteWireField` | 一根或多根平行无限长理想细直导线的三维磁感应强度 |
| `CircularLoopField` | 理想细圆电流环的三维磁感应强度 |
| `ChargedRingField` | 均匀带电理想细圆环的三维电场，圆环的静电孪生 |
| `DielectricSphereField` | 匀强外场中介质球或导体球的三维电场，分段解析 |
| `LaplaceField` | 二维谐波多项式势的负梯度；首版为解析场族，不是通用边值求解器 |
| `SlopeField` | 一阶 ODE $y'=f(x,y)$ 的二维辅助向量场 $(1,f)$ |
| `MagneticShieldCylinderField` | 无限长线性磁屏蔽圆柱壳在横向匀强外场中的二维 B 场 |
| `CompositeField` | 对多个同维场做线性叠加 |

所有物理场内部应采用一致单位。输入源参数的单位与坐标系不得只存在于图标题里。

奇点集合只由源几何决定：`PointChargeField` 与 `MagneticDipoleField` 在源位置、`CircularLoopField` 与 `ChargedRingField` 在细环上都返回显式 `NaN`，即使该源的电荷、磁矩或电流为 0。零强度源在 Web 场景层构建场之前就被剔除；核心不会把源点软化成普通采样点。

#### `InfiniteWireField`

```python
wires = InfiniteWireField(
    current=[1.0, -1.0],
    position=[[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]],
    direction=(0.0, 0.0, 1.0),
)
field = wires.evaluate(points)
```

`current` 使用 A，可为标量或与导线数一致的一维数组；`position` 是每根导线上的参考点，单位 m，形状为 `(3,)` 或 `(n, 3)`。所有导线共享非零 `direction`，构造器将它归一化；正电流沿该方向，磁场遵守右手定则。默认 `permeability` 为 `scipy.constants.mu_0`，输出为 T。`positions`、`currents` 与 `direction` 是只读数组。

整个理想导线都是奇点，零电流也保留这项几何定义；奇点返回 `NaN`。模型没有有限导线端部、回流路径或有限线径。浏览器显示垂直于导线的真实不变平面，排除圆只用于追踪和遮罩。

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

#### `ChargedRingField`

```python
ring = ChargedRingField(
    charge=1.0e-9,
    radius=0.8,
    center=(0.0, 0.0, 0.0),
    normal=(0.0, 0.0, 1.0),
)
field = ring.evaluate(points)
psi = ring.flux_function(points)
```

`charge` 是整个圆环的总电荷，单位 C；`radius` 与 `center` 使用 m；默认 `permittivity` 使用 `scipy.constants.epsilon_0`，因此 `evaluate()` 返回 V/m。几何参数与 `CircularLoopField` 完全相同，`normal` 只固定环面，不带方向意义。细环本身是显式 `NaN` 奇点，与电荷值无关。

`flux_function()` 返回电场的轴对称通量函数 $\Psi$：穿过与轴同心、半径为 $\rho$、轴向位置为 $z$ 的圆盘的电通量除以 $2\pi$，满足 $E_\rho=-\rho^{-1}\partial_z\Psi$、$E_z=\rho^{-1}\partial_\rho\Psi$。它在轴上为 0、关于 $z$ 为奇函数、细环上为 `NaN`；环外的平面 $z=0$ 是割线，$\Psi$ 在那里跳变 $Q/(2\pi\varepsilon_0)$，恰好落在割线上的点取 $z\to0^+$ 的值。实现用圆盘对环上一点所张立体角的 Paxton 闭式（完全与不完全椭圆积分），测试与沿半径的直接通量求积比较。

#### `DielectricSphereField`

```python
sphere = DielectricSphereField(
    applied_field=(1.0, 0.0, 0.0),
    radius=1.0,
    center=(0.0, 0.0, 0.0),
    relative_permittivity=4.0,  # math.inf for a conductor
)
field = sphere.evaluate(points)
psi = sphere.flux_function(points)
```

`applied_field` 是远离球的匀强外场（V/m，非零），`radius` 与 `center` 使用 m，`relative_permittivity` 不小于 1，`math.inf` 表示导体。球内是匀强场 $3\mathbf E_0/(\varepsilon_r+2)$（导体为零），球外是 $\mathbf E_0$ 加感应偶极 $\mathbf p=4\pi\varepsilon_0a^3\alpha\mathbf E_0$ 的场，$\alpha=(\varepsilon_r-1)/(\varepsilon_r+2)$。场处处有限：切向分量在球面连续，法向分量按束缚面电荷跳变；恰在球面上的点取球外值。`polarizability`、`interior_field` 与 `is_conductor` 是只读属性。

`flux_function()` 返回 $\mathbf D/\varepsilon_0$ 的轴对称通量函数：以外场方向为轴，球内取 $\varepsilon_r\mathbf E$、球外取 $\mathbf E$ 穿过同轴圆盘的通量除以 $2\pi$。因为 $\mathbf D$ 的法向分量在介质表面连续，它在球面连续，等值线同时是 $\mathbf E$ 与 $\mathbf D$ 的场线；导体的球内值为 0，函数在球面跳变，线终止于自由面电荷。

#### `LaplaceField`

```python
field = LaplaceField(
    order=2,
    strength=1.0,
    length_scale=1.0,
    angle=0.0,
    center=(0.0, 0.0),
)
vectors = field.evaluate(points)
phi = field.potential(points)
```

令 $w=((x-c_x)+i(y-c_y))e^{-i\theta}/L$，模型取 $\phi=FL\operatorname{Re}(w^n)/n$、$\mathbf F=-\nabla\phi$。`order` 是正整数 $n$，`strength` 是有限的场尺度 $F$，`length_scale` 是有限正长度 $L$，`angle` 是有限的弧度角 $\theta$，`center` 是有限二维坐标。所有参数为只读属性；`potential()` 的输出形状是 `points.shape[:-1]`，`evaluate()` 保留输入形状 `(..., 2)`。

调用方负责统一单位：势的单位是场单位乘坐标单位。Web 两个预设统一使用无量纲约定。非零强度下，$n=1$ 为匀强场，$n>1$ 在中心有真实零点；`strength=0` 则为全零场。无有限位置的解析奇点，非有限坐标或浮点溢出返回 `NaN`，不能靠 softening 隐藏。各阶是谐波多项式，不包括对数源、任意边界条件或数值 Laplace 求解。

#### `SlopeField`

```python
field = SlopeField(lambda x, y: x - y)
vectors = field.evaluate([[0.0, 1.0], [1.0, 0.0]])
```

`slope` 必须是接受 NumPy `x`、`y` 数组的 callable，返回标量或能广播到点批次形状的实数数组。模型返回 $(1,f(x,y))$；归一化追踪只改变参数化，不改变 $y'=f(x,y)$ 的解图像。输入与输出均为二维，Web 例子坐标和向量单位均为 `"1"`（无量纲）。斜率为零表示水平解，向量并不为零；热图 $\sqrt{1+f^2}$ 只是辅助向量的大小，不是速度或物理场强。非有限坐标或斜率使两个分量都返回 `NaN`，交给追踪器的 `nonfinite_field` 政策处理；复数输出或错误的广播形状抛出 `ValueError`。调用方仍需保证函数的光滑性，并为奇点配置排除区。

这里的 Python callable 不会由 HTTP 请求创建或执行；Web 只提供固定公式的两个预设，自定义表达式输入仍在后续路线图中。

#### `MagneticShieldCylinderField`

```python
shield = MagneticShieldCylinderField(
    applied_field=(1.0e-6, 0.0),
    inner_radius=0.7,
    outer_radius=1.0,
    center=(0.0, 0.0),
    relative_permeability=100.0,
)
b = shield.evaluate(points)
az = shield.flux_function(points)
```

模型维数为 2，横截面垂直于无限长筒的轴；内腔和外部为真空，筒壁是线性、各向同性、均匀材料。`applied_field` 是有限非零的二维远处磁感应强度（T），半径和 `center` 使用 m，必须满足 $0<a<b$；`relative_permeability` 是有限且不小于 1 的相对磁导率 $\mu_r$，不接受 `math.inf`。参数与 `interior_field`、`shielding_factor` 都是只读属性。内腔为匀强场 $\mathbf B_0/S$，其中

$$
S=1+\frac{(\mu_r-1)^2}{4\mu_r}\left(1-\frac{a^2}{b^2}\right).
$$

`evaluate()` 返回 B，场在两个界面都有限，恰在界面上采用径向外侧值：$r=a$ 属筒壁，$r=b$ 属外部。法向 B 连续，切向 H 连续，切向 B 一般跳变。`flux_function()` 返回二维 $A_z$，单位 T·m，形状为 `points.shape[:-1]`，满足 $\mathbf B=(\partial_y A_z,-\partial_x A_z)$；它跨两界面连续，可用作场线不变量。

手动使用追踪器时应把以 `center` 为圆心、半径分别为 $a,b$ 的两个 `SphericalExclusion` 登记为 `interfaces`，而非 `exclusions`。材料不是奇点区，壁内允许采样与追踪。首版不包含有限长度、端部开口、多层、非线性、饱和、磁滞或涡流；公式推导及原始来源见[场模型教程](tutorial/02-field-models.md#magnetic-shield-cylinder)。

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

内置排除区的步内检查会在需要时截断返回轨迹，并保留正确的种子、终点及终止原因。有限细分仍不能判定的接触（包括精确切触）返回 `solver_failure`，`message` 说明几何检查未能消除歧义；不会把它报告为安全出域或凭空认定穿入。几何检查只消费求解器的稠密输出，不重新计算场公式；为生成稠密输出而执行的额外 RHS 求值计入 `nfev`。与闭环一样，检查可能在求解分段完成后截断，因此 `nfev` 可以包含最终返回端点之后的探测。

`interfaces` 是场不连续的曲面（例如材料边界），用带符号 `margin()` 的区域给出，可与 `exclusions` 共用 `SphericalExclusion`。线到达界面时求解器在界面上以事件停下，再沿到达界面的运动方向前进几个 ulp 重新启动，所以折射点由事件定位，不被自适应步长抹平；跨越界面那一步本身的局部误差仍在容差量级（介质球场景实测 $\Psi$ 沿线相对漂移约 $10^{-5}$，无界面事件时约 $10^{-4}$）。积分按“到最近界面的距离”分段：单位速度的线走不到界面就不可能越过它，所以一段之内不会一进一出而漏检；分段长度有 $10^{-4}\,\mathrm{max\_step}$ 的下限，弦长短于这个下限的擦入仍可能漏掉（默认参数下对应深度约 $10^{-11}$ m）。恰在界面上的种子先沿其场方向移动几个 ulp。越过界面后场为零或非有限时，分支在界面上以相应原因终止；连续三次重启都没有前进（外侧场指回界面的滑动接触）时以 `solver_failure` 终止并在 `message` 中说明。稠密输出会把定位到的交点并入点序。未登记 `interfaces` 时不会执行界面分段或重启；内置排除区仍会执行独立的步内检查。

追踪器另有一个**停滞保护**：右端函数是单位流向，光滑线上一步之内它只转很小的角度；分支沿对称线正对零场点（例如相邻同号电荷之间的鞍点）时，求解器的各级点落在方向反转两侧，步端却进不了零场阈值，于是以 $10^{-5}$ 量级的小步来回跳动、流向反复反转，直到耗尽弧长预算。求解器记录最近至多 128 步的反转历史，其中有 48 步流向反转超过 $120^\circ$ 时停下（从第一步起累计，不必先走满 128 步；来回跳动时约有一半的步反转），追踪器沿流向把反转二分到零场阈值之内，以 `null_field` 终止并在 `message` 中说明；找不到零点（反转来自未登记的不连续面）则以 `solver_failure` 终止。未达到这个判据就由零场事件结束的线保持逐位不变：全部预设线如此，已验证的短跳动样本最多跳动 44 步、反转 14 次。达到判据之后才碰巧落入阈值的线改为由二分定位零点，这是有意的例外；记录在案的一例（$+1/-6.233$ nC 轴线种子，原先跳动 220 步）仍以 `null_field` 结束，终点到解析零点的距离从 $1.4\times10^{-7}$ m 降到 $3\times10^{-8}$ m，求值次数约减半。有限窗口是可执行的停滞策略，不是“以后永远不会落入阈值”的证明。

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
  "version": "0.3.9"
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

当前预设标识符为：

- `electric_dipole`
- `electric_quadrupole`
- `electric_hexagon`
- `electric_hexagon_alternating`
- `magnetic_dipole`
- `halbach_array`
- `current_loop`
- `straight_wire`
- `parallel_wires`
- `antiparallel_wires`
- `solenoid`
- `helmholtz_coils`
- `anti_helmholtz_coils`
- `charged_ring`
- `dielectric_sphere`
- `conducting_sphere`
- `uniform`
- `laplace_saddle`
- `laplace_multipole`
- `slope_linear`
- `slope_logistic`
- `magnetic_shield_cylinder`

此端点是可用预设及其交互能力的权威目录；当前无构建客户端随版本静态提供同一组选项，并由契约测试防止两边漂移。客户端不应假设列表永久不变。

v0.3.9 的五个预设也不接受 `sources` 覆盖，不提供 `source_separation`，仍接受同样的 `density`、`resolution` 与 `seeds`。`laplace_saddle` 与 `laplace_multipole` 分别取 `order=2/3`、其余参数为默认值；`slope_linear` 使用 $y'=x-y$，`slope_logistic` 使用 $y'=y(1-y)$。这四个预设的 `domain.unit`、`scalar.unit`、`vector.unit` 均为 `"1"`，向量标签为 `F`。`magnetic_shield_cylinder` 采用上述 Python 示例参数，坐标单位 m，向量标签 B、单位 T，`sources` 为空，`regions` 返回一个圆柱壳。五个场景的 `seed_mode` 都是 `coverage`，线密度不表示场强或磁通。

v0.3.8 新增的六个磁静态预设均为固定几何，不提供 `source_separation`，不接受 `sources` 覆盖（返回 422）；`density`、`resolution` 和手动 `seeds` 与其他预设相同。直导线标记表示横截面，螺线管与线圈组的每一对标记表示同一个圆环的两个截面；均复用 `wire_out`（⊙，电流沿 +z）与 `wire_into`（⊗，沿 −z），`strength` 是电流大小，单位 A。场和向量网格的量为 B，单位 T，曲线是 z=0 不变平面中的真实磁场线。播种为 `coverage`，图中的线密度不代表磁感应强度；各场景的具体布点和近似由 `metadata` 说明。闭合终止仍使用既有的 `closed_loop` 契约。

`electric_dipole`、`electric_quadrupole`、`electric_hexagon`、`electric_hexagon_alternating`、`magnetic_dipole` 与 `halbach_array` 是可编辑点源预设，因此返回 `source_separation`。`exclusive_minimum: 0.322` 表示任意两个实际拥有排除区域的源中心距离必须**严格大于** 0.322 m；等于该值仍冲突。其余固定几何预设没有这项能力，响应省略该字段；兼容客户端也应把缺失或 `null` 都解释为“不提供源间距交互”。

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
  ],
  "seeds": [{"x": 0.0, "y": 1.2}]
}
```

字段说明：

| 字段 | 必需 | 语义 |
|---|---:|---|
| `preset` | 否 | `/api/presets` 返回的稳定标识符；默认 `electric_dipole` |
| `density` | 否 | 种子总预算，整数范围 6–40，默认 18，不代表物理场强；每个参与播种的源至少分配 1 个种子 |
| `resolution` | 否 | 两个方向共同使用的标量与向量网格分辨率，整数范围 32–144 |
| `sources` | 否 | 电偶极预设接受 2–8 个电荷且至少各含一个正、负电荷；电四极子、六个正电荷和三对交替电荷预设接受 1–8 个任意符号的电荷；磁偶极和 Halbach 预设接受 1–8 个 `dipole`；若省略则使用预设源，显式空列表、未知字段以及其他固定几何预设的 override 会被拒绝 |
| `seeds` | 否 | 1–16 个手动种子 `{x, y}`，单位随预设的 `domain.unit`，范围 $[-3,3]$（即计算域）；每个种子在预设预算之外双向追踪一次，所有预设都接受；显式空列表、未知字段和越界坐标会被拒绝 |

`sources[].x` 与 `sources[].y` 是笛卡尔坐标，单位固定为 m，范围均为 $[-2.8,2.8]$。请求模型 `SourceInput.kind` 只接受 `positive`、`negative`、`dipole`、`uniform`；响应专用的 `wire_out`/`wire_into`/`ring_charge` 不能提交。电荷源的 `strength` 单位为 nC：`positive` 必须严格大于 0，`negative` 必须严格小于 0，二者都拒绝 0；省略时正电荷默认为 `1`，负电荷按 `kind` 默认为 `-1`。单位由预设决定，请求不得提交 `strength_unit`。

可编辑点源场景无论使用默认源还是 `sources` 覆盖，电荷都拥有圆形排除区域；磁偶极和 Halbach 则只有 `strength != 0` 的 active 偶极拥有排除区域。服务端对这些 active 源逐对检查 `/api/presets` 公布的下限；显式源中心距小于或等于 0.322 m 时返回 422，`detail` 精确指出原请求中的 0-based 下标，例如 `sources[1] 与 sources[2] 的中心距离必须大于 0.322 m`。浏览器的预判与吸附只改善交互，服务端校验仍是权威边界。

磁偶极子的 `strength` 单位为 A·m²，`angle_deg` 是从 $+x$ 朝 $+y$ 逆时针量取的面内角度，必须满足 $0\le\theta<360^\circ$。省略角度时使用 $90^\circ$，以兼容此前正强度指向 $+y$ 的行为；显式 `null`、非有限值和越界角度都非法。实际三维磁矩为

$$
\mathbf m=s\left(\cos\theta,\sin\theta,0\right),
$$

其中 $s$ 是有符号 `strength`，所以负值会把实际方向再翻转 $180^\circ$；单个 $s=0$ 仍是允许且会原样返回的显示 marker，但不参与场、播种、排除区域、源间距或种子预算。`magnetic_dipole` 与 `halbach_array` 场景必须至少有一个非零偶极；全零请求返回 422，`detail` 为 `至少需要一个非零磁偶极`。`angle_deg` 只能出现在 `dipole` 中，电荷即使提交 `null` 也会返回 422。这个参数化存在 $(s,\theta)$ 与 $(-s,\theta+180^\circ)$ 的等价表示，是为兼容 pre-1.0 已有的有符号强度契约而保留。

四个电荷预设共用同一套点电荷契约：默认源不同，其余（排除区域、源间距、按 $|q|$ 分配预算、返线抑制、`nC` 单位）完全相同。`electric_quadrupole` 把正负交替的四个 1 nC 电荷放在 $(\pm0.9,\pm0.9)$ m 的正方形顶点；`electric_hexagon` 把六个 $+1$ nC 电荷放在半径 0.9 m 的正六边形顶点；`electric_hexagon_alternating` 用同一六边形、符号交替。六边形顶点坐标由同一对舍入后的数值镜像生成，几何严格对称；逐源顺序求和的场则只对称到舍入误差（默认排布约 $10^{-13}\ \mathrm{V/m}$，对称轴上的横向分量约 $2\times10^{-15}\ \mathrm{V/m}$），正对中心的线因此保持在对称线附近，而中心汇点会把它们收进去。这三个预设一旦收到 `sources` 覆盖，标题改为“可编辑点电荷组的电场线”，不再冒充命名几何；`electric_dipole` 保持原有的“至少各含一个正、负电荷”约束和标题。

点电荷场景的零场终止阈值为 $10^{-6}\ \mathrm{V/m}$：默认源是相距约 1 m 的 1 nC 电荷，$|\mathbf E|$ 的量级为 10 V/m，线在离线性零点约 $10^{-7}$ m 处以 `null_field` 停止。这是与场同单位的数值截止值，不是新的物理零点；把整个场景乘以常数时它不会自动缩放：强度远小于 1 nC 的覆盖（例如 $10^{-10}$ nC）会让整个场景低于阈值，所有线在种子处就以 `null_field` 终止，$\pm0.01$ nC 则不受影响。凡是通向真实零场点的轴线（例如 $+1/-4$ nC 或其他 density 下的 $+1/-5$ nC）都会从此前的 `exclusion_hit` 改为 `null_field`；渲染数与抑制数不变。

服务端必须为密度、分辨率、源数量和数值范围设置上限，防止一次交互请求耗尽内存或 CPU。

`density` 是一次场景请求的**整数 trace-job 总预算**，不是“每个源各放多少条”，范围为 6–40；浏览器滑块步长为 1，因此圆环的奇数预算轴线分支可达。电偶极的全部正、负电荷都参与预算：每个源先得到 1 个 job，其余按 $|q|$ 用稳定最大余数法分配；正电荷正向追踪，负电荷反向追踪。自定义磁偶极和 Halbach 场景只让非零 active 偶极参与逐源预算；若参与数量超过 `density`，服务端返回 422，例如 `electric_dipole 有 8 个电荷参与播种，density 至少为 8`。默认 Halbach 使用两条独立轨道，所以最低密度仍是 API 下限 6，不受八个显示偶极数量约束。浏览器在所有 POST 路径上执行相同预判并自动抬高不足的预算。

各预设的策略分别是：

- v0.3.9 的两个 Laplace、两个斜率场和磁屏蔽筒使用 `coverage`，覆盖各自的方向区间、解曲线族或内腔/筒壁/外部区域；每个 job 双向积分，具体布点由 `seed_description` 说明；不宣称等通量；

- 电荷与自定义多磁偶极使用逐源几何覆盖，`seed_mode: "coverage"`；
- 单个 active 磁偶极沿随参数角度旋转的赤道线按距离等距覆盖并双向追踪，`seed_mode: "coverage"`；负强度会反转实际磁矩，但不会改变同一条赤道几何；
- 默认 Halbach 在 $x\in[-2.1,2.1]$、$y=\pm0.45$ m 的两条平行轨道上等距覆盖并双向追踪；上轨分到 $\lceil density/2\rceil$ 个 job，下轨分到 $\lfloor density/2\rfloor$ 个，`seed_mode: "coverage"`；
- v0.3.8 的直导线、双导线、螺线管与线圈组使用 `coverage`：按不同半径、主磁通族或局部闭环族选择种子；预算与显示电流截面数无关，允许 `density=6` 的 8 圈螺线管；这些预设不宣称等通量；
- 圆环在环内赤道段调用公开的 `CircularLoopField.flux_function()`，以求根方式选择等 $\psi$ 的镜像轮廓；奇数预算再增加一条轴线特征线，整体 `seed_mode: "equal_flux"`；
- 带电圆环在两个截面周围半径 0.162 m 的种子圆上调用公开的 `ChargedRingField.flux_function()`：环外赤道割线上下的 $\Psi$ 从 $+Q/(4\pi\varepsilon_0)$ 单调降到 $-Q/(4\pi\varepsilon_0)$，取 $\lfloor density/2\rfloor$ 个等分区间的中点为目标求根，再镜像到另一截面；目标恰为 0 的种子直接放在环内赤道上，因为它通向中心鞍点，任何离轴舍入都会把它偏转。奇数预算另加一条从 $+x$ 截面沿环面向外的赤道射线（上下两半场的分界线），整体 `seed_mode: "equal_flux"`；
- 介质球与导体球在左边界 $x=-3+10^{-4}$ m 上调用公开的 `DielectricSphereField.flux_function()`：从 $\Psi_D/\lfloor density/2\rfloor$ 到边界最大可用高度 $y=2.88$ m 处的 $\Psi_D$ 等分取目标求根，再镜像到轴两侧；奇数预算另加轴线（$\Psi_D=0$ 的成员），整体 `seed_mode: "equal_flux"`。介质球的线在球面折射后继续，导体球的线在球面以 `null_field` 终止；
- 匀强场仍从左边界等距覆盖播种，`seed_mode: "coverage"`。

每个 job 无论是否最终渲染，都恰好给 `metadata.termination_counts` 的一个原因加 1，所以 `sum(termination_counts.values()) == density`。对 `BOTH` job，`termination` 和 `termination_counts` 记录点序末端的正向分支，`start_termination` 与 `start_termination_counts` 另记点序起点的反向分支；非双向场景的起点端统计为空对象。双向曲线的点始终从反向端经过种子排到正向端，因此 `direction` 为 `1`。

电荷场会先完成全部 job，再按实际终止源对去重。若某条可渲染正电荷 $P\to N$ 轨迹已出现，随后终止于同一 $P$ 的负电荷反向 $N\to P$ 轨迹会被抑制；没有正向代表的源对以及负源到计算域边界的轨迹都保留。这里不做浮点几何相似判定。`suppressed_count` 记录被抑制的可渲染轨迹数，`rendered_line_count` 精确等于 `len(lines)`；若还存在不足两个有限点的退化结果，则 `rendered_line_count + suppressed_count <= density`（带手动种子时上界加上 `manual_seed_count`）。

固定回归场景 $+1\ \mathrm{nC}$（$x=-0.85$ m）、$-5\ \mathrm{nC}$（$x=0.85$ m）、`density=18` 会分配 4 个正向和 14 个反向 job。实际终止为 8 次 `exclusion_hit`（3 条 $P\to N$ 加 5 条 $N\to P$）、1 次 `null_field` 与 9 次 `domain_exit`：从正电荷背向负电荷出发的那条轴线在 $x=-0.85\,(1+\sqrt5)/(\sqrt5-1)\approx-2.225$ m 的真实零场点停止。去重只抑制 5 条返线，故最终 `suppressed_count=5`、`rendered_line_count=13`。九条从边界进入负源的线全部保留。

#### 手动种子 { #manual-seeds }

`seeds` 列出的每个点都用该预设的追踪器设置（同样的容差、排除区、界面与零场阈值）**双向**追踪一次，结果按种子顺序追加在预设线之后，并带 `seed_index`（请求中的 0-based 下标）；预设线不带这个字段。手动线的点序从反向端经种子到正向端，`direction` 为 `1`，`termination` 与 `start_termination` 都存在，种子本身是点列中的一个点。种子恰在球面这类界面上时，追踪器先把起点沿场方向推离界面几个 ulp 再积分（见 [`FieldLineTracer`](#fieldlinetracer)），返回的点列仍包含请求的种子：离它最近、且在 $10^{-6}$ m 内的推离起点被换成种子本身。沿对称轴正对零场点（例如六个正电荷的电荷对之间的鞍点）的种子在该零点以 `null_field` 停下，见追踪器的停滞保护。手动种子**不占** `density` 预算，也不参加电荷场的返线抑制：`termination_counts` 之和仍等于 `density`，手动线的两端原因另记在 `metadata.manual_termination_counts` 与 `metadata.manual_start_termination_counts`，二者之和都等于 `manual_seed_count`（即请求的种子数）。落在排除区内、导体球内或零场点上的种子只得到一个点，不产生线，但仍计入手动终止统计（例如 `exclusion_hit` 或 `null_field`），所以 `rendered_line_count + suppressed_count <= density + manual_seed_count`。

同一场景与预算下，预设线的追踪结果与不带 `seeds` 的请求相同并共用缓存；手动线按种子列表另行缓存，改分辨率时二者都不重算。

成功响应结构如下。为便于阅读，示意片段把网格缩成 $2\times2$，并只展示渲染轨迹中的 1 条预设线与 1 条手动线；实际端点接受的 `resolution` 不低于 32，数组会相应更长。

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
  "vector": {
    "nx": 2,
    "ny": 2,
    "u": [0.12, null, -12.4, 0.17],
    "v": [-0.09, null, 1.6, 0.06],
    "label": "E",
    "unit": "V/m"
  },
  "lines": [
    {
      "points": [[-1.8, 0.2], [-1.7, 0.21]],
      "direction": 1,
      "termination": "domain_exit"
    },
    {
      "points": [[-0.4, 3.0], [0.0, 1.2], [0.5, 0.36]],
      "direction": 1,
      "termination": "exclusion_hit",
      "start_termination": "domain_exit",
      "seed_index": 0
    }
  ],
  "regions": [],
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
    "termination_counts": {"exclusion_hit": 8, "null_field": 1, "domain_exit": 9},
    "start_termination_counts": {},
    "suppressed_count": 5,
    "rendered_line_count": 14,
    "manual_seed_count": 1,
    "manual_termination_counts": {"exclusion_hit": 1},
    "manual_start_termination_counts": {"domain_exit": 1}
  }
}
```

#### `domain`

给出数值坐标范围、坐标系和坐标单位。当前响应固定为 `coordinate_system: "cartesian"`，`unit` 为 `"m"` 或 `"1"`；四个数学预设使用 `"1"`，表示无量纲，其余物理预设使用 m。前端使用同一个范围映射标量栅格、曲线和源，不能为每个图层独立自动缩放；坐标轴与探针必须显示该单位或其无量纲含义。

#### `scalar`

- `nx`、`ny` 定义规则网格尺寸；
- `values` 是长度为 `nx * ny` 的 row-major 一维数组：第 0 行对应 `ymax`，每行从 `xmin` 到 `xmax`，随后向 `ymin` 进入下一行；未遮罩项是未按色标裁剪的原始有限浮点值；
- `mask` 与 `values` 等长并逐项满足 `mask[i] == (values[i] is null)`；`true`/`null` 表示源排除区或非有限采样；
- `scale` 当前只允许 `linear` 或 `log`；
- `label` 与 `unit` 必须一同显示；`unit: "1"` 明确表示无量纲，不能把未声明单位的物理量默认为无量纲。
- `vmin` 与 `vmax` 是服务端在排除 mask 后给出的建议色标范围，必须满足 `vmax > vmin`，对数尺度还要求 `vmin > 0`。

`vmin`/`vmax` 只控制颜色归一化；探针和数据消费者仍读取 `values` 的原值，即使它低于 `vmin` 或高于 `vmax`。对数尺度不能显示的非正值由显示层标为未着色（浏览器画成灰底斜线），不得改写成任意小正数；它们若本身有限，也仍是未遮罩原值。

#### `vector` { #vector }

与标量网格同一次求值得到的面内场分量，可选字段，当前服务端总是返回：

- `nx`、`ny` 与 `scalar` 相同，排列方式也相同（第 0 行对应 `ymax`，每行从 `xmin` 到 `xmax`）；
- `u`、`v` 分别是 $x$、$y$ 分量（沿用 matplotlib `quiver(X, Y, U, V)` 的命名），是未裁剪的 float64 原值，`null` 的位置与 `scalar.mask` 逐项一致，二者必须同时为 `null`；
- `label` 是场的名字（`E`、`B` 或数学辅助场 `F`），`unit` 是分量单位，与 `scalar.unit` 相同；当前 `scalar` 就是这个场的模长，`scalar.label` 为 `|E|`/`|B|`/`|F|`，$\sqrt{u^2+v^2}$ 与 `scalar.values` 只差浮点舍入。

导体球内部是真实的零场：`u`、`v` 为 `0.0`，不是 `null`。浏览器用这份网格画无向的斜率短线、在探针里显示方向角，并在光标周围用双线性插值加 RK4 画预览束；预览只是网格插值，不是服务端积分，点击后才把种子交给服务端得到精确线（见[手动种子](#manual-seeds)）。缺少这一字段的旧响应仍然有效，只是没有这些显示。

#### `lines`

每条线是按轨迹顺序排列的坐标点。`direction` 为 `1` 或 `-1`，表示点序相对场方向；`termination` 记录点序末端采用的主分支终止原因。`start_termination` 是可选的加法字段，只在双向轨迹中记录点序起点端的反向分支原因。方向元数据不是要求前端把点序颠倒。`seed_index` 只出现在[手动种子](#manual-seeds)的线上，是请求 `seeds` 中的 0-based 下标；预设线没有这个字段。后续可以增加场强和弧长，但客户端应忽略未知字段以保持向前兼容。

#### `sources`

源图层独立于标量网格。前端据 `kind` 选择符号或形状，并把 `strength` 与逐源 `strength_unit` 一同显示；不得从颜色像素反推源参数。电荷响应的单位为 `nC`，磁偶极矩响应的单位为 `A·m²`。`dipole` 响应总是显式返回规范范围内的 `angle_deg`；其他 kind 不返回该字段。前端画的是结合 `strength` 符号后的实际磁矩箭头，而不是只按参数角度画箭头。

`halbach_array` 的默认响应包含 8 个等间距 `dipole`：位置沿 $x\in[-2.1,2.1]$ 排列，`strength=1 A·m²`，角度序列为 $0^\circ,90^\circ,180^\circ,270^\circ$ 并重复两次。它是理想点偶极近似的教学预设，只承诺有限阵列的一侧场增强，不等同于有限尺寸永磁体，也不声称弱侧严格为零。用户一旦增删、移动或旋转源，响应元数据会把场景称为“可编辑面内磁偶极子阵列”，不再把任意排列冒充标准 Halbach 几何。

`charged_ring` 固定返回两个只读 `ring_charge` 标记，画作带正号的圆点：它们是同一个均匀带电圆环与 z=0 子午面的两个交点，`strength` 都是同一个非零总电荷 `1 nC`。该场景使用与点电荷相同的 $10^{-6}\ \mathrm{V/m}$ 零场终止阈值，环内赤道线在中心零场点以 `null_field` 停止。

`current_loop` 固定返回两个只读标记：`wire_out` 画作 ⊙（电流出屏），`wire_into` 画作 ⊗（电流入屏）。两者是**同一个环形导体与 z=0 子午面的两个交点**，不是两根独立导线；二者的 `strength` 完全相同，表示同一个非负回路电流幅值，单位 `A`。电流方向只由 `kind` 承载，不使用一正一负的有符号电流。`strength_unit` 与 `wire_*` 都是只读响应元数据，不得混入后续 `SourceInput` 请求；前端也不得拖动或用键盘移动这两个固定标记。

#### `regions`

材料区域是几何，不是场源：`dielectric_sphere`、`conducting_sphere` 与 `magnetic_shield_cylinder` 各返回一个区域，其他预设返回空列表 `[]`。每个区域给出 `kind`、中心 `x`/`y`、`radius`、`unit: "m"`。介质球另给 `relative_permittivity`（不小于 1），导体球该字段为 `null`；球体的 `inner_radius` 与 `relative_permeability` 只能省略或为 `null`，不能带有圆柱壳参数值。前端把区域画成该平面上的界面并标注材料，区域内的场是真实值、照常着色；缺少 `regions` 字段的旧响应按空列表处理，但存在时每个区域必须完整，否则整个场景被判为无效。

圆柱壳的 `radius` 是外半径，`inner_radius` 必须满足 `0 < inner_radius < radius`，`relative_permeability` 必须有限且不小于 1；其 `relative_permittivity` 为 `null`。前端显示内外两条圆界面与材料环，内腔不会被填成实心材料，也不会因材料区域而遮罩或禁止播种。

```json
{
  "kind": "magnetic_shield_cylinder",
  "x": 0.0,
  "y": 0.0,
  "radius": 1.0,
  "inner_radius": 0.7,
  "relative_permeability": 100.0,
  "relative_permittivity": null,
  "unit": "m"
}
```

```json
{
  "kind": "dielectric_sphere",
  "x": 0.0,
  "y": 0.0,
  "radius": 1.0,
  "relative_permittivity": 4.0,
  "unit": "m"
}
```

#### `metadata`

`projection_note` 不能省略。它说明曲线是二维真实场线、投影流线还是三维曲线切片。定义见[二维切片何时包含真实场线](tutorial/04-slices-and-validation.md#true-vs-projected)。

`field_model`、`seed_mode`、`seed_description` 与各项计数都是前端直接展示的科学解释。`seed_mode` 是 `coverage`、`equal_flux`、`feature` 三值枚举；`seed_description` 是不可为空的具体说明。当前没有预设使用 `feature`，它为将来从临界点、分离线等特征位置出发的策略保留。兼容客户端对未来未知 mode 应原样显示，而不是拒绝场景；v0.3.0 前缓存的自由文本 `seed_mode` 在缺少说明时也可原样展示。终止原因使用 `TerminationReason` 的字符串值；客户端可翻译已知值，但遇到未知键必须原样显示。

`termination_counts` 表示预设 job 的末端/主分支原因，和恒等于 `density`；`start_termination_counts` 只汇总预设 `BOTH` job 的起点端原因，和等于双向 job 数。`suppressed_count` 是已计算但因电荷源对已有正向代表而未画出的可渲染线数，`rendered_line_count` 是实际响应线数（含手动线）。抑制不从尝试预算或终止统计中扣除。`manual_seed_count`、`manual_termination_counts` 与 `manual_start_termination_counts` 只统计[手动种子](#manual-seeds)：没有 `seeds` 的请求返回 `0` 与两个空对象。

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

::: vectorviz.fields.InfiniteWireField
    options:
      heading_level: 4

::: vectorviz.fields.CircularLoopField
    options:
      heading_level: 4

::: vectorviz.fields.ChargedRingField
    options:
      heading_level: 4

::: vectorviz.fields.DielectricSphereField
    options:
      heading_level: 4

::: vectorviz.fields.CompositeField
    options:
      heading_level: 4

::: vectorviz.planar_fields.LaplaceField
    options:
      heading_level: 4

::: vectorviz.planar_fields.SlopeField
    options:
      heading_level: 4

::: vectorviz.material_fields.MagneticShieldCylinderField
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
