# 3. 数值追踪

场线追踪从种子点出发，沿局部场方向积分，同时控制每一步的误差。

VectorViz 解的是

$$
\frac{d\mathbf x}{ds}
=\sigma\frac{\mathbf F(\mathbf x)}{\lVert\mathbf F(\mathbf x)\rVert},
\qquad \sigma\in\{+1,-1\}.
$$

$\sigma$ 只是积分方向。物理箭头仍沿 $+\mathbf F$；反向分支用于从同一种子追出曲线的另一半。

## 一条曲线怎样算出来

对每个种子，程序按下面的顺序工作：

1. 检查种子是否在计算域和排除区之外；
2. 求 $\mathbf F(\mathbf x_0)$，拒绝非有限值或零场点；
3. 分别积分正、反两个分支；
4. 每个成功步后检查计算域、排除区和弱场事件；
5. 到达事件或最大弧长时停止，并记录原因；
6. 若用于绘图，再用稠密输出按弧长重采样。

自适应求解器的内部节点并不等距：弯曲或误差较大的区域节点更密，平直区域更疏。渲染点应另行按弧长重采样，无需为画面等距而固定积分步长。

## 为什么不用固定步长 RK4 作为默认值

手写 RK4 适合演示一次积分步，但不会同时给出误差估计。固定步长也难以兼顾平直区、急弯区、网格边界和源附近；步长过大时，还可能跨过很薄的事件面。

嵌入式 Runge–Kutta 方法同时算出两个不同阶的近似，用它们的差估计局部误差并调整下一步。Dormand–Prince 5(4) 是这类方法的经典来源。[^dormand-prince]

当前默认使用 SciPy 的 DOP853。它是八阶显式 Runge–Kutta 方法，适合要求较高精度的非刚性问题；`solve_ivp` 还提供事件定位与稠密输出。[^scipy-solve-ivp]

## 六个最常用的参数

```python
from vectorviz import TraceOptions

options = TraceOptions(
    rtol=1e-7,
    atol=1e-9,
    max_step=0.05,
    max_arc_length=12.0,
    null_threshold=1e-12,
    output_step=0.025,
)
```

| 参数 | 控制什么 | 容易误解的地方 |
|---|---|---|
| `rtol` | 随状态大小缩放的局部误差 | 不是整条曲线的最终误差保证 |
| `atol` | 很小坐标分量的绝对局部误差 | 若坐标单位是米，它也带有米的量纲 |
| `max_step` | 求解器单次最多走多长 | 过大可能漏过窄事件或削平弯曲 |
| `max_arc_length` | 一个分支最多追多远 | 不是计算域直径，也不是物理时间 |
| `null_threshold` | 弱场停止的绝对场强 | 场整体缩放后也要相应缩放 |
| `output_step` | 返回点的弧长间距 | 只改变采样，不提高积分本身精度 |

SciPy 的误差尺度近似为 `atol + rtol * abs(y)`。不同坐标分量的尺度相差很大时，先无量纲化，或为分量选择不同的绝对容差。

这个误差尺度还依赖坐标原点。同一幅局部几何若整体平移到很大的坐标值，`rtol * abs(y)` 会随之变大。远离原点的小尺度场景应改用以场景中心为原点的局部坐标，计算结束后再变换回全局坐标。

网格场的 `max_step` 还要受单元尺寸限制。取最小网格间距的 $0.25$ 到 $0.5$ 倍可作为第一次试算的保守值，但最终值必须靠网格与步长收敛实验确定，不是固定定律。

## 明确记录终止原因

每个分支都应记录停止原因。当前可能的状态有：

| `termination` | 含义 |
|---|---|
| `domain_exit` | 到达计算域边界 |
| `exclusion_hit` | 命中源或几何排除面 |
| `null_field` | 场强降到数值截止值；精确零点处方向才真正未定义 |
| `nonfinite_field` | 场求值出现 `NaN` 或无穷 |
| `max_arc_length` | 达到设定的最大弧长 |
| `seed_outside_domain` | 种子一开始就在域外 |
| `solver_failure` | 求解器未能继续 |
| `closed_loop` | 显式启用闭环判据后，轨迹以同向切向返回种子邻域 |

事件函数靠符号变化定位边界。若一步跨过同一事件的多个零点，求解器可能漏检；与事件面相切而不变号时也可能漏检。特别是 `null_threshold=0` 时，事件函数 $\lVert\mathbf F\rVert$ 恒非负，不能依赖普通变号检测找到所有精确零点。因此 `max_step`、正的物理或数值截止值，以及事后最小场强检查都很重要。[^scipy-solve-ivp]

闭环检测默认关闭，因为空间容差随坐标单位与场景尺度变化，不能用一个无量纲常数替所有问题决定。启用时同时设置 `closure_tolerance` 与 `closure_min_arc_length`，后者必须大于前者的两倍，让轨迹先完成可分辨的离开；追踪器还要求返回点切向与种子切向足够同向，阈值由 `closure_tangent_cosine` 控制，合法范围是 `[-1, 1)`。这三个条件分别避免“刚从种子出发就闭合”、近掠种子和反向穿越的误判。追踪器通过 $\tfrac12\lVert\mathbf x-\mathbf x_0\rVert^2$ 沿轨迹的导数定位局部最近点，再检查距离与切向；空间容差不会替换用户设置的 `max_step`。该候选是非终止辅助事件，候选本身不会立即停止求解器；当前分段继续到末端或更早的物理终止事件，追踪器随后裁剪返回点并停止下一分段。`nfev` 汇总已运行分段的 RHS 求值，可能包含闭合点后的 RHS 探测，但不计事件函数或结果场强重算。[^scipy-solve-ivp]

“靠近其他已有曲线就停止”仍未实现，它需要跨种子的空间索引与独立终止原因，不能与 `closed_loop` 混用。

## 查看一条线的诊断信息

```python
import numpy as np

from vectorviz import (
    Domain,
    FieldLineTracer,
    PointChargeField,
    SphericalExclusion,
    TraceOptions,
)

field = PointChargeField(1.0e-9, position=(0.0, 0.0))
tracer = FieldLineTracer(
    field,
    domain=Domain((-2.0, -2.0), (2.0, 2.0)),
    exclusions=[SphericalExclusion((0.0, 0.0), 0.05)],
    options=TraceOptions(max_step=0.02, output_step=0.02),
)

trace = tracer.trace(np.array([0.2, 0.1]), direction="both")

print(trace.terminations)
print(trace.forward.nfev if trace.forward else None)
print(trace.field_magnitude.min(), trace.field_magnitude.max())
```

点电荷场线是径向直线。除了查看终止原因，还可检查所有点的极角是否保持不变。若把容差收紧十倍，极角漂移和边界事件位置应收敛。

## 网格场：先插值向量，再归一化

规则网格通常存成

```text
(nx, ny, nz, 3)
```

最后一个轴是 $F_x,F_y,F_z$。SciPy 的 `RegularGridInterpolator` 可以一次插值整个向量。[^scipy-grid]

正确顺序是

```text
插值 Fx, Fy, Fz -> 得到 F(x) -> 计算 |F| -> 归一化
```

不要先把每个网格点归一化再插值。归一化是非线性操作，这样会丢掉幅值信息并改变零点附近的方向。也不要把缺失数据填成零；那会制造假零点。

域外默认终止，不做无依据的外推。材料 mask、无效单元和网格外部各自保留原因。

三线性插值不会自动保持 $\nabla\cdot\mathbf B=0$。磁场导入后要计算离散散度；精度要求高时，可改为插值矢势或使用保持散度结构的离散方法。

## 把误差拆开看

场线图的误差可分为五层：

1. **模型误差**：把有限线圈当成点偶极，或忽略材料边界；
2. **场离散误差**：FEM 网格或数值求积本身不够精；
3. **插值误差**：从离散节点重建 $\mathbf F(\mathbf x)$ 时产生；
4. **ODE 误差**：积分与事件定位产生；
5. **显示误差**：重采样、投影和像素化产生。

收紧 `rtol` 只会减小第 4 层误差。若场数据过粗，求解器只是更精确地沿错误的插值场积分。已有研究也会分别评估场重建误差和积分误差。[^steffen]

## 三个实用的误差量

### 1. 切向残差

在曲线段中点求场，令段方向为 $\widehat{\mathbf t}$：

$$
\epsilon_{\mathrm{tan}}
=\left\lVert\widehat{\mathbf t}\times\widehat{\mathbf F}\right\rVert.
$$

它是两者夹角正弦。理想值为零；减小输出间距后应下降。

### 2. 解析不变量漂移

- 旋转场 $\mathbf F=(-y,x)$：半径 $x^2+y^2$ 应保持不变；
- 鞍点场 $\mathbf F=(x,-y)$：乘积 $xy$ 应保持不变；
- 轴对称圆线圈：$\psi$ 应沿场线保持不变。

### 3. 曲线收敛

先分别收紧 ODE 容差和 `max_step`，把结果重采样到同一弧长网格，再比较对应点距离或 Hausdorff 距离。随后固定足够严格的 ODE 参数并加密场网格，以单独检查插值收敛。

二维光滑、非零、单值向量场的精确积分曲线不会在普通点相交。若同一平面上的数值曲线交叉，先检查步长和插值；三维投影的情况见[第 4 章](04-slices-and-validation.md)。

!!! example "对照阅读：离散网格上的场线"
    Pjer 的[《数值计算矢量场流线（适用于流场线，追磁力线）Python》](https://zhuanlan.zhihu.com/p/451474459)把三线性插值和双向 RK4 串成了最小例子。它适合看数据流；工具能力、坐标映射和配套代码中的问题见[参考页的勘误](../references.md#zhihu-grid-tracing)。

## 本章检查表

- [ ] 正、反分支分别积分，并各自保存终止原因。
- [ ] `rtol`、`atol` 和 `max_step` 经过收敛实验，不照抄默认值。
- [ ] 用稠密输出控制显示点，不把固定步长当作等距渲染方法。
- [ ] 网格数据先插值物理分量，再归一化。
- [ ] 分开报告模型、网格、插值、ODE 和显示误差。
- [ ] 至少检查切向残差与一个解析不变量。

下一章处理二维切片和播种。这两件事即使数值积分完全正确，也可能让图的物理含义出错。

## 本章引用

[^dormand-prince]: J. R. Dormand, P. J. Prince, [“A family of embedded Runge–Kutta formulae”](https://doi.org/10.1016/0771-050X(80)90013-3), *Journal of Computational and Applied Mathematics* 6 (1980), 19–26。
[^scipy-solve-ivp]: SciPy, [`scipy.integrate.solve_ivp` 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)，方法、容差、事件和稠密输出。
[^scipy-grid]: SciPy, [Regular grid interpolation 教程](https://docs.scipy.org/doc/scipy/tutorial/interpolate/ND_regular_grid.html)。
[^steffen]: M. Steffen et al., [“Investigation of Smoothness-Increasing Accuracy-Conserving Filters for Improving Streamline Integration through Discontinuous Fields”](https://doi.org/10.1109/TVCG.2008.9), *IEEE Transactions on Visualization and Computer Graphics* 14 (2008), 680–692。
