# 4. 切片、播种与验证

把三维场显示在二维屏幕上，不只是少画一个坐标。先要回答：屏幕上的线究竟是哪一种曲线？

## 二维切片何时包含真实场线 { #true-vs-projected }

设切片平面的单位法向为 $\mathbf n$。若平面内每一点都满足

$$
\mathbf F(\mathbf x)\cdot\mathbf n=0,
$$

场在这里没有法向分量。若场满足局部唯一性，从平面内出发的场线就不会离开。这样的平面称为不变平面；在其中做二维积分，得到的就是真实三维场线。

“看起来对称”还不够，直接检查法向分量。常见例子包括：

- 点电荷都位于 $z=0$ 时，Coulomb 场在该平面内；
- 磁偶极矩和偶极子位置都位于 $z=0$ 时，偶极场在该平面内；
- 轴对称圆线圈的任意子午面是不变平面。

## 任意切片上画出来的是什么

把三维场丢掉法向分量会得到

$$
\mathbf F_{\parallel}
=\mathbf F-(\mathbf F\cdot\mathbf n)\mathbf n.
$$

对 $\mathbf F_\parallel$ 积分得到的是**投影流线**。若法向分量不为零，它既不是真实三维场线，也不一定等于某条三维场线的二维投影。

两者偶尔会重合。例如匀强场在法向上不变化，先做三维积分再投影，与先投影向量再积分会得到同一条直线。一般场在离开切片后还会随法向坐标变化，这时两条曲线便不同。

还要单独处理 $\mathbf F_\parallel=0$ 而 $\mathbf F\ne0$ 的点。完整三维场在这里有方向，但方向完全垂直于切片，投影 ODE 没有方向。投影模式应按 $\lVert\mathbf F_\parallel\rVert$ 触发 mask 或终止，不能只检查完整场强。

界面和导出文件应区分四种模式：

| 模式 | 图上的曲线 |
|---|---|
| 不变平面 | 留在平面内的真实三维场线 |
| 投影流线 | $\mathbf F_\parallel$ 的二维积分曲线 |
| 平面播种的三维线 | 从切片出发，随后可离开切片的真实三维场线 |
| 三维线切片 | 已积分三维曲线与切片的交点或短线段 |

后两种不能只靠二维 ODE 得到。三维曲线投影后还可能相交；这不代表原空间中的场线相交。

当前三个浏览器预设没有使用任意平面投影：电偶极子与磁偶极子采用不变平面，匀强场则直接定义在二维平面中。`metadata.projection_note` 会明确记录切片方式。

## 背景颜色也要说明投影

同一切片至少能画三种不同标量：

$$
\lVert\mathbf F\rVert,
\qquad
\lVert\mathbf F_\parallel\rVert,
\qquad
F_n=\mathbf F\cdot\mathbf n.
$$

- $\lVert\mathbf F\rVert$ 是完整三维场强；
- $\lVert\mathbf F_\parallel\rVert$ 只看平面内分量；
- $F_n$ 能直接看出场穿出或穿入切片的位置。

前两者在不变平面上相等，在任意切片上可能差很多。色标必须写量名和单位，例如 `|B| [T]`，不能只放一条彩虹色带。

正值跨度很大时可使用对数色标；带符号的 $F_n$ 应用以零为中点的发散色图。奇点 mask 不参加自动色标统计，否则一个源附近的极大值会压平其余区域。

## 播种决定你看见哪些线

ODE 需要种子点 $\mathbf x_0$。不同播种方法回答不同问题。

### 覆盖播种

在边界、圆周或网格上按几何间距放种子。它让画面均匀，适合交互探索。Jobard–Lefer 算法还会让新曲线与已有曲线保持近似间距，从而减少空洞和拥挤。[^jobard-lefer]

这种“等间距”没有自动的物理权重。当前前端使用的就是覆盖播种。

### 等通量播种

三维场在种子面 $S$ 上的微元通量为

$$
d\Phi=\left|\mathbf F\cdot\mathbf n_S\right|dA.
$$

沿面累计通量，每隔相同 $\Delta\Phi$ 放一个种子。这样每条线近似代表相同通量，线数才可用于比较场强。费曼讲义中的线密度约定正是这种思想的几何表达。[^feynman-density]

二维模型在种子曲线 $C$ 上使用线元

$$
d\phi=\left|\mathbf F\cdot\mathbf n_C\right|d\ell.
$$

它表示二维通量，或挤出模型中单位厚度的通量。若二维图只是三维 Coulomb 场的平面限制，这个线积分不能自动当成真实三维通量。

“每条线代表相同通量”还要求相邻通量管之间没有源，即沿途区域满足 $\nabla\cdot\mathbf F=0$。穿过电荷或其他源后，线的通量权重会改变，必须重新说明统计面。

零净通量不等于没有局部通量。若一个面上同时有穿入和穿出，分别按正、负方向累计，不能先把它们相消。

### 特征播种

从零点附近的特征方向、源表面、材料界面或用户关心的位置播种。它用于找分界线和连接关系，不追求均匀覆盖。

不要只用含义不明的 `density` 表示播种设置。界面和导出数据还应保存 `seed_mode`，图注则注明“覆盖”“等通量”或“特征播种”。

## 箭头、场线和 LIC 各看什么

| 图层 | 最擅长表达 | 局限 |
|---|---|---|
| 稀疏箭头 | 局部方向与可选幅值 | 密集时遮挡，难看全局连接 |
| 离散场线 | 源到汇、闭环和分界结构 | 强烈依赖播种 |
| LIC 纹理 | 密集方向纹理 | 静态对称核通常分不出 $\mathbf F$ 与 $-\mathbf F$ |
| 标量背景 | 场强、势或某个分量 | 本身不表达方向 |

线积分卷积（LIC）沿局部流线卷积噪声纹理，适合显示复杂方向结构。原始方法由 Cabral 和 Leedom 提出。[^lic] 它是显示层，不是积分正确性的证明；若要表示正反方向，还需叠加箭头或动画。当前前端尚未实现 LIC。

场线上的箭头按弧长放置，不按求解器节点编号放置。自适应积分节点在弯曲区更密，按节点放箭头会把数值步长误画成物理密度。

## 用已知解验证结果

先从答案已知的模型开始，再进入复杂源。

| 模型 | 正确几何或不变量 | 主要检查量 |
|---|---|---|
| $\mathbf F=(1,0)$ | 平行直线 | 到解析直线的最大距离 |
| $\mathbf F=(-y,x)$ | 同心圆 | 半径漂移、闭环误差 |
| $\mathbf F=(x,-y)$ | 双曲线 | $xy$ 漂移 |
| 单点电荷 | 径向直线，$r^{-2}$ | 极角漂移、$r^2|E|$ |
| 理想磁偶极 | 轴/赤道值，$r^{-3}$ | 点值、对称性、远场标度 |
| 圆线圈 | 轴线闭式，$\psi$ 守恒 | 点值、求积差、$\psi$ 漂移 |
| 解析场采样到网格 | 向直接解析结果收敛 | 曲线距离、散度或旋度残差 |

`vtkStreamTracer` 这样的成熟工具同样保存积分方向和终止原因，并区分自适应积分参数。[^vtk-stream] 它可以用来交叉验证网格后端，但不能代替解析基准。

## 一套可重复的验证顺序

### 第一步：验证场值

先选几个有解析解的点，逐分量比较。再检查单位、符号以及旋转或镜面对称性。

### 第二步：验证几何

用匀强场、旋转场和径向场检查积分器。记录切向残差、解析不变量和终止点误差。

### 第三步：做 ODE 收敛

保持场模型不变，依次收紧 `rtol`、`atol` 和 `max_step`。把结果重采样到同一弧长坐标再比较。

### 第四步：做网格收敛

固定足够严格的 ODE 参数，逐步加密场网格。若曲线不向解析场结果靠近，问题在离散或插值，不在积分器。

### 第五步：检查画面说明

确认图中能直接读到：

- 物理量和单位；
- 完整场强还是平面内场强；
- 真实场线还是投影流线；
- 播种方式；
- mask 的含义；
- 终止原因统计。

## 计算切片不变性残差

对已知平面取一批点，计算法向分量相对场强：

```python
import numpy as np

def plane_invariance_residual(vectors, normal):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    magnitude = np.linalg.norm(vectors, axis=-1)
    normal_part = np.abs(vectors @ normal)
    return normal_part / np.maximum(magnitude, np.finfo(float).tiny)
```

残差小只说明这些采样点近似满足条件。解析模型还要从公式或对称性证明平面不变；数值网格则应随加密重复检查，以免漏掉局部法向分量。

!!! example "对照阅读：三维追踪与二维成像"
    [《一个粗鄙的 3D 场线的可视化框架》](https://zhuanlan.zhihu.com/p/649494323)展示了三维 RK4、相机和亮度编码怎样接在一起。它的正交投影、固定步长和停止条件各有边界，详见[参考页的勘误](../references.md#zhihu-3d-field-lines)。

## 本章检查表

- [ ] 每张二维图明确属于四种切片模式中的哪一种。
- [ ] 色标区分 $|\mathbf F|$、$|\mathbf F_\parallel|$ 和 $F_n$。
- [ ] `density` 不表示场强；等通量播种另有名称和记录。
- [ ] mask 不进入色标统计。
- [ ] 箭头按弧长放置，LIC 另加方向提示。
- [ ] 场值、几何、ODE、网格和显示分层验证。

下一章转向黑洞光线，并说明它为什么不能用场线方程处理。

## 本章引用

[^jobard-lefer]: B. Jobard, W. Lefer, [“Creating Evenly-Spaced Streamlines of Arbitrary Density”](https://doi.org/10.1007/978-3-7091-6876-9_5), 1997。
[^feynman-density]: R. P. Feynman, R. B. Leighton, M. Sands, [*The Feynman Lectures on Physics*, Vol. II, Ch. 4, §4-8](https://www.feynmanlectures.caltech.edu/II_04.html)。
[^lic]: B. Cabral, L. C. Leedom, [“Imaging Vector Fields Using Line Integral Convolution”](https://doi.org/10.1145/166117.166151), SIGGRAPH 1993；[开放存档](https://digital.library.unt.edu/ark:/67531/metadc1399414/)。
[^vtk-stream]: VTK, [`vtkStreamTracer` 官方文档](https://vtk.org/doc/nightly/html/classvtkStreamTracer.html)。
