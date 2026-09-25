# 2. 从物理源到向量场

场线积分器只需要知道：给定位置 $\mathbf x$，$\mathbf F(\mathbf x)$ 是多少。本章先构造这个函数，下一章再沿它积分。

本章只讨论真空中的静电场和稳恒磁场。它们满足

$$
\nabla\cdot\mathbf E=\frac{\rho}{\varepsilon_0},
\qquad
\nabla\times\mathbf E=0,
$$

$$
\nabla\cdot\mathbf B=0,
\qquad
\nabla\times\mathbf B=\mu_0\mathbf J.
$$

源 $\rho$、电流 $\mathbf J$ 和边界条件共同决定场；场线只是解的可视化。这里不讨论时变问题，因为那时还要保留法拉第项和位移电流项。Maxwell 方程的完整形式见费曼讲义。[^feynman-maxwell]

## 建模前先写四行

每个模型文件或笔记本开头都写清：

1. **物理量**：画 $\mathbf E$、$\mathbf B$ 还是 $\mathbf H$？
2. **坐标系**：笛卡尔、柱坐标还是球坐标？分量在哪个基底下？
3. **单位**：位置、源强和输出各用什么单位？
4. **定义域**：哪些点是奇点，哪些区域属于材料或计算域外？

真空中的 $\mathbf B$ 与 $\mathbf H$ 满足 $\mathbf B=\mu_0\mathbf H$。进入磁性材料后，两者不能混用。

代码使用 `scipy.constants` 的 $\varepsilon_0$ 和 $\mu_0$。2019 年后的 SI 中，$\mu_0$ 已不是精确等于 $4\pi\times10^{-7}\ \mathrm{N/A^2}$ 的定义常数。[^bipm-si]

## 点电荷：第一个可验证模型

位于 $\mathbf x_q$、电量为 $q$ 的点电荷在源点外产生

$$
\mathbf E(\mathbf x)
=\frac{q}{4\pi\varepsilon_0}
\frac{\mathbf x-\mathbf x_q}
{\lVert\mathbf x-\mathbf x_q\rVert^3},
\qquad \mathbf x\ne\mathbf x_q.
$$

多个点电荷直接做向量叠加。[^openstax-electric-field]

```python
import numpy as np

from vectorviz import PointChargeField

field = PointChargeField(
    charge=np.array([1.0e-9, -1.0e-9]),
    position=np.array([[-0.5, 0.0], [0.5, 0.0]]),
)

points = np.array([[0.0, 1.0], [0.0, 2.0]])
electric_field = field.evaluate(points)
print(electric_field)  # V/m
```

这里输入二维坐标，是把三维库仑场限制在包含电荷的对称平面上。它不是二维静电学的基本解；二维泊松方程使用不同的格林函数，随距离的衰减也不同。

浏览器场景 API 对这组量采用明确的 SI 契约：`domain` 和源位置使用笛卡尔坐标与 m，电荷 `strength` 使用 nC，响应再把源强单位逐源返回。`positive` 的数值必须严格为正，`negative` 必须严格为负，0 对两类电荷都非法；服务端不会用绝对值替调用者修正符号。Python 库的 `PointChargeField` 则直接接收 C，两层之间由场景编排显式换算 $1\,\mathrm{nC}=10^{-9}\,\mathrm C$。

### 三个立即可做的检查

- 单个正电荷的方向沿径向向外，负电荷沿径向向内；
- 单个点电荷的 $r^2\lVert\mathbf E\rVert$ 与距离无关；
- 若两个等量异号电荷沿 $x$ 轴放置，那么在 $y$ 轴上，$y$ 分量抵消，$x$ 分量相加并从正电荷指向负电荷。

源点没有有限场值。`PointChargeField` 在那里返回 `NaN`；画图和积分时用一个有物理尺度的排除区域遮住它。mask 表示“模型在这里不定义”，不是“这里的场为零”。这个奇点集合由源的几何位置决定，与源强无关：零电荷或零磁矩的点仍返回 `NaN`，圆环在零电流时同样如此。浏览器场景层因此在构建场之前就剔除零强度源，而不是让核心把源点伪装成普通采样点。

## 两点电荷不等于理想电偶极

当前浏览器的“电偶极子”是两个有限间距的异号点电荷，计算时精确叠加两项 Coulomb 场。

理想电偶极是另一个模型。令 $\mathbf p=q\mathbf d$，且观测距离远大于电荷间距，远场为

$$
\mathbf E_{\mathrm{dip}}(\mathbf r)
=\frac{1}{4\pi\varepsilon_0r^3}
\left[3(\mathbf p\cdot\widehat{\mathbf r})\widehat{\mathbf r}-\mathbf p\right].
$$

它按 $r^{-3}$ 衰减，只适合远场或理想点偶极。[^feynman-electric-dipole] 近源处不能拿它替代两个真实点电荷。

## 磁偶极：有限电流源的远场

理想磁偶极矩为 $\mathbf m$，源点外的磁感应强度为

$$
\mathbf B(\mathbf r)
=\frac{\mu_0}{4\pi r^3}
\left[3(\mathbf m\cdot\widehat{\mathbf r})\widehat{\mathbf r}-\mathbf m\right].
$$

`MagneticDipoleField` 实现的就是这个三维公式。它可描述理想点偶极，也可近似有限线圈或磁体的远场。小电流环的偶极矩是 $\mathbf m=I\mathbf A$；只有当观测距离远大于线圈尺寸时，偶极近似才可靠。[^feynman-magnetic-dipole]

这里不把磁偶极建模为一对正负磁荷。由于 $\nabla\cdot\mathbf B=0$，磁力线没有普通端点；曲线碰到绘图区边界，只说明它被裁断了。

```python
from vectorviz import MagneticDipoleField

dipole = MagneticDipoleField(moment=(0.0, 1.0, 0.0))  # A m²
value = dipole.evaluate((0.0, 2.0, 0.0))
print(value)  # T
```

浏览器把偶极矩和偶极子位置都放在 $z=0$ 平面内。这个平面的法向场分量为零，所以前端画到的曲线是真实三维磁力线留在该平面内的部分。

浏览器 API 中磁偶极 `strength` 的单位是 A·m²，`angle_deg` 从 $+x$ 朝 $+y$ 逆时针量取。实际磁矩映射为

$$
\mathbf m=s(\cos\theta,\sin\theta,0),
$$

其中 $s$ 是有符号 `strength`。省略角度时取 $90^\circ$，因此兼容原来正强度沿 $+y$、负强度沿 $-y$ 的行为；负强度也等价于把显示的实际方向再翻转 $180^\circ$。强度 0 表示该源没有场贡献，但仍占一个播种源预算。这与电荷 `kind` 的非零符号约束不同。

## 线性 Halbach 教学预设

连续平面磁化若保持幅值不变而让方向随位置旋转，可以使两侧的场相长与相消，从而得到理想的单侧磁通结构；旋转方向决定哪一侧较弱。[^mallinson-one-sided] 实际分段阵列用有限个不同磁化方向近似连续旋转，Halbach 的永磁多极设计奠定了这类分段结构的工程基础。[^halbach-multipole]

VectorViz 的 `halbach_array` 不是有限尺寸永磁体求解器，而是八个理想点磁偶极子的教学组合：源沿 $x\in[-2.1,2.1]$ 等距排列，强度均为 $1\ \mathrm{A\,m^2}$，角度按

$$
0^\circ,90^\circ,180^\circ,270^\circ,
0^\circ,90^\circ,180^\circ,270^\circ
$$

旋转。有限长度、离散化和点偶极近似都会留下边缘泄漏，因此这里准确的说法是“$+y$ 一侧增强、$-y$ 一侧减弱”，不是弱侧严格为零。数值测试在避开源 mask 的两条对称采样带上比较 $\operatorname{mean}|\mathbf B|^2$；当前默认几何的强弱比约为 13.5，并以大于 8 作为留有边缘效应余量的回归门槛。场线仍采用覆盖播种，线条数量不能当作磁通。

该预设允许增删、移动和旋转偶极子，便于观察排列被破坏后的变化。编辑后的场仍是严格计算的点偶极叠加，但不一定还是 Halbach 排列；API 元数据会相应改称“可编辑面内磁偶极子阵列”。

## 圆电流线圈：从积分模型到解析模型

`CircularLoopField` 已实现理想细圆电流环的三维解析场。`current` 使用 A，`radius` 和 `center` 使用 m，默认返回 T；`normal` 是线圈平面的单位法向，并与正电流按右手定则绑定。测试侧保留直接 Biot–Savart 求积作为独立 oracle，但不会把逐点自适应求积暴露为交互运行时模型。

半径为 $a$、电流为 $I$ 的理想细线圈先由 Biot–Savart 定律定义：

$$
\mathbf B(\mathbf x)
=\frac{\mu_0 I}{4\pi}
\oint
\frac{d\boldsymbol\ell'\times(\mathbf x-\mathbf x')}
{\lVert\mathbf x-\mathbf x'\rVert^3}.
$$

直接数值求积是很好的基准。在线圈轴线上，积分可化成

$$
B_z(0,z)
=\frac{\mu_0 I a^2}{2(a^2+z^2)^{3/2}}.
$$

中心值是 $\mu_0I/(2a)$，远场按 $r^{-3}$ 衰减并趋近磁偶极。[^openstax-current-loop]

设线圈中心为 $\mathbf c$、单位法向为 $\mathbf n$。对 $\mathbf d=\mathbf x-\mathbf c$ 定义

$$
z=\mathbf d\cdot\mathbf n,
\qquad
\boldsymbol\rho=\mathbf d-z\mathbf n,
\qquad
\rho=\lVert\boldsymbol\rho\rVert.
$$

再令

$$
D_\pm=(a\pm\rho)^2+z^2,
\qquad
m=k^2=\frac{4a\rho}{D_+},
\qquad
p=1-m=\frac{D_-}{D_+}.
$$

轴外闭式解使用第一、第二类完全椭圆积分 $K(m),E(m)$：

$$
B_\rho=
\frac{\mu_0Iz}{2\pi\rho\sqrt{D_+}}
\left[
-K(m)+\frac{a^2+\rho^2+z^2}{D_-}E(m)
\right],
$$

$$
B_z=
\frac{\mu_0I}{2\pi\sqrt{D_+}}
\left[
K(m)+\frac{a^2-\rho^2-z^2}{D_-}E(m)
\right],
\qquad
\mathbf B=B_\rho\frac{\boldsymbol\rho}{\rho}+B_z\mathbf n.
$$

NASA 技术报告给出了多种坐标系下的等价公式，并把定义域限定在导体之外。[^nasa-loop] SciPy 的 `ellipk`/`ellipe` 接收参数 $m=k^2$，不是模数 $k$；代码同时保留 $m$ 与 $p$，避免在两个极限中用一次浮点减法丢掉较小者。[^scipy-elliptic]

实现分别处理四个数值区域：

1. 轴线用上面的极限公式，避开柱坐标通式中的可消奇异；
2. 极近轴使用由轴线场导数得到的正则 Taylor 展开，不把很小的 $B_\rho$ 粗暴置零；
3. $m\to1$ 时把直接计算的 $p=D_-/D_+$ 交给 `ellipkm1`，因此导线邻点保持有限，只有导线本身为奇点；
4. 远场 $m\to0$ 时，以收敛级数计算两个从 $m^2$ 起始的 $K/E$ 组合，避免离轴磁偶极小量被相消误差淹没。

固定种子的随机轴外点由测试内 Gauss–Legendre Biot–Savart 求积交叉验证；中心和整条轴线另与闭式解比较，远场同时检查磁偶极极限和 $O((a/r)^2)$ 的误差收敛。

理想导线本身仍是奇点，解析公式并不会让它变成普通采样点。三维追踪可用 `ToroidalExclusion` 在圆形中心线周围定义有限半径终止管；这个半径只属于 mask/事件几何，不是给解析场加入有限线径或 epsilon 软化。

## 用磁通函数画轴对称场线

对轴对称场，取矢势的环向分量 $A_\phi(\rho,z)$，定义

$$
\psi(\rho,z)=\rho A_\phi,
$$

它决定极向磁场。在 $\rho>0$ 处，

$$
B_\rho=-\frac{1}{\rho}\frac{\partial\psi}{\partial z},
\qquad
B_z=\frac{1}{\rho}\frac{\partial\psi}{\partial\rho},
\qquad
\mathbf B_{\mathrm p}\cdot\nabla\psi=0.
$$

因此，$\psi$ 的等值线是子午面内的极向磁力线。对圆线圈，$B_\phi=0$，它们也是完整三维场线在子午面内的截线。一般轴对称场还可由矢势的其他分量产生 $B_\phi$；此时 $\psi=\text{常数}$ 给出的是磁面，三维场线会沿磁面绕行。[^ogilvie-flux]

不要画 $A_\phi=\text{常数}$。正确的不变量是 $\rho A_\phi$。轴上要使用正则极限；一般情况下，半径 $\rho$ 的圆盘所穿磁通为

$$
2\pi\,[\psi(\rho,z)-\psi(0,z)].
$$

把轴上的 $\psi$ 取为零后，它才简化为 $2\pi\psi$。

浏览器的 `current_loop` 预设取 $a=1\ \mathrm m$、$I=1\ \mathrm A$、圆心在原点且法向为 $+y$，并显示真实的 $z=0$ 子午面。按右手定则，圆环在 $x=-a$ 处的切向电流沿 $+z$ 出屏，画作 `wire_out`（⊙）；在 $x=+a$ 处沿 $-z$ 入屏，画作 `wire_into`（⊗）。两个标记都显示同一个非负电流幅值 `1 A`，因为它们是同一根环形导体的两个截面交点，而不是两根独立导线。[^openstax-current-loop]

二维 mask 是三维 `ToroidalExclusion` 与这个子午面的精确截面，因而表现为 $x=\pm a$ 处的两个圆盘；这两个 mask 圆盘的半径只控制数值终止和图上的未着色斜线区，不是物理线径。种子半径由环内赤道段上等间隔的 $\psi$ 目标求根得到，再镜像到轴线两侧，属于等通量播种，见[切片与验证](04-slices-and-validation.md)中的“等通量播种”。靠近导线的闭轨启用 `closed_loop` 后只走一周；会离开视域的轨迹从赤道正向积分，再用严格的赤道镜面对称补齐另一半，拼接后的点序仍沿 $+\mathbf B$。

## 三类场模型

| 类型 | 怎样得到 $\mathbf F(\mathbf x)$ | 优点 | 主要误差 |
|---|---|---|---|
| 解析场 | 直接代公式 | 快，适合单元测试 | 模型近似、奇点和公式数值稳定性 |
| 积分场 | 对连续源或边界做数值积分 | 贴近源定义，可用于交叉验证 | 求积误差、近奇异积分 |
| 网格场 | 从 FEM、实验或仿真网格插值 | 能处理复杂几何与材料 | 离散、插值和域边界误差 |

三类模型都实现同一个批量接口：

```python
vectors = field.evaluate(points)  # points.shape == vectors.shape == (..., D)
```

场模型只负责求值，不负责播种、积分或配色。同一个场对象因此可用于数值测试、浏览器显示和后续三维渲染。

!!! example "对照阅读：圆线圈推导"
    知乎回答[《电磁学中，载流圆线圈在全空间的磁场分布是怎样的？》](https://www.zhihu.com/question/446655531/answer/2089751442)展示了三条解析路线。主方法可用，但球谐级数和几处展开式不能原样照抄；逐条说明见[参考页的勘误](../references.md#zhihu-current-loop)。

## 本章检查表

- [ ] 写清物理量、坐标系、单位和定义域。
- [ ] 区分精确两点电荷与远场电偶极。
- [ ] 把磁偶极称为理想模型或有限源远场，不误作线圈全空间精确解。
- [ ] 圆线圈同时用轴线公式和直接求积验证。
- [ ] 奇点进入 mask，不填零，也不加入任意大的 epsilon。
- [ ] 轴对称磁场画 $\psi=\rho A_\phi$，并检查是否存在 $B_\phi$。

下一章把已经能求值的 $\mathbf F(\mathbf x)$ 交给自适应 ODE 求解器。

## 本章引用

[^feynman-maxwell]: R. P. Feynman, R. B. Leighton, M. Sands, [*The Feynman Lectures on Physics*, Vol. II, Ch. 18](https://www.feynmanlectures.caltech.edu/II_18.html)，Maxwell 方程的完整形式。
[^bipm-si]: BIPM, [*The International System of Units (SI Brochure)*, 9th ed.](https://doi.org/10.59161/AUEZ1291)，以及[安培定义附录](https://www.bipm.org/documents/20126/41489676/SI-App2-ampere.pdf/0987a90e-051b-dd7f-827d-3f7b32751a61)。
[^openstax-electric-field]: OpenStax, [*University Physics*, Vol. 2, §5.4 Electric Field](https://openstax.org/books/university-physics-volume-2/pages/5-4-electric-field)，点电荷场和叠加原理。
[^feynman-electric-dipole]: R. P. Feynman, R. B. Leighton, M. Sands, [Vol. II, Ch. 6, §6-2](https://www.feynmanlectures.caltech.edu/II_06.html)，电偶极远场。
[^feynman-magnetic-dipole]: R. P. Feynman, R. B. Leighton, M. Sands, [Vol. II, Ch. 14](https://www.feynmanlectures.caltech.edu/II_14.html)，电流环、Biot–Savart 定律与磁偶极近似。
[^mallinson-one-sided]: J. C. Mallinson, [“One-Sided Fluxes—A Magnetic Curiosity?”](https://doi.org/10.1109/TMAG.1973.1067714), *IEEE Transactions on Magnetics* 9 (1973), 678–682。论文直接求解恒幅旋转磁化的平面结构，并说明旋转方向如何选择弱场侧。
[^halbach-multipole]: K. Halbach, [“Design of Permanent Multipole Magnets with Oriented Rare Earth Cobalt Material”](https://doi.org/10.1016/0029-554X(80)90094-4), *Nuclear Instruments and Methods* 169 (1980), 1–10。
[^openstax-current-loop]: OpenStax, [§12.1 Biot–Savart Law](https://openstax.org/books/university-physics-volume-2/pages/12-1-the-biot-savart-law) 与 [§12.4 Magnetic Field of a Current Loop](https://openstax.org/books/university-physics-volume-2/pages/12-4-magnetic-field-of-a-current-loop)。
[^nasa-loop]: J. C. Simpson et al., [“Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop”](https://ntrs.nasa.gov/citations/20010038494), NASA Technical Reports Server, 2001。
[^scipy-elliptic]: SciPy，[`ellipk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipk.html)、[`ellipe`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipe.html) 与 [`ellipkm1`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipkm1.html)；前两者使用参数 $m$，后者直接使用 $p=1-m$。
[^ogilvie-flux]: G. I. Ogilvie, [“Astrophysical fluid dynamics”](https://doi.org/10.1017/S0022377816000489), *Journal of Plasma Physics* 82 (2016), §9.2。
