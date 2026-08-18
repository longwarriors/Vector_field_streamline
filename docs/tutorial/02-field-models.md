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

源点没有有限场值。`PointChargeField` 在那里返回 `NaN`；画图和积分时用一个有物理尺度的排除区域遮住它。mask 表示“模型在这里不定义”，不是“这里的场为零”。

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

浏览器 API 中磁偶极 `strength` 的单位是 A·m²，当前映射为 $y$ 方向偶极矩；它允许正、负和 0，分别表示方向、反向与零偶极矩。单源为 0 时退化为零场；多源场中某一个 0 只表示该源没有贡献。这与电荷 `kind` 的非零符号约束不同。

## 圆电流线圈：从积分模型到解析模型

!!! note "当前状态"
    圆线圈模型尚未进入代码。本节给出实现顺序和验证公式。

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

轴外闭式解要用第一、第二类完全椭圆积分。NASA 技术报告给出了多种坐标系下的公式，并把定义域限定在导体之外。[^nasa-loop] 实现时分别处理：

1. 轴线用上面的极限公式，避开柱坐标通式中的可消奇异；
2. 一般点用椭圆积分公式；
3. 随机抽样点与直接 Biot–Savart 求积对比。

理想导线本身仍是奇点，解析公式并不会让它变成普通采样点。

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
[^openstax-current-loop]: OpenStax, [§12.1 Biot–Savart Law](https://openstax.org/books/university-physics-volume-2/pages/12-1-the-biot-savart-law) 与 [§12.4 Magnetic Field of a Current Loop](https://openstax.org/books/university-physics-volume-2/pages/12-4-magnetic-field-of-a-current-loop)。
[^nasa-loop]: J. C. Simpson et al., [“Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop”](https://ntrs.nasa.gov/citations/20010038494), NASA Technical Reports Server, 2001。
[^ogilvie-flux]: G. I. Ogilvie, [“Astrophysical fluid dynamics”](https://doi.org/10.1017/S0022377816000489), *Journal of Plasma Physics* 82 (2016), §9.2。
