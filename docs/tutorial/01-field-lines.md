# 1. 场线是什么

一条场线就是一条曲线：你沿着它走时，脚下的场方向始终贴着曲线切向。费曼物理学讲义用同样的几何定义引入电磁场线。[^feynman-lines]

## 从箭头变成曲线

设静态向量场为 $\mathbf F(\mathbf x)$。用参数 $\lambda$ 描述曲线，最直接的方程是

$$
\frac{d\mathbf x}{d\lambda}=\mathbf F(\mathbf x).
$$

右边给出下一步的方向和前进速度。若只关心曲线形状，可以把右边归一化：

$$
\frac{d\mathbf x}{ds}
=\pm\frac{\mathbf F(\mathbf x)}{\lVert\mathbf F(\mathbf x)\rVert}.
$$

此时 $s$ 是弧长，$+$ 和 $-$ 分别沿场方向与逆场方向。只要 $\lVert\mathbf F\rVert>0$，两种参数化走过的是同一条几何曲线；区别只是沿曲线走得快不快。

更一般地，若 $a(\mathbf x)>0$，那么

$$
\widetilde{\mathbf F}=a(\mathbf x)\mathbf F
$$

它只改变场强，不改变场线形状。

## 先用匀强场检查追踪器

匀强场的答案应该是平行直线。它适合检查坐标、积分方向和计算域。

```python
import numpy as np

from vectorviz import Domain, FieldLineTracer, TraceOptions, UniformField

field = UniformField((1.0, 0.25))
domain = Domain(lower=(-2.0, -1.0), upper=(2.0, 1.0))
tracer = FieldLineTracer(
    field,
    domain=domain,
    options=TraceOptions(max_step=0.05, output_step=0.05),
)

result = tracer.trace(seed=np.array([0.0, 0.0]), direction="both")

print(result.points[:3])
print(result.terminations)
```

### 你应该看到什么

- 所有点都落在直线 $y=0.25x$ 上；
- 正、反两个分支从同一种子出发，分别碰到计算域边界；
- `result.points` 合并两支时只保留一个种子点；
- 改变向量大小，例如从 $(1,0.25)$ 改成 $(10,2.5)$，曲线形状不变。

用下面的残差检查直线关系：

```python
residual = result.points[:, 1] - 0.25 * result.points[:, 0]
assert np.max(np.abs(residual)) < 1e-10
```

## 场线不自动表示场强

场线方程只确定经过每个种子点的曲线，并不决定屏幕上放多少个种子。任意均匀播种、鼠标点击播种或“看起来舒服”的间距，都只是在安排画面。

传统教材还会加一条绘图约定：让垂直于场的单位面积所穿过的线数与通量成比例。[^feynman-electrostatics] 软件只有在按通量分配种子时才满足这条约定。因此：

> 等间距让图更清楚；等通量才让线数带有物理含义。

VectorViz 前端的 `density` 只控制场线覆盖的疏密，不表示场强。场强请看带单位的色标或探针数值。

## 场线、粒子轨迹和流体迹线

这三种曲线经常长得相似，方程却不同。

| 曲线 | 初始条件 | 方程说明 |
|---|---|---|
| 静态场线 | 位置 $\mathbf x_0$ | 切向由当前的 $\mathbf F(\mathbf x)$ 决定 |
| 带电粒子轨迹 | 位置和速度 $(\mathbf x_0,\mathbf v_0)$ | 由洛伦兹力决定加速度 |
| 流体迹线 | 位置和起始时刻 | 跟随随时间变化的速度场 $\mathbf u(\mathbf x,t)$ |

带电粒子的运动满足

$$
m\frac{d\mathbf v}{dt}
=q\left(\mathbf E+\mathbf v\times\mathbf B\right),
\qquad
\frac{d\mathbf x}{dt}=\mathbf v.
$$

它需要初速度。即使只有静电场，粒子也不一定沿某条场线走；磁场中的速度叉乘项更会让速度偏转而不是“顺着磁力线前进”。

对随时间变化的流场，某一时刻画出的流线只是瞬时方向图。只有稳态流动中，流线、迹线和脉线才重合。

## 零场点为什么必须停

当 $\lVert\mathbf F\rVert=0$ 时，$\widehat{\mathbf F}$ 没有方向。不要用

```text
F / (norm(F) + epsilon)
```

代替弱场终止事件。它在非零点仍保持方向，却不再以弧长速度前进；越靠近零点，前进越慢。这样会把“到达零场附近”混进最大积分长度或求解器停止条件，零点也不再有独立的诊断记录。

只有精确零点的方向在数学上没有定义。实际计算通常在更早的正阈值处停止，因为过小的场可能低于模型或浮点数能可靠分辨的尺度。VectorViz 把这一分支记为 `null_field`。

这个阈值与场强同单位，是数值截止值，不是新的物理零点。若把整个场乘以常数，阈值也要按同样比例调整，否则场线几何虽不变，停止位置却会改变。终止事件的实现见第 3 章。

## 本章检查表

- [ ] 能写出场线的切向方程。
- [ ] 知道归一化改变参数速度，不改变非零区的曲线形状。
- [ ] 不从任意播种的线密度读取场强。
- [ ] 不把场线当作带质量粒子的运动轨迹。
- [ ] 在零场点终止，不用 epsilon 伪造方向。

下一章从点电荷、磁偶极子和电流线圈出发，构造方程右边的 $\mathbf F(\mathbf x)$。

## 本章引用

[^feynman-lines]: R. P. Feynman, R. B. Leighton, M. Sands, [*The Feynman Lectures on Physics*, Vol. II, Ch. 1](https://www.feynmanlectures.caltech.edu/II_01.html)，场的向量与场线表示。
[^feynman-electrostatics]: R. P. Feynman, R. B. Leighton, M. Sands, [*The Feynman Lectures on Physics*, Vol. II, Ch. 4, §4-8](https://www.feynmanlectures.caltech.edu/II_04.html)，电场线、通量与线密度约定。
