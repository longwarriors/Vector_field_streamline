# 6. 非定常流：流线、迹线与脉线

同一个含时速度场可以生成三种曲线。要读懂它们，先明确固定的是观察时刻、粒子身份，还是释放位置。流线在某一时刻处处与速度相切；迹线记录同一个质点的运动；脉线则把经过同一固定位置的不同质点，在同一观察时刻的位置收集起来。定常流中，经过同一释放点的三者落在同一条几何曲线上，有限时间窗可能只覆盖其中一段。[^stern-flow]

## 三种曲线的方程

设位置为 $\mathbf x$（m），速度为 $\mathbf u(\mathbf x,t)$（m/s），观察时刻为 $T$（s）。记 $\Phi(t;\tau,\mathbf a)$ 为在时刻 $\tau$ 从位置 $\mathbf a$ 出发的流体质点，到时刻 $t$ 的位置。

| 曲线 | 固定的量 | 随曲线改变的量 | 定义 |
|---|---|---|---|
| 流线 streamline | 观察时刻 $T$ | 空间弧长 $s$ | $d\mathbf x/ds=\mathbf u(\mathbf x,T)/\lVert\mathbf u(\mathbf x,T)\rVert$ |
| 迹线 pathline | 同一个质点的初始位置 $\mathbf a$ 与初始时刻 $t_0$ | 物理时间 $t$ | $d\mathbf x/dt=\mathbf u(\mathbf x,t)$，$\mathbf x(t_0)=\mathbf a$ |
| 脉线 streakline | 释放位置 $\mathbf a$ 与观察时刻 $T$ | 释放时刻 $\tau$ | $\{\Phi(T;\tau,\mathbf a):\tau\in[\tau_a,\tau_b]\}$ |

流线积分时，右端的 $T$ 始终不变。迹线必须用真实速度推进时间；除以速度模长会改变粒子的位置和到达时刻。脉线需要多次独立的粒子积分，它既不是某个粒子的历史，也不是不同时间流线的拼接。[^stern-flow]

瞬时速度为零时，流线的单位切向没有定义；粒子方程却仍有意义。粒子可以暂时静止，随后在速度改变时继续运动。因此 `PathlineTracer` 没有场线追踪器的 `null_field` 终止条件。脉线的连线次序是释放时间次序，其切向一般不等于当前速度方向，不能在上面套用流线的方向箭头。

## 可独立验算的模型

v0.3.10 使用空间均匀、横向速度随时间振荡的二维运动学模型：

$$
\mathbf u(x,y,t)=\bigl(U,\ V+A\sin(\omega t)\bigr).
$$

$U,V,A$ 的单位为 m/s，$\omega$ 为 rad/s，$\omega t$ 是无量纲相位。`flow_oscillating` 取 $U=1$、$V=0$、$A=0.6$、$\omega=1$；`flow_steady` 取 $U=1$、$V=0.4$、$A=0$。空间导数全为零，因此该场无散，但这里仅给定速度，不求解压力、黏性应力或 Navier–Stokes 边值问题。

下列公式直接对速度积分得到，可作为独立验收依据。若 $U\ne0$，通过 $(x_a,y_a)$ 的冻结流线为

$$
y-y_a=\frac{V+A\sin(\omega T)}{U}(x-x_a).
$$

若 $\omega>0$，同一质点的迹线为

$$
\begin{aligned}
x(t)&=x_a+U(t-t_0),\\
y(t)&=y_a+V(t-t_0)+\frac A\omega\bigl[\cos(\omega t_0)-\cos(\omega t)\bigr].
\end{aligned}
$$

固定位置释放的脉线则为

$$
\begin{aligned}
x(T;\tau)&=x_a+U(T-\tau),\\
y(T;\tau)&=y_a+V(T-\tau)+\frac A\omega\bigl[\cos(\omega\tau)-\cos(\omega T)\bigr].
\end{aligned}
$$

这两组式子分别改变 $t$ 和 $\tau$，并不是同一种采样。$A=0$ 时，两者都落在 $y-y_a=(V/U)(x-x_a)$ 上；$\omega=0$ 时振荡项恒为零，应直接采用匀速解，不能除以零。定常对照验证的是几何一致，不要求各类曲线使用相同点数或相同排列方向。

非定常也不保证任意时刻都能区分三者。例如本模型在 $t_0=0,T=\pi/\omega$，且释放窗为 $[0,T]$ 时，迹线和脉线恰好描出同一条曲线。这是公式的对称性，不是程序错误；检验差异应选一般时刻，并同时检查采样时间。

## 在浏览器中比较

选择非定常流预设，图上的固定释放位置是 $(-2,0)$ m。先保持初始时刻为 0，把观察时刻设为 4 s，释放窗设为 $[0,4]$ s。经过该位置的流线向右下倾斜；迹线保存粒子先向上再转向的历史；脉线中的点都属于 4 s 这一张快照。比如在 $x=0$ m 处，三者的解析高度分别约为 $-0.9082$ m、$0.8497$ m 与 $0.1425$ m，可以明确区分。

再切到定常对照，三类结果应落在同一直线上。增加 `density` 只增加每个站点的脉线释放时间目标采样数，不能把这些点的疏密解释为质量密度。点击图面可增加释放站：每个站点各有一条冻结流线、一条迹线和一组脉线粒子，所有站点共享时间设置。热图、向量探针和斜率短线只描述观察时刻的速度；单张网格缺少时间历史，不能用来重建迹线或脉线。

时间窗口限制为 $0\le t_0\le T\le6$ s、$0\le\tau_a\le\tau_b\le T$。$t_0=T$ 的迹线只有初始点；$\tau_a=\tau_b$ 的脉线只有一个粒子；极窄非零窗口会合并无法用浮点数区分的重复释放时刻，同时保留两端；$\tau_b<T$ 表示已经停止释放，脉线不必连接释放站。详细请求与响应见 [API 时间契约](../api.md#flow-time-contract)。

## 积分、采样与终止 { #time-integration }

服务端直接使用 `scipy.integrate.DOP853` 步进求解物理时间初值问题，并通过稠密输出得到等时间间隔的迹线样本；误差容差控制位置积分，而释放时间样本数控制脉线的离散程度。DOP853 是八阶显式方法，其稠密输出为七阶插值；两种误差来源要分别检查。[^scipy-time]

粒子到达观察时刻时记为 `time_end`。矩形域检查在每个已接受步的 DOP853 稠密多项式上定位首次向外穿越，连一步内先出后入也会截在首次出口；这检查的是数值插值曲线，仍受积分容差和浮点根定位精度限制。只接触边界后返回域内不算出域，恰在最终时刻到达边界仍为 `time_end`，边界种子可向内移动或保持静止。

提前离开计算域、遇到非有限速度或求解失败时，迹线保留实际终点和时间；非有限值与失败仅保留最后通过域检查的有限状态，不声称定位到了奇点。脉线响应在对应释放下标处给出 `null`，保留实际终止时间和原因，不能把已停止的边界点画成观察时刻仍存活的粒子。绘制脉线时也不能跨过 `null` 连接两段。矩形计算域只限定可见与积分范围，不代表物理壁面，不会反射或黏住粒子。

核心模型接受有限的物理时间，0–6 s 是 Web 演示和工作量的限制。这里没有外部数值网格、任意障碍边界、湍流模型、粒子惯性或带电粒子的洛伦兹动力学；振荡电偶极子的迟滞电磁场仍属于后续阶段。

## 本章引用

[^stern-flow]: Fred Stern, [*ME:5160, Chapter 2: Pressure Distribution in a Fluid*](https://stern.lab.uiowa.edu/sites/stern.lab.uiowa.edu/files/2025-01/Chapter%202---58160-2024_final.pdf), University of Iowa, Fall 2024，pp. 40–42：流线、迹线、脉线的定义与定常关系。
[^scipy-time]: SciPy, [`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) 与 [`DOP853`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.DOP853.html)，初值问题、局部误差容差与稠密输出。
