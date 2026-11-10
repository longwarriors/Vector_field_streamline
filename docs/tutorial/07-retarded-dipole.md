# 7. 迟滞电偶极场

v0.3.11 的 `dipole_radiation` 展示真空中理想局域电偶极子的永恒简谐稳态。源的偶极矩为

$$
\mathbf p(t)=p_0\cos(\omega t+\phi)\,\hat{\mathbf z},\qquad
p_0=10^{-9}\ \mathrm{C\,m},\quad f=100\ \mathrm{MHz},\quad\omega=2\pi f,\quad\phi=0.
$$

源固定在原点。坐标用 m、核心时间用 s；周期 $T=10^{-8}$ s，即 10 ns。页面的 0–10 ns 对应 API 的 0–$10^{-8}$ s，默认 1.25 ns；$t_{\rm s}=10^{-9}t_{\rm ns}$。这与非定常流预设的 0–6 s 窗口是两个不同契约。

## 从同一个迟滞解得到三个观察量

令 $\mathbf r$ 是从源中心指向观察点的向量，$r=|\mathbf r|$、$\mathbf n=\mathbf r/r$。所有源量都在迟滞时刻 $t_r=t-r/c$ 求值，点表示对源时间求导：

$$
\begin{aligned}
\mathbf E(\mathbf r,t)&=\frac1{4\pi\varepsilon_0}\left[
\frac{3\mathbf n(\mathbf n\cdot\mathbf p)-\mathbf p}{r^3}
+\frac{3\mathbf n(\mathbf n\cdot\dot{\mathbf p})-\dot{\mathbf p}}{cr^2}
+\frac{\mathbf n\times(\mathbf n\times\ddot{\mathbf p})}{c^2r}\right]_{t_r},\\
\mathbf B(\mathbf r,t)&=\frac{\mu_0}{4\pi}\left[
\frac{\dot{\mathbf p}\times\mathbf n}{r^2}
+\frac{\ddot{\mathbf p}\times\mathbf n}{cr}\right]_{t_r},\\
\mathbf S(\mathbf r,t)&=\frac1{\mu_0}\mathbf E(\mathbf r,t)\times\mathbf B(\mathbf r,t).
\end{aligned}
$$

这里 $\dot{\mathbf p}=-\omega\mathbf p_0\sin(\omega t_r+\phi)$，$\ddot{\mathbf p}=-\omega^2\mathbf p_0\cos(\omega t_r+\phi)$。电场的三项分别按 $r^{-3}$、$r^{-2}$、$r^{-1}$ 衰减，对应近场、感应项、辐射项；磁场含后两种标度。公式核对自 Caltech 的[《费曼物理学讲义》II §21–4，式 21.23、21.26–27](https://www.feynmanlectures.caltech.edu/II_21.html#Ch21-S4)。

常数取 $c=299792458$ m/s（[NIST，精确值](https://physics.nist.gov/cgi-bin/cuu/Value?c)），$\varepsilon_0$ 取锁定 SciPy 的 `epsilon_0`，$\mu_0=1/(\varepsilon_0c^2)$ 保持浮点计算内部一致。E、B、S 分别以 V/m、T、W/m² 返回。各向量来自同一次三维求值，不从渲染后的颜色或二维投影重建能流。

## 哪个平面上的线才是真的

| 观察量 | 显示平面与轴 | 曲线含义 |
|---|---|---|
| 瞬时电场 E | $y=0$，横轴 x、纵轴 z | 固定时刻沿 E 的真实场线 |
| 磁场 B | $z=0$，横轴 x、纵轴 y | 绕 z 轴的真实环向磁场线 |
| 瞬时能流 S | $y=0$，横轴 x、纵轴 z | 固定时刻沿 $\mathbf E\times\mathbf B/\mu_0$ 的方向线 |

E 与 S 在子午面内，B 则垂直于这个面。因此切到 B 时切片也改变。B 在赤道面内呈环向；它不是把子午面面外分量丢掉后的零场。切换观察量后，探针的分量含义、纵轴名称和单位随之变化。

每条线都冻结同一个观察时刻；积分参数是空间弧长。能流线不等于光子路径或质点迹线，也不能沿曲线积分“飞行时间”。近区储能与源交换，瞬时 S 可以向内；代码保留这种方向，未将其强制改为向外。热图显示模长，向内或向外要看向量、箭头和探针。

## 如何看到迟滞和近远区

波长 $\lambda=c/f=2.99792458$ m，$c/\omega\approx0.4771$ m。可视域为 $[-6,6]^2$ m，含多个迟滞相位；原点半径 0.12 m 的圆只用于遮罩和追踪排除，不是有限天线的尺寸，也不进入物理公式。

先在默认观察时刻查看 E，再改变观察时刻并比较不同半径的方向。近区由多项共同决定，不能把每个方向翻转都解释成向外运动的粒子。远区应接近横波，$\mathbf B\approx\mathbf n\times\mathbf E/c$；但轴上辐射项消失，不能在轴上用 $E/B$ 作比。改变时刻是在比较一系列冻结的迟滞场；浏览器的网格预览已禁用，避免把插值曲线读作传播过程。点击播种仍由 Python 对解析场积分。

默认覆盖播种兼顾不同半径，线密度不代表电通量、磁通量或功率。不同半径的 B 环可能有相反方向；B 或 S 的零点处，归一化方向不可定义，场线会按零场阈值停止。源排除、出域、闭环、长度预算和非有限场沿用可重现的场线终止契约。自动磁场环沿 B 单向追踪；E/S 与手动种子双向追踪，结果保留双端原因。具体阈值与步长见 [API 契约](../api.md#electromagnetic-contract)。

## 瞬时能流与周期平均

页面显示瞬时 S，不提供周期平均切换。对完整周期积分后，理想简谐偶极在源外有

$$
\langle\mathbf S\rangle=
\frac{p_0^2\omega^4\sin^2\theta}{32\pi^2\varepsilon_0c^3r^2}\,\mathbf n,
\qquad
\langle P\rangle=\frac{p_0^2\omega^4}{12\pi\varepsilon_0c^3}.
$$

$\theta$ 是与 z 轴的夹角。周期平均的向外结果不能代替近区每一时刻的能流。辐射能流的均方场关系和球面功率积分可参见[《费曼物理学讲义》I §32–1](https://www.feynmanlectures.caltech.edu/I_32.html#Ch32-S1)；上式在本项目中也由完整 E×B 的周期积分独立验收。

## 验证与边界

核心测试分别检查静态与低频极限、轴上与赤道对称性、迟滞相位、近远场标度、远区横向性及 E/B 关系、近区能流反向和周期平均角分布。还从迟滞标量势与矢势做独立数值微分核对场的符号，避免只用同一实现的包装互相比对。

核心允许任意有限实数时刻，包括负数；这与永恒简谐源一致。源点始终是显式 NaN 奇点，即使偶极矩为零；无效坐标与无法表示的结果也返回 NaN，Web 把不可用格点变成 `null`。公共模型的零频率极限是静态偶极，B 和 S 为零。Web 固定模型参数，只开放观察量、观察时刻、采样分辨率和播种密度。

模型没有突然开启的源、有限天线结构、材料、近场探头反馈、自洽源动力学或带电粒子运动。页面不宣称展示源从无到有时的完整因果瞬态。三维视图、外部网格和通用 PDE 求解仍在后续路线中。
