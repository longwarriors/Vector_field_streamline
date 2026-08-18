# 参考资料

正文中的脚注会直接跳到原论文、官方教材或软件文档。本页把它们按主题排好，便于继续读。

选择来源时遵守三条规则：物理发现和算法尽量找原论文；公式入门优先用公开教材；软件行为只引用官方文档。历史论文的记号可能与现代写法不同，不能只抄公式而忽略原文约定。

## 场线与电磁学

1. M. Faraday, [“Experimental Researches in Electricity—Twenty-Eighth Series. On Lines of Magnetic Force”](https://doi.org/10.1098/rstl.1852.0004), *Philosophical Transactions of the Royal Society of London* **142** (1852), 25–56。场线思想的历史原始文献。

2. J. C. Maxwell, [“A Dynamical Theory of the Electromagnetic Field”](https://doi.org/10.1098/rstl.1865.0008), *Philosophical Transactions of the Royal Society of London* **155** (1865), 459–512。用于电磁场理论史；今天常见的四条紧凑向量方程是后来的现代整理。

3. R. P. Feynman, R. B. Leighton, M. Sands, *The Feynman Lectures on Physics*, Vol. II：

    - [Ch. 1 Electromagnetism](https://www.feynmanlectures.caltech.edu/II_01.html)：向量场与场线；
    - [Ch. 2 Differential Calculus of Vector Fields](https://www.feynmanlectures.caltech.edu/II_02.html)：梯度、散度和旋度；
    - [Ch. 4 Electrostatics](https://www.feynmanlectures.caltech.edu/II_04.html)：电场线、通量与等势面；
    - [Ch. 6 The Electric Field in Various Circumstances](https://www.feynmanlectures.caltech.edu/II_06.html)：电偶极远场；
    - [Ch. 14 The Magnetic Field in Various Situations](https://www.feynmanlectures.caltech.edu/II_14.html)：Biot–Savart、电流环与磁偶极；
    - [Ch. 18 The Maxwell Equations](https://www.feynmanlectures.caltech.edu/II_18.html)：Maxwell 方程的完整形式。

4. OpenStax, *University Physics*, Vol. 2：

    - [§5.4 Electric Field](https://openstax.org/books/university-physics-volume-2/pages/5-4-electric-field)；
    - [§5.6 Electric Field Lines](https://openstax.org/books/university-physics-volume-2/pages/5-6-electric-field-lines)；
    - [§12.1 The Biot–Savart Law](https://openstax.org/books/university-physics-volume-2/pages/12-1-the-biot-savart-law)；
    - [§12.4 Magnetic Field of a Current Loop](https://openstax.org/books/university-physics-volume-2/pages/12-4-magnetic-field-of-a-current-loop)；
    - [§16.1 Maxwell’s Equations and Electromagnetic Waves](https://openstax.org/books/university-physics-volume-2/pages/16-1-maxwells-equations-and-electromagnetic-waves)。

5. J. C. Simpson, J. E. Lane, C. D. Immer, R. C. Youngquist, [“Simple Analytic Expressions for the Magnetic Field of a Circular Current Loop”](https://ntrs.nasa.gov/citations/20010038494), NASA Technical Reports Server, 2001。给出理想细线圈在导体外的笛卡尔、柱坐标和球坐标解析式。

6. G. I. Ogilvie, [“Astrophysical fluid dynamics”](https://doi.org/10.1017/S0022377816000489), *Journal of Plasma Physics* **82** (2016)。§9.2 讨论轴对称磁通函数 $\psi=\rho A_\phi$。

7. BIPM, [*The International System of Units (SI Brochure)*, 9th ed.](https://doi.org/10.59161/AUEZ1291)，以及[安培定义附录](https://www.bipm.org/documents/20126/41489676/SI-App2-ampere.pdf/0987a90e-051b-dd7f-827d-3f7b32751a61)。用于 SI 常数和 2019 年后 $\mu_0$ 的地位。

## 数值积分与向量场可视化

1. J. R. Dormand, P. J. Prince, [“A family of embedded Runge–Kutta formulae”](https://doi.org/10.1016/0771-050X(80)90013-3), *Journal of Computational and Applied Mathematics* **6** (1980), 19–26。

2. SciPy 官方文档：

    - [`scipy.integrate.solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)：自适应 ODE、事件和稠密输出；
    - [Regular grid interpolation](https://docs.scipy.org/doc/scipy/tutorial/interpolate/ND_regular_grid.html)：规则网格多分量插值；
    - [`scipy.special.ellipk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipk.html) 与 [`ellipe`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipe.html)：完全椭圆积分。SciPy 的参数是 $m=k^2$。

3. M. Steffen et al., [“Investigation of Smoothness-Increasing Accuracy-Conserving Filters for Improving Streamline Integration through Discontinuous Fields”](https://doi.org/10.1109/TVCG.2008.9), *IEEE Transactions on Visualization and Computer Graphics* **14** (2008), 680–692。讨论场光滑性、重建与积分误差。

4. H. Bhatia et al., [“Flow Visualization with Quantified Spatial and Temporal Errors Using Edge Maps”](https://doi.org/10.1109/TVCG.2011.265), *IEEE Transactions on Visualization and Computer Graphics* **18** (2012), 1383–1396。包含数值积分造成错误曲线交叉的例子。

5. B. Jobard, W. Lefer, [“Creating Evenly-Spaced Streamlines of Arbitrary Density”](https://doi.org/10.1007/978-3-7091-6876-9_5), in *Visualization in Scientific Computing ’97*。覆盖播种与等间距流线的经典算法。

6. B. Cabral, L. C. Leedom, [“Imaging Vector Fields Using Line Integral Convolution”](https://doi.org/10.1145/166117.166151), SIGGRAPH 1993, 263–270；[开放存档](https://digital.library.unt.edu/ark:/67531/metadc1399414/)。

7. VTK, [`vtkStreamTracer` 官方文档](https://vtk.org/doc/nightly/html/classvtkStreamTracer.html)。说明正反向追踪、归一化向量、积分器设置和终止原因。

8. P. J. Roache, [“Code Verification by the Method of Manufactured Solutions”](https://doi.org/10.1115/1.1436090), *Journal of Fluids Engineering* **124** (2002)。制造解与代码验证。

9. I. Babuška, J. T. Oden, [“Verification and validation in computational engineering and science”](https://doi.org/10.1016/j.cma.2004.03.002), *Computer Methods in Applied Mechanics and Engineering* **193** (2004)。区分方程是否算对与模型是否描述现实。

## 引力、测地线与黑洞成像

1. OpenStax, [*University Physics*, Vol. 1, §13.2 Gravitation Near Earth’s Surface](https://openstax.org/books/university-physics-volume-1/pages/13-2-gravitation-near-earths-surface)。牛顿引力场和场线的公开入门教材。

2. R. P. Kerr, [“Gravitational Field of a Spinning Mass as an Example of Algebraically Special Metrics”](https://doi.org/10.1103/PhysRevLett.11.237), *Physical Review Letters* **11** (1963), 237–238。

3. B. Carter, [“Global Structure of the Kerr Family of Gravitational Fields”](https://doi.org/10.1103/PhysRev.174.1559), *Physical Review* **174** (1968), 1559–1571。Kerr 测地线的 Hamilton–Jacobi 可分离性与第四守恒量。

4. B. Carter, [“Hamilton-Jacobi and Schrödinger Separable Solutions of Einstein’s Equations”](https://doi.org/10.1007/BF03399503), *Communications in Mathematical Physics* **10** (1968), 280–310。

5. J. L. Synge, [“The Escape of Photons from Gravitationally Intense Stars”](https://doi.org/10.1093/mnras/131.3.463), *Monthly Notices of the Royal Astronomical Society* **131** (1966), 463–466。Schwarzschild 逃逸锥和临界捕获。

6. J. M. Bardeen, [“Timelike and Null Geodesics in the Kerr Metric”](https://inspirehep.net/literature/1361769), in *Black Holes / Les Astres Occlus* (1973), 215–240。Kerr 测地线与观察者天空中的阴影轮廓。

7. V. Perlick, O. Y. Tsupko, [“Calculating black hole shadows: review of analytical studies”](https://doi.org/10.1016/j.physrep.2021.10.004), *Physics Reports* **947** (2022), 1–39。Schwarzschild、Kerr 阴影和临界光线的系统推导。

8. F. H. Vincent et al., [“GYOTO: a new general relativistic ray-tracing code”](https://doi.org/10.1088/0264-9381/28/22/225011), *Classical and Quantum Gravity* **28** (2011), 225011。反向测地线追踪、终止事件和协变辐射传输。

9. P. A. González et al., [“OSIRIS: a new code for ray tracing around compact objects”](https://doi.org/10.1140/epjc/s10052-022-10054-0), *European Physical Journal C* **82** (2022)。Hamilton 形式、约束监控和阴影验证。

10. J.-P. Luminet, [“Image of a spherical black hole with thin accretion disk”](https://ui.adsabs.harvard.edu/abs/1979A%26A....75..228L/abstract), *Astronomy & Astrophysics* **75** (1979), 228–235。薄盘黑洞合成图像的经典工作。

11. S. E. Gralla, D. E. Holz, R. M. Wald, [“Black hole shadows, photon rings, and lensing rings”](https://doi.org/10.1103/PhysRevD.100.024018), *Physical Review D* **100** (2019), 024018。区分阴影、临界曲线、光子环和透镜环。

12. EinsteinPy, [Nulllike Geodesic 官方 API](https://docs.einsteinpy.org/en/latest/api/geodesic/geodesic.html)。API 使用 $G=c=M=1$；`return_cartesian=True` 只转换位置，动量仍为球坐标分量。可用于原型和交叉验证，正式验证仍以守恒量与解析极限为准。

13. E. Belbruno, F. Pretorius, [“A Dynamical Systems Approach to Schwarzschild Null Geodesics”](https://doi.org/10.1088/0264-9381/28/19/195007), *Classical and Quantum Gravity* **28** (2011), 195007。说明 Schwarzschild 零测地线与等效中心力动力系统的对应，适合核对视频使用的“偏折加速度”。

14. R. W. Lindquist, [“Relativistic transport theory”](https://doi.org/10.1016/0003-4916(66)90207-7), *Annals of Physics* **37** (1966), 487–518。协变辐射传输的基础。

15. C. T. Cunningham, [“The effects of redshifts and focusing on the spectrum of an accretion disk around a Kerr black hole”](https://doi.org/10.1086/154033), *The Astrophysical Journal* **202** (1975), 788–802。Kerr 薄盘的频移、聚焦与 transfer function。

16. D. N. Page, K. S. Thorne, [“Disk-Accretion onto a Black Hole. Time-Averaged Structure of Accretion Disk”](https://doi.org/10.1086/152990), *The Astrophysical Journal* **191** (1974), 499–506。相对论薄盘的径向通量模型。

17. O. James et al., [“Gravitational lensing by spinning black holes in astrophysics, and in the movie Interstellar”](https://doi.org/10.1088/0264-9381/32/6/065001), *Classical and Quantum Gravity* **32** (2015), 065001。DNGR 的 Kerr 光线追踪，以及电影画面为可读性所作的处理。

18. Event Horizon Telescope Collaboration, 2019 M87\* 系列：

    - [Paper I：阴影与整体结论](https://doi.org/10.3847/2041-8213/ab0ec7)；
    - [Paper IV：VLBI 数据到图像的重建](https://doi.org/10.3847/2041-8213/ab0e85)；
    - [Paper V：GRMHD 与不对称亮环](https://doi.org/10.3847/2041-8213/ab0f43)；
    - [Paper VI：阴影尺度与质量](https://doi.org/10.3847/2041-8213/ab1141)。

19. L. Medeiros et al., [“The Image of the M87 Black Hole Reconstructed with PRIMO”](https://doi.org/10.3847/2041-8213/acc32d), *The Astrophysical Journal Letters* **947** (2023), L7。视频 26 分多钟出现的 EHT/PRIMO 三联图来源。

20. Event Horizon Telescope Collaboration, [“The persistent shadow of the supermassive black hole of M 87. I. Observations, calibration, imaging, and analysis”](https://doi.org/10.1051/0004-6361/202347932), *Astronomy & Astrophysics* **681** (2024), A79。视频结尾 2017 与 2018 图像对比的来源。

## 黑洞视频案例与出处 { #black-hole-video }

这一组不是物理定律的依据，而是第 5 章所分析视频的来源记录。主体创作、外部素材和算法先例分开列，避免把“内容相似”误写成“已经证实搬运”。

1. 彭导分享，[Bilibili 完整版](https://www.bilibili.com/video/BV1RpZHBFE1C/)与[同作者 YouTube 版](https://www.youtube.com/watch?v=INrkLxPWxFk)，2026-02-15。两页与本地 MP4 的标题、时长、章节和主持人一致。YouTube 是同作者跨平台发布，不是另一位英文作者的原片。

2. A. Roussel / ScienceClic English, [*Let’s reproduce the calculations from Interstellar*](https://www.youtube.com/watch?v=ABFGKdKKKyg), 2024。彭导视频约 27:00–27:38 的高质量吸积盘片段在画面上直接标出这个链接和 `Simulation Alessandro Roussel 2024`。

3. T. Collett, [*Cosmology with Double-Source-Plane Lenses*](https://indico.ipmu.jp/event/38/attachments/991/1136/11_Collett.pdf)。彭导视频约 22:45 出现其中 `Double source plane strong lensing` 等页面。相应的 Hubble 双环图可从 [NASA 原图页](https://science.nasa.gov/asset/hubble/gravitational-lens-system-sdssj09461006-double-einstein-ring)核对署名。

4. yochichao577, [*Interstellar Reimagined*](https://steamcommunity.com/sharedfiles/filedetails/?id=3406276870), SpaceEngine Steam Workshop。B 站简介还注明使用了 SpaceEngine 内置 `Intermediate-mass Black Hole` 和黑洞大修 Mod。这里只保留来源链接，不把画面复制到仓库。

5. R. Antonelli, [*Raytracing a Black Hole / Starless*](https://rantonels.github.io/starless/)及[源码](https://github.com/rantonels/starless)；D. Brant, [“Ray tracing black holes”](https://dmitrybrant.com/2018/12/11/ray-tracing-black-holes)。两者把 Schwarzschild 轨道方程、虚拟中心力、反向追踪、背景天球、薄盘、多重像和频移串成了与视频很相似的实现链。它们是强技术来源候选，但没有作者工程或逐句证据时，不能写成已经证实的抄袭来源。

## 数据与外部求解器

1. FEMM, [官方文档](https://www.femm.info/doku/doku.php?id=documentation) 与 [pyFEMM](https://www.femm.info/doku/doku.php?id=pyfemm)。用于二维/轴对称有限元磁场导出。

2. PyVista, [`streamlines` 官方文档](https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.streamlines)。它封装 VTK 流线功能，适合后续三维交互原型。

## 知乎案例与勘误 { #zhihu-cases }

下面三篇保留了本项目的出发点，适合看推导过程、最小实现和显示效果。它们是社区案例，不替代前面的原论文、公开教材和官方文档。以下附注按 2026-08-18 可见的正文与配套代码核对；只列会影响公式、程序或物理解读的地方。

### 圆电流环的解析磁场 { #zhihu-current-loop }

**原文：** 杂然赋流形丶，[《电磁学中，载流圆线圈在全空间的磁场分布是怎样的？》](https://www.zhihu.com/question/446655531/answer/2089751442)，2021-08-28 发布，2021-08-29 更新。

**适合参考：** 回答从线电流的分布式电流密度出发，利用轴对称性把磁矢势化为 $A_\phi$，再介绍完全椭圆积分、球谐展开和柱坐标修正贝塞尔展开。椭圆积分闭式、轴上磁场和远场磁偶极极限适合用作数值程序的测试基准。

**勘误与边界：**

- 标题中的“全空间”应排除理想线电流所在的圆周 $\rho=a,z=0$；该处场发散。有限截面导线是另一个模型。[NASA 圆电流环报告](https://ntrs.nasa.gov/citations/20010038494)也把解析式的适用域写为导体之外。
- 消去 Dirac $\delta$ 后，积分系数应为 $\mu_0Ia/(4\pi)$，不是原文的 $\mu_0I/(4\pi a)$。后面的椭圆积分闭式已恢复正确归一化。
- $E(k)$ 的级数推导有一处 Pochhammer 符号误写：应为 $(-\tfrac12)_n$，不是 $(\tfrac12)_n$；最终列出的 $E(k)$ 级数是对的。小 $k$ 展开中，$A_\phi$ 校正项的分母应为 $8(a^2+r^2)^2$，原文少了平方。
- 球谐展开末式的 $A_\phi$ 应含伴随勒让德函数 $P^1_{2n+1}(\cos\theta)$，原文漏了上标 $1$。$B_r$ 末式的径向因子应为 $r_<^{2n+1}/r_>^{2n+2}$；原文把分母指数写成 $2n+1$，量纲和 $r^{-3}$ 远场都不成立。$B_\theta$ 分段式的说明也写反了：上行对应 $r<a$，下行对应 $r>a$。$B_r$ 推导首行的 $1/\partial\theta$ 则是排印错误，应为 $\partial/\partial\theta$。
- 原文用模数 $k$ 定义 $K(k),E(k)$；SciPy 的 [`ellipk`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipk.html) 和 [`ellipe`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipe.html) 接收的是参数 $m=k^2$。轴上 $k\to0$ 要走极限分支，不能把闭式中的 $0/0$ 直接交给浮点计算。

这些错误不推翻回答的主线，也不影响它给出的轴上式和磁偶极远场；球谐级数部分不能原样抄进代码。

### 一个 3D 场线渲染框架 { #zhihu-3d-field-lines }

**原文：** 欣快的犀牛，[《Jackson 电动力学笔记(37)：一个粗鄙的 3D 场线的可视化框架（开源代码）》](https://zhuanlan.zhihu.com/p/649494323)，编辑于 2023-08-11；[配套 Go 代码](https://github.com/euphoricrhino/jackson-em-notes/tree/main/go/pkg/field-line)。

**适合参考：** 文章给出一条完整的最小实现链：用户提供三维向量场、种子和停止条件，程序用经典 RK4 追踪曲线，再作正交投影，并把场强映射为线段透明度。六边形点电荷、极化圆柱壳与球、四面体点电荷适合作为三维渲染和相机动画案例。

**勘误与边界：**

- 文中写“6 个正电荷、每个 30 根，一共 360 根”，源码实际生成 $6\times30=180$ 条。
- “RK4 最终误差为 $O(h^4)$”是光滑问题、固定有限积分区间和足够小步长下的全局阶，不是这份程序已经给出的误差保证。代码直接积分 $d\mathbf x/dt=\mathbf F$，单步空间位移随 $|\mathbf F|$ 变化：奇点附近可能跨步过大，零点附近又会停滞。通用追踪器应使用弧长参数或等价归一化，并设置误差控制、最大弧长、最大步数和事件终止。[VTK `vtkStreamTracer`](https://vtk.org/doc/nightly/html/classvtkStreamTracer.html)可作参数设计的对照。
- 弱场阈值只是数值停止条件，不等于一般意义上的物理终点。六个等量正电荷的原点只在所画的 $z=0$ 不变平面内表现为吸引方向，在完整三维中是鞍点。正负交替的六电荷模型因对称性整条 $z$ 轴都是零场，并不是孤立的“不稳定平衡点”。
- 输出是没有深度遮挡的正交二维投影。屏幕上的交叉不表示三维场线相交；透明度的 gamma 映射只是一种显示编码，线密度仍由种子决定。
- 配套代码没有最大弧长或最大步数，闭合且不离开画幅的线可能一直积分；均匀场还会让亮度的 min–max 归一化出现 `max == min`。这两项进入本项目时都要显式处理。

### 网格场上的 RK4 与三线性插值 { #zhihu-grid-tracing }

**原文：** Pjer，[《数值计算矢量场流线（适用于流场线，追磁力线）Python》](https://zhuanlan.zhihu.com/p/451474459)，2021-12-30；[配套代码](https://github.com/peijin94/stream3py)。

**适合参考：** 文章把“已有离散三维向量场 → 三线性插值 → 归一化方向 → 从种子点双向 RK4 积分”串成一个很小的例子。固定步长 RK4 和规则网格三线性插值作为教学实现没有问题。

**勘误与边界：**

- 文中说 MATLAB、PyVista 等工具不能指定步长和精度，这不正确。MATLAB [`stream3`](https://www.mathworks.com/help/matlab/ref/stream3.html) 支持 `step` 和 `maxvert`；文章发表时的 [PyVista 0.32.1 源码](https://github.com/pyvista/pyvista/blob/0.32.1/pyvista/core/filters/data_set.py#L2738-L2744)已经暴露 RK2、RK4、RK45、初始/最小/最大步长和 `max_error`。固定步长 RK4 只能指定步长，本身也不能直接指定误差。
- 配套代码公开了 `dr` 参数，却在积分时把它写死为 $\pm0.25$，调用者传入的值不起作用。代码还把 $y,z$ 的边界都拿 `Bx.shape[0]` 判断，非立方网格会出错。
- 自写 `trilerp` 把数组下标直接当物理坐标，只适用于单位间距的正交规则网格；示例结果也没有映射回传入的 $x,y,z$ 坐标。非等距规则网格可对照 SciPy [`RegularGridInterpolator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html)。
- 场线方程应明确写成 $d\mathbf x/ds=\mathbf F/\lVert\mathbf F\rVert$，并注明只在 $\lVert\mathbf F\rVert>0$ 时成立。零场、非有限值、源邻域和计算域边界都需要终止条件。归一化只保留曲线形状，不保留场强或粒子飞行时间。
- 这篇文章处理的是**已有采样场之后的场线后处理**，没有求解产生 $B_x,B_y,B_z$ 的物理方程。场求解、网格、插值和 ODE 误差仍要分别做收敛检查。非定常速度场中，流线还是固定时刻的瞬时曲线，不等于粒子轨迹线。
