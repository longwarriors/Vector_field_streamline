import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 示例，使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题


class MagneticDipole:
    """
    表示磁场中的磁偶极子类
    """

    def __init__(self, moment, position, orientation=None, label=None):
        """
        初始化一个磁偶极子

        参数:
        moment: 磁矩，单位A·m²
        position: 偶极子位置，格式为[x, y, z]，单位米(m)
        orientation: 磁矩方向，单位矢量[mx, my, mz]，如果为None则默认为z轴方向[0, 0, 1]
        label: 偶极子的标签，如果为None则自动生成
        """
        self.moment_magnitude = float(moment)  # 磁矩大小
        self.position = np.array(position)  # 位置向量

        # 确保位置是3D向量
        if len(self.position) == 2:
            self.position = np.append(self.position, 0)  # 添加z=0

        # 处理磁矩方向
        if orientation is None:
            self.orientation = np.array([0, 0, 1])  # 默认指向z轴正方向
        else:
            self.orientation = np.array(orientation)
            # 归一化方向向量
            norm = np.linalg.norm(self.orientation)
            if norm > 0:
                self.orientation = self.orientation / norm

        # 计算磁矩向量（方向 * 大小）
        self.moment = self.orientation * self.moment_magnitude

        # 根据磁矩大小确定颜色（红色表示强磁矩，蓝色表示弱磁矩）
        # 这里可以根据需要调整颜色方案
        if self.moment_magnitude > 0:
            self.color = 'red'
        else:
            self.color = 'blue'

        # 确定标签
        self.label = label if label else f'm={self.moment_magnitude:.2e}A·m²'

        # 确定显示大小（基于磁矩大小的绝对值）
        self.size = np.abs(self.moment_magnitude) * 1e3  # 缩放因子可根据需要调整
        # 限制尺寸范围
        self.size = max(20, min(500, self.size))

    def magnetic_field_at(self, point):
        """
        计算该磁偶极子在指定点产生的磁场

        参数:
        point: 需要计算磁场的点坐标 [x, y, z]

        返回值:
        磁场矢量 [Bx, By, Bz]
        """
        mu0 = 4 * np.pi * 1e-7  # 真空磁导率 (H/m)

        # 确保点是3D向量
        point = np.array(point)
        if len(point) == 2:
            point = np.append(point, 0)  # 添加z=0

        # 计算位置矢量（从偶极子指向场点）
        r_vec = point - self.position

        # 计算距离
        r = np.linalg.norm(r_vec)

        # 防止除以零
        if r < 1e-10:
            return np.zeros(3)

        # 计算单位方向向量
        r_hat = r_vec / r

        # 磁偶极子磁场公式: B = (μ0/4π) * (3(m·r̂)r̂ - m) / r³
        # 其中m是磁矩向量，r̂是单位距离向量

        # 计算 m·r̂ （磁矩与方向向量的点积）
        m_dot_r = np.dot(self.moment, r_hat)

        # 计算磁场
        B = (mu0 / (4 * np.pi)) * (3 * m_dot_r * r_hat - self.moment) / (r ** 3)

        return B

    def __str__(self):
        """字符串表示"""
        sign = '+' if self.moment_magnitude > 0 else '-' if self.moment_magnitude < 0 else ''
        return f"MagneticDipole({sign}{abs(self.moment_magnitude):.2e}A·m² at {self.position}, dir={self.orientation})"


class MagneticFieldSimulator:
    """
    磁场模拟和可视化类
    """

    def __init__(self):
        """初始化模拟器"""
        self.dipoles = []  # 存储磁偶极子列表

    def add_dipole(self, dipole):
        """
        添加一个磁偶极子到模拟系统

        参数:
        dipole: MagneticDipole对象
        """
        self.dipoles.append(dipole)

    def add_dipoles(self, dipoles):
        """
        添加多个磁偶极子到模拟系统

        参数:
        dipoles: MagneticDipole对象列表
        """
        self.dipoles.extend(dipoles)

    def clear_dipoles(self):
        """清除所有磁偶极子"""
        self.dipoles = []

    def magnetic_field_at(self, point):
        """
        计算所有磁偶极子在指定点产生的合成磁场

        参数:
        point: 需要计算磁场的点坐标 [x, y, z]

        返回值:
        合成磁场矢量 [Bx, By, Bz]
        """
        if not self.dipoles:
            return np.zeros(3)

        # 计算每个磁偶极子产生的磁场并求和
        B_total = np.zeros(3)
        for dipole in self.dipoles:
            B_total += dipole.magnetic_field_at(point)

        return B_total

    def calculate_field_grid(self, xlim, ylim, zlim=None, grid_size=20):
        """
        计算网格上每个点的磁场

        参数:
        xlim, ylim: 区域范围 [min, max]
        zlim: Z轴范围，默认为None（表示2D平面，z=0）
        grid_size: 网格大小

        返回值:
        如果zlim为None（2D模式）:
            X, Y: 网格坐标
            Bx, By, Bz: 网格上每点的磁场分量
            B_mag: 磁场强度
        否则（3D模式）:
            X, Y, Z: 网格坐标
            Bx, By, Bz: 网格上每点的磁场分量
            B_mag: 磁场强度
        """
        # 确定是2D还是3D模式
        is_3d = zlim is not None

        if is_3d:
            # 3D网格
            x = np.linspace(xlim[0], xlim[1], grid_size)
            y = np.linspace(ylim[0], ylim[1], grid_size)
            z = np.linspace(zlim[0], zlim[1], grid_size)
            X, Y, Z = np.meshgrid(x, y, z)

            # 初始化磁场数组
            Bx = np.zeros_like(X)
            By = np.zeros_like(Y)
            Bz = np.zeros_like(Z)
            B_mag = np.zeros_like(X)

            # 计算每个网格点的磁场
            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(grid_size):
                        point = np.array([X[j, i, k], Y[j, i, k], Z[j, i, k]])

                        # 检查该点是否与任何磁偶极子重合
                        skip = False
                        for dipole in self.dipoles:
                            if np.allclose(point, dipole.position, atol=0.1):
                                skip = True
                                break

                        if not skip:
                            B = self.magnetic_field_at(point)
                            Bx[j, i, k] = B[0]
                            By[j, i, k] = B[1]
                            Bz[j, i, k] = B[2]
                            B_mag[j, i, k] = np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)

            return X, Y, Z, Bx, By, Bz, B_mag

        else:
            # 2D网格（z=0的平面）
            x = np.linspace(xlim[0], xlim[1], grid_size)
            y = np.linspace(ylim[0], ylim[1], grid_size)
            X, Y = np.meshgrid(x, y)

            # 初始化磁场数组
            Bx = np.zeros_like(X)
            By = np.zeros_like(Y)
            Bz = np.zeros_like(X)
            B_mag = np.zeros_like(X)

            # 计算每个网格点的磁场
            for i in range(grid_size):
                for j in range(grid_size):
                    point = np.array([X[j, i], Y[j, i], 0])  # z=0平面

                    # 检查该点是否与任何磁偶极子重合
                    skip = False
                    for dipole in self.dipoles:
                        if np.allclose(point[:2], dipole.position[:2], atol=0.1):
                            skip = True
                            break

                    if not skip:
                        B = self.magnetic_field_at(point)
                        Bx[j, i] = B[0]
                        By[j, i] = B[1]
                        Bz[j, i] = B[2]
                        B_mag[j, i] = np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)

            return X, Y, Bx, By, Bz, B_mag

    def auto_range(self, padding=2.0):
        """
        根据磁偶极子位置自动确定绘图范围

        参数:
        padding: 在最大最小值之外的填充空间

        返回值:
        xlim, ylim, zlim: 区域范围 [min, max]
        """
        if not self.dipoles:
            return [-10, 10], [-10, 10], [-10, 10]

        positions = np.array([dipole.position for dipole in self.dipoles])

        xmin, ymin, zmin = positions.min(axis=0) - padding
        xmax, ymax, zmax = positions.max(axis=0) + padding

        return [xmin, xmax], [ymin, ymax], [zmin, zmax]

    def visualize_2d(self, xlim=None, ylim=None, grid_size=30, figsize=(10, 8),
                     cmap='viridis', streamplot_density=1.5, plane='xy'):
        """
        可视化2D平面上的磁场

        参数:
        xlim, ylim: 区域范围 [min, max]，如果为None则自动确定
        grid_size: 网格大小
        figsize: 图形大小
        cmap: 颜色映射
        streamplot_density: 流线密度
        plane: 要可视化的平面，'xy', 'xz', 或 'yz'
        """
        if not self.dipoles:
            print("没有磁偶极子，无法可视化")
            return

        # 如果没有指定范围，则自动确定
        if xlim is None or ylim is None:
            xlim, ylim, zlim = self.auto_range()

            # 根据选定的平面调整范围
            if plane == 'xz':
                ylim = zlim
            elif plane == 'yz':
                xlim = ylim
                ylim = zlim

        # 创建正确的网格
        x = np.linspace(xlim[0], xlim[1], grid_size)
        y = np.linspace(ylim[0], ylim[1], grid_size)
        X, Y = np.meshgrid(x, y)

        # 初始化磁场数组
        Bx = np.zeros((grid_size, grid_size))
        By = np.zeros((grid_size, grid_size))
        Bz = np.zeros((grid_size, grid_size))
        B_mag = np.zeros((grid_size, grid_size))

        # 根据选择的平面计算对应点的磁场
        for i in range(grid_size):
            for j in range(grid_size):
                if plane == 'xy':
                    # xy平面 (z=0)
                    point = np.array([X[j, i], Y[j, i], 0])
                    B = self.magnetic_field_at(point)
                    Bx[j, i], By[j, i], Bz[j, i] = B
                    # 流线使用xy平面内的分量
                    u, v = Bx[j, i], By[j, i]
                elif plane == 'xz':
                    # xz平面 (y=0)
                    point = np.array([X[j, i], 0, Y[j, i]])  # 注意这里y=0，z用Y的值
                    B = self.magnetic_field_at(point)
                    Bx[j, i], By[j, i], Bz[j, i] = B
                    # 流线使用xz平面内的分量
                    u, v = Bx[j, i], Bz[j, i]
                elif plane == 'yz':
                    # yz平面 (x=0)
                    point = np.array([0, X[j, i], Y[j, i]])  # 注意这里x=0，y用X的值，z用Y的值
                    B = self.magnetic_field_at(point)
                    Bx[j, i], By[j, i], Bz[j, i] = B
                    # 流线使用yz平面内的分量
                    u, v = By[j, i], Bz[j, i]

                B_mag[j, i] = np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)

        # 根据平面选择正确的流线分量
        if plane == 'xy':
            u, v = Bx, By
            xlabel, ylabel = 'x (m)', 'y (m)'
        elif plane == 'xz':
            u, v = Bx, Bz
            xlabel, ylabel = 'x (m)', 'z (m)'
        elif plane == 'yz':
            u, v = By, Bz
            xlabel, ylabel = 'y (m)', 'z (m)'

        # 对磁场强度取对数，便于可视化
        B_log = np.log10(B_mag + 1e-10)

        # 创建图形
        plt.figure(figsize=figsize)

        # 使用磁场大小的对数作为颜色映射
        norm = colors.Normalize(vmin=B_log.min(), vmax=B_log.max())

        # 绘制磁场线，并保存返回的对象用于创建颜色条
        strm = plt.streamplot(X, Y, u, v, color=B_log, linewidth=1,
                              cmap=cmap, density=streamplot_density,
                              arrowstyle='->', arrowsize=1.5, norm=norm)

        # 添加磁偶极子
        for dipole in self.dipoles:
            # 确定在当前平面上的位置和方向
            if plane == 'xy':
                pos_x, pos_y = dipole.position[0], dipole.position[1]
                dir_x, dir_y = dipole.orientation[0], dipole.orientation[1]
                show_dipole = abs(dipole.position[2]) < 0.5  # 接近xy平面的磁偶极子
            elif plane == 'xz':
                pos_x, pos_y = dipole.position[0], dipole.position[2]
                dir_x, dir_y = dipole.orientation[0], dipole.orientation[2]
                show_dipole = abs(dipole.position[1]) < 0.5  # 接近xz平面的磁偶极子
            elif plane == 'yz':
                pos_x, pos_y = dipole.position[1], dipole.position[2]
                dir_x, dir_y = dipole.orientation[1], dipole.orientation[2]
                show_dipole = abs(dipole.position[0]) < 0.5  # 接近yz平面的磁偶极子

            # 只显示接近当前平面的磁偶极子
            if show_dipole:
                # 绘制磁偶极子位置
                plt.scatter(pos_x, pos_y, c=dipole.color, s=dipole.size, zorder=10)

                # 绘制磁矩方向（平面内的投影）
                moment_len = np.sqrt(dir_x ** 2 + dir_y ** 2)
                if moment_len > 0.1:  # 如果在平面内有显著分量
                    # 归一化平面内的方向向量
                    dir_x, dir_y = dir_x / moment_len, dir_y / moment_len
                    # 绘制方向箭头
                    plt.arrow(pos_x, pos_y, dir_x, dir_y, color=dipole.color,
                              width=0.1, head_width=0.3, head_length=0.3, zorder=11)

                # 添加标签
                plt.text(pos_x + 0.5, pos_y + 0.5, dipole.label, fontsize=10, zorder=12)

        # 添加颜色条
        cbar = plt.colorbar(strm.lines)
        cbar.set_label('log10(|B|) (T)')

        plt.title(f'磁场分布 ({plane}平面)')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def visualize_3d(self, xlim=None, ylim=None, zlim=None, grid_size=10, figsize=(12, 10),
                     scale=15, skip=2):
        """
        可视化3D空间中的磁场（使用箭头表示）

        参数:
        xlim, ylim, zlim: 区域范围 [min, max]，如果为None则自动确定
        grid_size: 每个维度的网格点数
        figsize: 图形大小
        scale: 箭头的缩放因子
        skip: 每隔多少个点绘制一个箭头（减少视觉混乱）
        """
        if not self.dipoles:
            print("没有磁偶极子，无法可视化")
            return

        # 如果没有指定范围，则自动确定
        if xlim is None or ylim is None or zlim is None:
            xlim, ylim, zlim = self.auto_range()

        # 创建3D图形
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # 创建网格点
        x = np.linspace(xlim[0], xlim[1], grid_size)
        y = np.linspace(ylim[0], ylim[1], grid_size)
        z = np.linspace(zlim[0], zlim[1], grid_size)

        # 计算并绘制磁场向量
        for i in range(0, grid_size, skip):
            for j in range(0, grid_size, skip):
                for k in range(0, grid_size, skip):
                    # 当前点坐标
                    point = np.array([x[i], y[j], z[k]])

                    # 检查该点是否与任何磁偶极子重合
                    skip_point = False
                    for dipole in self.dipoles:
                        if np.allclose(point, dipole.position, atol=0.3):
                            skip_point = True
                            break

                    if skip_point:
                        continue

                    # 计算磁场
                    B = self.magnetic_field_at(point)
                    Bx, By, Bz = B
                    B_mag = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

                    # 跳过磁场太小的点
                    if B_mag < 1e-12:
                        continue

                    # 计算归一化长度和颜色
                    length = np.log10(B_mag + 1e-10) + 1  # 使用对数缩放
                    color = plt.cm.viridis(
                        (np.log10(B_mag + 1e-10) - np.log10(1e-12)) / (np.log10(1) - np.log10(1e-12))
                    )

                    # 归一化方向向量
                    if B_mag > 0:
                        u, v, w = Bx / B_mag, By / B_mag, Bz / B_mag
                    else:
                        u, v, w = 0, 0, 0

                    # 绘制箭头
                    ax.quiver(point[0], point[1], point[2], u, v, w,
                              length=length * scale, color=color,
                              arrow_length_ratio=0.3, alpha=0.7)

        # 添加磁偶极子
        for dipole in self.dipoles:
            x, y, z = dipole.position
            u, v, w = dipole.orientation

            # 绘制磁偶极子的位置
            ax.scatter([x], [y], [z], color=dipole.color, s=dipole.size, zorder=10)

            # 绘制磁矩方向
            arrow_length = 1.5  # 固定长度
            ax.quiver(x, y, z, u, v, w,
                      length=arrow_length, color=dipole.color,
                      arrow_length_ratio=0.3, linewidth=3)

            # 添加标签
            ax.text(x, y, z, dipole.label, fontsize=10)

        # 设置图形标签和标题
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('磁场3D分布')

        # 设置坐标轴范围
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        # 添加颜色条说明磁场强度
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label('log10(|B|) (T)')

        plt.tight_layout()
        plt.show()

    def visualize(self, mode='2d', plane='xy', **kwargs):
        """
        可视化磁场

        参数:
        mode: '2d'或'3d'，决定使用哪种可视化方法
        plane: 在2d模式下，指定要显示的平面：'xy', 'xz', 或 'yz'
        **kwargs: 传递给对应可视化函数的额外参数
        """
        if mode.lower() == '2d':
            self.visualize_2d(plane=plane, **kwargs)
        else:
            self.visualize_3d(**kwargs)

    def interactive_mode(self):
        """运行交互式模式，让用户添加磁偶极子"""
        print("=== 磁场模拟器交互模式 ===")
        print("添加磁偶极子，输入0结束添加")

        while True:
            try:
                m_str = input("输入磁矩(单位:A·m²，输入0结束): ")
                m = float(m_str)

                if m == 0:
                    break

                x = float(input("输入x坐标(单位:米): "))
                y = float(input("输入y坐标(单位:米): "))
                z = float(input("输入z坐标(单位:米): "))

                print("输入磁矩方向（单位向量）:")
                mx = float(input("方向x分量: "))
                my = float(input("方向y分量: "))
                mz = float(input("方向z分量: "))

                label = input("输入磁偶极子标签(可选，按Enter跳过): ")
                if not label:
                    label = None

                dipole = MagneticDipole(m, [x, y, z], [mx, my, mz], label)
                self.add_dipole(dipole)
                print(f"已添加: {dipole}")

            except ValueError:
                print("请输入有效数字")

        if self.dipoles:
            print(f"共添加了 {len(self.dipoles)} 个磁偶极子")

            while True:
                vis_mode = input("选择可视化模式 (2d/3d，默认2d): ").lower() or '2d'

                if vis_mode == '2d':
                    plane = input("选择可视化平面 (xy/xz/yz，默认xy): ").lower() or 'xy'
                    if plane not in ['xy', 'xz', 'yz']:
                        print("无效的平面，使用xy平面")
                        plane = 'xy'
                    self.visualize(mode='2d', plane=plane)
                else:
                    self.visualize(mode='3d')

                again = input("是否使用其他方式可视化? (y/n): ").lower()
                if again != 'y':
                    break
        else:
            print("未添加任何磁偶极子，退出")


# 示例使用
if __name__ == "__main__":
    # 创建模拟器
    simulator = MagneticFieldSimulator()

    # 方法1：直接添加MagneticDipole对象
    dipole1 = MagneticDipole(1.0, [0, 0, 0], [0, 0, 1], "竖直磁铁")
    dipole2 = MagneticDipole(0.8, [5, 0, 0], [1, 0, 0], "水平磁铁")
    dipole3 = MagneticDipole(1.2, [2.5, 5, 1], [0.5, 0.5, 0.7], "倾斜磁铁")

    simulator.add_dipoles([dipole1, dipole2, dipole3])

    # 可视化磁场，2D模式
    print("2D模式可视化 (xy平面):")
    simulator.visualize(mode='2d', plane='xy')

    # 可视化xz平面
    print("2D模式可视化 (xz平面):")
    simulator.visualize(mode='2d', plane='xz')

    # 可视化3D模式
    print("3D模式可视化:")
    simulator.visualize(mode='3d', grid_size=8)  # 3D模式使用较小的网格以提高性能


    # 方法2：使用交互模式
    # 取消注释下面两行以使用交互模式
    # simulator.clear_dipoles()  # 清除之前的磁偶极子
    # simulator.interactive_mode()

    # 方法3：添加预设配置
    def setup_bar_magnet(simulator, position=[0, 0, 0], orientation=[0, 0, 1], strength=1.0, label="条形磁铁"):
        """
        创建一个简单的条形磁铁模型（使用一个磁偶极子）
        """
        dipole = MagneticDipole(strength, position, orientation, label)
        simulator.add_dipole(dipole)
        return dipole


    def setup_electromagnet(simulator, position=[0, 0, 0], orientation=[0, 0, 1],
                            current=1.0, loops=10, radius=0.5, label="电磁铁"):
        """
        创建一个简单的环形电磁铁模型（磁矩 = 电流 * 面积 * 匝数）
        """
        area = np.pi * radius ** 2
        moment = current * area * loops
        dipole = MagneticDipole(moment, position, orientation, label)
        simulator.add_dipole(dipole)
        return dipole


    def setup_earth_field(simulator, strength=3e-5, inclination=60):
        """
        模拟地球磁场（简化为一个全局均匀场）

        参数:
        strength: 磁场强度，默认约30微特斯拉（地球表面平均值）
        inclination: 磁场倾角（与水平面的夹角，度），正值表示向下
        """
        # 将倾角转换为弧度
        inc_rad = np.radians(inclination)

        # 计算指向北方的水平分量和垂直向下分量
        h_component = strength * np.cos(inc_rad)
        v_component = strength * np.sin(inc_rad)

        # 假设北方对应y轴正方向，上方对应z轴正方向
        # 地球磁场指向北方的同时向下倾斜
        # 因此方向向量为[0, h_component, -v_component]
        # 我们需要一个非常大的磁矩才能在整个区域内产生基本均匀的场
        # 将磁偶极子放在远处（比可视化区域大很多）
        distance = 1000  # 足够远
        moment = distance ** 3 * strength * 10  # 放大磁矩以产生所需场强

        # 创建一个远距离磁偶极子来模拟近似均匀场
        dipole = MagneticDipole(moment, [0, -distance, 0], [0, h_component, -v_component], "地球磁场")
        simulator.add_dipole(dipole)
        return dipole


    # 创建一个新的模拟器实例来演示这些预设配置
    def demo_presets():
        earth_sim = MagneticFieldSimulator()

        # 添加地球磁场
        setup_earth_field(earth_sim)

        # 添加一个条形磁铁
        setup_bar_magnet(earth_sim, position=[0, 0, 0], orientation=[0, 0, 1], strength=0.5, label="垂直磁铁")

        # 添加一个电磁铁
        setup_electromagnet(earth_sim, position=[5, 0, 0], orientation=[1, 0, 0], current=2.0, loops=20,
                            label="水平电磁铁")

        # 可视化结果
        earth_sim.visualize(mode='2d', plane='xy')
        earth_sim.visualize(mode='2d', plane='xz')
        earth_sim.visualize(mode='3d', grid_size=7)


    # 取消注释下行以运行预设配置演示
    # demo_presets()

    # 方法4：创建两个磁铁之间的相互作用场景
    def two_magnets_interaction():
        """演示两个磁铁之间的磁场分布"""
        sim = MagneticFieldSimulator()

        # 添加两个平行放置的条形磁铁，磁极方向相反
        # 磁铁1：N极朝上
        dipole1 = MagneticDipole(1.0, [0, 0, 0], [0, 0, 1], "磁铁1 (N↑)")
        # 磁铁2：N极朝下，与磁铁1相互排斥
        dipole2 = MagneticDipole(1.0, [0, 0, 3], [0, 0, -1], "磁铁2 (N↓)")

        sim.add_dipoles([dipole1, dipole2])

        # 可视化不同平面的磁场分布
        print("两个条形磁铁的磁场相互作用:")
        sim.visualize(mode='2d', plane='xz', xlim=[-5, 5], ylim=[-1, 4])
        sim.visualize(mode='3d', grid_size=6, xlim=[-3, 3], ylim=[-3, 3], zlim=[-1, 4])


    # 取消注释下行以演示两磁铁相互作用
    # two_magnets_interaction()

    # 方法5：模拟一个复杂的多磁极系统
    def multi_pole_system():
        """创建一个复杂的多磁极系统"""
        sim = MagneticFieldSimulator()

        # 添加一个四极磁场配置（两个N极和两个S极交替排列）
        # 这可以近似模拟一些四极磁铁装置
        r = 3.0  # 磁极的放置半径
        strength = 0.8  # 磁矩大小

        # 添加四个磁偶极子，形成四极场
        # 第一个磁偶极子 - 指向中心
        dipole1 = MagneticDipole(strength, [r, 0, 0], [-1, 0, 0], "磁极1")
        # 第二个磁偶极子 - 远离中心
        dipole2 = MagneticDipole(strength, [0, r, 0], [0, 1, 0], "磁极2")
        # 第三个磁偶极子 - 指向中心
        dipole3 = MagneticDipole(strength, [-r, 0, 0], [1, 0, 0], "磁极3")
        # 第四个磁偶极子 - 远离中心
        dipole4 = MagneticDipole(strength, [0, -r, 0], [0, -1, 0], "磁极4")

        sim.add_dipoles([dipole1, dipole2, dipole3, dipole4])

        # 可视化四极场
        print("四极磁场配置:")
        sim.visualize(mode='2d', plane='xy', xlim=[-5, 5], ylim=[-5, 5])
        sim.visualize(mode='3d', grid_size=6, xlim=[-5, 5], ylim=[-5, 5], zlim=[-5, 5])

    # 取消注释下行以演示多极系统
    # multi_pole_system()
