import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 示例，使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题


class Charge:
    """
    表示电场中的点电荷类
    """

    def __init__(self, q, position, label=None):
        """
        初始化一个电荷

        参数:
        q: 电荷量，单位库仑(C)
        position: 电荷位置，格式为[x, y]，单位米(m)
        label: 电荷的标签，如果为None则自动生成
        """
        self.q = q  # 电荷量
        self.position = np.array(position)  # 位置向量

        # 根据电荷量确定颜色
        self.color = 'red' if q > 0 else 'blue' if q < 0 else 'gray'

        # 确定标签
        self.label = label if label else f'q={q:.2e}C'

        # 确定显示大小（基于电荷量的绝对值）
        self.size = np.abs(q) * 5e10  # 缩放因子可根据需要调整
        # 限制尺寸范围
        self.size = max(20, min(500, self.size))

    def electric_field_at(self, point):
        """
        计算该电荷在指定点产生的电场

        参数:
        point: 需要计算电场的点坐标 [x, y]

        返回值:
        电场矢量 [Ex, Ey]
        """
        k = 8.99e9  # 库仑常数 (N·m²/C²)
        epsilon = 1e-10  # 避免除以零的小量

        # 计算距离矢量
        r_vec = np.array(point) - self.position
        # 计算距离
        r = np.sqrt(np.sum(r_vec ** 2))
        # 计算单位方向向量
        r_hat = r_vec / (r + epsilon)
        # 计算电场
        E = k * self.q * r_hat / ((r + epsilon) ** 2)

        return E

    def __str__(self):
        """字符串表示"""
        sign = '+' if self.q > 0 else '-' if self.q < 0 else ''
        return f"Charge({sign}{abs(self.q):.2e}C at {self.position})"


class ElectricFieldSimulator:
    """
    电场模拟和可视化类
    """

    def __init__(self):
        """初始化模拟器"""
        self.charges = []  # 存储电荷列表

    def add_charge(self, charge):
        """
        添加一个电荷到模拟系统

        参数:
        charge: Charge对象
        """
        self.charges.append(charge)

    def add_charges(self, charges):
        """
        添加多个电荷到模拟系统

        参数:
        charges: Charge对象列表
        """
        self.charges.extend(charges)

    def clear_charges(self):
        """清除所有电荷"""
        self.charges = []

    def electric_field_at(self, point):
        """
        计算所有电荷在指定点产生的合成电场

        参数:
        point: 需要计算电场的点坐标 [x, y]

        返回值:
        合成电场矢量 [Ex, Ey]
        """
        if not self.charges:
            return np.zeros(2)

        # 计算每个电荷产生的电场并求和
        E_total = np.zeros(2)
        for charge in self.charges:
            E_total += charge.electric_field_at(point)

        return E_total

    def calculate_field_grid(self, xlim, ylim, grid_size):
        """
        计算网格上每个点的电场

        参数:
        xlim, ylim: 区域范围 [min, max]
        grid_size: 网格大小

        返回值:
        X, Y: 网格坐标
        Ex, Ey: 网格上每点的电场分量
        E_mag: 电场强度
        """
        # 创建网格点
        x = np.linspace(xlim[0], xlim[1], grid_size)
        y = np.linspace(ylim[0], ylim[1], grid_size)
        X, Y = np.meshgrid(x, y)

        # 计算每个网格点的电场
        Ex = np.zeros_like(X)
        Ey = np.zeros_like(Y)
        E_mag = np.zeros_like(X)

        for i in range(grid_size):
            for j in range(grid_size):
                point = np.array([X[j, i], Y[j, i]])

                # 检查该点是否与任何电荷重合
                skip = False
                for charge in self.charges:
                    if np.allclose(point, charge.position, atol=0.1):
                        skip = True
                        break

                if not skip:
                    E = self.electric_field_at(point)
                    Ex[j, i] = E[0]
                    Ey[j, i] = E[1]
                    E_mag[j, i] = np.sqrt(E[0] ** 2 + E[1] ** 2)

        return X, Y, Ex, Ey, E_mag

    def auto_range(self, padding=2.0):
        """
        根据电荷位置自动确定绘图范围

        参数:
        padding: 在最大最小值之外的填充空间

        返回值:
        xlim, ylim: 区域范围 [min, max]
        """
        if not self.charges:
            return [-10, 10], [-10, 10]

        positions = np.array([charge.position for charge in self.charges])

        xmin, ymin = positions.min(axis=0) - padding
        xmax, ymax = positions.max(axis=0) + padding

        return [xmin, xmax], [ymin, ymax]

    def visualize(self, xlim=None, ylim=None, grid_size=30, figsize=(10, 8),
                  cmap='viridis', streamplot_density=1.5):
        """
        可视化电场

        参数:
        xlim, ylim: 区域范围 [min, max]，如果为None则自动确定
        grid_size: 网格大小
        figsize: 图形大小
        cmap: 颜色映射
        streamplot_density: 流线密度
        """
        if not self.charges:
            print("没有电荷，无法可视化")
            return

        # 如果没有指定范围，则自动确定
        if xlim is None or ylim is None:
            xlim, ylim = self.auto_range()

        # 计算电场网格
        X, Y, Ex, Ey, E_mag = self.calculate_field_grid(xlim, ylim, grid_size)

        # 对电场强度取对数，便于可视化
        E_log = np.log10(E_mag + 1e-10)

        # 创建图形
        plt.figure(figsize=figsize)

        # 使用电场大小的对数作为颜色映射
        norm = colors.Normalize(vmin=E_log.min(), vmax=E_log.max())

        # 绘制电场线，并保存返回的对象用于创建颜色条
        strm = plt.streamplot(X, Y, Ex, Ey, color=E_log, linewidth=1,
                              cmap=cmap, density=streamplot_density,
                              arrowstyle='->', arrowsize=1.5, norm=norm)

        # 添加电荷
        for charge in self.charges:
            plt.scatter(charge.position[0], charge.position[1],
                        c=charge.color, s=charge.size, zorder=10)
            plt.text(charge.position[0] + 0.5, charge.position[1] + 0.5,
                     charge.label, fontsize=10)

        # 添加颜色条
        cbar = plt.colorbar(strm.lines)
        cbar.set_label('log10(|E|) (V/m)')

        plt.title('电场分布')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def interactive_mode(self):
        """运行交互式模式，让用户添加电荷"""
        print("=== 电场模拟器交互模式 ===")
        print("添加电荷，输入0结束添加")

        while True:
            try:
                q_str = input("输入电荷量(单位:纳库仑，输入0结束): ")
                q = float(q_str) * 1e-9  # 转换为库仑

                if q == 0:
                    break

                x = float(input("输入x坐标(单位:米): "))
                y = float(input("输入y坐标(单位:米): "))

                label = input("输入电荷标签(可选，按Enter跳过): ")
                if not label:
                    label = None

                charge = Charge(q, [x, y], label)
                self.add_charge(charge)
                print(f"已添加: {charge}")

            except ValueError:
                print("请输入有效数字")

        if self.charges:  # 是否为空列表
            print(f"共添加了 {len(self.charges)} 个电荷")
            self.visualize()
        else:
            print("未添加任何电荷，退出")


# 示例使用
if __name__ == "__main__":
    # 创建模拟器
    simulator = ElectricFieldSimulator()

    # 方法1：直接添加Charge对象
    charge1 = Charge(1e-9, [0, 0], "正电荷")
    charge2 = Charge(-1e-9, [5, 0], "负电荷")
    charge3 = Charge(2e-9, [2.5, 5], "大正电荷")
    print(charge3)

    simulator.add_charges([charge1, charge2, charge3])

    # 可视化电场
    simulator.visualize()


    # 方法2：使用交互模式
    # 取消注释下行以使用交互模式
    # simulator.clear_charges()  # 清除之前的电荷
    # simulator.interactive_mode()

    # 方法3：添加随机电荷
    def add_random_charges(simulator: ElectricFieldSimulator, n=5, q_range=(-2e-9, 2e-9), xy_range=(-10, 10)):
        """添加随机电荷"""
        simulator.clear_charges()

        for i in range(n):
            q = np.random.uniform(*q_range)
            x = np.random.uniform(*xy_range)
            y = np.random.uniform(*xy_range)

            charge = Charge(q, [x, y], f"随机{i + 1}")
            simulator.add_charge(charge)

        print(f"已添加 {n} 个随机电荷")


    add_random_charges(simulator, n=7)
    simulator.visualize()
