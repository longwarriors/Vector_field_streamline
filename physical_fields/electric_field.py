import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 示例，使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题


def electric_field(charges, positions, point):
    """
    计算N个点电荷在给定点产生的电场

    参数:
    charges: 电荷量数组 [q1, q2, ..., qN]
    positions: 电荷位置数组 [[x1,y1], [x2,y2], ..., [xN,yN]]
    point: 需要计算电场的点 [x, y]

    返回值:
    电场矢量 [Ex, Ey]
    """
    k = 8.99e9  # 库仑常数 (N·m²/C²)
    epsilon = 1e-10  # 避免除以零的小量

    E_total = np.zeros(2)

    for q, pos in zip(charges, positions):
        # 计算距离矢量
        r_vec = point - pos
        # 计算距离
        r = np.sqrt(np.sum(r_vec ** 2))
        # 计算单位方向向量
        r_hat = r_vec / (r + epsilon)
        # 计算该电荷产生的电场并累加
        E_total += k * q * r_hat / ((r + epsilon) ** 2)

    return E_total


def plot_electric_field(charges, positions, xlim=(-10, 10), ylim=(-10, 10), grid_size=20):
    """
    绘制电场图

    参数:
    charges: 电荷量数组 [q1, q2, ..., qN]
    positions: 电荷位置数组 [[x1,y1], [x2,y2], ..., [xN,yN]]
    xlim, ylim: 绘图区域范围
    grid_size: 网格大小，决定箭头密度
    """
    positions = np.array(positions)
    charges = np.array(charges)

    # 创建网格点
    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x, y)

    # 计算每个网格点的电场
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(Y)
    E_mag = np.zeros_like(X)

    for i in range(len(x)):
        for j in range(len(y)):
            point = np.array([X[j, i], Y[j, i]])

            # 检查该点是否与任何电荷重合
            skip = False
            for pos in positions:
                if np.allclose(point, pos, atol=0.1):
                    skip = True
                    break

            if not skip:
                E = electric_field(charges, positions, point)
                Ex[j, i] = E[0]
                Ey[j, i] = E[1]
                E_mag[j, i] = np.sqrt(E[0] ** 2 + E[1] ** 2)

    # 对电场强度取对数，便于可视化
    E_log = np.log10(E_mag + 1e-10)

    # 创建图形
    plt.figure(figsize=(10, 8))

    # 使用电场大小的对数作为颜色映射
    norm = colors.Normalize(vmin=E_log.min(), vmax=E_log.max())

    # 绘制电场线，并保存返回的对象用于创建颜色条
    strm = plt.streamplot(X, Y, Ex, Ey, color=E_log, linewidth=1,
                          cmap='viridis', density=1.5, arrowstyle='->', arrowsize=1.5,
                          norm=norm)

    # 添加电荷
    for q, pos in zip(charges, positions):
        color = 'red' if q > 0 else 'blue'
        size = np.abs(q) * 50  # 根据电荷大小调整圆圈大小
        plt.scatter(pos[0], pos[1], c=color, s=size, zorder=10)
        plt.text(pos[0] + 0.5, pos[1] + 0.5, f'q={q}C', fontsize=10)

    # 添加颜色条，使用streamplot的color属性
    cbar = plt.colorbar(strm.lines)
    cbar.set_label('log10(|E|) (V/m)')

    plt.title('N个点电荷的电场分布')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# 示例使用
if __name__ == "__main__":
    # 示例：3个电荷系统
    # 格式：charges = [q1, q2, q3, ...], positions = [[x1, y1], [x2, y2], [x3, y3], ...]
    charges = [1e-9, -1e-9, 2e-9]  # 电荷量，单位库仑(C)
    positions = [[0, 0], [5, 0], [2.5, 5]]  # 电荷位置，单位米(m)

    # 绘制电场
    plot_electric_field(charges, positions, xlim=(-2, 7), ylim=(-2, 7), grid_size=30)


    # 交互式使用：允许用户添加多个电荷
    def interactive_charges():
        charges = []
        positions = []

        while True:
            try:
                charge = float(input("输入电荷量(单位:纳库仑, 输入0结束): ")) * 1e-9
                if charge == 0:
                    break
                x = float(input("输入x坐标(单位:米): "))
                y = float(input("输入y坐标(单位:米): "))

                charges.append(charge)
                positions.append([x, y])

            except ValueError:
                print("请输入有效数字")

        if charges:  # 是否为空列表
            xlim = [min(pos[0] for pos in positions) - 2, max(pos[0] for pos in positions) + 2]
            ylim = [min(pos[1] for pos in positions) - 2, max(pos[1] for pos in positions) + 2]
            plot_electric_field(charges, positions, xlim=xlim, ylim=ylim)

    # interactive_charges()
