import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

class UAV:
    def __init__(self, x0, y0, h0, vmax, hmax, xmax, ymax, Na, search_radius):
        self.x = x0  # 初始 x 坐标
        self.y = y0  # 初始 y 坐标
        self.h = h0  # 初始朝向角度
        self.vmax = vmax  # 最大线速度
        self.hmax = hmax  # 最大角速度
        self.xmax = xmax
        self.ymax = ymax
        self.Na = Na  # 动作空间离散化数量
        self.search_radius = search_radius  # 搜索范围半径

    def update_position(self, dt, a_idx):
        a = self.discrete_action(a_idx)
        dx = dt * self.vmax * np.cos(self.h)  # x 方向位移
        dy = dt * self.vmax * np.sin(self.h)  # y 方向位移
        self.x += dx
        self.y += dy
        self.h += dt * a  # 更新朝向角度
        self.x = np.clip(self.x, 0, self.xmax)  # 确保位置在边界内
        self.y = np.clip(self.y, 0, self.ymax)  # 确保位置在边界内
        self.h = (self.h + np.pi) % (2 * np.pi) - np.pi  # 确保朝向角度在 [-pi, pi) 范围内
        return self.x, self.y, self.h

    def discrete_action(self, a_idx):
        na = a_idx + 1  # 从 1 开始索引
        return (2 * na - self.Na - 1) * self.hmax / (self.Na - 1)

class Target:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.found = False

# 更新函数
def update(frame):
    for i, uav in enumerate(uavs):
        a_idx = np.random.randint(-1 * Na, Na)  # 随机选择动作索引
        x, y, _ = uav.update_position(dt, a_idx)
        uav_plots[i].set_data(x, y)
        # 更新搜索范围扇形的位置和角度
        search_patches[i].center = (x, y)
        search_patches[i].set_theta1(np.degrees(uav.h - hmax))
        search_patches[i].set_theta2(np.degrees(uav.h + hmax))
    return uav_plots + search_patches

if __name__ == "__main__":
    # 定义场景参数
    n = 3  # 无人机数量
    m = 5  # 目标数量
    xmax = 100  # x 最大边界
    ymax = 100  # y 最大边界
    dt = 0.1  # 时间步长
    vmax = 10  # 无人机最大线速度
    hmax = np.pi / 4  # 无人机最大角速度
    Na = 5  # 动作空间离散化数量
    search_radius = 20  # 搜索范围半径

    # 创建无人机对象
    uavs = [UAV(np.random.uniform(0, xmax), np.random.uniform(0, ymax), np.random.uniform(-np.pi, np.pi), vmax, hmax, xmax, ymax, Na, search_radius) for _ in range(n)]

    # 初始化动画
    fig, ax = plt.subplots()
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    uav_plots = [ax.plot([], [], marker='o', linestyle='None')[0] for _ in range(n)]
    search_patches = [patches.Wedge((uav.x, uav.y), uav.search_radius, np.degrees(uav.h - hmax), np.degrees(uav.h + hmax), color='blue', alpha=0.2) for uav in uavs]

    # 添加搜索范围扇形到图中
    for patch in search_patches:
        ax.add_patch(patch)

    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=100, blit=True, interval=50, repeat=True)
    # 保存动画为gif格式
    ani.save('animated_plot.gif', writer='imagemagick')
    plt.show()
