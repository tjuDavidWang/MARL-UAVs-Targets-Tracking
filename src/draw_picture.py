import matplotlib.pyplot as plt
import numpy as np

def draw(uav_xs, uav_ys, target_xs, target_ys, num_steps, n_uav, m_target, dp, ep_num):
    fig, ax = plt.subplots()
    alphas = np.linspace(0, 1, num_steps)
    for i in range(num_steps):
        for j in range(n_uav):
            ax.scatter(uav_xs[i][j], uav_ys[i][j], color='b', marker='o', label='UAV', alpha=alphas[i], s=1)

    for i in range(num_steps):
        for j in range(m_target):
            ax.scatter(target_xs[i][j], target_ys[i][j], color='r', marker='o', label='TARGET', alpha=alphas[i], s=1)

    # 取二维数组的最后一行元素
    last_row_xs = uav_xs[-1]
    last_row_ys = uav_ys[-1]

    for x, y in zip(last_row_xs, last_row_ys):
        circle = plt.Circle((x, y), radius=dp, color='lightblue', alpha=0.2)
        ax.add_patch(circle)

    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)
    ax.set_xticks(range(0, 2000 + 1, 500))
    ax.set_yticks(range(0, 2000 + 1, 500))
    ax.set_aspect('equal')
    ax.set_title('episode %d' % ep_num)
    plt.show()

