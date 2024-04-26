import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches


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


def update(frame, env, uav_plots, target_plots, uav_search_patches, steps):
    for i, uav in enumerate(env.uav_list):
        uav_x = env.position['all_uav_xs'][frame * steps][i]
        uav_y = env.position['all_uav_ys'][frame * steps][i]
        uav_plots[i].set_data(uav_x, uav_y)
        # 更新搜索范围扇形的位置和角度
        uav_search_patches[i].center = (uav_x, uav_y)
        # uav_search_patches[i].set_theta1(np.degrees(uav.h - uav.h_max))
        # uav_search_patches[i].set_theta2(np.degrees(uav.h + uav.h_max))

    for i in range(env.m_targets):
        target_x = env.position['all_target_xs'][frame * steps][i]
        target_y = env.position['all_target_ys'][frame * steps][i]
        target_plots[i].set_data(target_x, target_y)

    return target_plots + uav_plots + uav_search_patches


def draw_animation(env, num_steps, ep_num, frames=100):  # num_steps % frames must be 0
    if num_steps % frames != 0:
        print("num_steps % frames must be 0!")
        return

    fig, ax = plt.subplots()
    ax.set_xlim(-env.x_max / 3, env.x_max / 3 * 4)
    ax.set_ylim(-env.y_max / 3, env.y_max / 3 * 4)
    uav_plots = [ax.plot([], [], marker='o', linestyle='None')[0] for _ in range(env.n_uav)]
    target_plots = [ax.plot([], [], marker='o', linestyle='None')[0] for _ in range(env.m_targets)]

    uav_search_patches = [patches.Circle((uav.x, uav.y), uav.dp, color='blue', alpha=0.2) for uav in env.uav_list]

    for patch in uav_search_patches:
        ax.add_patch(patch)

    ani = animation.FuncAnimation(fig, update, frames=frames,
                                  fargs=(env, uav_plots, target_plots, uav_search_patches, num_steps // frames),
                                  blit=True, interval=50, repeat=True)
    # 保存动画为gif格式
    ani.save('../../results/' + 'animated_plot_' + str(ep_num + 1) + '.gif', writer='imagemagick')
    # plt.show()


def plot_reward_curve(return_list):
    plt.figure()
    plt.plot(return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Total Return')
    plt.title('Reward Curve')
    plt.grid(True)
    plt.savefig("../../results/result-curve" + ".png")
    plt.show()
