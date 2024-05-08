import os.path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches


def update(frame, env, uav_plots, target_plots, uav_search_patches, frames, num_steps, interval=5):
    for i, uav in enumerate(env.uav_list):
        uav_x = env.position['all_uav_xs'][
                frame + (num_steps - frames) // 2: frame + num_steps - frames: interval]
        uav_x = [sublist[i] for sublist in uav_x]
        uav_y = env.position['all_uav_ys'][
                frame + (num_steps - frames) // 2: frame + num_steps - frames: interval]
        uav_y = [sublist[i] for sublist in uav_y]
        uav_plots[i].set_data(uav_x, uav_y)
        # 更新搜索范围扇形的位置和角度
        uav_search_patches[i].center = (uav_x[-1], uav_y[-1])
        # uav_search_patches[i].set_theta1(np.degrees(uav.h - uav.h_max))
        # uav_search_patches[i].set_theta2(np.degrees(uav.h + uav.h_max))

    for i in range(env.m_targets):
        target_x = env.position['all_target_xs'][
                   frame + (num_steps - frames) // 2: frame + num_steps - frames: interval]
        target_x = [sublist[i] for sublist in target_x]
        target_y = env.position['all_target_ys'][
                   frame + (num_steps - frames) // 2: frame + num_steps - frames: interval]
        target_y = [sublist[i] for sublist in target_y]
        target_plots[i].set_data(target_x, target_y)

    return target_plots + uav_plots + uav_search_patches


def draw_animation(config, env, num_steps, ep_num, frames=100):

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-env.x_max / 3, env.x_max / 3 * 4)
    ax.set_ylim(-env.y_max / 3, env.y_max / 3 * 4)
    uav_plots = [ax.plot([], [], marker='o', color='b', linestyle='None', markersize=1)[0]
                 for _ in range(env.n_uav)]
    target_plots = [ax.plot([], [], marker='o', color='r', linestyle='None', markersize=1)[0]
                    for _ in range(env.m_targets)]

    uav_search_patches = [patches.Circle((0, 0), uav.dp, color='lightblue', alpha=0.2)
                          for uav in env.uav_list]
    for patch in uav_search_patches:
        ax.add_patch(patch)

    ani = animation.FuncAnimation(fig, update, frames=frames,
                                  fargs=(env,
                                         uav_plots,
                                         target_plots,
                                         uav_search_patches,
                                         frames,
                                         num_steps),
                                  blit=True, interval=10, repeat=True)
    # 保存动画为gif格式
    ani.save(os.path.join(config["save_dir"], 'animated_plot_' + str(ep_num + 1) + '.gif'), writer='imagemagick')
    plt.close(fig)


def plot_reward_curve(config, return_list, name):
    plt.figure(figsize=(6, 6))
    plt.plot(return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Total Return')
    plt.title(name)
    plt.grid(True)
    plt.savefig(os.path.join(config["save_dir"], name + ".png"))
    # plt.show()
