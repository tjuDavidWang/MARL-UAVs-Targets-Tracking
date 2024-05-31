import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import imageio
from PIL import Image
import numpy as np

# 初始化文本对象为None
text_obj = None

def resize_image(image_path):
    img = Image.open(image_path).convert('RGB')
    # Resize the image to be divisible by 16
    new_width = (img.width // 16) * 16
    new_height = (img.height // 16) * 16
    if new_width != img.width or new_height != img.height:
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)  # Updated to use Image.Resampling.LANCZOS
    return np.array(img)

def get_gradient_color(start_color, end_color, num_points, idx):
    start_rgba = np.array(mcolors.to_rgba(start_color))
    end_rgba = np.array(mcolors.to_rgba(end_color))
    ratio = idx / max(1, num_points - 1)
    gradient_rgba = start_rgba + (end_rgba - start_rgba) * ratio
    return mcolors.to_hex(gradient_rgba)

def update(ax, env, uav_plots, target_plots, uav_search_patches, frame, frames, num_steps, interval=2, paint_all=True):
    global text_obj

    if frame == 0:
        return
    for i, uav in enumerate(env.uav_list):
        uav_x = env.position['all_uav_xs'][0: frame: interval]
        uav_y = env.position['all_uav_ys'][0: frame: interval]
        uav_x = [sublist[i] for sublist in uav_x]
        uav_y = [sublist[i] for sublist in uav_y]
        if uav_x and uav_y:  # Ensure the lists are not empty
            colors = [get_gradient_color('#E1FFFF', '#0000FF', frame, idx) for idx in range(len(uav_x))]
            uav_plots[i].set_offsets(np.column_stack([uav_x, uav_y]))
            uav_plots[i].set_color(colors)
            uav_search_patches[i].center = (uav_x[-1], uav_y[-1])
        else:
            print(f"Warning: UAV {i} position list is empty at frame {frame}.")

    for i in range(env.m_targets):
        target_x = env.position['all_target_xs'][0: frame: interval]
        target_y = env.position['all_target_ys'][0: frame: interval]
        target_x = [sublist[i] for sublist in target_x]
        target_y = [sublist[i] for sublist in target_y]
        if target_x and target_y:  # Ensure the lists are not empty
            colors = [get_gradient_color('#FFC0CB', '#DC143C', frame, idx) for idx in range(len(target_x))]
            target_plots[i].set_offsets(np.column_stack([target_x, target_y]))
            target_plots[i].set_color(colors)
        else:
            print(f"Warning: Target {i} position list is empty at frame {frame}.")

    text_str = (
        f"detected target num = {env.covered_target_num[frame]}\n"
        f"detected target rate = {env.covered_target_num[frame] / env.m_targets * 100:.2f}%"
    )

    # 清除之前的文本对象（如果存在）
    if text_obj is not None:
        text_obj.remove()

    # 绘制新的文本对象，没有边框，颜色为深蓝色
    text_obj = ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       color='black')


def draw_animation(config, env, num_steps, ep_num, frames=100):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-env.x_max / 3, env.x_max / 3 * 4)
    ax.set_ylim(-env.y_max / 3, env.y_max / 3 * 4)
    uav_plots = [ax.scatter([], [], marker='o', color='b', linestyle='None', s=2,alpha=1) for _ in range(env.n_uav)]
    target_plots = [ax.scatter([], [], marker='o', color='r', linestyle='None', s=3,alpha=1) for _ in range(env.m_targets)]
    uav_search_patches = [patches.Circle((0, 0), uav.dp, color='lightblue', alpha=0.2) for uav in env.uav_list]
    for patch in uav_search_patches:
        ax.add_patch(patch)

    save_dir = os.path.join(config["save_dir"], "frames")
    os.makedirs(save_dir, exist_ok=True)

    # Save frames at intervals of 5 num_steps
    step_interval = 5
    for frame in range(0, num_steps, step_interval):
        update(ax, env, uav_plots, target_plots, uav_search_patches, frame, frames, num_steps)
        plt.draw()
        plt.savefig(os.path.join(save_dir, f'frame_{frame:04d}.png'))
        plt.pause(0.001)  # Pause to ensure the plot updates visibly if needed

    plt.close(fig)

    # Generate MP4
    video_path = os.path.join(config["save_dir"], "animated", f'animated_plot_{ep_num + 1}.mp4')
    writer = imageio.get_writer(video_path, fps=5, codec='libx264', format='FFMPEG', pixelformat='yuv420p')

    for frame in range(0, num_steps, step_interval):
        frame_path = os.path.join(save_dir, f'frame_{frame:04d}.png')
        if os.path.exists(frame_path):
            img_array = resize_image(frame_path)
            writer.append_data(img_array)
    writer.close()

    # Optionally remove PNG files
    for frame in range(0, num_steps, step_interval):
        frame_path = os.path.join(save_dir, f'frame_{frame:04d}.png')
        if os.path.exists(frame_path):
            os.remove(frame_path)


def plot_reward_curve(config, return_list, name):
    plt.figure(figsize=(6, 6))
    plt.plot(return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Total Return')
    plt.title(name)
    plt.grid(True)
    plt.savefig(os.path.join(config["save_dir"], name + ".png"))
    # plt.show()

