from moviepy.editor import VideoFileClip

def convert_mp4_to_gif(input_path, output_path):
    # 加载视频文件
    clip = VideoFileClip(input_path)
    
    # 设置GIF的帧率，可以根据需要调整
    clip = clip.set_fps(10)

    # 转换为GIF并保存
    clip.write_gif(output_path)

# 调用函数，指定输入视频和输出GIF的路径
convert_mp4_to_gif("C:\\Users\\12920\\Desktop\\figma.mp4", "C:\\Users\\12920\\Desktop\\figma.gif")
