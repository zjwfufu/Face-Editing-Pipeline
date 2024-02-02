import os
import imageio
from PIL import ImageDraw
import numpy as np


def get_file_list(folder_path, is_abspath=True, extensions=['jpg', 'png', 'bmp', 'jpeg']):
    file_list = []
    file_name = None
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for file in filenames:
            ext = file.split('.')[-1]
            file_name = file.split('.')[0]
            if ext in extensions:
                if is_abspath:
                    file_list.append(os.path.join(dirpath, file))
                else:
                    file_list.append(file)

    if file_name is not None and file_name.isdigit():
        file_list.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        return file_list
    elif file_name is not None and (file_name.split('_')[-1]).isdigit():
        file_list.sort(key=lambda x: int((os.path.basename(x).split('_')[-1]).split('.')[0]))
        return file_list
    else:
        return sorted(file_list)
    

def save_snap_shot(snap_shot, write_dir):
    video = imageio.get_writer(write_dir, mode='I', fps=10, codec='libx264', bitrate='16M')
    for frame in snap_shot:
        video.append_data(frame)
    video.close()


def save_transition_video(orig_img, paste_img, write_dir):
    width, height = orig_img.size

    frames_per_second = 10

    draw = ImageDraw.Draw(orig_img)

    frames = []
    for i in range(0, width, 10):
        draw.line([(i, 0), (i, height)], fill=(0, 0, 0), width=4)
        
        current_frame = paste_img.copy()
        current_frame.paste(orig_img.crop((i, 0, width, height)), (i, 0))
        frames.append(current_frame.copy())

    with imageio.get_writer(write_dir, fps=frames_per_second, macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(np.array(frame))