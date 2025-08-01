from calendar import c
import numpy as np
from PIL import Image
import noise

def create_pit_image(width=256, height=256, pit_start=100, pit_end=156):
    """
    生成一张带有黑色条带（坑）的白色灰度图。

    Args:
        width (int): 图片宽度。
        height (int): 图片高度。
        pit_start (int): 坑在宽度上的起始像素位置。
        pit_end (int): 坑在宽度上的结束像素位置。
        filename (str): 输出图片的文件名。
    """
    # 创建一个全白的 NumPy 数组 (255 代表白色)
    # uint8 是 0-255 的无符号整数
    terrain_array = np.full((height, width), 255, dtype=np.uint8)

    # 在指定区域填充黑色 (0 代表黑色) 来制造“坑”
    terrain_array[:, pit_start:pit_end] = 0

    # 从 NumPy 数组创建图片对象
    img = Image.fromarray(terrain_array, 'L')  # 'L' 表示灰度模式

    # 保存图片
    img.save("assets/mujoco_terrains/pit_terrain.png")
    img.save("/tmp/mujoco_terrains/pit_terrain.png")
    print(f"成功生成地形图片")

import numpy as np
from PIL import Image
import noise

def create_climbing_hfield_section(
    width=4096, # 使用更高分辨率以获得更平滑的过渡
    height=256,
    mountain_start_ratio=0.5125,
    mountain_end_ratio=0.5625,
    slope_start_height=0.0,
    slope_end_height=1.0,
    noise_scale=0.2,
    noise_octaves=5,
    noise_persistence=0.5,
    noise_lacunarity=2.0,
    holds_config=None,
    ):
    """
    生成一个只在指定区间内有山体、其余部分为平地的高度图。
    """
    # 1. 计算山体区间的像素索引
    start_x_px = int(width * mountain_start_ratio)
    end_x_px = int(width * mountain_end_ratio)
    mountain_width_px = end_x_px - start_x_px

    if mountain_width_px <= 0:
        raise ValueError("mountain_start_ratio must be less than mountain_end_ratio.")

    # 2. 【只在山体区间内】生成剖面
    # a. 基础斜坡
    mountain_slope = np.linspace(slope_start_height, slope_end_height, mountain_width_px)
    
    # b. 噪音
    mountain_noise = np.zeros(mountain_width_px)
    for x in range(mountain_width_px):
        mountain_noise[x] = noise.pnoise1(
            (start_x_px + x) * noise_scale, # 使用全局坐标以保证噪音连续性
            octaves=noise_octaves,
            persistence=noise_persistence,
            lacunarity=noise_lacunarity,
            base=0
        )
    mountain_noise = (mountain_noise - np.min(mountain_noise)) / (np.max(mountain_noise) - np.min(mountain_noise))
    mountain_noise *= (1.0 - np.max(mountain_slope))

    # c. 结合斜坡和噪音
    mountain_profile = mountain_slope + mountain_noise

    # d. 添加攀岩点 (只在山体区域)
    if holds_config is None:
        holds_config = [(0.05, 8, 0.2), (0.15, 8, 0.3), (0.3, 12, 0.32), (0.45, 9, 0.35), (0.6, 8, 0.28), (0.75, 9, 0.25), (0.9, 7, 0.3)]

    X_mountain = np.arange(mountain_width_px)
    for hold_ratio_in_mountain, h_radius, h_height in holds_config:
        h_x = int(mountain_width_px * hold_ratio_in_mountain)
        dist_from_center = np.abs(X_mountain - h_x)
        mask = np.exp(-(dist_from_center**2 / (2 * h_radius**2)))
        mountain_profile += mask * h_height

    # 3. 组合成完整的1D剖面
    final_profile_1d = np.zeros(width)
    
    # a. 山体前方的平地 (高度与山体起点相同)
    start_height = mountain_profile[0]
    final_profile_1d[0:start_x_px] = start_height
    
    # b. 山体部分
    final_profile_1d[start_x_px:end_x_px] = mountain_profile
    
    # c. 山体后方的平地 (高度与山体终点相同)
    end_height = mountain_profile[-1]
    final_profile_1d[end_x_px:] = end_height

    # 4. 归一化、平铺并保存
    final_profile_1d = (final_profile_1d - np.min(final_profile_1d)) / (np.max(final_profile_1d) - np.min(final_profile_1d))
    hfield_data_2d = np.tile(final_profile_1d, (height, 1))
    hfield_data_2d_uint8 = (hfield_data_2d * 255).astype(np.uint8)
    image = Image.fromarray(hfield_data_2d_uint8, 'L')
    image.save("assets/mujoco_terrains/climbing_hfield_section.png")
    image.save("/tmp/mujoco_terrains/climbing_hfield_section.png")
    print(f"成功创建分段式高度图")

def create_sand_hfield(
    width=4096, # 使用更高分辨率以获得更平滑的过渡
    height=256,
    sand_start_ratio=0.5125,
    sand_end_ratio=1,
    sand_min_height=0.,
    sand_max_height=1.,
    sand_unit=40
    ):
    start_x_px = int(width * sand_start_ratio)
    end_x_px = int(width * sand_end_ratio)
    sand_width_px = end_x_px - start_x_px

    if sand_width_px <= 0:
        raise ValueError("sand_start_ratio must be less than sand_end_ratio.")

    hfield_data_2d = np.ones((height, width), dtype=np.float32) * 0.5
    hfield_data_2d[:, start_x_px:end_x_px] = np.sin(np.linspace(0, 2 * np.pi * sand_width_px / sand_unit, sand_width_px)) * (sand_max_height - sand_min_height) / 2 + (sand_max_height + sand_min_height) / 2


    hfield_img = (hfield_data_2d * 255).astype(np.uint8)
    image = Image.fromarray(hfield_img, 'L')
    image.save("assets/mujoco_terrains/sand_hfield.png")
    image.save("/tmp/mujoco_terrains/sand_hfield.png")

if __name__ == '__main__':
    create_pit_image(800, 800, 410, 415)
    # create_climbing_hfield_section()
    # create_sand_hfield()