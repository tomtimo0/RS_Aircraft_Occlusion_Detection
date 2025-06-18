import os
import cv2
import numpy as np
import math

# ==============================================================================
# 脚本目标:
# 1. 整合两种高斯光斑生成函数于一处。
# 2. 分别在黑色背景上生成光斑，并保存为图片，用于直观对比。
#    - 方法一: 来自 generate.py (椭圆、可旋转)
#    - 方法二: 来自 GaussianSpot_add_combine.py (圆形、不可旋转)
# ==============================================================================


# ==============================================================================
# 方法一: 来自 generate.py (椭圆、可旋转)
# ==============================================================================

def get_rotated_ellipse_canvas_size(sigma_x, sigma_y, rotation_angle_deg, k=3.5):
    """
    计算包含旋转椭圆的最小画布尺寸 (源自 generate.py)。

    @param {float} sigma_x x轴标准差
    @param {float} sigma_y y轴标准差
    @param {float} rotation_angle_deg 旋转角度(度)
    @param {float} k sigma倍数(默认3.5)
    @return {tuple} (画布宽, 画布高)
    """
    if sigma_x <= 0 or sigma_y <= 0: return (10, 10)
    rotation_angle_rad = np.deg2rad(rotation_angle_deg)
    cos_theta, sin_theta = np.cos(rotation_angle_rad), np.sin(rotation_angle_rad)
    a, b = k * sigma_x, k * sigma_y
    AABB_half_width = math.sqrt(a**2 * cos_theta**2 + b**2 * sin_theta**2)
    AABB_half_height = math.sqrt(a**2 * sin_theta**2 + b**2 * cos_theta**2)
    padding = 20
    return (int(round(2 * AABB_half_width + padding)), int(round(2 * AABB_half_height + padding)))

def generate_elliptical_gaussian_spot_rgba(spot_canvas_size_wh, amplitude, sigma_x, sigma_y, rotation_angle_deg, spot_color_rgb=(255, 255, 255)):
    """
    生成椭圆高斯光斑的 RGBA 图像 (源自 generate.py)。

    @return {np.ndarray} RGBA格式光斑图像 (H,W,4) 或 None
    """
    if sigma_x <= 0 or sigma_y <= 0: return None
    width, height = int(round(spot_canvas_size_wh[0])), int(round(spot_canvas_size_wh[1]))
    if width <= 0 or height <= 0: return None
    
    center_x, center_y = width / 2.0, height / 2.0
    rotation_angle_rad = np.deg2rad(rotation_angle_deg)
    cos_theta, sin_theta = np.cos(rotation_angle_rad), np.sin(rotation_angle_rad)
    
    x_coords, y_coords = np.arange(width), np.arange(height)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    x_shifted = xx - center_x
    y_shifted = yy - center_y
    x_prime = x_shifted * cos_theta + y_shifted * sin_theta
    y_prime = -x_shifted * sin_theta + y_shifted * cos_theta
    
    sigma_x_sq = sigma_x**2 if sigma_x > 1e-6 else 1e-6
    sigma_y_sq = sigma_y**2 if sigma_y > 1e-6 else 1e-6
    
    exponent = -((x_prime**2) / (2 * sigma_x_sq) + (y_prime**2) / (2 * sigma_y_sq))
    intensity = amplitude * np.exp(exponent)
    
    alpha_channel = np.clip(intensity, 0, 1) * 255
    alpha_channel = alpha_channel.astype(np.uint8)
    
    spot_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    spot_rgba[..., 0:3] = spot_color_rgb
    spot_rgba[..., 3] = alpha_channel
    
    return spot_rgba

# ==============================================================================
# 方法二: 来自 GaussianSpot_add_combine.py (圆形)
# ==============================================================================

def generate_circular_spot_on_black(canvas_size_wh, radius, intensity):
    """
    在黑色画布上生成一个圆形高斯光斑 (逻辑源自 GaussianSpot_add_combine.py)。

    @param {tuple} canvas_size_wh 画布尺寸 (宽, 高)
    @param {float} radius 光斑半径 (用于计算sigma)
    @param {float} intensity 光斑峰值强度
    @return {np.ndarray} BGR格式的带有光斑的黑色画布
    """
    width, height = canvas_size_wh
    center_x, center_y = width / 2, height / 2
    sigma = radius / 2.0  # 按原脚本逻辑

    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    gauss_mask = np.exp(-(dist_from_center**2 / (2 * sigma**2)))
    alpha_mask = np.clip(intensity * gauss_mask, 0, 1) # 最终透明度
    
    # 创建黑色背景和白色光斑层
    black_background = np.zeros((height, width, 3), dtype=np.float32)
    white_glare_layer = np.ones((height, width, 3), dtype=np.float32) * 255
    
    # Alpha混合
    alpha_mask_3ch = alpha_mask[..., np.newaxis]
    blended_image = black_background * (1 - alpha_mask_3ch) + white_glare_layer * alpha_mask_3ch
    
    return np.clip(blended_image, 0, 255).astype(np.uint8)


# ==============================================================================
# 主执行流程: 生成并保存对比图像
# ==============================================================================

if __name__ == '__main__':
    # --- 公共配置 ---
    OUTPUT_DIR = "spot_comparison_output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    CANVAS_SIZE_WH = (400, 400) # 为圆形光斑指定一个固定大小的画布
    
    print(f"开始生成高斯光斑对比图，将保存到 '{OUTPUT_DIR}' 文件夹...")

    # --- 1. 生成椭圆光斑 (方法一) ---
    print("  (1/2) 生成椭圆、可旋转光斑 (来自 generate.py)...")
    ellipse_params = {
        'sigma_x': 80.0,
        'sigma_y': 40.0,
        'rotation_angle_deg': 30,
        'amplitude': 2.5,
        'spot_color_rgb': (255, 255, 255)
    }
    
    # 根据椭圆参数自动计算合适的画布大小
    ellipse_canvas_size = get_rotated_ellipse_canvas_size(
        ellipse_params['sigma_x'], ellipse_params['sigma_y'], ellipse_params['rotation_angle_deg']
    )
    
    # 生成带透明通道的椭圆光斑
    spot_rgba = generate_elliptical_gaussian_spot_rgba(
        spot_canvas_size_wh=ellipse_canvas_size,
        **ellipse_params
    )

    if spot_rgba is not None:
        # 创建一个黑色背景
        black_bg_for_ellipse = np.zeros((ellipse_canvas_size[1], ellipse_canvas_size[0], 3), dtype=np.uint8)
        
        # 将RGBA光斑图像覆盖到黑色背景上
        # 简单实现alpha混合
        alpha_mask = (spot_rgba[:, :, 3] / 255.0)[..., np.newaxis]
        spot_rgb = spot_rgba[:, :, :3]
        
        blended_ellipse = black_bg_for_ellipse * (1 - alpha_mask) + spot_rgb * alpha_mask
        final_ellipse_image = blended_ellipse.astype(np.uint8)

        # 保存图像
        output_path_ellipse = os.path.join(OUTPUT_DIR, "spot_1_elliptical_rotated.png")
        cv2.imwrite(output_path_ellipse, final_ellipse_image)
        print(f"      -> 已保存: {output_path_ellipse}")
    else:
        print("      -> 生成椭圆光斑失败。")

    # --- 2. 生成圆形光斑 (方法二) ---
    print("  (2/2) 生成圆形光斑 (来自 GaussianSpot_add_combine.py)...")
    circle_params = {
        'radius': 100.0,
        'intensity': 2.5  # 强度可以大于1，最终会clip
    }

    circular_spot_image = generate_circular_spot_on_black(
        canvas_size_wh=CANVAS_SIZE_WH,
        **circle_params
    )
    
    # 保存图像
    output_path_circular = os.path.join(OUTPUT_DIR, "spot_2_circular_combined_style.png")
    cv2.imwrite(output_path_circular, circular_spot_image)
    print(f"      -> 已保存: {output_path_circular}")

    print("\n处理完成！") 