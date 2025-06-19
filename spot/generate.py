import numpy as np
import cv2
import math
import random # 用于随机化参数
import os
from shapely.geometry import Polygon, Point # 导入 Shapely
import argparse
from glob import glob
from tqdm import tqdm
import json
from multiprocessing import Pool, cpu_count

# =====================
# 高斯光斑生成与遮挡主流程
# =====================

# ---------------------------------------------
# 新增：生成圆形高斯光斑的 mask
# ---------------------------------------------
"""
@param {int} height mask高度
@param {int} width mask宽度
@param {float} center_x 高斯中心x
@param {float} center_y 高斯中心y
@param {float} sigma 标准差
@param {float} intensity 峰值强度（可>1）
@return {np.ndarray} 单通道高斯mask，值域0~1
"""
def generate_circular_gaussian_mask(height, width, center_x, center_y, sigma, intensity=1.0):
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    gauss_mask = intensity * np.exp(-(dist_from_center ** 2) / (2 * sigma ** 2))
    gauss_mask = np.clip(gauss_mask, 0, 1)
    return gauss_mask.astype(np.float32)

# ---------------------------------------------
# 生成椭圆高斯光斑的 RGBA 图像
# ---------------------------------------------
"""
@param {tuple} spot_canvas_size_wh 光斑画布尺寸 (宽, 高)
@param {float} amplitude 高斯光斑峰值强度 (0~1)
@param {float} sigma_x x轴标准差
@param {float} sigma_y y轴标准差
@param {float} rotation_angle_deg 旋转角度(度)
@param {tuple} spot_color_rgb 光斑颜色 (R,G,B)
@return {np.ndarray} RGBA格式光斑图像 (H,W,4) 或 None
"""
def generate_gaussian_spot(spot_canvas_size_wh, amplitude, sigma_x, sigma_y, rotation_angle_deg, spot_color_rgb=(255, 255, 255)):
    if sigma_x <= 0 or sigma_y <= 0:
        return None
    width, height = int(round(spot_canvas_size_wh[0])), int(round(spot_canvas_size_wh[1]))
    if width <= 0 or height <= 0:
        return None
    center_x, center_y = width / 2.0, height / 2.0
    rotation_angle_rad = np.deg2rad(rotation_angle_deg)
    cos_theta = np.cos(rotation_angle_rad)
    sin_theta = np.sin(rotation_angle_rad)
    x_coords = np.arange(width)
    y_coords = np.arange(height)
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
    spot_rgba[..., 0] = spot_color_rgb[2]
    spot_rgba[..., 1] = spot_color_rgb[1]
    spot_rgba[..., 2] = spot_color_rgb[0]
    spot_rgba[..., 3] = alpha_channel
    return spot_rgba

# ---------------------------------------------
# 计算包含旋转椭圆的最小画布尺寸
# ---------------------------------------------
"""
@param {float} sigma_x x轴标准差
@param {float} sigma_y y轴标准差
@param {float} rotation_angle_deg 旋转角度(度)
@param {float} k sigma倍数(默认3.5)
@return {tuple} (画布宽, 画布高)
"""
def get_rotated_ellipse_canvas_size(sigma_x, sigma_y, rotation_angle_deg, k=3.5):
    if sigma_x <= 0 or sigma_y <= 0:
        return (10, 10)
    rotation_angle_rad = np.deg2rad(rotation_angle_deg)
    cos_theta = np.cos(rotation_angle_rad)
    sin_theta = np.sin(rotation_angle_rad)
    a = k * sigma_x
    b = k * sigma_y
    AABB_half_width = math.sqrt(a**2 * cos_theta**2 + b**2 * sin_theta**2)
    AABB_half_height = math.sqrt(a**2 * sin_theta**2 + b**2 * cos_theta**2)
    padding = 20
    canvas_width = 2 * AABB_half_width + padding
    canvas_height = 2 * AABB_half_height + padding
    return (int(round(canvas_width)), int(round(canvas_height)))

# ---------------------------------------------
# 将光斑叠加到背景图像
# ---------------------------------------------
"""
@param {np.ndarray} background_image_bgr 背景BGR图像
@param {np.ndarray} spot_rgba RGBA光斑图像
@param {int} top_left_x_on_bg 光斑左上角x
@param {int} top_left_y_on_bg 光斑左上角y
@return {np.ndarray} 叠加结果BGR图像
"""
def blend_spot_on_image(background_image_bgr, spot_rgba, top_left_x_on_bg, top_left_y_on_bg):
    bg_h, bg_w = background_image_bgr.shape[:2]
    spot_h, spot_w = spot_rgba.shape[:2]
    roi_x_start_bg = top_left_x_on_bg
    roi_y_start_bg = top_left_y_on_bg
    roi_x_end_bg = roi_x_start_bg + spot_w
    roi_y_end_bg = roi_y_start_bg + spot_h
    spot_crop_x_start = 0
    spot_crop_y_start = 0
    spot_crop_x_end = spot_w
    spot_crop_y_end = spot_h
    if roi_x_start_bg < 0:
        spot_crop_x_start = -roi_x_start_bg
        roi_x_start_bg = 0
    if roi_y_start_bg < 0:
        spot_crop_y_start = -roi_y_start_bg
        roi_y_start_bg = 0
    if roi_x_end_bg > bg_w:
        spot_crop_x_end = spot_w - (roi_x_end_bg - bg_w)
        roi_x_end_bg = bg_w
    if roi_y_end_bg > bg_h:
        spot_crop_y_end = spot_h - (roi_y_end_bg - bg_h)
        roi_y_end_bg = bg_h
    eff_overlay_w = roi_x_end_bg - roi_x_start_bg
    eff_overlay_h = roi_y_end_bg - roi_y_start_bg
    if eff_overlay_w <= 0 or eff_overlay_h <= 0:
        return background_image_bgr
    spot_to_overlay = spot_rgba[spot_crop_y_start:spot_crop_y_end, spot_crop_x_start:spot_crop_x_end]
    bg_roi = background_image_bgr[roi_y_start_bg:roi_y_end_bg, roi_x_start_bg:roi_x_end_bg]
    if bg_roi.shape[0] != spot_to_overlay.shape[0] or bg_roi.shape[1] != spot_to_overlay.shape[1]:
        return background_image_bgr
    spot_rgb = spot_to_overlay[:, :, :3]
    spot_alpha = spot_to_overlay[:, :, 3].astype(np.float32) / 255.0
    spot_alpha_3channel = np.stack([spot_alpha]*3, axis=-1)
    blended_roi = (spot_rgb.astype(np.float32) * spot_alpha_3channel) + \
                  (bg_roi.astype(np.float32) * (1.0 - spot_alpha_3channel))
    blended_roi = blended_roi.astype(np.uint8)
    result_image = background_image_bgr.copy()
    result_image[roi_y_start_bg:roi_y_end_bg, roi_x_start_bg:roi_x_end_bg] = blended_roi
    return result_image

# ---------------------------------------------
# 获取飞机OBB的中心和尺寸
# ---------------------------------------------
"""
@param {list} obb_coords_8_points 8点OBB坐标[x1,y1,...,x4,y4]
@return {tuple} (中心x, 中心y, 近似宽, 近似高)
"""
def get_aircraft_obb_properties(obb_coords_8_points):
    points = np.array(obb_coords_8_points).reshape(4, 2)
    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    try:
        rect = cv2.minAreaRect(points.astype(np.float32))
        approx_width, approx_height = rect[1]
    except Exception:
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
        approx_width = max_x - min_x
        approx_height = max_y - min_y
        if approx_width < 1: approx_width = 1
        if approx_height < 1: approx_height = 1
    return center_x, center_y, approx_width, approx_height

# ---------------------------------------------
# 创建椭圆多边形（用于OOAP计算）
# ---------------------------------------------
"""
@param {float} center_x 椭圆中心x
@param {float} center_y 椭圆中心y
@param {float} sigma_x x轴标准差
@param {float} sigma_y y轴标准差
@param {float} rotation_angle_deg 旋转角度
@param {float} k_sigma_level k倍sigma
@param {int} N_points 多边形点数
@return {Polygon} 椭圆多边形
"""
def create_ellipse_polygon(center_x, center_y, sigma_x, sigma_y, rotation_angle_deg, k_sigma_level=2.0, N_points=50):
    if sigma_x <=0 or sigma_y <= 0:
        return Polygon()
    a = k_sigma_level * sigma_x
    b = k_sigma_level * sigma_y
    if a <= 1e-3 or b <= 1e-3:
        return Polygon()
    t = np.linspace(0, 2 * np.pi, N_points, endpoint=False)
    ellipse_x_unrotated = a * np.cos(t)
    ellipse_y_unrotated = b * np.sin(t)
    rotation_angle_rad = np.deg2rad(rotation_angle_deg)
    cos_theta = np.cos(rotation_angle_rad)
    sin_theta = np.sin(rotation_angle_rad)
    ellipse_x_rotated = ellipse_x_unrotated * cos_theta - ellipse_y_unrotated * sin_theta + center_x
    ellipse_y_rotated = ellipse_x_unrotated * sin_theta + ellipse_y_unrotated * cos_theta + center_y
    ellipse_points = list(zip(ellipse_x_rotated, ellipse_y_rotated))
    try:
        ellipse_poly = Polygon(ellipse_points)
        if not ellipse_poly.is_valid:
            ellipse_poly = ellipse_poly.convex_hull
            if not ellipse_poly.is_valid or ellipse_poly.area < 1e-3:
                return Polygon()
        return ellipse_poly
    except Exception as e:
        return Polygon()

# ---------------------------------------------
# 计算 OOAP (光斑与飞机OBB的重叠面积百分比)
# ---------------------------------------------
"""
@param {list} aircraft_obb_pixel_coords 飞机OBB像素坐标
@param {dict} spot_params 光斑参数
@param {tuple} spot_center_on_bg_xy 光斑中心坐标
@param {float} k_sigma_level_for_ooap k倍sigma
@return {float} OOAP百分比
"""
def calculate_ooap(aircraft_obb_pixel_coords, spot_params, spot_center_on_bg_xy, k_sigma_level_for_ooap=2.0):
    try:
        aircraft_points = np.array(aircraft_obb_pixel_coords).reshape(4, 2)
        aircraft_poly = Polygon(aircraft_points)
        if not aircraft_poly.is_valid or aircraft_poly.area < 1e-3:
            return 0.0
        aircraft_area = aircraft_poly.area
        if aircraft_area < 1e-6:
            return 0.0
        spot_poly = create_ellipse_polygon(
            center_x=spot_center_on_bg_xy[0],
            center_y=spot_center_on_bg_xy[1],
            sigma_x=spot_params['sigma_x'],
            sigma_y=spot_params['sigma_y'],
            rotation_angle_deg=spot_params['rotation_deg'],
            k_sigma_level=k_sigma_level_for_ooap
        )
        if not spot_poly.is_valid or spot_poly.area < 1e-3:
            return 0.0
        intersection_poly = aircraft_poly.intersection(spot_poly)
        intersection_area = intersection_poly.area
        ooap = (intersection_area / aircraft_area) * 100.0
        return ooap
    except Exception as e:
        print(f"计算 OOAP 时发生错误: {e}")
        return -1.0

# ---------------------------------------------
# 为飞机目标生成高斯光斑遮挡并计算OOAP (此为核心调用函数)
# ---------------------------------------------
"""
@param {np.ndarray} background_img BGR图像数据
@param {list} aircraft_obb_coords 飞机OBB像素坐标 [x1,y1,...,x4,y4]
@param {float} ooap_k_sigma OOAP计算时椭圆边界k倍sigma
@param {dict} config 光斑生成参数配置（可选）
@return {tuple} (叠加遮挡后的BGR图像, OOAP百分比)
"""
def apply_occlusion_to_aircraft(background_img, aircraft_obb_coords, ooap_k_sigma=2.0, config=None):
    # 默认配置 (如果未提供)
    if config is None:
        config = {
            "placement_offset_range_factor": (-0.6, 0.6), # 偏移范围因子 (相对于飞机OBB尺寸)
            "sigma_scale_range": (0.2, 0.8),          # Sigma 缩放因子范围 (相对于飞机最大尺寸)
            "aspect_ratio_range": (0.5, 1.5),         # 椭圆长宽比范围
            "min_absolute_sigma": 7.0,                # Sigma 的最小绝对像素值
            "rotation_range_deg": (0, 360),           # 旋转角度范围
            "amplitude_range": (1.6, 2.5),            # 光斑强度范围
            "spot_color_rgb": (255, 255, 255),        # 光斑颜色
            "spot_shape": "ellipse"                  # 新增：光斑形状，可选 'ellipse' 或 'circle'
        }

    if background_img is None:
        return None, -1.0

    img_h, img_w = background_img.shape[:2]
    ac_center_x, ac_center_y, ac_width, ac_height = get_aircraft_obb_properties(aircraft_obb_coords)

    spot_params = {}

    # 1. 光斑中心位置（允许一定范围偏移）
    offset_x_pixels = random.uniform(config["placement_offset_range_factor"][0] * ac_width,
                                     config["placement_offset_range_factor"][1] * ac_width)
    offset_y_pixels = random.uniform(config["placement_offset_range_factor"][0] * ac_height,
                                     config["placement_offset_range_factor"][1] * ac_height)
    spot_params['center_x_on_bg'] = ac_center_x + offset_x_pixels
    spot_params['center_y_on_bg'] = ac_center_y + offset_y_pixels
    # clip到图像边界（允许略微超出）
    spot_params['center_x_on_bg'] = np.clip(spot_params['center_x_on_bg'], -img_w*0.2, img_w*1.2)
    spot_params['center_y_on_bg'] = np.clip(spot_params['center_y_on_bg'], -img_h*0.2, img_h*1.2)

    # 2. 光斑Sigma（基于飞机尺寸，允许一定比例）
    base_dim_for_sigma = max(ac_width, ac_height)
    if base_dim_for_sigma < 10: base_dim_for_sigma = 10
    sigma_scale = random.uniform(config["sigma_scale_range"][0], config["sigma_scale_range"][1])
    aspect_ratio = random.uniform(config["aspect_ratio_range"][0], config["aspect_ratio_range"][1])
    spot_params['sigma_x'] = max(sigma_scale * base_dim_for_sigma, config["min_absolute_sigma"])
    spot_params['sigma_y'] = max(spot_params['sigma_x'] * aspect_ratio, config["min_absolute_sigma"])

    # 3. 其他参数
    spot_params['rotation_deg'] = random.uniform(config["rotation_range_deg"][0], config["rotation_range_deg"][1])
    spot_params['amplitude'] = random.uniform(config["amplitude_range"][0], config["amplitude_range"][1])
    spot_params['color_rgb'] = config["spot_color_rgb"]

    # 4. 计算光斑画布尺寸
    canvas_w, canvas_h = get_rotated_ellipse_canvas_size(
        spot_params['sigma_x'], spot_params['sigma_y'], spot_params['rotation_deg']
    )
    if canvas_w <= 0 or canvas_h <=0:
        return background_img, 0.0

    # 5. 生成高斯光斑图像
    if config.get("spot_shape", "ellipse") == "circle":
        # 圆形高斯光斑（无旋转，sigma_x=sigma_y，忽略rotation）
        sigma = float(spot_params['sigma_x'])
        center_x = canvas_w / 2.0
        center_y = canvas_h / 2.0
        mask = generate_circular_gaussian_mask(canvas_h, canvas_w, center_x, center_y, sigma, spot_params['amplitude'])
        # 构造RGBA
        spot_rgba = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        spot_rgba[..., 0] = spot_params['color_rgb'][2]
        spot_rgba[..., 1] = spot_params['color_rgb'][1]
        spot_rgba[..., 2] = spot_params['color_rgb'][0]
        spot_rgba[..., 3] = (mask * 255).astype(np.uint8)
    else:
        # 椭圆+旋转
        spot_rgba = generate_gaussian_spot(
        spot_canvas_size_wh=(canvas_w, canvas_h),
        amplitude=spot_params['amplitude'],
        sigma_x=spot_params['sigma_x'],
        sigma_y=spot_params['sigma_y'],
        rotation_angle_deg=spot_params['rotation_deg'],
        spot_color_rgb=spot_params['color_rgb']
    )
    if spot_rgba is None:
        return background_img, 0.0

    # 6. 计算光斑在背景图像上的左上角放置位置
    top_left_x = int(round(spot_params['center_x_on_bg'] - canvas_w / 2.0))
    top_left_y = int(round(spot_params['center_y_on_bg'] - canvas_h / 2.0))
    occluded_image = blend_spot_on_image(background_img, spot_rgba, top_left_x, top_left_y)

    # 7. 计算 OOAP
    current_ooap = calculate_ooap(
        aircraft_obb_coords,
        spot_params,
        spot_center_on_bg_xy=(spot_params['center_x_on_bg'], spot_params['center_y_on_bg']),
        k_sigma_level_for_ooap=ooap_k_sigma
    )
    return occluded_image, current_ooap

# ---------------------------------------------
# 新增: 在图像上应用随机背景光斑 (负样本)
# ---------------------------------------------
"""
@param {np.ndarray} background_img BGR图像数据
@param {dict} config 背景光斑的参数配置
@return {np.ndarray} 叠加了背景光斑的BGR图像
"""
def apply_random_background_spot(background_img, config):
    if background_img is None:
        return background_img

    img_h, img_w = background_img.shape[:2]
    spot_params = {}

    # 1. 光斑中心位置 (完全随机)
    spot_params['center_x_on_bg'] = random.uniform(0, img_w)
    spot_params['center_y_on_bg'] = random.uniform(0, img_h)

    # 2. 光斑Sigma (在绝对像素范围内随机)
    aspect_ratio = random.uniform(config["aspect_ratio_range"][0], config["aspect_ratio_range"][1])
    spot_params['sigma_x'] = random.uniform(config["sigma_range_px"][0], config["sigma_range_px"][1])
    spot_params['sigma_y'] = spot_params['sigma_x'] * aspect_ratio

    # 3. 其他参数
    spot_params['rotation_deg'] = random.uniform(config["rotation_range_deg"][0], config["rotation_range_deg"][1])
    spot_params['amplitude'] = random.uniform(config["amplitude_range"][0], config["amplitude_range"][1])
    spot_params['color_rgb'] = config["spot_color_rgb"]

    # 4. 计算光斑画布尺寸
    canvas_w, canvas_h = get_rotated_ellipse_canvas_size(
        spot_params['sigma_x'], spot_params['sigma_y'], spot_params['rotation_deg']
    )
    if canvas_w <= 0 or canvas_h <= 0:
        return background_img

    # 5. 生成高斯光斑图像 (只支持椭圆)
    spot_rgba = generate_gaussian_spot(
        spot_canvas_size_wh=(canvas_w, canvas_h),
        amplitude=spot_params['amplitude'],
        sigma_x=spot_params['sigma_x'],
        sigma_y=spot_params['sigma_y'],
        rotation_angle_deg=spot_params['rotation_deg'],
        spot_color_rgb=spot_params['color_rgb']
    )
    if spot_rgba is None:
        return background_img

    # 6. 计算光斑在背景图像上的左上角放置位置并叠加
    top_left_x = int(round(spot_params['center_x_on_bg'] - canvas_w / 2.0))
    top_left_y = int(round(spot_params['center_y_on_bg'] - canvas_h / 2.0))
    occluded_image = blend_spot_on_image(background_img, spot_rgba, top_left_x, top_left_y)
    
    return occluded_image

# ---------------------------------------------
# 新增: 多进程工作函数
# ---------------------------------------------
def process_single_image(args_tuple):
    """
    处理单张图片的所有逻辑，包括读取、为每个目标生成多种遮挡并保存。
    设计为与 multiprocessing.Pool 配合使用。
    
    @param {tuple} args_tuple 包含所有必要参数的元组:
        - image_path (str): 待处理图片的完整路径。
        - data_dir (str): 数据集根目录。
        - output_dir (str): 输出根目录。
        - config (dict): 光斑生成参数。
        - k_sigma (float): 用于计算OOAP的k西格玛值。
        - bg_spots (int): 背景光斑(负样本)的数量。
    """
    image_path, data_dir, output_dir, config, k_sigma, bg_spots = args_tuple
    try:
        # --- 1. 构建路径 ---
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        relative_path = os.path.relpath(os.path.dirname(image_path), os.path.join(data_dir, 'images'))
        
        label_path = os.path.join(data_dir, 'labels', relative_path, f"{base_name}.txt")
        output_image_subdir = os.path.join(output_dir, 'images', relative_path)
        output_label_subdir = os.path.join(output_dir, 'labels', relative_path)
        output_occlusion_subdir = os.path.join(output_dir, 'occlusions', relative_path)
        
        os.makedirs(output_image_subdir, exist_ok=True)
        os.makedirs(output_label_subdir, exist_ok=True)
        os.makedirs(output_occlusion_subdir, exist_ok=True)
        
        # --- 2. 读取图像 ---
        background_img = cv2.imread(image_path)
        if background_img is None:
            print(f"\n警告: 无法加载图像，跳过: {image_path}")
            return
        img_h, img_w = background_img.shape[:2]

        # --- 3. 读取和解析标签 (如果标签文件存在) ---
        aircraft_to_process = []
        original_label_lines = []
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    original_label_lines = f.readlines()
                    for line_idx, line in enumerate(original_label_lines):
                        parts = line.strip().split()
                        if len(parts) == 9 and int(parts[0]) == 0: # DOTA飞机类别为0
                            norm_coords = np.array([float(x) for x in parts[1:]])
                            pixel_coords = np.zeros_like(norm_coords)
                            pixel_coords[0::2] = norm_coords[0::2] * img_w
                            pixel_coords[1::2] = norm_coords[1::2] * img_h
                            aircraft_to_process.append({
                                'coords': pixel_coords.tolist(),
                                'line_index': line_idx
                            })
            except Exception as e:
                print(f"\n错误: 读取或解析标签文件 {label_path} 时失败: {e}")
                return # 如果标签损坏，则跳过

        # --- 4. 累积应用遮挡 ---
        final_occluded_image = background_img.copy()
        all_occlusions_metadata = []

        # 4.1 为飞机目标添加遮挡
        if aircraft_to_process:
            for aircraft_data in aircraft_to_process:
                final_occluded_image, ooap_value = apply_occlusion_to_aircraft(
                    final_occluded_image,
                    aircraft_data['coords'],
                    ooap_k_sigma=k_sigma,
                    config=config
                )
                if ooap_value >= 0:
                    all_occlusions_metadata.append({
                        "occluded_object_line_index": aircraft_data['line_index'],
                        "ooap_percent": round(ooap_value, 2),
                        "ooap_normalized": round(ooap_value / 100.0, 6)
                    })
        
        # 4.2 添加随机背景光斑 (对所有图片都执行)
        if bg_spots > 0:
            bg_spot_config = {
                "sigma_range_px": (15, 60),
                "aspect_ratio_range": (0.5, 1.5),
                "rotation_range_deg": (0, 360),
                "amplitude_range": (1.6, 2.5),
                "spot_color_rgb": (255, 255, 255),
            }
            for _ in range(bg_spots):
                final_occluded_image = apply_random_background_spot(final_occluded_image, bg_spot_config)

        # --- 5. 保存所有结果 ---
        output_image_filename = f"{base_name}.jpg"
        output_label_filename = f"{base_name}.txt"
        output_occlusion_filename = f"{base_name}.json"

        # 5.1 保存处理后的图像 (以高质量JPG格式)
        output_image_path = os.path.join(output_image_subdir, output_image_filename)
        cv2.imwrite(output_image_path, final_occluded_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 5.2 保存标签文件 (如果无原始标签，则创建空文件)
        output_label_path = os.path.join(output_label_subdir, output_label_filename)
        with open(output_label_path, 'w') as f:
            if original_label_lines:
                f.writelines(original_label_lines)
            else:
                f.write("") # 确保为无标签的图片创建空的txt文件

        # 5.3 保存遮挡元数据 (如果存在)
        if all_occlusions_metadata:
            output_occlusion_path = os.path.join(output_occlusion_subdir, output_occlusion_filename)
            with open(output_occlusion_path, 'w') as f:
                json.dump(all_occlusions_metadata, f, indent=4)

    except Exception as e:
        print(f"\n错误: 处理文件 {image_path} 时发生严重异常: {e}")


# ---------------------------------------------
# 主流程：为每个飞机目标生成多样化遮挡样本并计算OOAP
# ---------------------------------------------
def main():
    """
    主函数，用于解析命令行参数并批量处理整个数据集。
    """
    parser = argparse.ArgumentParser(
        description="【多进程高效版】为数据集中每张图片的飞机目标添加高斯光斑遮挡，并为每张原图生成一张合成图。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
使用示例:
python spot/generate.py --data-dir E:/datasets/DOTA_planes --output-dir E:/datasets/DOTA_planes_occluded_multi --bg-spots 5

新版说明:
- 此版本为每个输入图像生成一个对应的输出图像。
- 图像中的所有飞机目标都会被依次添加一个随机遮挡。
- 新增 `--bg-spots` 参数，用于在图片上随机添加N个与飞机无关的背景光斑，作为负样本，以提升模型泛化能力。
- 不再有'attempts'参数，数据不会爆炸式增长。

输出目录结构:
<output-dir>/
├── images/
│   └── (存放带多个遮挡的合成图片, 与原图一一对应)
├── labels/
│   └── (存放复制的原始标签, 与原图一一对应)
└── occlusions/
    └── (存放.json文件, 每个文件包含对应图片上所有遮挡点的信息列表)
"""
    )
    parser.add_argument('--data-dir', type=str, required=True, help='数据集根目录路径。')
    parser.add_argument('--output-dir', type=str, required=True, help='处理后图像的输出根目录。')
    parser.add_argument('--bg-spots', type=int, default=3, help='每张图片额外添加的背景干扰光斑(负样本)数量。')
    args = parser.parse_args()

    # ========== 光斑生成参数配置 ==========
    custom_spot_config = {
        "placement_offset_range_factor": (-0.6, 0.6),
        "sigma_scale_range": (0.1, 0.5),
        "aspect_ratio_range": (0.4, 1.6),
        "min_absolute_sigma": 8.0,
        "rotation_range_deg": (0, 360),
        "amplitude_range": (1.6, 2.5),
        "spot_color_rgb": (255, 255, 255),
        "spot_shape": "ellipse"
    }

    K_SIGMA_FOR_OOAP_BOUNDARY = 1.8

    # ========== 查找所有图片文件 ==========
    print(f"正在从 '{args.data_dir}' 搜索图片...")
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(args.data_dir, 'images', '**', ext), recursive=True))

    if not image_paths:
        print(f"错误: 在 '{os.path.join(args.data_dir, 'images')}' 目录下没有找到任何支持的图片文件。请检查路径和文件格式。")
        return
        
    # ========== 设置多进程任务 ==========
    tasks = [(path, args.data_dir, args.output_dir, custom_spot_config, K_SIGMA_FOR_OOAP_BOUNDARY, args.bg_spots) for path in image_paths]
    
    num_processes = cpu_count()
    print(f"找到 {len(image_paths)} 张图片。将使用 {num_processes} 个CPU核心进行并行处理...")

    # ========== 使用进程池执行任务 ==========
    with Pool(processes=num_processes) as pool:
        # 使用 tqdm 显示总体进度
        # imap_unordered可以更快地得到结果，即使任务完成顺序不同
        list(tqdm(pool.imap_unordered(process_single_image, tasks), total=len(tasks), desc="批量处理进度"))


    print("\n批量处理完成！")

if __name__ == "__main__":
    main()

