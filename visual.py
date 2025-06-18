import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import cv2
import os
from shapely.geometry import Polygon, Point #导入 Shapely
import math
import random


# 初始化全局变量
center_x, center_y = 320, 320
image_path = ""
txt_file_path = ""

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
# 为飞机目标生成高斯光斑遮挡并计算OOAP
# ---------------------------------------------
"""
@param {str} image_path 图像文件路径
@param {list} aircraft_obb_coords 飞机OBB像素坐标 [x1,y1,...,x4,y4]
@param {float} ooap_k_sigma OOAP计算时椭圆边界k倍sigma
@param {dict} config 光斑生成参数配置（可选）
@return {tuple} (叠加遮挡后的BGR图像, OOAP百分比)
"""
def apply_occlusion_to_aircraft(image_path, aircraft_obb_coords, ooap_k_sigma=2.0, config=None, center_x_on_bg=None, center_y_on_bg=None, sigma_x_override=None, sigma_y_override=None):
    # 默认配置 (如果未提供)
    if config is None:
        config = {
            "placement_offset_range_factor": (-0.6, 0.6), # 偏移范围因子 (相对于飞机OBB尺寸)
            "sigma_scale_range": (0.2, 0.8),          # Sigma 缩放因子范围 (相对于飞机最大尺寸)
            "aspect_ratio_range": (0.5, 1.5),         # 椭圆长宽比范围
            "min_absolute_sigma": 7.0,                # Sigma 的最小绝对像素值
            "rotation_range_deg": (0, 360),           # 旋转角度范围
            "amplitude_range": (0.9, 1.0),            # 光斑强度范围
            "spot_color_rgb": (255, 255, 255)         # 光斑颜色
        }
    
    if os.path.exists(image_path):
        print("文件存在")
    else:
        print("文件不存在，请检查路径")


    background_img = cv2.imread(image_path)

    if background_img is None:
        print("图像读取失败，请检查文件格式或内容")
        return None, -1.0

    img_h, img_w = background_img.shape[:2]
    ac_center_x, ac_center_y, ac_width, ac_height = get_aircraft_obb_properties(aircraft_obb_coords)

    spot_params = {}

    # 1. 光斑中心位置（使用传入值或默认值）
    if center_x_on_bg is not None and center_y_on_bg is not None:
        spot_params['center_x_on_bg'] = center_x_on_bg
        spot_params['center_y_on_bg'] = center_y_on_bg
    else:
        offset_x_pixels = random.uniform(config["placement_offset_range_factor"][0] * ac_width,
                                         config["placement_offset_range_factor"][1] * ac_width)
        offset_y_pixels = random.uniform(config["placement_offset_range_factor"][0] * ac_height,
                                         config["placement_offset_range_factor"][1] * ac_height)
        spot_params['center_x_on_bg'] = ac_center_x + offset_x_pixels
        spot_params['center_y_on_bg'] = ac_center_y + offset_y_pixels
        # clip到图像边界（允许略微超出）
        spot_params['center_x_on_bg'] = np.clip(spot_params['center_x_on_bg'], -img_w*0.2, img_w*1.2)
        spot_params['center_y_on_bg'] = np.clip(spot_params['center_y_on_bg'], -img_h*0.2, img_h*1.2)

    # 2. 光斑Sigma（使用传入值或默认值）
    if sigma_x_override is not None and sigma_y_override is not None:
        spot_params['sigma_x'] = sigma_x_override
        spot_params['sigma_y'] = sigma_y_override
    else:
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
    if canvas_w <= 0 or canvas_h <= 0:
        return background_img, 0.0

    # 5. 生成高斯光斑图像
    gaussian_spot_img = generate_gaussian_spot(
        spot_canvas_size_wh=(canvas_w, canvas_h),
        amplitude=spot_params['amplitude'],
        sigma_x=spot_params['sigma_x'],
        sigma_y=spot_params['sigma_y'],
        rotation_angle_deg=spot_params['rotation_deg'],
        spot_color_rgb=spot_params['color_rgb']
    )
    if gaussian_spot_img is None:
        return background_img, 0.0

    # 6. 计算光斑在背景图像上的左上角放置位置
    top_left_x = int(round(spot_params['center_x_on_bg'] - canvas_w / 2.0))
    top_left_y = int(round(spot_params['center_y_on_bg'] - canvas_h / 2.0))
    occluded_image = blend_spot_on_image(background_img, gaussian_spot_img, top_left_x, top_left_y)

    # 7. 计算 OOAP
    current_ooap = calculate_ooap(
        aircraft_obb_coords,
        spot_params,
        spot_center_on_bg_xy=(spot_params['center_x_on_bg'], spot_params['center_y_on_bg']),
        k_sigma_level_for_ooap=ooap_k_sigma
    )
    return occluded_image, current_ooap


def predict_image(image_path):
    try:
        background_img=Image.open(image_path)
        tk_image = display_image(background_img, image_label)
        image_label.image = tk_image  # 保持对图像的引用，防止被垃圾回收
        
        # 获取参数
        try:
            center_x = int(center_x_entry.get())
            center_y = int(center_y_entry.get())
            radius = int(radius_entry.get())
        except ValueError:
            return
        
        K_SIGMA_FOR_OOAP_BOUNDARY = 1.8  # OOAP计算时椭圆边界k倍sigma

        # 自定义光斑生成参数配置
        custom_spot_config = {
            "placement_offset_range_factor": (-0.6, 0.6),  # 光斑中心偏移范围
            "sigma_scale_range": (0.1, 0.4),  # 光斑sigma范围
            "aspect_ratio_range": (0.4, 1.6),
            "min_absolute_sigma": 8.0,
            "rotation_range_deg": (0, 360),
            "amplitude_range": (1.6, 2.5),
            "spot_color_rgb": (255, 255, 255),
            "spot_shape": "ellipse"
        }

        # 获取图像尺寸
        img_h, img_w = background_img.size

        # 读取OBB标签
        aircraft_pixel_obbs = []
        try:
            with open(txt_file_path, 'r') as f:
                lines = f.readlines()
                for line_num, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) == 9:
                        class_idx = int(parts[0])
                        if class_idx == 0:  # 只处理飞机类别
                            norm_coords = np.array([float(x) for x in parts[1:]])
                            pixel_coords = np.zeros_like(norm_coords)
                            pixel_coords[0::2] = norm_coords[0::2] * img_w
                            pixel_coords[1::2] = norm_coords[1::2] * img_h
                            aircraft_pixel_obbs.append(pixel_coords.tolist())
        except FileNotFoundError:
            print(f"错误：找不到标签文件 {txt_file_path}")
        except Exception as e:
            print(f"解析标签文件 {txt_file_path} 时出错: {e}")

        if not aircraft_pixel_obbs:
            print(f"在 {txt_file_path} 中没有找到飞机目标的有效标注。")

        print(f"从标签文件中解析得到 {len(aircraft_pixel_obbs)} 个飞机 OBB。")

        # 计算 sigma_x 和 sigma_y
        sigma_x = radius
        sigma_y = radius

        # 为每个飞机目标生成遮挡并计算 OOAP
        max_ooap = 0.0
        for i, obb_coords_for_one_plane in enumerate(aircraft_pixel_obbs):
            image_with_gaussian, original_ooap = apply_occlusion_to_aircraft(
                image_path,
                obb_coords_for_one_plane,
                ooap_k_sigma=K_SIGMA_FOR_OOAP_BOUNDARY,
                config=custom_spot_config,
                center_x_on_bg=center_x,
                center_y_on_bg=center_y,
                sigma_x_override=sigma_x,
                sigma_y_override=sigma_y
            )
            if original_ooap > max_ooap:
                max_ooap = original_ooap

        # 保存生成高斯光斑后的图像
        gaussian_path = save_image(image_with_gaussian, image_path)
        file_name = os.path.basename(gaussian_path)
        name, ext = os.path.splitext(file_name)
        new_file_name = f"{name}.jpg"
        # 获取预测结果
        results = model(gaussian_path, imgsz=640, conf=0.25, save=True)

        predict_image_path = os.path.join("E:/python/huangnanqi/runs/obb/predict/", new_file_name)
        print(predict_image_path)
        if os.path.exists(predict_image_path):
            final_image = Image.open(predict_image_path)

            # 显示最终带有检测框的图像
            tk_image_with_boxes = display_image(final_image, image_label_with_gaussian)
            image_label_with_gaussian.image = tk_image_with_boxes

            # 更新 OOAP 标签的文本
            ooap_label.config(text=f"最大OOAP: {max_ooap:.2f}%")
            image_path_label.config(text=f"生成的图片路径: {predict_image_path}")
        else:
            messagebox.showerror("错误", "未找到预测后的图像")
    except Exception as e:
        messagebox.showerror("错误", f"处理图像时出错: {e}")


def display_image(image, label):
    image = image.resize((640, 640), Image.LANCZOS)  # 替换 ANTIALIAS 为 LANCZOS
    tk_image = ImageTk.PhotoImage(image)
    label.config(image=tk_image)
    label.image = tk_image  # 保持对图像的引用，防止被垃圾回收
    return tk_image

import cv2
from PIL import Image

def save_image(image, original_image_path):
    """
    保存生成高斯光斑后的图像到文件夹
    :param image: 生成高斯光斑后的图像 (numpy.ndarray 或 PIL Image)
    :param original_image_path: 原始图像的路径
    """
    # 获取原始图像的文件名和扩展名
    file_name = os.path.basename(original_image_path)
    name, ext = os.path.splitext(file_name)
    
    # 创建保存文件夹
    save_folder = os.path.join(os.path.dirname(original_image_path), "gaussian_blurred_images")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # 生成新的文件名
    new_file_name = f"{name}_gaussian_blurred{ext}"
    save_path = os.path.join(save_folder, new_file_name)
    
    # 检查图像类型并保存
    if isinstance(image, np.ndarray):
        # 如果是 NumPy 数组，使用 cv2.imwrite 保存
        cv2.imwrite(save_path, image)
    elif isinstance(image, Image.Image):
        # 如果是 PIL Image 对象，使用 save 方法保存
        image.save(save_path)
    else:
        raise TypeError("Unsupported image type. Expected numpy.ndarray or PIL.Image.")
    
    return save_path


def set_gaussian_center(event):
    global center_x, center_y
    # 记录点击位置的坐标
    center_x = event.x
    center_y = event.y
    
    # 将点击位置的坐标转换为预测图的坐标
    original_width, original_height = Image.open(image_path).size
    scaled_center_x = (center_x / 640) * original_width
    scaled_center_y = (center_y / 640) * original_height
    
    # 更新输入框中的值
    center_x_entry.delete(0, tk.END)
    center_x_entry.insert(0, str(int(scaled_center_x)))
    center_y_entry.delete(0, tk.END)
    center_y_entry.insert(0, str(int(scaled_center_y)))

def open_image():
    global image_path, center_x, center_y
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
    if image_path:
        # 将路径显示在标签中
        image_path_label.config(text=f"选择的图片路径: {image_path}")
        predict_image(image_path)
        # 初始化变量
        center_x, center_y = 320, 320
        center_x_entry.delete(0, tk.END)
        center_x_entry.insert(0, str(center_x))
        center_y_entry.delete(0, tk.END)
        center_y_entry.insert(0, str(center_y))

# 创建函数以获取TXT文件路径
def get_txt_file():
    global txt_file_path
    # 打开文件对话框以获取TXT文件路径
    txt_file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if txt_file_path:
        # 将路径显示在标签中
        file_path_label.config(text=f"选择的TXT文件路径: {txt_file_path}")
    else:
        messagebox.showinfo("信息", "未选择任何文件")


def update_and_predict(*args):
    try:
        global image_path
        if image_path:
            predict_image(image_path)
    except Exception as e:
        messagebox.showerror("错误", f"更新预测时出错: {e}")

# 初始化 YOLO 模型
model = YOLO('yolov8n-obb.pt')
#model=YOLO('runs/train/yolov8n_obb_train_v1/weights/best.pt')



# 创建主窗口
root = tk.Tk()
root.title("实时高斯光斑生成和飞机检测工具")

# 创建一个按钮以打开图片
open_button = tk.Button(root, text="选择图片进行预测", command=open_image)
open_button.pack(pady=10)

# 添加标签来显示生成的图片路径
image_path_label = tk.Label(root, text="选择的图片路径: 未生成")
image_path_label.pack(pady=10)

# 创建按钮以获取TXT文件路径
open_txt_button = tk.Button(root, text="选择图像的label", command=get_txt_file)
open_txt_button.pack(pady=10)

# 创建标签以显示TXT文件路径
file_path_label = tk.Label(root, text="选择的label路径: 未选择")
file_path_label.pack(pady=10)


# 创建标签来显示原始图像
image_label = tk.Label(root)
image_label.pack(side=tk.LEFT, padx=10)
image_label.bind("<Button-1>", set_gaussian_center)  # 绑定鼠标左键单击事件

# 创建标签来显示生成高斯光斑后的图像
image_label_with_gaussian = tk.Label(root)
image_label_with_gaussian.pack(side=tk.RIGHT, padx=10)
image_label_with_gaussian.bind("<Button-1>", set_gaussian_center)  # 绑定鼠标左键单击事件

# 添加输入框以选择高斯光斑的中心位置和半径
tk.Label(root, text="高斯光斑中心 X:").pack(pady=5)
center_x_entry = tk.Entry(root)
center_x_entry.pack(pady=5)
center_x_entry.insert(0, "320")  # 默认值
center_x_var = tk.StringVar(root)
center_x_var.trace("w", update_and_predict)
center_x_entry.config(textvariable=center_x_var)

tk.Label(root, text="高斯光斑中心 Y:").pack(pady=5)
center_y_entry = tk.Entry(root)
center_y_entry.pack(pady=5)
center_y_entry.insert(0, "320")  # 默认值
center_y_var = tk.StringVar(root)
center_y_var.trace("w", update_and_predict)
center_y_entry.config(textvariable=center_y_var)

tk.Label(root, text="高斯光斑半径:").pack(pady=5)
radius_entry = tk.Entry(root)
radius_entry.pack(pady=5)
radius_entry.insert(0, "30")  # 默认值
radius_var = tk.StringVar(root)
radius_var.trace("w", update_and_predict)
radius_entry.config(textvariable=radius_var)

# 添加标签来显示 OOAP 的值
ooap_label = tk.Label(root, text="OOAP: 0.0%")
ooap_label.pack(pady=5)

# 运行 GUI 应用
root.mainloop()
