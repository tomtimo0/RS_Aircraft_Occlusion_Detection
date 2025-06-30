# src/image_utils.py

import cv2
import numpy as np
import math
from typing import Optional

# ==============================================================================
# 核心光斑生成与叠加函数 (基于你提供的代码进行整理和适配)
# ==============================================================================

def get_rotated_ellipse_canvas_size(sigma_x: float, sigma_y: float, rotation_angle_deg: float, k: float = 3.5) -> tuple:
    """
    计算包含旋转椭圆的最小画布尺寸，以确保高斯光斑在k*sigma范围内能完整显示。

    参数:
    - sigma_x (float): x轴标准差。
    - sigma_y (float): y轴标准差。
    - rotation_angle_deg (float): 旋转角度 (度)。
    - k (float): sigma倍数，决定光斑边界的范围，默认为3.5，覆盖约99.9%的能量。

    返回:
    - tuple: (画布宽, 画布高)，均为整数。
    """
    if sigma_x <= 0 or sigma_y <= 0:
        # 提供一个最小的默认尺寸以避免后续计算错误
        return (20, 20) 
    
    rotation_angle_rad = np.deg2rad(rotation_angle_deg)
    cos_theta = np.cos(rotation_angle_rad)
    sin_theta = np.sin(rotation_angle_rad)

    # 椭圆在 k*sigma 处的半长轴和半短轴
    a = k * sigma_x
    b = k * sigma_y

    # 计算旋转后椭圆的轴对齐包围盒 (AABB) 的半宽和半高
    # https://en.wikipedia.org/wiki/Ellipse#Minimum_bounding_box
    aabb_half_width = math.sqrt(a**2 * cos_theta**2 + b**2 * sin_theta**2)
    aabb_half_height = math.sqrt(a**2 * sin_theta**2 + b**2 * cos_theta**2)
    
    # 增加一些padding确保边缘不会被裁剪，并向上取整
    padding = 10 # 像素
    canvas_width = math.ceil(2 * aabb_half_width + padding)
    canvas_height = math.ceil(2 * aabb_half_height + padding)
    
    return (int(canvas_width), int(canvas_height))


def generate_gaussian_spot_rgba(
    spot_canvas_size_wh: tuple,
    amplitude: float, # 高斯函数的峰值振幅 (可以 > 1 来模拟过曝)
    sigma_x: float,
    sigma_y: float,
    rotation_angle_deg: float,
    spot_color_rgb: tuple = (255, 255, 255) # 光斑颜色 (R,G,B)
) -> Optional[np.ndarray]:
    """
    生成一个椭圆高斯光斑的 RGBA 图像。

    参数:
    - spot_canvas_size_wh (tuple): 光斑画布尺寸 (宽, 高)。
    - amplitude (float): 高斯函数的峰值振幅。最终alpha会clip到[0,1]。
    - sigma_x (float): x轴标准差。
    - sigma_y (float): y轴标准差。
    - rotation_angle_deg (float): 旋转角度 (度)。
    - spot_color_rgb (tuple): 光斑的RGB颜色 (默认白色)。

    返回:
    - np.ndarray: RGBA格式的光斑图像 (H,W,4), uint8类型。如果参数无效则返回 None。
    """
    width, height = int(round(spot_canvas_size_wh[0])), int(round(spot_canvas_size_wh[1]))
    if width <= 0 or height <= 0 or sigma_x <= 1e-6 or sigma_y <= 1e-6: # sigma过小可能导致除零
        print(f"警告: 无效的画布尺寸或sigma值. W:{width}, H:{height}, SX:{sigma_x}, SY:{sigma_y}")
        return None

    center_x, center_y = (width -1) / 2.0, (height -1) / 2.0 # 画布中心

    rotation_angle_rad = np.deg2rad(rotation_angle_deg)
    cos_theta = np.cos(rotation_angle_rad)
    sin_theta = np.sin(rotation_angle_rad)

    x_coords = np.arange(width)
    y_coords = np.arange(height)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # 从画布中心平移坐标
    x_shifted = xx - center_x
    y_shifted = yy - center_y

    # 旋转坐标系
    x_prime = x_shifted * cos_theta + y_shifted * sin_theta
    y_prime = -x_shifted * sin_theta + y_shifted * cos_theta
    
    # 防止sigma为0导致除零错误
    sigma_x_sq = sigma_x**2 if sigma_x > 1e-6 else 1e-6
    sigma_y_sq = sigma_y**2 if sigma_y > 1e-6 else 1e-6
    
    # 计算高斯强度
    exponent = -((x_prime**2) / (2 * sigma_x_sq) + (y_prime**2) / (2 * sigma_y_sq))
    intensity_profile = amplitude * np.exp(exponent)

    # Alpha通道: 将强度归一化到[0,1]并转换为0-255
    # np.clip确保了即使amplitude > 1 (模拟过曝)，alpha值也在正确范围内
    alpha_channel = np.clip(intensity_profile, 0, 1) * 255
    alpha_channel = alpha_channel.astype(np.uint8)

    # 创建RGBA图像
    spot_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    spot_rgba[..., 0] = spot_color_rgb[0] # R
    spot_rgba[..., 1] = spot_color_rgb[1] # G
    spot_rgba[..., 2] = spot_color_rgb[2] # B
    spot_rgba[..., 3] = alpha_channel     # Alpha
    
    return spot_rgba


def generate_circular_gaussian_mask_rgba(
    spot_canvas_size_wh: tuple,
    amplitude: float, # 高斯函数的峰值振幅 (可以 > 1 来模拟过曝)
    sigma: float,     # 圆形高斯的标准差
    spot_color_rgb: tuple = (255, 255, 255) # 光斑颜色 (R,G,B)
) -> Optional[np.ndarray]:
    """
    生成一个圆形高斯光斑的 RGBA 图像。

    参数:
    - spot_canvas_size_wh (tuple): 光斑画布尺寸 (宽, 高)。
    - amplitude (float): 高斯函数的峰值振幅。最终alpha会clip到[0,1]。
    - sigma (float): 圆形高斯的标准差。
    - spot_color_rgb (tuple): 光斑的RGB颜色 (默认白色)。

    返回:
    - np.ndarray: RGBA格式的光斑图像 (H,W,4), uint8类型。如果参数无效则返回 None。
    """
    width, height = int(round(spot_canvas_size_wh[0])), int(round(spot_canvas_size_wh[1]))
    if width <= 0 or height <= 0 or sigma <= 1e-6:
        print(f"警告: 无效的画布尺寸或sigma值. W:{width}, H:{height}, S:{sigma}")
        return None

    center_x, center_y = (width -1) / 2.0, (height -1) / 2.0

    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    dist_from_center_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
    
    sigma_sq = sigma**2 if sigma > 1e-6 else 1e-6
    
    intensity_profile = amplitude * np.exp(-dist_from_center_sq / (2 * sigma_sq))
    
    alpha_channel = np.clip(intensity_profile, 0, 1) * 255
    alpha_channel = alpha_channel.astype(np.uint8)

    spot_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    spot_rgba[..., 0] = spot_color_rgb[0] # R
    spot_rgba[..., 1] = spot_color_rgb[1] # G
    spot_rgba[..., 2] = spot_color_rgb[2] # B
    spot_rgba[..., 3] = alpha_channel     # Alpha

    return spot_rgba


def blend_spot_on_image(
    background_image_bgr: np.ndarray,
    spot_rgba: np.ndarray,
    top_left_x_on_bg: int, # 光斑左上角在背景图上的x坐标
    top_left_y_on_bg: int  # 光斑左上角在背景图上的y坐标
) -> np.ndarray:
    """
    将RGBA光斑图像通过Alpha混合叠加到背景BGR图像上。
    处理边界情况，确保光斑不会超出背景图像范围。

    参数:
    - background_image_bgr (np.ndarray): BGR格式的背景图像。
    - spot_rgba (np.ndarray): RGBA格式的光斑图像。
    - top_left_x_on_bg (int): 光斑左上角在背景图上的x坐标。
    - top_left_y_on_bg (int): 光斑左上角在背景图上的y坐标。

    返回:
    - np.ndarray: 叠加了光斑的BGR图像。
    """
    bg_h, bg_w = background_image_bgr.shape[:2]
    spot_h, spot_w = spot_rgba.shape[:2]

    # 计算光斑在背景图上的实际有效叠加区域的起始和结束坐标
    # 同时计算光斑本身需要裁剪的区域
    
    # 背景上的ROI起始点
    roi_x_start_bg = top_left_x_on_bg
    roi_y_start_bg = top_left_y_on_bg
    
    # 光斑上的裁剪起始点
    spot_crop_x_start = 0
    spot_crop_y_start = 0

    # 修正左边界和上边界超出情况
    if roi_x_start_bg < 0:
        spot_crop_x_start = -roi_x_start_bg # 光斑需要从这里开始裁剪
        roi_x_start_bg = 0                  # 背景从0开始
    if roi_y_start_bg < 0:
        spot_crop_y_start = -roi_y_start_bg
        roi_y_start_bg = 0

    # 背景上的ROI结束点 (初始)
    roi_x_end_bg = top_left_x_on_bg + spot_w
    roi_y_end_bg = top_left_y_on_bg + spot_h
    
    # 光斑上的裁剪结束点 (初始)
    spot_crop_x_end = spot_w
    spot_crop_y_end = spot_h

    # 修正右边界和下边界超出情况
    if roi_x_end_bg > bg_w:
        spot_crop_x_end = spot_w - (roi_x_end_bg - bg_w) # 光斑裁剪到这里结束
        roi_x_end_bg = bg_w                             # 背景到bg_w结束
    if roi_y_end_bg > bg_h:
        spot_crop_y_end = spot_h - (roi_y_end_bg - bg_h)
        roi_y_end_bg = bg_h
        
    # 计算实际有效叠加区域的宽高
    effective_overlay_w = roi_x_end_bg - roi_x_start_bg
    effective_overlay_h = roi_y_end_bg - roi_y_start_bg

    # 如果有效叠加区域尺寸<=0，说明光斑完全在图像外，直接返回原图
    if effective_overlay_w <= 0 or effective_overlay_h <= 0:
        return background_image_bgr.copy() # 返回副本以保持一致性

    # 裁剪光斑图像以匹配有效叠加区域
    spot_to_overlay = spot_rgba[spot_crop_y_start:spot_crop_y_end, spot_crop_x_start:spot_crop_x_end]

    # 提取背景中的ROI
    bg_roi = background_image_bgr[roi_y_start_bg:roi_y_end_bg, roi_x_start_bg:roi_x_end_bg]
    
    # 再次检查裁剪后的光斑和背景ROI尺寸是否一致（理论上应该一致）
    if bg_roi.shape[0] != spot_to_overlay.shape[0] or bg_roi.shape[1] != spot_to_overlay.shape[1]:
        print(f"警告: 裁剪后的光斑和背景ROI尺寸不匹配。BG ROI: {bg_roi.shape}, Spot Overlay: {spot_to_overlay.shape}")
        # 尝试调整spot_to_overlay的尺寸以匹配bg_roi，这是一种补救措施
        spot_to_overlay = cv2.resize(spot_to_overlay, (bg_roi.shape[1], bg_roi.shape[0]))
        if bg_roi.shape[0] != spot_to_overlay.shape[0] or bg_roi.shape[1] != spot_to_overlay.shape[1]:
            print("错误: 尺寸调整失败，返回原图。")
            return background_image_bgr.copy()


    # 分离光斑的RGB和Alpha通道
    spot_rgb = spot_to_overlay[:, :, :3].astype(np.float32)
    spot_alpha = spot_to_overlay[:, :, 3].astype(np.float32) / 255.0 # 归一化到 [0, 1]
    
    # 将Alpha通道扩展为3通道，以便与RGB图像进行元素级乘法
    spot_alpha_3channel = np.stack([spot_alpha]*3, axis=-1)

    # Alpha混合: Output = Foreground * Alpha + Background * (1 - Alpha)
    blended_roi = (spot_rgb * spot_alpha_3channel) + \
                  (bg_roi.astype(np.float32) * (1.0 - spot_alpha_3channel))
    
    blended_roi = np.clip(blended_roi, 0, 255).astype(np.uint8) # 确保值在0-255范围内

    # 将混合后的ROI放回背景图像副本
    result_image = background_image_bgr.copy()
    result_image[roi_y_start_bg:roi_y_end_bg, roi_x_start_bg:roi_x_end_bg] = blended_roi
    
    return result_image


# ==============================================================================
# 主调用函数 (将被Web UI或其他应用逻辑调用)
# ==============================================================================

def add_gaussian_spot_to_image(
    background_image_bgr: np.ndarray,
    spot_center_abs_xy: tuple,      # 光斑在背景图上的绝对中心坐标 (x_center, y_center)
    spot_sigma_x: float,            # x轴标准差 (像素)
    spot_sigma_y: float,            # y轴标准差 (像素) (圆形光斑时，此值会被设为sigma_x)
    spot_rotation_angle_deg: float, # 旋转角度 (度) (圆形光斑时，此值无效)
    spot_amplitude: float,          # 高斯光斑峰值强度 (可以 > 1 来模拟过曝)
    spot_color_rgb: tuple = (255, 255, 255), # 光斑颜色 (R,G,B)
    spot_shape: str = 'ellipse'     # 'ellipse' 或 'circle'
) -> np.ndarray:
    """
    在背景图像的指定位置添加一个参数化的高斯光斑。
    此函数适合被交互式UI（如Streamlit滑块）调用。

    参数:
    - background_image_bgr: OpenCV BGR格式的背景图像。
    - spot_center_abs_xy: 光斑在背景图像上的中心绝对坐标 (x_center, y_center)。
    - spot_sigma_x: 光斑的x轴标准差。
    - spot_sigma_y: 光斑的y轴标准差 (对于圆形光斑，此值将被设为等于sigma_x)。
    - spot_rotation_angle_deg: 光斑的旋转角度（度）。对于圆形光斑，此参数将被忽略并设为0。
    - spot_amplitude: 高斯函数的峰值振幅 (例如 0.1 到 3.0)。
    - spot_color_rgb: 光斑的RGB颜色。
    - spot_shape: 光斑的形状, 'ellipse' 或 'circle'。

    返回:
    - np.ndarray: 叠加了高斯光斑的BGR图像。若参数无效或生成失败，返回原始图像的副本。
    """
    if background_image_bgr is None:
        print("错误: 背景图像为空。")
        return None # 或者可以引发一个异常

    # 参数校验和调整
    current_sigma_x = max(1.0, spot_sigma_x) # sigma至少为1像素
    current_sigma_y = max(1.0, spot_sigma_y)
    current_rotation_deg = spot_rotation_angle_deg
    current_amplitude = max(0.0, spot_amplitude)

    if spot_shape == 'circle':
        current_sigma_y = current_sigma_x  # 圆形光斑 sigma_x = sigma_y
        current_rotation_deg = 0           # 圆形光斑无特定旋转方向

    # 1. 计算光斑画布尺寸 (使用k=3.5，覆盖高斯分布主要部分)
    # 对于圆形光斑，传入调整后的sigma_x, sigma_y (相等) 和 rotation=0
    canvas_w, canvas_h = get_rotated_ellipse_canvas_size(
        current_sigma_x, current_sigma_y, current_rotation_deg, k=3.5 
    )
    if canvas_w <= 0 or canvas_h <= 0:
        print("警告: 计算得到的光斑画布尺寸无效，返回原图副本。")
        return background_image_bgr.copy()

    # 2. 生成高斯光斑 RGBA 图像
    spot_rgba = None
    if spot_shape == 'ellipse':
        spot_rgba = generate_gaussian_spot_rgba(
            spot_canvas_size_wh=(canvas_w, canvas_h),
            amplitude=current_amplitude,
            sigma_x=current_sigma_x,
            sigma_y=current_sigma_y,
            rotation_angle_deg=current_rotation_deg,
            spot_color_rgb=spot_color_rgb
        )
    elif spot_shape == 'circle':
        # 可以选择使用特定的圆形光斑生成函数，或用椭圆函数模拟 (sigma_x=sigma_y, angle=0)
        # 这里为了代码复用和一致性，继续使用 generate_gaussian_spot_rgba
        # 但传入的是为圆形调整后的参数
        spot_rgba = generate_gaussian_spot_rgba( # 或者用 generate_circular_gaussian_mask_rgba
            spot_canvas_size_wh=(canvas_w, canvas_h),
            amplitude=current_amplitude,
            sigma_x=current_sigma_x, # 圆形的sigma
            sigma_y=current_sigma_x, # 圆形的sigma (等于sigma_x)
            rotation_angle_deg=0,    # 圆形无旋转
            spot_color_rgb=spot_color_rgb
        )
        # 如果你希望严格使用你提供的 generate_circular_gaussian_mask_rgba:
        # spot_rgba = generate_circular_gaussian_mask_rgba(
        #     spot_canvas_size_wh=(canvas_w, canvas_h),
        #     amplitude=current_amplitude,
        #     sigma=current_sigma_x, # 圆形只需要一个sigma
        #     spot_color_rgb=spot_color_rgb
        # )

    else:
        print(f"警告: 未知的光斑形状 '{spot_shape}'，返回原图副本。")
        return background_image_bgr.copy()

    if spot_rgba is None:
        print("警告: 生成光斑 RGBA 图像失败，返回原图副本。")
        return background_image_bgr.copy()

    # 3. 计算光斑在背景图像上的左上角放置位置
    # spot_center_abs_xy 是光斑的中心，而画布的中心是 (canvas_w-1)/2, (canvas_h-1)/2
    # 所以左上角坐标是 中心 - 画布半尺寸
    top_left_x_on_bg = int(round(spot_center_abs_xy[0] - (canvas_w -1) / 2.0))
    top_left_y_on_bg = int(round(spot_center_abs_xy[1] - (canvas_h -1) / 2.0))

    # 4. 叠加光斑到背景图像
    occluded_image = blend_spot_on_image(
        background_image_bgr, # 传入原图，blend_spot_on_image内部会创建副本
        spot_rgba,
        top_left_x_on_bg,
        top_left_y_on_bg
    )
    
    return occluded_image


# ==============================================================================
# 测试代码块
# ==============================================================================
if __name__ == '__main__':
    # 创建一个简单的测试背景图像 (例如，黑色)
    test_bg_height, test_bg_width = 600, 800
    test_bg_image = np.zeros((test_bg_height, test_bg_width, 3), dtype=np.uint8)
    # 添加一些文字以便观察
    cv2.putText(test_bg_image, "Test Background", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    cv2.line(test_bg_image, (0, test_bg_height//2), (test_bg_width, test_bg_height//2), (50,50,50), 1)
    cv2.line(test_bg_image, (test_bg_width//2, 0), (test_bg_width//2, test_bg_height), (50,50,50), 1)


    print("测试开始...")

    # --- 测试1: 椭圆光斑 ---
    print("\n--- 测试椭圆光斑 ---")
    ellipse_params = {
        'spot_center_abs_xy': (test_bg_width // 2, test_bg_height // 2), # 中心
        'spot_sigma_x': 100,
        'spot_sigma_y': 50,
        'spot_rotation_angle_deg': 30,
        'spot_amplitude': 1.5, # 峰值强度
        'spot_color_rgb': (255, 230, 200), # 暖白色
        'spot_shape': 'ellipse'
    }
    img_with_ellipse = add_gaussian_spot_to_image(test_bg_image, **ellipse_params)
    
    if img_with_ellipse is not None:
        cv2.imshow("Image with Ellipse Spot", img_with_ellipse)
        print(f"椭圆光斑参数: {ellipse_params}")
    else:
        print("椭圆光斑生成失败。")
    cv2.waitKey(0)

    # --- 测试2: 圆形光斑 ---
    print("\n--- 测试圆形光斑 ---")
    circle_params = {
        'spot_center_abs_xy': (test_bg_width // 4, test_bg_height // 4), # 左上角区域
        'spot_sigma_x': 60, # 对于圆形，sigma_y 将被忽略或设为等于sigma_x
        'spot_sigma_y': 30, # 此值在spot_shape='circle'时会被spot_sigma_x覆盖
        'spot_rotation_angle_deg': 45, # 此值在spot_shape='circle'时会被设为0
        'spot_amplitude': 2.0, # 更强的光斑
        'spot_color_rgb': (200, 255, 255), # 青色光斑
        'spot_shape': 'circle'
    }
    img_with_circle = add_gaussian_spot_to_image(test_bg_image, **circle_params)

    if img_with_circle is not None:
        cv2.imshow("Image with Circle Spot", img_with_circle)
        print(f"圆形光斑输入参数: {circle_params}")
        # 实际使用的参数会被调整
        print(f"实际圆形sigma: {circle_params['spot_sigma_x']}, 实际旋转: 0")
    else:
        print("圆形光斑生成失败。")
    cv2.waitKey(0)

    # --- 测试3: 光斑部分超出边界 ---
    print("\n--- 测试光斑部分超出边界 ---")
    edge_params = {
        'spot_center_abs_xy': (20, 30), # 光斑中心靠近左上角，使其部分超出
        'spot_sigma_x': 80,
        'spot_sigma_y': 80, # 使其为圆形，更容易观察裁剪
        'spot_rotation_angle_deg': 0,
        'spot_amplitude': 1.0,
        'spot_color_rgb': (255, 200, 255), # 粉色
        'spot_shape': 'ellipse' # 即使sigma_x=sigma_y, 设为ellipse测试通用路径
    }
    img_with_edge_spot = add_gaussian_spot_to_image(test_bg_image, **edge_params)
    if img_with_edge_spot is not None:
        cv2.imshow("Image with Edge Spot", img_with_edge_spot)
        print(f"边界光斑参数: {edge_params}")
    else:
        print("边界光斑生成失败。")
    cv2.waitKey(0)

    # --- 测试4: 极小光斑 ---
    print("\n--- 测试极小光斑 ---")
    tiny_params = {
        'spot_center_abs_xy': (test_bg_width *3//4, test_bg_height *3//4),
        'spot_sigma_x': 5, # 非常小的sigma
        'spot_sigma_y': 3,
        'spot_rotation_angle_deg': 0,
        'spot_amplitude': 2.5, # 即使小，也可以很亮
        'spot_color_rgb': (200, 200, 255), 
        'spot_shape': 'ellipse'
    }
    img_with_tiny_spot = add_gaussian_spot_to_image(test_bg_image, **tiny_params)
    if img_with_tiny_spot is not None:
        cv2.imshow("Image with Tiny Spot", img_with_tiny_spot)
        print(f"小光斑参数: {tiny_params}")
    else:
        print("小光斑生成失败。")
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print("\n测试完成。关闭所有窗口。")