# src/metrics.py

import numpy as np
from shapely.geometry import Polygon
from typing import Optional

def create_ellipse_polygon(
    center_x: float, 
    center_y: float, 
    sigma_x: float, 
    sigma_y: float, 
    rotation_angle_deg: float, 
    k_sigma_level: float = 2.0, # 与你之前代码一致的默认值
    n_points: int = 50
) -> Optional[Polygon]:
    """
    根据高斯光斑的参数创建一个近似的椭圆多边形 (Shapely Polygon)。
    用于计算光斑的有效区域以进行OOAP计算。

    参数:
    - center_x, center_y: 椭圆中心在图像上的绝对坐标。
    - sigma_x, sigma_y: 光斑的x轴和y轴标准差。
    - rotation_angle_deg: 光斑的旋转角度（度）。
    - k_sigma_level: 定义椭圆边界的sigma倍数 (例如, 2.0 表示在2*sigma处)。
    - n_points: 用于近似椭圆的多边形顶点数量。

    返回:
    - shapely.geometry.Polygon 对象，如果参数无效则返回 None。
    """
    if sigma_x <= 1e-3 or sigma_y <= 1e-3: # 避免sigma过小导致问题
        return None
    
    # 椭圆在 k_sigma_level 处的半长轴和半短轴
    a = k_sigma_level * sigma_x
    b = k_sigma_level * sigma_y

    if a <= 1e-3 or b <= 1e-3: # 避免轴长过小
        return None

    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    ellipse_x_unrotated = a * np.cos(t)
    ellipse_y_unrotated = b * np.sin(t)

    rotation_angle_rad = np.deg2rad(rotation_angle_deg)
    cos_theta = np.cos(rotation_angle_rad)
    sin_theta = np.sin(rotation_angle_rad)

    # 旋转并平移到图像坐标系
    ellipse_x_rotated = ellipse_x_unrotated * cos_theta - ellipse_y_unrotated * sin_theta + center_x
    ellipse_y_rotated = ellipse_x_unrotated * sin_theta + ellipse_y_unrotated * cos_theta + center_y
    
    ellipse_points = list(zip(ellipse_x_rotated, ellipse_y_rotated))
    
    try:
        ellipse_poly = Polygon(ellipse_points)
        if not ellipse_poly.is_valid:
            # 尝试修复无效多边形，例如通过凸包
            ellipse_poly = ellipse_poly.convex_hull 
            if not ellipse_poly.is_valid or ellipse_poly.area < 1e-3: # 面积太小也视为无效
                return None
        return ellipse_poly
    except Exception:
        return None


def calculate_single_ooap(
    aircraft_obb_vertices: np.ndarray, # 飞机OBB的4个顶点 [[x1,y1],...,[x4,y4]] 像素坐标
    spot_params: dict,                 # 光斑参数字典，应包含:
                                       # 'spot_center_abs_xy', 
                                       # 'spot_sigma_x', 'spot_sigma_y', 'spot_rotation_angle_deg'
    k_sigma_level_for_ooap: float = 2.0 # 定义光斑有效边界的k倍sigma
) -> float:
    """
    计算单个飞机OBB被给定高斯光斑遮挡的面积百分比 (OOAP)。

    参数:
    - aircraft_obb_vertices: 飞机OBB的4个顶点 (像素坐标, [[x1,y1],...,[x4,y4]])。
    - spot_params: 包含光斑中心、sigmas和旋转角度的字典。
        - 'spot_center_abs_xy': 光斑中心在背景图上的绝对坐标 (tuple or list)。
        - 'spot_sigma_x', 'spot_sigma_y': 光斑的x轴和y轴标准差。
        - 'spot_rotation_angle_deg': 光斑的旋转角度（度）。
    - k_sigma_level_for_ooap: 定义光斑有效边界的k倍sigma。

    返回:
    - float: OOAP值 (0.0 到 100.0)。如果计算失败或无效，则返回 -1.0。
    """
    try:
        # 1. 创建飞机多边形
        aircraft_poly = Polygon(aircraft_obb_vertices)
        if not aircraft_poly.is_valid or aircraft_poly.area < 1e-3: # 目标面积过小
            # print("警告 (calculate_single_ooap): 无效的飞机多边形或面积过小。")
            return 0.0 # 或者 -1.0，取决于如何处理这种情况
        
        aircraft_area = aircraft_poly.area
        if aircraft_area < 1e-6: # 避免除以零
            return 0.0

        # 2. 创建光斑椭圆多边形
        spot_poly = create_ellipse_polygon(
            center_x=spot_params['spot_center_abs_xy'][0], # 假设spot_params中是这个键
            center_y=spot_params['spot_center_abs_xy'][1],
            sigma_x=spot_params['spot_sigma_x'],
            sigma_y=spot_params['spot_sigma_y'],
            rotation_angle_deg=spot_params['spot_rotation_angle_deg'], # 假设是这个键
            k_sigma_level=k_sigma_level_for_ooap
        )

        if spot_poly is None or not spot_poly.is_valid or spot_poly.area < 1e-3:
            # print("警告 (calculate_single_ooap): 无效的光斑多边形或面积过小。")
            return 0.0 # 如果光斑无效，则遮挡为0

        # 3. 计算交集面积
        intersection_poly = aircraft_poly.intersection(spot_poly)
        intersection_area = intersection_poly.area

        # 4. 计算OOAP
        ooap = (intersection_area / aircraft_area) * 100.0
        return np.clip(ooap, 0.0, 100.0) #确保在合理范围

    except Exception as e:
        print(f"错误 (calculate_single_ooap): 计算OOAP时发生错误: {e}")
        return -1.0 # 表示计算出错


def calculate_iou_obb(obb1_vertices: np.ndarray, obb2_vertices: np.ndarray) -> float:
    """
    计算两个旋转边界框 (OBB) 之间的交并比 (IoU)。
    每个OBB由4个顶点定义。

    参数:
    - obb1_vertices (np.ndarray): 第一个OBB的4个顶点 [[x1,y1],...,[x4,y4]]。
    - obb2_vertices (np.ndarray): 第二个OBB的4个顶点 [[x1,y1],...,[x4,y4]]。

    返回:
    - float: IoU值 (0.0 到 1.0)。如果多边形无效或计算出错，返回0.0。
    """
    try:
        poly1 = Polygon(obb1_vertices)
        poly2 = Polygon(obb2_vertices)

        if not poly1.is_valid or not poly2.is_valid or poly1.area < 1e-3 or poly2.area < 1e-3:
            return 0.0

        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - intersection_area

        if union_area < 1e-6: # 避免除以零
            return 0.0
        
        iou = intersection_area / union_area
        return np.clip(iou, 0.0, 1.0)
    except Exception as e:
        print(f"错误 (calculate_iou_obb): 计算IoU时发生错误: {e}")
        return 0.0

if __name__ == '__main__':
    # 测试 calculate_single_ooap
    print("--- 测试 OOAP 计算 ---")
    mock_aircraft_obb = np.array([[100, 100], [300, 100], [300, 200], [100, 200]]) # 200x100的矩形
    mock_spot_params_partial_overlap = {
        'spot_center_abs_xy': (200, 150), # 中心部分重叠
        'spot_sigma_x': 80,
        'spot_sigma_y': 40,
        'spot_rotation_angle_deg': 0
    }
    ooap1 = calculate_single_ooap(mock_aircraft_obb, mock_spot_params_partial_overlap, k_sigma_level_for_ooap=2.0)
    print(f"OOAP (部分重叠): {ooap1:.2f}%")

    mock_spot_params_full_overlap = {
        'spot_center_abs_xy': (200, 150), # 飞机中心
        'spot_sigma_x': 150, # 较大sigma，应该能覆盖整个飞机
        'spot_sigma_y': 100,
        'spot_rotation_angle_deg': 0
    }
    ooap2 = calculate_single_ooap(mock_aircraft_obb, mock_spot_params_full_overlap, k_sigma_level_for_ooap=2.0)
    print(f"OOAP (完全覆盖，预期接近100%): {ooap2:.2f}%")

    mock_spot_params_no_overlap = {
        'spot_center_abs_xy': (500, 500), # 远离飞机
        'spot_sigma_x': 50,
        'spot_sigma_y': 50,
        'spot_rotation_angle_deg': 0
    }
    ooap3 = calculate_single_ooap(mock_aircraft_obb, mock_spot_params_no_overlap)
    print(f"OOAP (无重叠): {ooap3:.2f}%")

    # 测试 calculate_iou_obb
    print("\n--- 测试 IoU 计算 ---")
    obb1 = np.array([[0,0], [10,0], [10,10], [0,10]])
    obb2 = np.array([[5,5], [15,5], [15,15], [5,15]]) # 部分重叠
    iou1 = calculate_iou_obb(obb1, obb2)
    print(f"IoU (部分重叠, 预期0.25*100 / (200-25) = 25/175 approx 0.14): {iou1:.4f}") # (5x5) / (100+100-25) = 25/175

    obb3 = np.array([[0,0], [10,0], [10,10], [0,10]]) # 完全相同
    iou2 = calculate_iou_obb(obb1, obb3)
    print(f"IoU (完全相同): {iou2:.4f}")

    obb4 = np.array([[100,100], [110,100], [110,110], [100,110]]) # 不重叠
    iou3 = calculate_iou_obb(obb1, obb4)
    print(f"IoU (不重叠): {iou3:.4f}")