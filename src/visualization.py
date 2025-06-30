# src/visualization.py

import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import os

def generate_random_color(class_id: int) -> tuple:
    """
    根据类别ID生成一个伪随机但相对固定的颜色。
    """
    # 使用固定的种子，确保对于相同的 class_id，颜色是相同的
    # 但不同的 class_id 会有不同的颜色系列
    random.seed(class_id + 42) # 加上一个偏移量以区分与Python内置random的种子
    blue = random.randint(50, 255)
    green = random.randint(50, 255)
    red = random.randint(50, 255)
    return (blue, green, red)

def draw_chinese_text_on_cv_image(cv_image, text, position, font_size=20, color=(255, 255, 255), thickness=2):
    """
    在OpenCV图像上绘制中文文本。
    
    参数:
    - cv_image: OpenCV图像 (BGR格式)
    - text: 要绘制的文本
    - position: 文本位置 (x, y)
    - font_size: 字体大小
    - color: 颜色 (BGR格式)
    - thickness: 线条粗细
    
    返回:
    - 绘制了文本的图像
    """
    try:
        # 将OpenCV图像转换为PIL图像
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv_image_rgb)
        
        # 创建绘图对象
        draw = ImageDraw.Draw(pil_image)
        
        # 尝试加载中文字体，如果失败则使用默认字体
        try:
            # 尝试使用系统中文字体
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # Windows 黑体
                "C:/Windows/Fonts/simsun.ttc",  # Windows 宋体
                "C:/Windows/Fonts/msyh.ttc",    # Windows 微软雅黑
                "/System/Library/Fonts/PingFang.ttc",  # macOS
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            ]
            
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except:
                        continue
            
            if font is None:
                # 如果找不到中文字体，使用默认字体
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 绘制文本
        draw.text(position, text, font=font, fill=color[::-1])  # PIL使用RGB，所以需要反转颜色
        
        # 将PIL图像转换回OpenCV格式
        cv_image_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return cv_image_with_text
        
    except Exception as e:
        print(f"绘制中文文本失败: {e}")
        # 如果PIL绘制失败，回退到OpenCV
        cv2.putText(cv_image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_size/20, color, thickness, cv2.LINE_AA)
        return cv_image

def draw_single_detection(
    image: np.ndarray,
    detection_info: dict,
    color: tuple = None, # BGR color
    show_label: bool = True,
    show_confidence: bool = True,
    show_center: bool = True,
    box_thickness: int = 2,
    font_scale: float = 0.6,
    font_thickness: int = 1,
    label_bg_alpha: float = 0.5 # 透明度 for label background
) -> np.ndarray:
    """
    在图像上绘制单个检测到的目标（旋转框、中心点、标签）。

    参数:
    - image (np.ndarray): 要在其上绘制的OpenCV BGR图像。
    - detection_info (dict): 包含单个检测目标信息的字典，应包含:
        'class_name' (str), 'confidence' (float),
        'obb_vertices' (np.ndarray, shape (4,2), dtype int32),
        'center_xy' (tuple, (cx, cy))
    - color (tuple, optional): BGR格式的颜色。如果为None，会生成随机颜色。
    - show_label (bool): 是否显示类别标签。
    - show_confidence (bool): 是否在标签中显示置信度。
    - show_center (bool): 是否绘制目标中心点。
    - box_thickness (int): 旋转框的线条粗细。
    - font_scale (float): 标签字体大小。
    - font_thickness (int): 标签字体粗细。
    - label_bg_alpha (float): 标签背景的透明度 (0.0 完全透明, 1.0 完全不透明)。

    返回:
    - np.ndarray: 绘制了检测结果的图像副本。
    """
    img_to_draw_on = image.copy() # 操作副本以避免修改原始图像

    obb_vertices = detection_info.get('obb_vertices')
    center_xy = detection_info.get('center_xy')
    class_name = detection_info.get('class_name', 'Unk')
    confidence = detection_info.get('confidence', 0.0)
    class_id = detection_info.get('class_id', 0)
    model_name_tag = detection_info.get('model_name', None) # 新增：获取模型名称

    if obb_vertices is None or not isinstance(obb_vertices, np.ndarray) or obb_vertices.shape != (4, 2):
        print("警告 (draw_single_detection): 'obb_vertices' 缺失或格式不正确。")
        return img_to_draw_on

    current_color = color if color is not None else generate_random_color(class_id)

    # 1. 绘制旋转边界框
    # OpenCV polylines 需要的顶点格式是 (num_points, 1, 2)
    pts = obb_vertices.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(img_to_draw_on, [pts], isClosed=True, color=current_color, thickness=box_thickness)

    # 2. 绘制中心点
    if show_center and center_xy is not None:
        try:
            center_int = (int(round(center_xy[0])), int(round(center_xy[1])))
            cv2.circle(img_to_draw_on, center_int, radius=max(3, box_thickness + 1), color=current_color, thickness=-1) # 实心圆
        except Exception as e:
            print(f"警告 (draw_single_detection): 绘制中心点失败: {e}")


    # 3. 准备并绘制标签（类别和置信度）
    if show_label:
        label_text = class_name
        if model_name_tag:
            label_text = f"[{model_name_tag[:3]}] {label_text}"
        if show_confidence:
            label_text += f": {confidence:.2f}"

        # 确保标签文本是有效的UTF-8字符串
        try:
            if isinstance(label_text, bytes):
                label_text = label_text.decode('utf-8', errors='replace')
            elif isinstance(label_text, str):
                # 检查字符串是否包含无效字符
                label_text.encode('utf-8').decode('utf-8')
        except UnicodeError:
            # 如果编码有问题，使用简化的标签
            label_text = f"class_{class_id}: {confidence:.2f}" if show_confidence else f"class_{class_id}"

        # 选择标签位置（通常在OBB的第一个点附近）
        # 确保标签在图像边界内
        label_origin_x = int(obb_vertices[0, 0])
        label_origin_y = int(obb_vertices[0, 1]) - 10 # 稍微向上偏移

        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # 标签背景框的坐标
        label_bg_rect_x1 = label_origin_x
        label_bg_rect_y1 = label_origin_y - text_height - baseline // 2
        label_bg_rect_x2 = label_origin_x + text_width
        label_bg_rect_y2 = label_origin_y + baseline // 2

        # 调整标签位置以避免超出图像顶部或左侧
        if label_bg_rect_y1 < 0:
            # 如果标签会超出顶部，则将其移到OBB第一个点的下方
            label_origin_y = int(obb_vertices[0, 1]) + text_height + baseline + 5 
            label_bg_rect_y1 = label_origin_y - text_height - baseline // 2
            label_bg_rect_x2 = label_origin_x + text_width
            label_bg_rect_y2 = label_origin_y + baseline // 2

        if label_bg_rect_x1 < 0:
            label_origin_x = 0
            # 重新计算背景框x坐标
            label_bg_rect_x1 = label_origin_x
            label_bg_rect_x2 = label_origin_x + text_width
        
        if label_bg_rect_x2 > img_to_draw_on.shape[1]: # 超出右边界
            label_origin_x -= (label_bg_rect_x2 - img_to_draw_on.shape[1])
            # 重新计算背景框x坐标
            label_bg_rect_x1 = label_origin_x
            label_bg_rect_x2 = label_origin_x + text_width


        # 绘制带透明度的标签背景
        if label_bg_alpha > 0:
            try:
                sub_img = img_to_draw_on[label_bg_rect_y1:label_bg_rect_y2, label_bg_rect_x1:label_bg_rect_x2]
                if sub_img.size > 0: # 确保ROI有效
                    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 50 # 深灰色背景
                    res = cv2.addWeighted(sub_img, 1 - label_bg_alpha, white_rect, label_bg_alpha, 1.0)
                    img_to_draw_on[label_bg_rect_y1:label_bg_rect_y2, label_bg_rect_x1:label_bg_rect_x2] = res
            except Exception as e:
                # print(f"警告 (draw_single_detection): 绘制标签背景失败: {e}")
                # 可能是由于坐标超出图像边界或ROI太小导致
                pass


        # 绘制标签文本
        try:
            # 使用PIL绘制中文文本
            img_to_draw_on = draw_chinese_text_on_cv_image(
                img_to_draw_on, 
                label_text, 
                (label_origin_x, label_origin_y - text_height),  # 调整位置以匹配PIL的坐标系统
                font_size=int(font_scale * 20),  # 将OpenCV的font_scale转换为PIL的字体大小
                color=current_color,
                thickness=font_thickness
            )
        except Exception as e:
            # 如果PIL绘制失败，回退到OpenCV
            print(f"警告: 绘制标签文本失败，使用OpenCV回退: {e}")
            try:
                cv2.putText(img_to_draw_on, label_text, (label_origin_x, label_origin_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, current_color, font_thickness, cv2.LINE_AA)
            except Exception as e2:
                # 如果还是失败，使用简化的文本
                print(f"警告: OpenCV绘制也失败，使用简化文本: {e2}")
                simple_text = f"class_{class_id}: {confidence:.2f}" if show_confidence else f"class_{class_id}"
                cv2.putText(img_to_draw_on, simple_text, (label_origin_x, label_origin_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, current_color, font_thickness, cv2.LINE_AA)
            
    return img_to_draw_on


def draw_all_detections(
    image: np.ndarray,
    detections_list: list,
    color_map: dict = None, # 例如: {'plane': (0,255,0), 'ship': (255,0,0)}
    **kwargs # 其他传递给 draw_single_detection 的参数
) -> np.ndarray:
    """
    在图像上绘制所有检测到的目标。

    参数:
    - image (np.ndarray): 要在其上绘制的OpenCV BGR图像。
    - detections_list (list): YOLODetectorOBB.detect() 返回的检测结果列表。
    - color_map (dict, optional): 类别名称到BGR颜色的映射。
    - **kwargs: 其他可以传递给 draw_single_detection 的参数
                (show_label, show_confidence, show_center, box_thickness, etc.)

    返回:
    - np.ndarray: 绘制了所有检测结果的图像副本。
    """
    output_image = image.copy()
    if not detections_list:
        return output_image

    for detection in detections_list:
        class_name = detection.get('class_name')
        class_id = detection.get('class_id', 0)
        
        obj_color = None
        if color_map and class_name in color_map:
            obj_color = color_map[class_name]
        else:
            obj_color = generate_random_color(class_id) # 如果没有指定颜色，则生成一个

        output_image = draw_single_detection(output_image, detection, color=obj_color, **kwargs)
            
    return output_image


# ==============================================================================
# 测试代码块
# ==============================================================================
if __name__ == '__main__':
    print("开始测试 visualization 模块...")

    # 1. 创建一个简单的测试背景图像
    test_bg_height, test_bg_width = 600, 800
    base_image = np.full((test_bg_height, test_bg_width, 3), (30, 30, 30), dtype=np.uint8) # 深灰色背景
    cv2.putText(base_image, "Visualization Test", (test_bg_width//2 - 150, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    # 2. 准备一些模拟的检测数据 (与 YOLODetectorOBB 输出格式一致)
    mock_detections = [
        {
            'class_id': 0, 'class_name': 'plane', 'confidence': 0.92,
            'obb_vertices': np.array([[100, 100], [300, 150], [280, 250], [80, 200]], dtype=np.int32),
            'center_xy': (190, 175)
        },
        {
            'class_id': 1, 'class_name': 'car', 'confidence': 0.78,
            'obb_vertices': np.array([[400, 300], [500, 320], [480, 420], [380, 400]], dtype=np.int32),
            'center_xy': (440, 360)
        },
        { # 一个标签可能超出边界的例子
            'class_id': 0, 'class_name': 'plane', 'confidence': 0.85,
            'obb_vertices': np.array([[20, 30], [120, 40], [110, 90], [10, 80]], dtype=np.int32),
            'center_xy': (65, 60)
        },
        { # 另一个例子，测试不同颜色
            'class_id': 2, 'class_name': 'ship', 'confidence': 0.65,
            'obb_vertices': np.array([[550, 100], [750, 120], [730, 220], [530, 200]], dtype=np.int32),
            'center_xy': (640, 160)
        }
    ]

    # 3. 定义颜色映射 (可选)
    custom_color_map = {
        'plane': (0, 255, 0),   # 绿色
        'car': (255, 0, 0),     # 蓝色
        # 'ship' 类没有在map中，将会使用随机颜色
    }

    # --- 测试 draw_single_detection ---
    print("\n--- 测试 draw_single_detection ---")
    image_single_test = base_image.copy()
    if mock_detections:
        # 只绘制第一个检测目标
        image_single_test = draw_single_detection(
            image_single_test, 
            mock_detections[0], 
            color=custom_color_map.get(mock_detections[0]['class_name']),
            show_center=True,
            box_thickness=3
        )
        cv2.imshow("Single Detection Test", image_single_test)
        cv2.waitKey(0)

    # --- 测试 draw_all_detections ---
    print("\n--- 测试 draw_all_detections ---")
    image_all_test = draw_all_detections(
        base_image.copy(), 
        mock_detections, 
        color_map=custom_color_map,
        show_label=True,
        show_confidence=True,
        show_center=True,
        font_scale=0.5,
        box_thickness=2,
        label_bg_alpha=0.6 # 半透明标签背景
    )
    cv2.imshow("All Detections Test", image_all_test)
    cv2.waitKey(0)

    # --- 测试不显示中心点和置信度 ---
    print("\n--- 测试自定义显示选项 ---")
    image_custom_options = draw_all_detections(
        base_image.copy(),
        mock_detections,
        color_map=custom_color_map,
        show_confidence=False,
        show_center=False,
        label_bg_alpha=0.0 # 完全透明标签背景
    )
    cv2.imshow("Custom Options Test", image_custom_options)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("\nVisualization 模块测试完成。")