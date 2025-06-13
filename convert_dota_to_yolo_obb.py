#将DOTA标签转换为yolo obb标签
import cv2
import numpy as np
import math
import os

def convert_dota_to_yolo_obb(dota_line, image_width, image_height, class_mapping):
    # """
    # 将单行 DOTA 格式的标注转换为 YOLO OBB 格式。

    # 参数:
    # dota_line (str): DOTA 格式的一行标注字符串。
    # image_width (int): 对应图像的宽度（像素）。
    # image_height (int): 对应图像的高度（像素）。
    # class_mapping (dict): 类别名称到类别ID的映射字典,例如 {"plane": 0}。

    # 返回:
    # str: YOLO OBB 格式的标注字符串，如果转换失败或类别无效则返回 None。
    # """
    parts = dota_line.strip().split()
    if len(parts) < 9:
        print(f"警告：行 '{dota_line.strip()}' 的部分不足9个，跳过。")
        return None

    try:
        coords_px = np.array([float(c) for c in parts[:8]], dtype=np.float32).reshape(4, 2)
        category_name = parts[8].lower()
    except ValueError:
        print(f"警告：无法解析行 '{dota_line.strip()}' 中的坐标。跳过。")
        return None

    if category_name not in class_mapping:
        # print(f"提示：类别 '{category_name}' 不在定义的类别映射中，跳过此标注。")
        return None

    class_id = class_mapping[category_name]

    # 使用 OpenCV 计算最小旋转矩形
    rect = cv2.minAreaRect(coords_px)

    cx_px, cy_px = rect[0]   # 中心点坐标
    w_cv, h_cv = rect[1]     # 宽、高（OpenCV 定义）
    angle_deg_cv = rect[2]   # OpenCV 角度（范围 [-90, 0)）

    # Ultralytics YOLOv8-OBB 的角度约定：
    # - 角度是弧度制，范围 [0, π)
    # - 宽度是长边，高度是短边
    # - 角度是从 x 轴顺时针旋转到宽度向量的角度

    if w_cv < h_cv:
        # 如果 OpenCV 返回的 width < height，则交换宽高
        w_px = h_cv
        h_px = w_cv
        # 调整角度：-angle_deg_cv + 90° 再转为弧度
        angle_rad = (-angle_deg_cv + 90.0) * np.pi / 180.0
    else:
        w_px = w_cv
        h_px = h_cv
        # 否则直接将 OpenCV 的角度转为正数并转为弧度
        angle_rad = -angle_deg_cv * np.pi / 180.0

    # 归一化中心点坐标（相对于图像尺寸）
    x_center_norm = cx_px / image_width
    y_center_norm = cy_px / image_height
    # 归一化宽高
    width_norm = w_px / image_width
    height_norm = h_px / image_height

    # 限制归一化值在 [0, 1] 范围内
    x_center_norm = np.clip(x_center_norm, 0., 1.)
    y_center_norm = np.clip(y_center_norm, 0., 1.)
    width_norm = np.clip(width_norm, 0., 1.)
    height_norm = np.clip(height_norm, 0., 1.)

    # 角度归一化到 [0, π)
    angle_rad = angle_rad % np.pi
    if angle_rad < 0:
        angle_rad += np.pi

    return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f} {angle_rad:.6f}"


def process_dataset(input_label_dir, output_yolo_label_dir, image_dir, class_mapping):
    # """
    # 处理整个数据集,将DOTA标签转换为YOLO OBB格式。
    # """

    # 创建输出目录（如不存在）
    os.makedirs(output_yolo_label_dir, exist_ok=True)

    print("开始转换 DOTA 标签...")
    print(f"输入 DOTA 标签目录: {input_label_dir}")
    print(f"输出 YOLO OBB 标签目录: {output_yolo_label_dir}")
    print(f"图像目录 (用于获取尺寸): {image_dir}")
    print(f"类别映射: {class_mapping}")

    processed_files = 0
    skipped_files_no_image = 0
    skipped_files_img_read_error = 0
    total_labels_converted = 0

    # 获取所有 .txt 文件
    label_files = [f for f in os.listdir(input_label_dir) if f.endswith(".txt")]
    if not label_files:
        print(f"警告: 在输入目录 '{input_label_dir}' 中没有找到 .txt 标签文件。")
        return

    for filename in label_files:
        input_label_path = os.path.join(input_label_dir, filename)
        output_label_path = os.path.join(output_yolo_label_dir, filename)

        base_name, _ = os.path.splitext(filename)
        possible_image_names = [f"{base_name}.png", f"{base_name}.jpg", f"{base_name}.jpeg", f"{base_name}.bmp", f"{base_name}.tif"]
        image_path_found = None

        # 查找对应的图像文件
        for img_name_candidate in possible_image_names:
            potential_path = os.path.join(image_dir, img_name_candidate)
            if os.path.exists(potential_path):
                image_path_found = potential_path
                break

        if not image_path_found:
            skipped_files_no_image += 1
            continue

        # 读取图像尺寸
        img = cv2.imread(image_path_found)
        if img is None:
            skipped_files_img_read_error += 1
            continue
        img_h, img_w = img.shape[:2]

        converted_labels_in_file = 0
        with open(input_label_path, 'r') as f_in, open(output_label_path, 'w') as f_out:
            for line_num, dota_line in enumerate(f_in):
                # DOTA 文件前两行通常是头信息（imagesource 和 gsd），跳过
                if line_num < 2 and ("imagesource" in dota_line.lower() or "gsd" in dota_line.lower()):
                    continue

                yolo_obb_line = convert_dota_to_yolo_obb(dota_line, img_w, img_h, class_mapping)
                if yolo_obb_line:
                    f_out.write(yolo_obb_line + "\n")
                    converted_labels_in_file += 1

        if converted_labels_in_file > 0:
            processed_files += 1
            total_labels_converted += converted_labels_in_file

    # 打印统计信息
    print("\n--- 转换统计 ---")
    print(f"总共检查的标签文件数: {len(label_files)}")
    print(f"成功处理并生成输出的标签文件数: {processed_files}")
    print(f"因未找到对应图像而跳过的标签文件数: {skipped_files_no_image}")
    print(f"因图像读取错误而跳过的标签文件数: {skipped_files_img_read_error}")
    print(f"总共转换的有效标注行数: {total_labels_converted}")
    print("标签转换完成。")


if __name__ == '__main__':
    # --- 用户配置区域 ---
    # 1. 设置原始 DOTA 标签文件所在的目录
    INPUT_DOTA_LABEL_DIR = "E:/RS_Aircraft_Occlusion_Detection/dataset/val/original_dota_labels"

    # 2. 设置转换后的 YOLO OBB 格式标签文件输出的目录
    OUTPUT_YOLO_LABEL_DIR = "E:/RS_Aircraft_Occlusion_Detection/dataset/val/labels"

    # 3. 设置图像文件所在的目录（需要读取图像以获取其尺寸）
    IMAGE_DIR = "E:/RS_Aircraft_Occlusion_Detection/dataset/val/images"

    # 4. 定义类别名称到 ID 的映射
    # 键是 DOTA 标签中的类别名（小写），值是 YOLO 中使用的整数 ID（从 0 开始）
    CLASS_MAPPING = {
        "plane": 0,
        "aircraft": 0
        # "small-vehicle": 1, # 如果想检测小车，给它ID 1
        # "large-vehicle": 2, # 如果想检测大车，给它ID 2
    }
    # --- 用户配置区域结束 ---

    # 检查路径是否存在
    if not os.path.isdir(INPUT_DOTA_LABEL_DIR):
        print(f"错误: 输入DOTA标签目录 '{INPUT_DOTA_LABEL_DIR}' 不存在。请检查路径。")
    elif not os.path.isdir(IMAGE_DIR):
        print(f"错误: 图像目录 '{IMAGE_DIR}' 不存在。请检查路径。")
    else:
        process_dataset(INPUT_DOTA_LABEL_DIR, OUTPUT_YOLO_LABEL_DIR, IMAGE_DIR, CLASS_MAPPING)