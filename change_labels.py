import os
import cv2
import numpy as np

def convert_dota_to_yolo_obb(input_label_dir, output_label_dir, image_dir, class_mapping):
    label_files = [f for f in os.listdir(input_label_dir) if f.endswith('.txt')]
    processed_files = 0
    skipped_files_no_image = 0
    skipped_files_img_read_error = 0
    total_labels_converted = 0

    for label_file in label_files:
        label_path = os.path.join(input_label_dir, label_file)
        output_label_path = os.path.join(output_label_dir, label_file)
        image_path = os.path.join(image_dir, os.path.splitext(label_file)[0] + '.png')

        if not os.path.exists(image_path):
            print(f"跳过标签文件 {label_file}: 对应的图像文件未找到")
            skipped_files_no_image += 1
            continue

        try:
            image = cv2.imread(image_path)
            height, width, _ = image.shape
        except Exception as e:
            print(f"跳过标签文件 {label_file}: 图像读取错误 {e}")
            skipped_files_img_read_error += 1
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()

        with open(output_label_path, 'w') as f_out:
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 10:
                    print(f"忽略损坏的标签: {label_path} - 标签需要 10 列，检测到 {len(parts)} 列")
                    continue
                
                x1, y1 = float(parts[0]), float(parts[1])
                x2, y2 = float(parts[2]), float(parts[3])
                x3, y3 = float(parts[4]), float(parts[5])
                x4, y4 = float(parts[6]), float(parts[7])
                class_name = parts[8]
                rotation = int(parts[9])  # 这里假设类别名称和旋转标志是分开的，如果合在一起则需要调整

                class_id = class_mapping.get(class_name.lower())
                if class_id is None:
                    print(f"忽略损坏的标签: {label_path} - 未知类别 {class_name}")
                    continue
                
                # 归一化角点坐标
                x1_norm, y1_norm = x1 / width, y1 / height
                x2_norm, y2_norm = x2 / width, y2 / height
                x3_norm, y3_norm = x3 / width, y3 / height
                x4_norm, y4_norm = x4 / width, y4 / height

                f_out.write(f"{class_id} {x1_norm} {y1_norm} {x2_norm} {y2_norm} {x3_norm} {y3_norm} {x4_norm} {y4_norm}\n")
                total_labels_converted += 1

        processed_files += 1

    print(f"总共检查的标签文件数: {len(label_files)}")
    print(f"成功处理并生成输出的标签文件数: {processed_files}")
    print(f"因未找到对应图像而跳过的标签文件数: {skipped_files_no_image}")
    print(f"因图像读取错误而跳过的标签文件数: {skipped_files_img_read_error}")
    print(f"总共转换的有效标注行数: {total_labels_converted}")
    print("标签转换完成。")

if __name__ == '__main__':
    # --- 用户配置区域 ---
    # 1. 设置原始 DOTA 标签文件所在的目录
    INPUT_DOTA_LABEL_DIR = "E:/python/huangnanqi/datasets/val/original_dota_labels"

    # 2. 设置转换后的 YOLO OBB 格式标签文件输出的目录
    OUTPUT_YOLO_LABEL_DIR = "E:/python/huangnanqi/datasets/val/labels"

    # 3. 设置图像文件所在的目录（需要读取图像以获取其尺寸）
    IMAGE_DIR = "E:/python/huangnanqi/datasets/val/images"

    # 4. 定义类别名称到 ID 的映射
    # 键是 DOTA 标签中的类别名（小写），值是 YOLO 中使用的整数 ID（从 0 开始）
    CLASS_MAPPING = {
        "plane": 0,
        "aircraft": 0
    }

    convert_dota_to_yolo_obb(INPUT_DOTA_LABEL_DIR, OUTPUT_YOLO_LABEL_DIR, IMAGE_DIR, CLASS_MAPPING)
