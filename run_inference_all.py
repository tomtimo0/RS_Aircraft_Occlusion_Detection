# run_inference.py

import os
from ultralytics import YOLO

# --- 1. 配置 ---
model_path = "yolov8n-obb.pt"
# 直接将整个文件夹路径作为输入
input_folder = r"E:\RS_Aircraft_Occlusion_Detection\dataset\train_split_with_multi_glare\images_test"

# --- 2. 加载模型 ---
print(f"正在加载模型: {model_path}")
model = YOLO(model_path)

# --- 3. 对整个文件夹进行推理 ---
print(f"开始对文件夹 '{input_folder}' 中的所有图片进行推理...")

# 推理结果会自动保存在 runs/detect/predict* 目录下
# stream=True 适用于处理大量图片或视频，可以节省内存
results = model(input_folder, save=True, stream=True)

# 遍历处理结果
count = 0
for result in results:
    count += 1
    print(f"  处理图片 {count}: {os.path.basename(result.path)}")
    # 你可以在这里添加对每个结果的单独处理逻辑
    # 例如，只打印检测到目标的图片信息
    if len(result.obb) > 0:
        print(f"    -> 检测到 {len(result.obb)} 个目标。")

print(f"\n全部推理完成！共处理了 {count} 张图片。")
print("结果已保存在最新的 'runs_test/detect/' 文件夹中。")