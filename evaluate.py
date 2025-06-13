from ultralytics import YOLO

#选择已训练好的模型路径
model = YOLO('runs/obb/train/yolov8n_obb_train_test_v1/weights/best.pt')

# 验证项目的名称，结果会保存在 runs/obb/val/project_name 目录下
project_name = 'runs/obb/val'
# 本次实验的名称，结果会保存在 project_name/exp_name 目录下,对于不同的训练实验应更改不同实验name
exp_name = 'yolov8n_obb_val_test_v1' 

# 验证模型
metrics = model.val(
    data='dota_aircraft.yaml',  # 指定用于验证的数据集配置文件
    imgsz=640,            # 输入图像的尺寸，应与训练时使用的图像尺寸一致或兼容
    batch=4,              # 每批次处理的图像数量，根据你的GPU显存进行调整
    split='val',          # 指定使用数据集配置文件中定义的 'val' (验证集) 部分进行评估
    project=project_name,
    name=exp_name 
)

# 打印验证指标
print("验证指标 (Validation Metrics):")
# mAP50-95(B): 表示在 IoU (Intersection over Union) 阈值从 0.5 到 0.95，步长为 0.05 的范围内的平均 mAP (mean Average Precision)。
# (B) 通常表示这是针对边界框 (Box) 的指标，对于 OBB (Oriented Bounding Box) 任务，这是旋转框的 mAP。
print(f"mAP50-95(B): {metrics.box.map}")
# mAP50(B): 表示在 IoU 阈值为 0.5 时的 mAP。
print(f"mAP50(B): {metrics.box.map50}")
# mAP75(B): 表示在 IoU 阈值为 0.75 时的 mAP。这是一个更严格的指标。
print(f"mAP75(B): {metrics.box.map75}")