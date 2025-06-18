from ultralytics import YOLO

# 加载官方预训练模型（可选n, s, m, l, x等版本）
model = YOLO("yolov8n-obb.pt")  # 也可用yolo11n.pt等

# 对单张图片推理
results = model("testimage/P0000.png")
results[0].show()  # 可视化结果