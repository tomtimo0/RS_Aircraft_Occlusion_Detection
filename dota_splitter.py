from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 关键代码

from ultralytics.data.split_dota import split_trainval, split_test

print("开始切割 DOTA 数据集...")

# 切割训练/验证集（带标签）
split_trainval(
    data_root="E:/RS_Aircraft_Occlusion_Detection/dota_obb_dataset",      # DOTA 原始数据根目录
    save_dir="E:/RS_Aircraft_Occlusion_Detection/yolo_obb_dataset", # 切割后保存目录
    rates=[0.5, 1.0, 1.5],              # 多尺度切割
    gap=500                             # 切块重叠像素
)

print("切割完成，请检查输出目录。")