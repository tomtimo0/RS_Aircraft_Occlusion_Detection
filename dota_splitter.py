from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 关键代码

from ultralytics.data.split_dota import split_trainval, split_test

print("开始切割 DOTA 数据集...")

# 切割训练/验证集（带标签）
split_trainval(
    data_root="E:/RemoteSensingObjectDetectionApp/cut",      # DOTA 原始数据根目录
    save_dir="E:/RemoteSensingObjectDetectionApp/res", # 切割后保存目录
    gap=0,                           # 切块重叠像素
    crop_size=300,
    rates=[1.0] 
)

print("切割完成，请检查输出目录。")