import os
from ultralytics import YOLO
from multiprocessing import freeze_support # 导入 freeze_support

def main(): # 将主要逻辑封装在一个函数中

    yaml_path = 'dota_aircraft.yaml'
    
    # 加载预训练模型
    model = YOLO('yolov8n-obb.pt')  # OBB专用模型
    
    # 训练项目的名称，结果会保存在 runs/train/project_name 目录下
    project_name = 'runs/train'
    # 本次实验的名称，结果会保存在 project_name/exp_name 目录下,对于不同的训练实验应更改不同实验name
    exp_name = 'yolov8n_obb_train_v1' 

    # 开始训练
    results = model.train(
        data= yaml_path,  # 数据集配置文件
        task= 'obb',    #指定任务类型为 OBB (旋转目标检测)
        # 训练参数
        epochs=1,        # 训练历元总数,每个历元代表对整个数据集进行一次完整的训练
        imgsz=640,         # 用于训练的图像目标尺寸
        batch=4,          # 批量大小
        # patience=30,       # 当性能趋于稳定时提前停止训练的轮次，防止过拟合
        device=0,         # 自动选择最空闲的GPU # 可以尝试明确指定 device=0 或 device='cpu'
        workers=8,         # 加载数据的工作线程数，影响预处理和输入速度
        optimizer='AdamW', # 推荐优化器
        lr0=0.001,         # 初始学习率
        weight_decay=0.05, # L2正则化，对大权重进行惩罚，防止过拟合
        warmup_epochs=3,   # 学习率预热的历元数，学习率从低值逐渐增加到初始学习率，以在早期稳定训练

        # 数据增强参数
        # degrees=45.0,    # 旋转角度
        # fliplr=0.5,      # 水平翻转概率50%
        # mosaic=1.0,      # 提升小目标检测
        # hsv_h=0.02,      # 色调增强2.0%（适应不同光照）
        # hsv_s=0.7,       # 饱和度增强70%（增强色彩差异）
        # hsv_v=0.4,       # 亮度增强40%（适应不同亮度）
        # scale=0.5,       # 缩放范围±50%（多尺度训练）
        # copy_paste=0.2,  # 小目标增强
        
        project=project_name,
        name=exp_name
    )

    print(f"训练完成。最佳模型保存在: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    freeze_support() # 在 Windows 上创建冻结的可执行文件时需要
    main()           # 调用主函数