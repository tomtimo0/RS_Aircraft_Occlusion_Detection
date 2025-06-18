import yaml

config = {
    #数据集路径配置
    'path': 'E:/RS_Aircraft_Occlusion_Detection/dataset',
    'train': 'train_split_with_targeted_glare1/images',#训练集图像路径
    'val': 'val_split_with_targeted_glare1/images',#验证集图像路径

    #类别信息
    'nc': 1,
    'names': ['aircraft'],

}

with open('dota_aircraft.yaml', 'w') as f:
    f.write("# YOLOv8 OBB 配置文件\n\n")
    for key, value in config.items():
        if isinstance(value, dict):
            f.write(f"{key}:\n")
            for k, v in value.items():
                f.write(f"  {k}: {v}\n")
        elif isinstance(value, list):
            f.write(f"{key}:\n")
            for item in value:
                f.write(f"  - {item}\n")
        else:
            f.write(f"{key}: {value}\n")

print("YAML配置文件已生成!")