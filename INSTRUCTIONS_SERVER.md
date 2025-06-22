# 服务器训练与评估指令

本文档提供了在服务器上运行此项目所需的指令，所有路径均已调整为相对路径，确保了可移植性。

## 训练流程

### 1. (可选) 生成带遮挡的数据集

如果你需要重新生成带遮挡的数据集，请运行此脚本。它会读取 `yolo_obb_dataset_original` 的内容，在图像上添加模拟高斯光斑，并将结果保存到 `yolo_obb_dataset_occluded` 目录。

```bash
python spot/generate.py --data-dir ./yolo_obb_dataset_original --output-dir ./yolo_obb_dataset_occluded
```
**注意**: 运行此脚本后，请确保 `yolo_obb_dataset_occluded` 目录中也有一个相应的 `dota_planes.yaml` 文件。你可以从 `yolo_obb_dataset_original` 目录中复制一份过来。

### 2. 训练基线模型 (在无遮挡数据集上)

```bash
yolo obb train data=./yolo_obb_dataset_original/dota_planes.yaml model=yolov8n-obb.pt epochs=50 imgsz=620 device=0 batch=16 name=dota_plane_baseline
```

### 3. 训练遮挡模型 (在有遮挡数据集上)

```bash
# 在 tmux 或 nohup 中运行!
yolo detect train \
    data=/home/matting/RS_Aircraft_Occlusion_Detection/yolo_obb_dataset_occluded/dota_planes.yaml \
    model=yolov8l-obb.pt \
    epochs=100 \
    imgsz=640 \
    device=0,1,2,3 \
    batch=128 \
    workers=32 \
    amp=True \
    lr0=0.01 \
    warmup_epochs=5
```

```bash
    # -s 后面是你的会话名字，可以自定义，比如 'yolo_training'
    tmux new -s yolo_training
    tmux ls
    tmux attach -t yolo_training
    #重新连接到会话里之后，像正常一样用Ctrl + c停止你的程序，然后输入exit或按Ctrl + d即可关闭该tmux会话
```

## 评估流程

评估时，请确保 `model=` 指向正确的模型权重路径（通常在 `runs/obb/.../weights/best.pt`）。

```bash
# 1. 基线模型 vs 遮挡数据 (测试基线模型在恶劣环境下的表现)
yolo obb val model=runs/obb/dota_plane_baseline/weights/best.pt data=./yolo_obb_dataset_occluded/dota_planes.yaml name=baseline_on_occluded

# 2. 遮挡模型 vs 无遮挡数据 (测试遮挡模型是否过拟合，泛化能力如何)
yolo obb val model=runs/obb/dota_plane_occluded/weights/best.pt data=./yolo_obb_dataset_original/dota_planes.yaml name=occluded_on_baseline

# 3. 遮挡模型 vs 遮挡数据 (获取遮挡模型在目标任务上的最终性能)
yolo obb val model=runs/obb/dota_plane_occluded/weights/best.pt data=./yolo_obb_dataset_occluded/dota_planes.yaml name=occluded_on_occluded
``` 