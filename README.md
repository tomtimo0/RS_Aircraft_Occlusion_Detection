# 《模式识别与机器学习》课程设计

[课程设计要求、项目详情、考核与报告规范（点击查看）](./project_spec.md)

[Git、GitHub操作与团队协作指南（点击查看）](./collab_guide.md)

[项目技术路线概要](./technical_path_summary_updated.md)

[阶段性成果](./results.md)检出率很高，但基线场景下误检率偏高

## Usage
## 训练用例
### 1. 训练基线模型 (在无遮挡数据集上)
```bash
yolo obb train data=E:/RS_Aircraft_Occlusion_Detection/yolo_obb_dataset_original/dota_planes.yaml model=yolov8n-obb.pt epochs=50 imgsz=620 device=0 batch=16 name=dota_plane_baseline
```
### 2. 生成带遮挡的数据集
```bash
python E:/RS_Aircraft_Occlusion_Detection/spot/generate.py --data-dir E:/RS_Aircraft_Occlusion_Detection/yolo_obb_dataset_original --output-dir E:/RS_Aircraft_Occlusion_Detection/yolo_obb_dataset
```
### 3. 训练遮挡模型 (在有遮挡数据集上)
```bash
yolo obb train data=E:/RS_Aircraft_Occlusion_Detection/yolo_obb_dataset/dota_planes.yaml model=yolov8n-obb.pt epochs=50 imgsz=620 device=0 batch=16 name=dota_plane_baseline3
```
## 评估用例
```bash
# 1. 基线模型 vs 遮挡数据 (测试基线模型在恶劣环境下的表现)
yolo obb val model=runs/obb/dota_plane_baseline/weights/best.pt data=E:/RS_Aircraft_Occlusion_Detection/yolo_obb_dataset/dota_planes.yaml name=baseline_on_occluded_real

# 2. 遮挡模型 vs 无遮挡数据 (测试遮挡模型是否过拟合，泛化能力如何)
yolo obb val model=runs/obb/dota_plane_baseline3/weights/best.pt data=E:/RS_Aircraft_Occlusion_Detection/yolo_obb_dataset_original/dota_planes.yaml name=occluded_on_baseline

# 3. 遮挡模型 vs 遮挡数据 (获取遮挡模型在目标任务上的最终性能)
yolo obb val model=runs/obb/dota_plane_baseline3/weights/best.pt data=E:/RS_Aircraft_Occlusion_Detection/yolo_obb_dataset/dota_planes.yaml name=occluded_on_occluded

# 4. 基线模型 vs 无遮挡数据 
yolo obb val model=runs/obb/dota_plane_baseline/weights/best.pt data=E:/RS_Aircraft_Occlusion_Detection/yolo_obb_dataset_original/dota_planes.yaml name=baseline_on_basline
```
git clone https://github.com/tomtimo0/RS_Aircraft_Occlusion_Detection.git