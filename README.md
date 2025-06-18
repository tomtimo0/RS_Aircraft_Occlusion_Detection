# 《模式识别与机器学习》课程设计

[课程设计要求、项目详情、考核与报告规范（点击查看）](./project_spec.md)

[Git、GitHub操作与团队协作指南（点击查看）](./collab_guide.md)

[项目技术路线概要](./technical_path_summary_updated.md)

## Usage

```bash
yolo obb train data=E:/RS_Aircraft_Occlusion_Detection/yolo_obb_dataset/dota_planes.yaml model=yolov8n-obb.pt epochs=50 imgsz=1024 device=0 batch=4 name=dota_plane_baseline
```
```bash
python E:/RS_Aircraft_Occlusion_Detection/spot/generate.py --data-dir E:/RS_Aircraft_Occlusion_Detection/testimage --output-dir E:/RS_Aircraft_Occlusion_Detection/testoutput
```
训练跑通了但显存炸了，下一步考虑加光斑，减小batch。即便指定了yolov8n也会下载yolo11n，很奇怪。
实现裁剪，光斑生成函数。
实现遮挡指标函数。