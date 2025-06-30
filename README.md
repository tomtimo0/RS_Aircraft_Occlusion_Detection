# 遥感图像飞机目标检测与遮挡模拟APP

本项目旨在开发一个应用程序，用于在遥感图像中检测飞机等目标，并特别关注在模拟遮挡（通过高斯光斑）条件下的检测性能。应用支持加载本地YOLO模型（特别是支持旋转边界框的版本），并在图像上可视化检测结果，包括旋转框和目标中心点。

## 功能特性

*   **高斯光斑模拟遮挡**: 在图像指定位置添加可调节参数（位置、大小、强度、方向）的高斯光斑。
*   **本地YOLO模型调用**: 支持加载用户提供的本地YOLO模型文件进行目标检测。
*   **旋转目标检测**: 专注于处理和可视化旋转边界框 (OBB)。
*   **结果可视化**: 在输出图像上绘制旋转框和目标中心点。

## 技术栈

*   Python 3.x
*   PyTorch (通过 `ultralytics` 库使用YOLOv8-OBB)
*   OpenCV-Python
*   NumPy

## 项目结构
RemoteSensingObjectDetectionApp/
├── README.md
├── data/
│ └── images/
├── models_yolo/
├── src/
│ ├── init.py
│ ├── image_utils.py
│ ├── yolo_detector.py
│ ├── visualization.py
│ └── main_cli.py
├── results/
├── requirements.txt
└── configs/ (可选)

## 安装与运行 (待补充)

### 环境设置

(后续补充详细步骤)

### 运行指令

(后续补充详细步骤)

## To-Do / 开发计划

1.  [x] 项目初始化和结构创建。
2.  [ ] 实现高斯光斑生成模块 (`src/image_utils.py`)。
3.  [ ] 实现YOLO模型加载与推理模块 (`src/yolo_detector.py`)，支持旋转框。
4.  [ ] 实现结果可视化模块 (`src/visualization.py`)，绘制旋转框和中心点。
5.  [ ] 编写命令行主程序 (`src/main_cli.py`) 串联所有功能。
6.  [ ] 准备示例图像和模型。
7.  [ ] 编写详细的运行和使用说明。
8.  [ ] 增加配置文件支持。
9.  [ ] 考虑简单的GUI界面 (e.g., Streamlit)。