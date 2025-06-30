# 遥感图像飞机目标检测与遮挡模拟分析应用

**项目状态：** ✅ 已完成

## 项目简介

本项目是一个基于 Streamlit 的交互式 Web 应用，旨在为遥感图像中的飞机目标检测提供遮挡模拟与分析功能。用户可以上传自己的遥感图像和对应的标注文件，通过在图像上实时、交互式地生成和调整高斯模糊遮挡，来直观地评估和对比不同 YOLOv8-OBB 模型在遮挡条件下的检测性能。

应用的核心功能在于建立一个**从"模拟遮挡"到"模型响应"的快速、可视化反馈闭环**，帮助研究人员和开发者高效地分析模型对不同程度、位置遮挡的鲁棒性。

### GIF 演示

*(此处未来可插入一个展示应用核心交互功能的GIF动图)*

![HUST-Logo](hust_logo.png)

## 主要功能

*   **交互式遮挡模拟**:
    *   在上传的图片上**实时点击**以确定遮挡中心。
    *   通过滑块**实时调整**高斯遮挡的强度 (Amplitude)、范围 (Sigma) 和旋转角度 (Rotation)。
    *   所有调整**即时生效**，提供流畅的"所见即所得"体验。
*   **多模型对比分析**:
    *   支持**同时加载多个**本地 YOLOv8-OBB 模型（`.pt` 文件）。
    *   在同一张遮挡图像上运行所有选定模型，并将结果**并排展示**。
    *   每个模型使用**不同颜色的边界框**进行区分，便于直观对比。
*   **真值（Ground Truth）集成**:
    *   支持上传与图像配套的 YOLO OBB 格式标签文件（`.txt`）。
    *   在结果图上绘制**真值框**（红色），作为评估基准。
*   **实时 OOAP 计算**:
    *   在调整遮挡区域时，应用会**实时计算**并显示每个真值目标被遮挡的面积百分比 (Object-Occlusion Area Percentage, OOAP)。
    *   OOAP 值直接标注在对应的真值框旁边，将遮挡程度**数据化、可视化**。
*   **结果可视化与导出**:
    *   清晰展示原始图像、遮挡后的图像以及各个模型的检测结果图。
    *   检测结果图包含**模型预测框（带置信度）、真值框和 OOAP 值**。

## 技术栈

*   **核心框架**: Streamlit
*   **深度学习**: PyTorch, Ultralytics (YOLOv8-OBB)
*   **图像处理**: OpenCV, Pillow
*   **科学计算**: NumPy

## 项目结构

```
RemoteSensingObjectDetectionApp/
├── app_streamlit.py                # Streamlit 应用主程序
├── requirements.txt                # Python 依赖
├── config.yaml                     # 应用配置文件
├── models_yolo/                    # 存放 YOLOv8-OBB 模型 (.pt)
├── data/                           # 存放示例图片和标签
│   ├── images/
│   └── labels/
├── src/                            # 后端功能模块
│   ├── image_utils.py              # 图像处理与遮挡生成工具
│   ├── yolo_detector.py            # YOLOv8 检测器封装
│   ├── visualization.py            # 结果可视化工具
│   └── metrics.py                  # 性能度量工具 (如 OOAP 计算)
├── hust_logo.png                   # Logo
├── 启动APP.bat                     # Windows 快速启动脚本
└── 安装依赖.bat                    # Windows 快速安装依赖脚本
```

## 本地部署与运行

1.  **克隆或下载项目**

2.  **安装依赖**:
    *   建议首先创建一个虚拟环境（如 venv 或 conda）。
    *   在 Windows 系统下，可以直接运行根目录下的 `安装依赖.bat` 脚本。
    *   或者手动通过 pip 安装：
        ```bash
        pip install -r requirements.txt
        ```

3.  **准备模型与数据**:
    *   将你的 YOLOv8-OBB 模型文件（`.pt`）放入 `models_yolo` 文件夹。
    *   （可选）将你的测试图片放入 `data/images`，对应的 YOLO OBB 标签（`.txt`）放入 `data/labels`。

4.  **启动应用**:
    *   在 Windows 系统下，可以直接运行根目录下的 `启动APP.bat` 脚本。
    *   或者在项目根目录下打开终端，运行以下命令：
        ```bash
        streamlit run app_streamlit.py
        ```

5.  **浏览器访问**:
    *   应用启动后，终端会显示一个 URL (通常是 `http://localhost:8501`)，在浏览器中打开即可使用。