# src/yolo_detector.py

import torch
from ultralytics import YOLO
import numpy as np
import cv2 # 主要用于测试时的图像加载

class YOLODetectorOBB:
    """
    一个封装了 Ultralytics YOLOv8-OBB 模型加载和推理的类。
    """
    def __init__(self, model_path: str):
        """
        初始化 YOLOv8-OBB 检测器。

        参数:
        - model_path (str): 本地 YOLOv8-OBB 模型文件 (.pt) 的路径。
        """
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"YOLO Detector: 使用设备 '{self.device}'")
        try:
            self.model = YOLO(model_path) # 加载模型
            # 将模型移至指定设备 (通常YOLO构造函数会自动处理，但明确指定也无妨)
            self.model.to(self.device) 
            print(f"YOLO Detector: 模型 '{model_path}' 加载成功。")
        except Exception as e:
            print(f"YOLO Detector: 模型 '{model_path}' 加载失败: {e}")
            self.model = None
            raise  # 重新引发异常，让调用者知道初始化失败

    def _calculate_obb_center(self, obb_points_8: np.ndarray) -> tuple:
        """
        从OBB的8个点坐标计算中心点。
        obb_points_8: [x1,y1,x2,y2,x3,y3,x4,y4]
        返回: (center_x, center_y)
        """
        if obb_points_8 is None or len(obb_points_8) != 8:
            return (0, 0) # 或者返回 None 并做相应处理
        points_reshaped = obb_points_8.reshape(4, 2)
        center_x = np.mean(points_reshaped[:, 0])
        center_y = np.mean(points_reshaped[:, 1])
        return (int(round(center_x)), int(round(center_y)))

    def _get_class_name_safely(self, res, class_id: int) -> str:
        """
        安全地获取类别名称，处理编码问题。
        
        参数:
        - res: YOLO结果对象
        - class_id: 类别ID
        
        返回:
        - str: 类别名称
        """
        try:
            if res.names and class_id in res.names:
                class_name = res.names[class_id]
                # 确保返回的字符串是有效的UTF-8编码
                if isinstance(class_name, bytes):
                    class_name = class_name.decode('utf-8', errors='replace')
                elif isinstance(class_name, str):
                    # 检查字符串是否包含无效字符
                    try:
                        class_name.encode('utf-8').decode('utf-8')
                    except UnicodeError:
                        # 如果编码有问题，使用默认名称
                        class_name = f"class_{class_id}"
                return class_name
            else:
                return f"class_{class_id}"
        except Exception as e:
            print(f"警告: 获取类别名称时出错 (class_id={class_id}): {e}")
            return f"class_{class_id}"

    def detect(self, image: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> list:
        """
        对输入的图像执行目标检测。

        参数:
        - image (np.ndarray): OpenCV BGR 格式的输入图像。
        - conf_threshold (float): 置信度阈值。
        - iou_threshold (float): OBB NMS（非极大值抑制）的IoU阈值。

        返回:
        - list: 一个字典列表，每个字典包含一个检测到的目标的信息:
            {
                'class_id': int,            # 类别ID
                'class_name': str,          # 类别名称
                'confidence': float,        # 置信度
                'obb_xyxyxyxy': np.ndarray, # OBB的8个点 [x1,y1,x2,y2,x3,y3,x4,y4]
                'obb_vertices': np.ndarray, # OBB的4个顶点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (方便绘制)
                'center_xy': tuple          # OBB的中心点 (cx, cy)
            }
          如果模型未加载或检测出错，返回空列表。
        """
        if self.model is None:
            print("YOLO Detector: 模型未加载，无法执行检测。")
            return []

        detections = []
        try:
            # 使用YOLO模型进行预测
            # 'source=image' 会让ultralytics处理图像到模型的转换
            # 'verbose=False' 减少控制台输出
            results = self.model.predict(
                source=image, 
                conf=conf_threshold, 
                iou=iou_threshold, 
                verbose=False,
                # imgsz=640, # 可以指定推理尺寸，如果需要的话
                # half=True, # 如果GPU支持且希望加速，可以开启半精度
            )

            if not results or not hasattr(results[0], 'obb') or results[0].obb is None:
                # print("YOLO Detector: 未检测到任何目标或结果不包含OBB信息。")
                return []

            # results[0] 对应第一张输入图像的结果
            res = results[0]
            
            # OBB坐标通常在 res.obb.xyxyxyxy (Tensor)
            # 类别ID在 res.obb.cls (Tensor)
            # 置信度在 res.obb.conf (Tensor)
            # 类别名称需要从 res.names 获取
            
            if res.obb.xyxyxyxy is not None and len(res.obb.xyxyxyxy) > 0:
                obb_coords_tensor = res.obb.xyxyxyxy.cpu().numpy() # [N, 8]
                class_ids_tensor = res.obb.cls.cpu().numpy()       # [N]
                confidences_tensor = res.obb.conf.cpu().numpy()    # [N]
                
                for i in range(len(obb_coords_tensor)):
                    class_id = int(class_ids_tensor[i])
                    class_name = self._get_class_name_safely(res, class_id)
                    confidence = float(confidences_tensor[i])
                    obb_8points = obb_coords_tensor[i] # [x1,y1,x2,y2,x3,y3,x4,y4]
                    
                    # 将8点坐标转换为4x2的顶点数组，方便后续处理（如绘制）
                    obb_vertices = obb_8points.reshape(4, 2).astype(np.int32)
                    
                    center_xy = self._calculate_obb_center(obb_8points)

                    detections.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'obb_xyxyxyxy': obb_8points, 
                        'obb_vertices': obb_vertices,
                        'center_xy': center_xy
                    })
            # else:
            #     print("YOLO Detector: res.obb.xyxyxyxy 为空或 None。")

        except Exception as e:
            print(f"YOLO Detector: 检测过程中发生错误: {e}")
            import traceback
            traceback.print_exc() # 打印详细的堆栈跟踪
            return []
            
        return detections

# ==============================================================================
# 测试代码块
# ==============================================================================
if __name__ == '__main__':
    print("开始测试 YOLODetectorOBB...")

    # --- 用户配置 ---
    # !!重要!!: 请将此路径替换为你本地的YOLOv8-OBB预训练模型或你训练好的模型路径
    # 例如，你可以从 Ultralytics 下载 yolov8n-obb.pt:
    # https://github.com/ultralytics/ultralytics/releases
    # 或者使用你基于DOTA数据集训练的模型
    MODEL_PATH = "../models_yolo/yolov8n-obb.pt" # <--- 修改这里为你模型的实际路径
    
    # !!重要!!: 请将此路径替换为你本地的测试图像路径
    # 这张图像最好包含可以用OBB模型检测到的物体（例如飞机，如果模型是DOTA训练的）
    IMAGE_PATH = "../data/images/P0022__682__1396___4537.jpg" # <--- 修改这里为你的测试图片路径

    # 检查模型和图像路径是否存在
    import os
    if not os.path.exists(MODEL_PATH) or not MODEL_PATH.endswith(".pt"):
        print(f"错误: YOLO模型文件 '{MODEL_PATH}' 不存在或非.pt文件。请下载yolov8n-obb.pt或使用你自己的模型，并更新路径。")
        print("可以从 https://github.com/ultralytics/ultralytics/releases 下载预训练的OBB模型。")
        exit()
    
    if not os.path.exists(IMAGE_PATH):
        print(f"错误: 测试图像文件 '{IMAGE_PATH}' 不存在。请提供一张测试图像并更新路径。")
        exit()

    # --- 测试流程 ---
    try:
        # 1. 初始化检测器
        print(f"\n1. 初始化检测器 (模型: {MODEL_PATH})...")
        detector = YOLODetectorOBB(model_path=MODEL_PATH)
        if detector.model is None:
            print("检测器初始化失败，退出测试。")
            exit()
        print("检测器初始化成功。")

        # 2. 加载测试图像
        print(f"\n2. 加载测试图像 ({IMAGE_PATH})...")
        test_image = cv2.imread(IMAGE_PATH)
        if test_image is None:
            print(f"无法加载测试图像 '{IMAGE_PATH}'。")
            exit()
        print(f"测试图像加载成功，尺寸: {test_image.shape[:2]} (H, W)")

        # 3. 执行检测
        print("\n3. 执行检测...")
        detected_objects = detector.detect(test_image, conf_threshold=0.3, iou_threshold=0.5)
        print(f"检测完成，找到 {len(detected_objects)} 个目标。")

        # 4. 打印检测结果
        if detected_objects:
            print("\n4. 检测到的目标详情:")
            for i, obj in enumerate(detected_objects):
                print(f"  目标 {i+1}:")
                print(f"    类别: {obj['class_name']} (ID: {obj['class_id']})")
                print(f"    置信度: {obj['confidence']:.4f}")
                print(f"    中心点: {obj['center_xy']}")
                # print(f"    OBB 8点坐标: {obj['obb_xyxyxyxy']}") # 较长，可选打印
                print(f"    OBB 顶点: {obj['obb_vertices'].tolist()}") 
        else:
            print("\n4. 未检测到满足条件的目标。")

        # 5. (可选) 可视化结果 - 需要 visualization.py 模块或简单的cv2绘制
        # 暂时在这里用cv2简单绘制以便快速验证
        if detected_objects and test_image is not None:
            vis_image = test_image.copy()
            for obj in detected_objects:
                vertices = obj['obb_vertices'].reshape((-1, 1, 2)) # OpenCV polylines 需要的格式
                cv2.polylines(vis_image, [vertices], isClosed=True, color=(0, 255, 0), thickness=2)
                
                center = obj['center_xy']
                cv2.circle(vis_image, center, radius=5, color=(0, 0, 255), thickness=-1)
                
                label = f"{obj['class_name']}: {obj['confidence']:.2f}"
                cv2.putText(vis_image, label, (vertices[0][0][0], vertices[0][0][1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 50), 2)
            
            cv2.imshow("YOLO OBB Detections", vis_image)
            print("\n按任意键关闭可视化窗口并结束测试...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"\n测试过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()

    print("\nYOLODetectorOBB 测试结束。")