from ultralytics import YOLO
import multiprocessing
import torch # 导入torch，可能需要检查或设置其多进程行为
import platform # 用于检查操作系统

# ----------------------------------------------------------------------
# 将所有核心逻辑封装在一个函数中
# ----------------------------------------------------------------------
def run_yolo_validation():
    print(f"Script starting on platform: {platform.system()}")
    print(f"Current multiprocessing start method: {multiprocessing.get_start_method(allow_none=True)}")

    # 尝试显式设置多进程启动方法为 'spawn' (主要针对Windows/macOS)
    # 这应该在任何进程启动之前完成，所以放在这里

    print("Initializing YOLO model...")
    # 选择已训练好的模型路径
    model_path = 'runs/train/yolov8n_obb_train_v1/weights/best.pt'
    try:
        model = YOLO(model_path)
        print(f"Model '{model_path}' loaded successfully.")
    except Exception as e:
        print(f"FATAL: Error loading YOLO model: {e}")
        print("Please ensure the model path is correct and the model file is accessible.")
        return # 如果模型加载失败，则退出

    # 验证项目的名称
    project_name = 'runs/val'
    exp_name = 'yolov8n_obb_val_o'
    data_config = 'dota_aircraft.yaml'

    print(f"\nStarting validation...")
    print(f"  Dataset config: {data_config}")
    print(f"  Project: {project_name}")
    print(f"  Experiment Name: {exp_name}")
    print(f"  Input image size: 640")
    print(f"  Batch size: 4")

    # --- 关键调试点：尝试将 workers 设置为 0 ---
    # 这将禁用数据加载的多进程，如果错误消失，则问题与 DataLoader 的多进程有关。
    num_workers = 0 # <--- 重要：从这里开始调试
    print(f"  Using num_workers for DataLoader: {num_workers} (0 means no subprocesses for data loading)")

    try:
        metrics = model.val(
            data=data_config,
            imgsz=640,
            batch=4,
            split='val',
            project=project_name,
            name=exp_name,
            workers=num_workers,  # <--- 将 workers 参数传递给 val 方法
            # verbose=True,       # 可以尝试开启详细输出，看是否有更多信息
        )
        print("\nmodel.val() completed.")
    except RuntimeError as e:
        print(f"\nFATAL: RuntimeError during model.val(): {e}")
        print("This strongly indicates a multiprocessing issue, even with 'if __name__ == \"__main__\":'.")
        if num_workers > 0:
            print("RECOMMENDATION: Try setting 'workers=0' in model.val() to disable multiprocessing for data loading.")
        else:
            print("Issue persists even with 'workers=0'. This is unusual and might point to a deeper issue or a bug.")
        return
    except Exception as e:
        print(f"\nFATAL: An unexpected error occurred during model.val(): {e}")
        return

    # 打印验证指标
    print("\n验证指标 (Validation Metrics):")
    if metrics and hasattr(metrics, 'box') and hasattr(metrics.box, 'map') and metrics.box.map is not None:
        print(f"  mAP50-95(B): {metrics.box.map:.4f}")
        print(f"  mAP50(B): {metrics.box.map50:.4f}")
        print(f"  mAP75(B): {metrics.box.map75:.4f}")
    else:
        print("  Could not retrieve valid box metrics or metrics.box.map was None.")
        if metrics:
            print(f"  Raw metrics object: {metrics}")
            if hasattr(metrics, 'box'):
                print(f"  Box metrics content: {metrics.box}")
            else:
                print("  Metrics object does not have 'box' attribute.")
        else:
            print("  Metrics object is None.")

# ----------------------------------------------------------------------
# 主程序入口点
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 1. freeze_support() 是第一件事，尤其是在 Windows 或打包时
    #    它应该在任何多进程代码之前被调用。
    multiprocessing.freeze_support()
    print("multiprocessing.freeze_support() called.")

    # 2. 调用包含所有核心逻辑的函数
    run_yolo_validation()

    print("\nScript finished.")