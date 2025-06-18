from ultralytics import YOLO
import multiprocessing
import torch  # 导入torch，可能需要检查或设置其多进程行为
import platform  # 用于检查操作系统

# ----------------------------------------------------------------------
# 将所有核心逻辑封装在一个函数中
# ----------------------------------------------------------------------
def run_yolo_validation():
    print(f"脚本在平台启动: {platform.system()}")  # 显示当前操作系统（Windows/macOS/Linux）
    print(f"当前使用的多进程启动方法: {multiprocessing.get_start_method(allow_none=True)}")
    # 在 Windows 上，推荐使用 spawn 方法，以避免一些子进程创建问题。

    print("正在初始化YOLO模型...")
    # 选择已训练好的模型路径
    model_path = 'runs/train/yolov8n_obb_train_v3/weights/best.pt'
    try:
        model = YOLO(model_path)
        print(f"模型 '{model_path}' 加载成功。")
    except Exception as e:
        print(f"致命错误: 加载YOLO模型时出错: {e}")
        print("请确保模型路径正确且模型文件可访问。")
        return  # 如果模型加载失败，则退出

    # 验证项目的名称
    project_name = 'runs/val'
    exp_name = 'yolov8n_obb_val_v'
    data_config = 'dota_aircraft.yaml'

    print(f"\n开始验证...")
    print(f"  数据集配置: {data_config}")
    print(f"  项目: {project_name}")
    print(f"  实验名称: {exp_name}")
    print(f"  输入图像大小: 1024")
    print(f"  批量大小: 4")

    # 这里设置为 0 表示不使用多进程加载数据（即单线程加载），主要用于调试。
    # 在 Windows 或打包环境中，多进程加载容易导致错误，因此从这里开始排查。
    num_workers = 0
    print(f"  DataLoader 使用的 num_workers: {num_workers} (0 表示没有用于数据加载的子进程)")

    try:
        metrics = model.val(
            data=data_config,
            imgsz=1024,
            batch=4,
            split='val',
            project=project_name,
            name=exp_name,
            workers=num_workers,  # <--- 将 workers 参数传递给 val 方法
            # verbose=True,       # 可以尝试开启详细输出，看是否有更多信息
        )
        print("\nmodel.val() 完成。")
    except RuntimeError as e:
        print(f"\n致命错误: model.val() 运行期间发生运行时错误: {e}")
        print("这强烈表明存在多进程问题，即使有 'if __name__ == \"__main__\":'。")
        if num_workers > 0:
            print("建议：尝试在 model.val() 中设置 'workers=0' 禁用数据加载的多进程。")
        else:
            print("即使设置了 'workers=0' 问题仍然存在。这不常见，可能指向更深层次的问题或错误。")
        return
    except Exception as e:
        print(f"\n致命错误: model.val() 运行期间发生了意外错误: {e}")
        return

    # 打印验证指标
    print("\n验证指标:")
    if metrics and hasattr(metrics, 'box') and hasattr(metrics.box, 'map') and metrics.box.map is not None:
        print(f"  mAP50-95(B): {metrics.box.map:.4f}")
        print(f"  mAP50(B): {metrics.box.map50:.4f}")
        print(f"  mAP75(B): {metrics.box.map75:.4f}")
    else:
        print("  无法获取有效的 box 指标或 metrics.box.map 为空。")
        if metrics:
            print(f"  原始指标对象: {metrics}")
            if hasattr(metrics, 'box'):
                print(f"  Box 指标内容: {metrics.box}")
            else:
                print("  指标对象没有 'box' 属性。")
        else:
            print("  指标对象为空。")

# ----------------------------------------------------------------------
# 主程序入口点
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 1. freeze_support() 是第一件事，尤其是在 Windows 或打包时
    #    它应该在任何多进程代码之前被调用。
    multiprocessing.freeze_support()
    print("调用了 multiprocessing.freeze_support()。")

    # 2. 调用包含所有核心逻辑的函数
    run_yolo_validation()

    print("\n脚本结束。")