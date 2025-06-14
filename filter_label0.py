import os
from pathlib import Path
from tqdm import tqdm # 用于显示进度条，可选，需 pip install tqdm

def filter_labels_by_class_index(labels_dir_root: str, target_class_index: int = 0):
    """
    遍历指定目录下的所有 .txt 标签文件，
    只保留那些以 target_class_index 开头的行。
    此函数会直接修改原文件，请务必在操作前备份你的标签数据！

    Args:
        labels_dir_root (str): 包含标签文件的根目录，
                               脚本会递归查找该目录及其子目录下的 .txt 文件。
                               例如: 'E:/RS_Aircraft_Occlusion_Detection/yolo_obb_dataset/labels'
        target_class_index (int): 希望保留的目标类别索引。
    """
    root_path = Path(labels_dir_root)
    if not root_path.is_dir():
        print(f"错误: 目录 '{labels_dir_root}' 不存在。")
        return

    print(f"开始处理目录 '{labels_dir_root}' 下的标签文件...")
    print(f"只保留类别索引为 '{target_class_index}' 的行。")
    print("重要提示：此操作将直接修改原始标签文件！建议先备份数据。")
    input("按 Enter键 继续，或按 Ctrl+C 中止...")


    # 使用 rglob 递归查找所有子目录中的 .txt 文件
    label_files = list(root_path.rglob("*.txt"))

    if not label_files:
        print(f"在 '{labels_dir_root}' 及其子目录中没有找到 .txt 标签文件。")
        return

    modified_files_count = 0
    processed_files_count = 0

    for txt_file_path in tqdm(label_files, desc="处理标签文件"):
        processed_files_count += 1
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"\n错误：无法读取文件 {txt_file_path}: {e}")
            continue

        original_line_count = len(lines)
        filtered_lines = []
        file_was_modified = False

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line: # 跳过空行
                continue

            parts = stripped_line.split()
            if not parts: # 再次确认，虽然 strip 后不太可能
                continue

            try:
                # 检查第一个元素是否为数字（类别索引）
                line_class_index = int(parts[0])
                if line_class_index == target_class_index:
                    # 检查坐标数量是否符合 OBB 格式 (类别索引 + 8个坐标值)
                    if len(parts) == 9:
                        filtered_lines.append(line) # 保留原始行，包括换行符
                    else:
                        print(f"\n警告：文件 {txt_file_path} 中，行 '{stripped_line}' 的类别索引正确 ({target_class_index})，"
                              f"但坐标数量 ({len(parts) - 1}个) 不符合预期的8个。此行将被忽略。")
                        file_was_modified = True # 因为有行被删除了
                else:
                    # 类别索引不是目标索引，标记为修改
                    file_was_modified = True
            except ValueError:
                # 第一个元素不是有效的整数，可能不是标准的YOLO标签行，直接保留或根据需要处理
                # 为安全起见，这里选择保留非标准行，但会打印警告
                # 如果确定非标准行都应该删除，可以将这行也标记为 file_was_modified = True 并不加入 filtered_lines
                print(f"\n警告：文件 {txt_file_path} 中，行 '{stripped_line}' 的第一个元素不是有效的类别索引。此行将被保留。")
                filtered_lines.append(line)


        if file_was_modified or len(filtered_lines) != original_line_count:
            try:
                with open(txt_file_path, 'w', encoding='utf-8') as f:
                    f.writelines(filtered_lines)
                if len(filtered_lines) != original_line_count: # 只有行数变化才算作“已修改”并计数
                    modified_files_count += 1
            except Exception as e:
                print(f"\n错误：无法写入文件 {txt_file_path}: {e}")

    print(f"\n处理完成。共检查 {processed_files_count} 个文件。")
    print(f"{modified_files_count} 个文件被修改（即，有非目标类别的行被移除，或格式不正确的行被移除）。")

if __name__ == "__main__":
    # **************************************************************************
    # 重要：请将下面的路径修改为你的实际标签文件所在的根目录！
    # 这个目录应该包含 train 和 val (或其他) 子目录，里面是 .txt 标签文件。
    # **************************************************************************
    labels_directory = r"E:\RS_Aircraft_Occlusion_Detection\yolo_obb_dataset\labels"

    # 你想保留的类别索引 (例如，飞机类别是 0)
    target_class_to_keep = 0

    filter_labels_by_class_index(labels_directory, target_class_to_keep)

    print("\n脚本执行完毕。请检查你的标签文件是否已按预期更新。")
    print("建议在运行YOLO训练之前，删除旧的标签缓存文件 (例如 labels/train.cache 和 labels/val.cache)。")