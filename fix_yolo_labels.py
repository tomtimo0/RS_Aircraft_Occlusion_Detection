import os
import glob
from tqdm import tqdm

def clamp(value, min_value=0.0, max_value=1.0):
    """
    将值限制在 [min_value, max_value] 区间内。
    """
    return max(min_value, min(value, max_value))

def fix_labels_in_dir(dataset_dir):
    """
    扫描并修复指定数据集目录下的所有YOLO OBB标签文件。

    @param {str} dataset_dir: 数据集根目录，例如 'yolo_obb_dataset'。
    """
    labels_path = os.path.join(dataset_dir, 'labels')
    if not os.path.exists(labels_path):
        print(f"错误：找不到标签目录 '{labels_path}'")
        return

    # 使用glob递归搜索train和val目录下的所有.txt文件
    files_to_check = glob.glob(os.path.join(labels_path, '**', '*.txt'), recursive=True)

    if not files_to_check:
        print(f"在 '{labels_path}' 中没有找到任何.txt标签文件。")
        return

    print(f"开始扫描 {len(files_to_check)} 个标签文件...")

    fixed_files_count = 0
    total_lines_checked = 0
    total_values_fixed = 0

    for file_path in tqdm(files_to_check, desc="修复进度"):
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            file_was_modified = False

            for line in lines:
                total_lines_checked += 1
                parts = line.strip().split()
                if len(parts) < 2:  # 至少需要类别+1个坐标
                    new_lines.append(line)
                    continue

                class_id = parts[0]
                coords = parts[1:]

                fixed_coords = []
                line_was_modified = False
                for coord in coords:
                    try:
                        val = float(coord)
                        clamped_val = clamp(val)
                        if val != clamped_val:
                            line_was_modified = True
                            total_values_fixed += 1
                        fixed_coords.append(str(clamped_val))
                    except ValueError:
                        # 如果某个部分无法转换为浮点数，保持原样
                        fixed_coords.append(coord)
                
                if line_was_modified:
                    file_was_modified = True
                    new_line = f"{class_id} {' '.join(fixed_coords)}\n"
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            
            if file_was_modified:
                fixed_files_count += 1
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)

        except Exception as e:
            print(f"\\n处理文件 {file_path} 时出错: {e}")

    print("\\n--- 修复完成 ---")
    print(f"总共扫描文件数: {len(files_to_check)}")
    print(f"总共检查行数: {total_lines_checked}")
    print(f"已修复的文件数: {fixed_files_count}")
    print(f"已修正的坐标值数: {total_values_fixed}")
    if fixed_files_count > 0:
        print("\\n你的标签文件现已清理完毕！可以重新开始训练了。")
    else:
        print("\\n未发现需要修复的标签文件，数据集是干净的。")

if __name__ == "__main__":
    # 将此路径替换为你的数据集根目录
    target_dataset_directory = 'yolo_obb_dataset' 
    fix_labels_in_dir(target_dataset_directory) 