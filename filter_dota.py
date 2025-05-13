import os
import shutil # 虽然没直接用 shutil.rmtree，但保留 import 以备将来扩展

def filter_dota_files_by_category(base_dataset_dir, category_to_keep="plane"):
    """
    遍历 DOTA 数据集目录，根据指定类别保留相关文件，删除无关文件。

    Args:
        base_dataset_dir (str): 数据集根目录路径 (例如 E:\\模式识别课设\\数据集).
        category_to_keep (str): 要保留的目标类别名称 (例如 "plane").
    """
    subdirs_to_process = ["train", "val"] # 只处理训练集和验证集
    print(f"开始处理目录: {base_dataset_dir}")
    print(f"将只保留包含类别 '{category_to_keep}' 的样本文件。")
    print("-" * 30)

    total_deleted_files = 0
    total_kept_samples = 0 # 基于找到 plane 标签的样本数

    for subdir in subdirs_to_process:
        print(f"正在处理子目录: {subdir}")
        label_dir = os.path.join(base_dataset_dir, subdir, "labelTxt-v2.0")
        image_dir = os.path.join(base_dataset_dir, subdir, "images")
        meta_dir = os.path.join(base_dataset_dir, subdir, "meta")

        # 检查必要目录是否存在
        if not os.path.isdir(label_dir):
            print(f"  错误: 标签目录不存在: {label_dir}")
            continue
        if not os.path.isdir(image_dir):
            print(f"  错误: 图像目录不存在: {image_dir}")
            continue
        if not os.path.isdir(meta_dir):
            print(f"  警告: 元数据目录不存在: {meta_dir} (将跳过元数据文件处理)")
            meta_dir_exists = False
        else:
            meta_dir_exists = True

        # 1. 找出所有包含目标类别的样本基础文件名
        files_to_keep = set()
        print(f"  正在扫描标签文件以查找 '{category_to_keep}'...")
        try:
            label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt") and os.path.isfile(os.path.join(label_dir, f))]
            for label_filename in label_files:
                label_filepath = os.path.join(label_dir, label_filename)
                base_name = os.path.splitext(label_filename)[0]
                try:
                    with open(label_filepath, 'r', encoding='utf-8') as f:
                        found_category_in_file = False # Flag for the entire file
                        for line in f:
                            line_content = line.strip()
                            if not line_content:
                                continue
                            # ---> 修复：使用空格分割 <---
                            parts = line_content.split() # Split by whitespace
                            # 确保分割后至少有10个部分 (8 coords + category + difficult)
                            if len(parts) >= 10:
                                # 类别通常是倒数第二个字段
                                category = parts[-2].strip()
                                if category.lower() == category_to_keep.lower():
                                    found_category_in_file = True
                                    break # 在此行找到，无需检查其他字段
                            # 如果需要检查所有字段（以防万一格式不规范），可以使用以下注释掉的代码块
                            # for part in parts:
                            #     if part.strip().lower() == category_to_keep.lower():
                            #         found_category_in_file = True
                            #         break # 在此行找到
                            # if found_category_in_file:
                            #     break # 在此文件找到

                        if found_category_in_file:
                            files_to_keep.add(base_name)
                            # 注意：这里 total_kept_samples 在循环外统一计算，避免重复加
                except Exception as e:
                    print(f"    读取标签文件时出错 {label_filepath}: {e}")

            num_kept_in_subdir = len(files_to_keep)
            total_kept_samples += num_kept_in_subdir # 更新总保留样本数
            print(f"  在 '{subdir}' 子目录中找到 {num_kept_in_subdir} 个包含 '{category_to_keep}' 的样本。")
        except Exception as e:
            print(f"  扫描标签目录时出错 {label_dir}: {e}")
            continue # 跳过这个子目录的处理

        # --------> 确认步骤 <--------
        confirm_subdir = input(f"  是否要删除 '{subdir}' 目录中不含 '{category_to_keep}' 的样本文件? (yes/no): ")
        if confirm_subdir.lower() != 'yes':
            print(f"  跳过删除操作 for '{subdir}' 目录。")
            print("-" * 30)
            continue # 跳到下一个子目录
        # --------> 确认步骤结束 <--------

        # 2. 删除不包含目标类别的样本文件
        print(f"  开始删除 '{subdir}' 中的不相关文件...")
        deleted_in_subdir = 0
        kept_files_in_subdir = 0 # 只统计此子目录内实际保留的文件数

        # 处理图像文件
        try:
            image_files = [f for f in os.listdir(image_dir) if f.endswith(".png") and os.path.isfile(os.path.join(image_dir, f))]
            for image_filename in image_files:
                base_name = os.path.splitext(image_filename)[0]
                if base_name not in files_to_keep:
                    filepath_to_delete = os.path.join(image_dir, image_filename)
                    try:
                        os.remove(filepath_to_delete)
                        deleted_in_subdir += 1
                    except Exception as e:
                        print(f"    删除图像文件时出错 {filepath_to_delete}: {e}")
                else:
                    kept_files_in_subdir += 1
        except Exception as e:
             print(f"  处理图像目录时出错 {image_dir}: {e}")

        # 处理标签文件 (再次遍历以删除)
        try:
            label_files_to_check = [f for f in os.listdir(label_dir) if f.endswith(".txt") and os.path.isfile(os.path.join(label_dir, f))]
            for label_filename in label_files_to_check:
                 base_name = os.path.splitext(label_filename)[0]
                 if base_name not in files_to_keep:
                     filepath_to_delete = os.path.join(label_dir, label_filename)
                     try:
                         os.remove(filepath_to_delete)
                         deleted_in_subdir += 1
                     except Exception as e:
                         print(f"    删除标签文件时出错 {filepath_to_delete}: {e}")
                 else:
                     kept_files_in_subdir += 1
        except Exception as e:
             print(f"  处理标签目录时出错 {label_dir}: {e}")

        # 处理元数据文件
        if meta_dir_exists:
            try:
                meta_files = [f for f in os.listdir(meta_dir) if f.endswith(".txt") and os.path.isfile(os.path.join(meta_dir, f))]
                for meta_filename in meta_files:
                    base_name = os.path.splitext(meta_filename)[0]
                    if base_name not in files_to_keep:
                        filepath_to_delete = os.path.join(meta_dir, meta_filename)
                        try:
                            os.remove(filepath_to_delete)
                            deleted_in_subdir += 1
                        except Exception as e:
                            print(f"    删除元数据文件时出错 {filepath_to_delete}: {e}")
                    else:
                       kept_files_in_subdir += 1
            except Exception as e:
                print(f"  处理元数据目录时出错 {meta_dir}: {e}")

        total_deleted_files += deleted_in_subdir
        print(f"  子目录 {subdir} 处理完成。删除了 {deleted_in_subdir} 个文件，保留了 {kept_files_in_subdir} 个文件（对应 {num_kept_in_subdir} 个样本）。")
        print("-" * 30)

    print("=" * 30)
    print("所有指定子目录处理完毕。")
    print(f"总共删除文件数: {total_deleted_files}")
    print(f"总共保留样本数 (基于找到'plane'标签的样本): {total_kept_samples}")
    print("=" * 30)

# --- 执行脚本 ---
# !!! 重要：请确保下面的路径是正确的 DOTA 数据集根目录 !!!
dota_root_directory = r"E:\模式识别课设\数据集"

# !!! 警告：此脚本会永久删除文件，请在运行前确认路径无误，并最好备份原始数据 !!!

# 准备运行
print("脚本即将开始执行文件过滤操作。")
filter_dota_files_by_category(dota_root_directory, category_to_keep="plane")
print("脚本执行完毕。")
