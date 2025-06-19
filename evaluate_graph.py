from ultralytics import YOLO
import os
import re
import matplotlib.pyplot as plt
import numpy as np # 导入 numpy 用于均值计算和更复杂的分箱操作

# 选择已训练好的模型路径
model_path = 'runs/train/best.pt' # 或者你的 best.pt 路径
model = YOLO(model_path)

print(f"模型 '{model_path}' 已加载。")

airplane_class_id = 0  # 飞机的类别ID为0，根据你的数据集配置文件
print(f"将使用飞机类别ID: {airplane_class_id}")

# 图片文件夹路径
image_folder = 'E:/RS_Aircraft_Occlusion_Detection/dataset/train_split_ooap/images'
# 保存结果的文件夹路径 (绘图结果仍会保存在这里)
save_folder = 'E:/RS_Aircraft_Occlusion_Detection/runs/obb/graph_results'

# 创建保存结果的文件夹 (主要用于保存绘图)
os.makedirs(save_folder, exist_ok=True)

# 存储每张图片中检测到的飞机的最小置信度
image_min_airplane_confidences = [] # 用于绘图的Y轴
# 存储每张图片的 ooap 值
ooap_values = [] # 用于绘图的X轴

# image_limit = 200 # 取消图片数量限制
processed_image_count = 0

print(f"开始处理图片...") # 更新提示信息

for image_name in os.listdir(image_folder):
    # if processed_image_count >= image_limit: # 取消图片数量限制
    #     print(f"已达到处理图片数量上限 ({image_limit})。")
    #     break

    if not (image_name.lower().endswith('.jpg') or image_name.lower().endswith('.png') or image_name.lower().endswith('.jpeg')):
        # print(f"跳过非图片文件: {image_name}") # 保持此注释，如果需要可以取消
        continue

    image_path = os.path.join(image_folder, image_name)
    base_name = os.path.splitext(image_name)[0]

    pattern = re.compile(r'ooap([0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?)')
    matches = list(pattern.finditer(base_name))

    if matches:
        last_match = matches[-1]
        ooap = float(last_match.group(1))
    else:
        print(f"警告: 文件名 {image_name} 中没有找到有效的 ooap 值。跳过该图片。")
        continue

    # --- 模型推理和结果提取 ---
    try:
        results = model(image_path, verbose=False) # verbose=False 减少控制台输出
    except Exception as e:
        print(f"错误: 模型推理失败于图片 {image_name}. 错误信息: {e}")
        continue

    current_image_airplane_confidences = []

    if results and isinstance(results, list):
        for result in results: # 通常 results 列表只包含一个对应单张输入图片的结果对象
            if hasattr(result, 'obb') and result.obb is not None and \
               hasattr(result.obb, 'conf') and result.obb.conf is not None and \
               hasattr(result.obb, 'cls') and result.obb.cls is not None and \
               len(result.obb.conf) > 0:
                
                confs = result.obb.conf.tolist()
                clss = result.obb.cls.tolist()

                for i in range(len(confs)):
                    confidence = confs[i]
                    class_id = int(clss[i])
                    if class_id == airplane_class_id:
                        current_image_airplane_confidences.append(confidence)

    # --- 处理当前图片的飞机检测结果 ---
    if current_image_airplane_confidences:
        min_confidence_for_this_image = min(current_image_airplane_confidences)
        image_min_airplane_confidences.append(min_confidence_for_this_image)
        ooap_values.append(ooap)
        print(f"图片 {image_name}: 检测到飞机, 最小置信度 {min_confidence_for_this_image:.4f}, OOAP: {ooap}")
    else:
        image_min_airplane_confidences.append(0.0)
        ooap_values.append(ooap)
        print(f"警告: 图片 {image_name} (OOAP: {ooap}) 中未检测到指定类别 ({airplane_class_id}) 的飞机。最小置信度记为0。")
            
    processed_image_count += 1

print(f"\n总共处理了 {processed_image_count} 张图片。")
print(f"提取到的ooap值数量: {len(ooap_values)}")
print(f"提取到的最小飞机置信度数量: {len(image_min_airplane_confidences)}")


# --- 绘图 ---
if not ooap_values or not image_min_airplane_confidences:
    print("没有足够的数据用于绘图。")
elif len(ooap_values) != len(image_min_airplane_confidences):
    print(f"警告: ooap_values ({len(ooap_values)}) 和 image_min_airplane_confidences ({len(image_min_airplane_confidences)}) 长度不一致，无法绘图。")
else:
    # 为了更好的可视化，按ooap值排序
    paired_values = sorted(zip(ooap_values, image_min_airplane_confidences))
    sorted_ooap_values = np.array([p[0] for p in paired_values]) # 转换为numpy数组以便于布尔索引
    sorted_confidences = np.array([p[1] for p in paired_values]) # 转换为numpy数组

    # 1. 原始散点/折线图 (可选，但有助于对比)
    plt.figure(figsize=(12, 7))
    plt.plot(sorted_ooap_values, sorted_confidences, marker='.', linestyle='-', color='lightblue', alpha=0.5, label='单个最小置信度')
    plt.title('Relationship between Occlusion Degree (OOAP) and Minimum Aircraft Detection Confidence')  
    plt.xlabel('Occlusion Degree (OOAP Value)')  
    plt.ylabel('Minimum Aircraft Detection Confidence')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_save_path = os.path.join(save_folder, 'ooap_vs_min_confidence_plot_original.png')
    plt.savefig(plot_save_path)
    print(f"原始折线图已保存到: {plot_save_path}")
    # plt.show() # 如果需要，显示原始图

    # 2. 按10%区间划分并取平均值绘图
    num_bins = 10 # 区间数量
    bin_edges = np.linspace(0, 1, num_bins + 1) # 0.0, 0.1, 0.2, ..., 1.0，定义区间的边界
    
    # 确保最后一个区间正确包含 1.0。
    # 例如，ooap < bin_edges[i+1] 可能会漏掉恰好为 1.0 的值
    # np.digitize 可以很好地处理这个问题，或者我们可以为比较稍微调整最后一个区间的边缘
    
    binned_average_confidences = [] # 存储每个区间的平均置信度
    bin_centers = [] # 存储每个区间的中心点
    bin_counts = [] # 查看每个区间包含多少数据点

    print("\n区间平均值计算:")
    for i in range(num_bins):
        lower_bound = bin_edges[i] # 区间下界
        upper_bound = bin_edges[i+1] # 区间上界
        
        # 定义当前区间内值的掩码
        # 对于除最后一个区间外的所有区间，它是 [下界, 上界)
        # 对于最后一个区间，它是 [下界, 上界]，以包含 ooap = 1.0
        if i < num_bins - 1:
            mask = (sorted_ooap_values >= lower_bound) & (sorted_ooap_values < upper_bound)
        else: # 最后一个区间，使上界具有包含性
            mask = (sorted_ooap_values >= lower_bound) & (sorted_ooap_values <= upper_bound)
            
        confidences_in_bin = sorted_confidences[mask] # 获取当前区间内的置信度值
        
        if len(confidences_in_bin) > 0:
            average_confidence = np.mean(confidences_in_bin) # 计算平均置信度
        else:
            average_confidence = 0.0 # 如果区间为空，平均置信度设为0.0 (或者用 np.nan，如果你希望跳过绘制空区间)

        binned_average_confidences.append(average_confidence)
        bin_centers.append((lower_bound + upper_bound) / 2) # 计算区间中心点
        bin_counts.append(len(confidences_in_bin)) # 记录该区间的数据点数量
        print(f"  区间 [{lower_bound:.1f}-{upper_bound:.1f}): {len(confidences_in_bin)} 个点, 平均置信度: {average_confidence:.4f}")

    # 绘制区间平均值图 (可以是折线图或条形图)
    plt.figure(figsize=(12, 7))
    
    # 作为折线图绘制
    plt.plot(bin_centers, binned_average_confidences, marker='o', linestyle='-', color='red', linewidth=2, label='Average confidence per interval')
    
    # # 可选：作为条形图绘制
    # plt.bar(bin_centers, binned_average_confidences, width=0.08, color='green', alpha=0.7, label='每个区间的平均置信度')

    plt.title('Average Minimum Aircraft Detection Confidence for Each OOAP Interval (10% Intervals)')  
    plt.xlabel('Midpoint of OOAP Interval')  
    plt.ylabel('Average Minimum Aircraft Detection Confidence')
    plt.xticks(bin_centers, [f"{bc:.2f}" for bc in bin_centers], rotation=45) # 在x轴上显示区间中心点
    # 或者，显示区间范围：
    # plt.xticks(bin_centers, [f"[{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f})" for i in range(num_bins)], rotation=45, ha='right')
    plt.grid(True)
    plt.legend() # 显示图例
    plt.tight_layout()

    binned_plot_save_path = os.path.join(save_folder, 'ooap_vs_avg_confidence_binned_plot.png')
    plt.savefig(binned_plot_save_path)
    print(f"\n区间平均值折线图已保存到: {binned_plot_save_path}")
    
    # 显示区间数据点数量以供验证
    plt.figure(figsize=(10, 5))
    plt.bar([f"[{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f})" for i in range(num_bins)], bin_counts, color='skyblue')
    plt.title('Number of Data Points in Each OOAP Interval')  
    plt.xlabel('OOAP Interval')  
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    bin_counts_save_path = os.path.join(save_folder, 'ooap_bin_counts.png')
    plt.savefig(bin_counts_save_path)
    print(f"区间数据点数量图已保存到: {bin_counts_save_path}")

    plt.show() # 在最后显示所有绘图