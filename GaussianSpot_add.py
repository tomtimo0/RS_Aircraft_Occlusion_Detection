import os
import cv2
import numpy as np
import random
from tqdm import tqdm

def add_multiple_gaussian_glares(image, num_spots_range=(1, 3), min_radius_ratio=0.05, max_radius_ratio=0.15, min_intensity=1.8, max_intensity=2.5):
    """
    在图像上随机位置添加多个、属性随机的高斯光斑。

    参数:
    - image: 输入的OpenCV图像 (NumPy数组)。
    - num_spots_range: 一个元组(min, max)，指定要添加的光斑数量的随机范围。
    - min_radius_ratio: 每个光斑半径相对于图像较小尺寸的最小比例。
    - max_radius_ratio: 每个光斑半径相对于图像较小尺寸的最大比例。
    - min_intensity: 每个光斑中心的最大亮度。
    - max_intensity: 每个光斑中心的最大亮度。

    返回:
    - 带有多个高斯光斑的图像。
    """
    height, width, _ = image.shape
    
    # 1. 随机确定要添加的光斑数量
    num_spots_to_add = random.randint(num_spots_range[0], num_spots_range[1])
    
    if num_spots_to_add == 0:
        return image

    # 2. 创建一个累积的Alpha蒙版，用于存储所有光斑的效果
    # 初始化为全0，尺寸与图像相同（单通道）
    cumulative_alpha_mask = np.zeros((height, width), dtype=np.float32)

    # 创建坐标网格，只需一次即可
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    
    # 3. 循环生成每个光斑并累加到蒙版上
    for _ in range(num_spots_to_add):
        # --- 每个光斑都有自己独立的随机属性 ---
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)
        
        smaller_dim = min(height, width)
        radius = random.randint(int(smaller_dim * min_radius_ratio), int(smaller_dim * max_radius_ratio))
        
        intensity = random.uniform(min_intensity, max_intensity)
        
        # --- 计算当前光斑的高斯分布 ---
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        sigma = radius / 2  # sigma控制光斑的衰减速度
        gauss_mask = np.exp(-(dist_from_center**2 / (2 * sigma**2)))
        
        # 将当前光斑的强度应用到其蒙版上
        single_glare_alpha = intensity * gauss_mask
        
        # 将当前光斑的效果累加到总蒙版上
        cumulative_alpha_mask += single_glare_alpha

    # 4. 剪辑累积蒙版，防止重叠区域的alpha值超过1
    # np.clip的`out`参数可以原地修改，节省内存
    np.clip(cumulative_alpha_mask, 0, 1, out=cumulative_alpha_mask)
    
    # 5. 将最终的累积蒙版应用到图像上
    # 将蒙版扩展到3个通道
    final_alpha_mask = cumulative_alpha_mask[..., np.newaxis]

    # 创建一个纯白色的眩光层
    white_glare_layer = np.ones_like(image, dtype=np.float32) * 255

    # 将原图转换为浮点数以进行精确计算
    image_float = image.astype(np.float32)

    # 使用alpha混合公式: new = old * (1 - alpha) + glare * alpha
    blended_image_float = image_float * (1 - final_alpha_mask) + white_glare_layer * final_alpha_mask
    
    # 将结果转换回8位无符号整数格式
    final_image = np.clip(blended_image_float, 0, 255).astype(np.uint8)
    
    return final_image


if __name__ == '__main__':
    # ======================= 配置区域 =======================
    
    # 1. 输入文件夹：包含已裁剪好的图片
    input_dir = r"E:\RS_Aircraft_Occlusion_Detection\dataset\train_split\images"

    # 2. 输出文件夹：用于存放添加了光斑的新图片
    output_dir = r"E:\RS_Aircraft_Occlusion_Detection\dataset\train_split_with_multi_glare\images"
    
    # 3. (可选) 控制光斑外观的参数
    #    <<<<<<<<<<<<<< 新增：控制每张图上光斑的数量范围 >>>>>>>>>>>>>>
    MIN_SPOTS_PER_IMAGE = 1  # 每张图最少1个光斑
    MAX_SPOTS_PER_IMAGE = 3  # 每张图最多3个光斑
    
    # 每个光斑的大小和强度范围
    MIN_RADIUS_RATIO = 0.05  # 光斑最小半径
    MAX_RADIUS_RATIO = 0.18  # 光斑最大半径
    MIN_INTENSITY = 1.6      # 最小亮度
    MAX_INTENSITY = 2.5      # 最大亮度
    
    # =======================================================
    
    print(f"输入路径: {input_dir}")
    print(f"输出路径: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    
    try:
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not image_files:
            print(f"错误：在输入目录 '{input_dir}' 中未找到任何图片。")
            exit()
    except FileNotFoundError:
        print(f"错误：找不到输入目录 '{input_dir}'。")
        exit()

    print(f"找到 {len(image_files)} 张图片，开始处理...")
    
    for filename in tqdm(image_files, desc="添加多个随机光斑"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            continue
            
        try:
            image = cv2.imread(input_path)
            if image is not None:
                # 调用新的函数来添加多个随机光斑
                image_with_glares = add_multiple_gaussian_glares(
                    image,
                    num_spots_range=(MIN_SPOTS_PER_IMAGE, MAX_SPOTS_PER_IMAGE),
                    min_radius_ratio=MIN_RADIUS_RATIO,
                    max_radius_ratio=MAX_RADIUS_RATIO,
                    min_intensity=MIN_INTENSITY,
                    max_intensity=MAX_INTENSITY
                )
                cv2.imwrite(output_path, image_with_glares)
            else:
                print(f"\n警告：无法读取文件 {filename}，已跳过。")
        except Exception as e:
            print(f"\n处理文件 {filename} 时发生错误: {e}")

    print("\n处理完成！所有带多个光斑的图片已保存到输出文件夹。")