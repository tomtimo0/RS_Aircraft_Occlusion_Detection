import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import shutil

# --- Helper Functions (No changes needed) ---
def parse_dota_labels_for_planes(label_path):
    """解析DOTA格式的标签文件，仅用于定位飞机以生成定向光斑。"""
    plane_bboxes = []
    if not os.path.exists(label_path):
        return plane_bboxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # 兼容'plane'或'aircraft'
            if len(parts) >= 9 and parts[8].lower() in ['plane', 'aircraft']:
                try:
                    coords = [float(p) for p in parts[:8]]
                    plane_bboxes.append(coords)
                except (ValueError, IndexError):
                    continue
    return plane_bboxes

# --- Upgraded Glare Generation Function (No changes needed) ---
def add_glares_and_get_labels(image, plane_4pt_bboxes, max_extra_spots=3, min_radius_ratio=0.06, max_radius_ratio=0.25, min_intensity=1.8, max_intensity=3.0):
    """
    在图像上添加多样化的光斑，并返回光斑的标签。
    返回: (带有光斑的图像, 光斑标签列表)
    每个光斑标签格式为: {'center_x': cx, 'center_y': cy, 'radius': r}
    """
    height, width, _ = image.shape
    spots_to_generate = []
    glare_labels = []

    # 定向在飞机上生成光斑
    if plane_4pt_bboxes:
        target_plane_pts = random.choice(plane_4pt_bboxes)
        coords = np.array(target_plane_pts).reshape(4, 2)
        xmin, ymin = int(np.min(coords[:, 0])), int(np.min(coords[:, 1]))
        xmax, ymax = int(np.max(coords[:, 0])), int(np.max(coords[:, 1]))
        if xmax > xmin and ymax > ymin:
            center_x, center_y = random.randint(xmin, xmax), random.randint(ymin, ymax)
            spots_to_generate.append({'center_x': center_x, 'center_y': center_y})

 
    # 生成额外光斑的位置（全图随机）
    num_extra_spots = random.randint(1, max_extra_spots)
    for _ in range(num_extra_spots):
        center_x = random.randint(0, width)
        center_y = random.randint(0, height) # y坐标应该在0到height之间
        spots_to_generate.append({'center_x': center_x, 'center_y': center_y})
        
    if not spots_to_generate: 
        return image, []

    # 逐层绘制光斑并记录标签
    image_float = image.astype(np.float32)
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    smaller_dim = min(height, width)

    for spot_props in spots_to_generate:
        cx, cy = spot_props['center_x'], spot_props['center_y']
        radius = random.randint(int(smaller_dim * min_radius_ratio), int(smaller_dim * max_radius_ratio))
        intensity = random.uniform(min_intensity, max_intensity)
        sigma = radius / 2.5
        
        rand_val = random.random()
        if rand_val < 0.8: color = np.array([255, 255, 255])
        elif rand_val < 0.9: color = np.array([230, 255, 255])
        else: color = np.array([255, 230, 220])
        
        glare_layer = np.full(image.shape, color, dtype=np.float32)

        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        gauss_mask = np.exp(-(dist_from_center**2 / (2 * sigma**2)))
        alpha_mask = intensity * gauss_mask
        np.clip(alpha_mask, 0, 1, out=alpha_mask)
        alpha_mask_3ch = alpha_mask[..., np.newaxis]
        image_float = image_float * (1 - alpha_mask_3ch) + glare_layer * alpha_mask_3ch

        glare_labels.append({'center_x': cx, 'center_y': cy, 'radius': radius})

    final_image = np.clip(image_float, 0, 255).astype(np.uint8)
    return final_image, glare_labels

if __name__ == '__main__':
    # ======================= 配置区域 =======================
    # 1. 原始数据集路径
    ORIGINAL_IMAGE_DIR = r"dataset/train_split/images"
    ORIGINAL_DOTA_LABEL_DIR = r"dataset/train_split/labelTxt" # DOTA格式标签，仅用于辅助生成光斑

    # 2. 新的“仅光斑”数据集的输出根目录
    AUGMENTED_DATASET_ROOT = r"dataset/train_split_with_multi_glare2"

    # 3. 类别定义 (现在只有一个类别)
    CLASS_MAPPING = {
        'glare': 0
    }
    # =======================================================

    # --- 设置新数据集的目录结构 ---
    # 图片文件夹
    aug_image_dir = os.path.join(AUGMENTED_DATASET_ROOT, "images")
    # ** 标签文件夹，按你的要求命名为 labels_spot **
    aug_label_dir = os.path.join(AUGMENTED_DATASET_ROOT, "labels_spot")
    
    os.makedirs(aug_image_dir, exist_ok=True)
    os.makedirs(aug_label_dir, exist_ok=True)
    
    print(f"将生成“仅光斑”数据集到: {AUGMENTED_DATASET_ROOT}")
    
    try:
        image_files = [f for f in os.listdir(ORIGINAL_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except FileNotFoundError:
        print(f"错误：找不到原始图像目录 '{ORIGINAL_IMAGE_DIR}'。")
        exit()

    print(f"找到 {len(image_files)} 张图片，开始生成数据集...")

    for filename in tqdm(image_files, desc="生成光斑及对应标签"):
        basename = os.path.splitext(filename)[0]
        
        original_image_path = os.path.join(ORIGINAL_IMAGE_DIR, filename)
        original_dota_label_path = os.path.join(ORIGINAL_DOTA_LABEL_DIR, basename + '.txt')
        
        augmented_image_path = os.path.join(aug_image_dir, filename)
        augmented_label_path = os.path.join(aug_label_dir, basename + '.txt')

        # --- 读取原始数据 ---
        image = cv2.imread(original_image_path)
        if image is None: continue
        height, width, _ = image.shape
        
        plane_4pt_bboxes = parse_dota_labels_for_planes(original_dota_label_path)

        # --- 生成带光斑的图片和光斑的标签 ---
        image_with_glares, glare_labels = add_glares_and_get_labels(image, plane_4pt_bboxes)
        
        # 保存增强后的图片
        cv2.imwrite(augmented_image_path, image_with_glares)

        # --- 核心步骤：只生成并写入光斑的标签 ---
        with open(augmented_label_path, 'w') as f_out:
            # 检查是否有光斑生成
            if not glare_labels:
                continue # 如果没有生成光斑，就创建一个空的标签文件

            glare_class_id = CLASS_MAPPING.get('glare')
            for glare in glare_labels:
                cx, cy, r = glare['center_x'], glare['center_y'], glare['radius']
                
                # 归一化
                cx_norm = cx / width
                cy_norm = cy / height
                w_norm = (2 * r) / width
                h_norm = (2 * r) / height
                angle_rad = 0 

                f_out.write(f"{glare_class_id} {cx_norm} {cy_norm} {w_norm} {h_norm} {angle_rad}\n")

    print("\n“仅光斑”数据集生成完毕！")