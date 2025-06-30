# app_streamlit.py

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import io
import yaml
from streamlit_image_coordinates import streamlit_image_coordinates

# 导入我们自己的模块
from src.image_utils import add_gaussian_spot_to_image
from src.yolo_detector import YOLODetectorOBB # YOLODetectorOBB 现在需要能被多次实例化或切换模型
from src.visualization import draw_all_detections
from src.metrics import calculate_single_ooap

# --- 辅助函数：加载配置 ---
def load_app_config(config_path="config.yaml"):
    """从YAML文件加载应用配置。"""
    if not os.path.exists(config_path):
        st.error(f"配置文件 '{config_path}' 未找到！请确保它在项目根目录下。")
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        st.error(f"加载配置文件 '{config_path}' 失败: {e}")
        return None

# --- 加载应用配置 ---
APP_CONFIG = load_app_config()
if APP_CONFIG is None:
    st.stop() # 如果配置加载失败，则停止应用

# --- 页面配置 ---
st.set_page_config(layout="wide", page_title=APP_CONFIG.get("page_title", "遥感图像飞机目标检测与遮挡模拟"))

# --- 显示校徽 ---
def show_logo():
    """
    在Streamlit页面顶部居中显示华中科技大学校徽。
    """
    logo_path = "hust_logo.png"
    if os.path.exists(logo_path):
        cols = st.columns([1, 2, 1])
        with cols[1]:
            st.image(logo_path, width=120)

show_logo()

# --- 应用状态管理 (调整) ---
if 'uploaded_image_bytes' not in st.session_state:
    st.session_state.uploaded_image_bytes = None
if 'uploaded_label_bytes' not in st.session_state:
    st.session_state.uploaded_label_bytes = None
if 'original_cv_image' not in st.session_state:
    st.session_state.original_cv_image = None
if 'spot_center_abs_xy' not in st.session_state:
    st.session_state.spot_center_abs_xy = None
# 'detector' 状态现在可能不需要全局存储单个实例，因为我们会按需创建或缓存多个
if 'loaded_detectors' not in st.session_state: # 缓存已加载的模型实例
    st.session_state.loaded_detectors = {}
if 'image_with_spot_preview' not in st.session_state:
    st.session_state.image_with_spot_preview = None
if 'multi_model_detection_images' not in st.session_state: # 新增: 存储多模型结果图
    st.session_state.multi_model_detection_images = {} # key: model_name, value: image_cv
if 'ground_truths_obb' not in st.session_state:
    st.session_state.ground_truths_obb = []
if 'ooap_results' not in st.session_state:
    st.session_state.ooap_results = [] # 保持，OOAP是针对GT和光斑的，与模型无关

# --- 从配置中获取模型列表 ---
config_models = APP_CONFIG.get('models', [])
AVAILABLE_MODELS_CONFIG = {}
for model_conf in config_models:
    if model_conf.get('name') and model_conf.get('path'):
        AVAILABLE_MODELS_CONFIG[model_conf['name']] = model_conf['path']
    else:
        st.warning(f"配置文件中的模型条目格式不正确: {model_conf}")

existing_models = {}
for name, path in AVAILABLE_MODELS_CONFIG.items():
    if os.path.exists(path):
        existing_models[name] = path
    else:
        st.warning(f"模型文件未找到 (来自配置): {path}。该模型将不可用。")

if not existing_models and config_models: # 只有当配置了模型但都找不到时才报错
    st.error("配置文件中指定的所有YOLO模型均未找到。请检查 `config.yaml` 中的路径。")
    st.stop()
elif not config_models:
    st.warning("`config.yaml` 中没有配置任何模型。")
    # 可以选择停止，或者允许在没有模型的情况下运行（仅图像处理）

# --- 从配置中获取默认UI参数 ---
DEFAULT_UI_PARAMS = APP_CONFIG.get('default_ui_params', {})

# --- 辅助函数 ---
def load_image_from_bytes(image_bytes):
    try:
        image_stream = io.BytesIO(image_bytes)
        pil_image = Image.open(image_stream)
        if pil_image.mode == 'RGBA' or pil_image.mode == 'LA' or (pil_image.mode == 'P' and 'transparency' in pil_image.info):
            pil_image = pil_image.convert('RGB')
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"图像加载失败: {e}")
        return None

def parse_yolo_obb_label_file(file_content_str: str, img_width: int, img_height: int, target_class_id: int = 0) -> list:
    gt_obbs_pixels = []
    lines = file_content_str.strip().split('\n')
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
            if class_id == target_class_id:
                coords_norm = np.array([float(x) for x in parts[1:]])
                if len(coords_norm) == 8:
                    pixel_coords = np.zeros_like(coords_norm)
                    pixel_coords[0::2] = coords_norm[0::2] * img_width
                    pixel_coords[1::2] = coords_norm[1::2] * img_height
                    gt_obbs_pixels.append(pixel_coords.reshape(4, 2).astype(np.int32))
        except ValueError:
            st.warning(f"标签解析警告: 跳过无效行: '{line}'")
            continue
    return gt_obbs_pixels

def get_detector_instance(model_name, model_path):
    """
    获取或创建指定模型的 YOLODetectorOBB 实例，避免重复加载。
    """
    if 'loaded_detectors' not in st.session_state:
        st.session_state.loaded_detectors = {}
    if model_name in st.session_state.loaded_detectors:
        return st.session_state.loaded_detectors[model_name]
    else:
        try:
            detector = YOLODetectorOBB(model_path=model_path)
            st.session_state.loaded_detectors[model_name] = detector
            return detector
        except Exception as e:
            st.error(f"模型 {model_name} 加载失败: {e}")
            return None

# --- UI 布局 ---
st.title(APP_CONFIG.get("app_title", "🛰️ 遥感图像目标检测与高斯光斑模拟"))
st.markdown(APP_CONFIG.get("app_description", "应用描述..."))

col1, col2 = st.columns([0.4, 0.6]) # 主布局

with col1: # 控制面板
    st.header("⚙️ 控制面板")

    # 1. 文件上传
    st.subheader("1. 上传图像和标签")
    uploaded_image_file = st.file_uploader("选择遥感图像...", type=APP_CONFIG.get('app_settings', {}).get('allowed_image_types', ["png", "jpg"]), key="uploader_img")
    uploaded_label_file = st.file_uploader("选择对应标签文件 (.txt, YOLO OBB, 可选)", type=["txt"], key="uploader_lbl")

    new_img_bytes = uploaded_image_file.getvalue() if uploaded_image_file else None
    new_lbl_bytes = uploaded_label_file.getvalue() if uploaded_label_file else None

    if new_img_bytes and st.session_state.get('uploaded_image_bytes') != new_img_bytes:
        st.session_state.uploaded_image_bytes = new_img_bytes
        st.session_state.original_cv_image = load_image_from_bytes(new_img_bytes)
        st.session_state.spot_center_abs_xy = None
        st.session_state.image_with_spot_preview = None
        st.session_state.multi_model_detection_images = {} # 图像变了，旧检测结果无效
        st.session_state.ground_truths_obb = []
        st.session_state.ooap_results = []
        st.session_state.uploaded_label_bytes = None
        if st.session_state.original_cv_image is not None: st.success("图像已更新。")

    if st.session_state.original_cv_image is not None and new_lbl_bytes and st.session_state.get('uploaded_label_bytes') != new_lbl_bytes:
        st.session_state.uploaded_label_bytes = new_lbl_bytes
        try:
            lbl_content = new_lbl_bytes.decode("utf-8")
            h, w = st.session_state.original_cv_image.shape[:2]
            target_id = int(APP_CONFIG.get('detection_params', {}).get('target_plane_class_id', 0))
            st.session_state.ground_truths_obb = parse_yolo_obb_label_file(lbl_content, w, h, target_id)
            st.session_state.ooap_results = []
            st.info(f"标签文件解析,找到 {len(st.session_state.ground_truths_obb)} 个真值。")
        except Exception as e:
            st.error(f"解析标签失败: {e}")

    # 2. 模型选择 (多选)
    st.subheader("2. 选择检测模型 (可多选)")
    if AVAILABLE_MODELS_CONFIG:
        model_names_options = list(AVAILABLE_MODELS_CONFIG.keys())
        # 默认选择第一个模型，如果列表不为空
        default_selection = [model_names_options[0]] if model_names_options else []
        
        selected_model_names_for_run = st.multiselect(
            "选择要运行的模型:",
            options=model_names_options,
            default=default_selection, # 默认选中第一个
            key="model_multiselect"
        )
    else:
        st.info("无可用模型。")
        selected_model_names_for_run = []

    # 3. 光斑参数 (与之前类似，使用 DEFAULT_UI_PARAMS)
    st.subheader("3. 高斯光斑模拟")
    enable_spot = st.checkbox("启用光斑模拟", value=DEFAULT_UI_PARAMS.get('enable_spot_default', False), key="cb_enable_spot")
    spot_shape = st.radio("光斑形状:", ('ellipse', 'circle'), index=0, horizontal=True, key="radio_spot_shape", disabled=not enable_spot)
    # ... (sigma_x, sigma_y, rotation, amplitude 滑块，与之前代码类似，确保使用唯一的key) ...
    if spot_shape == 'ellipse':
        spot_sigma_x = st.slider("Sigma X:", 1, 200, DEFAULT_UI_PARAMS.get('spot_sigma_x', 50), 1, key="slider_sx", disabled=not enable_spot)
        spot_sigma_y = st.slider("Sigma Y:", 1, 200, DEFAULT_UI_PARAMS.get('spot_sigma_y', 30), 1, key="slider_sy", disabled=not enable_spot)
        spot_rotation_angle_deg = st.slider("旋转角度:", 0, 360, DEFAULT_UI_PARAMS.get('spot_rotation_angle_deg', 0), 1, key="slider_rot", disabled=not enable_spot)
    else: # Circle
        spot_sigma_x = st.slider("Sigma:", 1, 200, DEFAULT_UI_PARAMS.get('spot_sigma_circle', 40), 1, key="slider_sc", disabled=not enable_spot)
        spot_sigma_y = spot_sigma_x
        spot_rotation_angle_deg = 0
    spot_amplitude = st.slider("光斑强度:", 0.1, 3.0, DEFAULT_UI_PARAMS.get('spot_amplitude', 1.5), 0.1, key="slider_amp", disabled=not enable_spot)
    spot_color_rgb = (255, 255, 255)


    # --- 实时光斑预览 和 OOAP 计算 (与之前类似) ---
    current_spot_params = {} # 用于传递给OOAP和光斑函数
    if st.session_state.original_cv_image is not None and enable_spot:
        if st.session_state.spot_center_abs_xy is not None:
            current_spot_params = {
                'spot_center_abs_xy': st.session_state.spot_center_abs_xy,
                'spot_sigma_x': spot_sigma_x, 'spot_sigma_y': spot_sigma_y,
                'spot_rotation_angle_deg': spot_rotation_angle_deg,
                'spot_amplitude': spot_amplitude, 'spot_color_rgb': spot_color_rgb, 'spot_shape': spot_shape
            }
            st.session_state.image_with_spot_preview = add_gaussian_spot_to_image(
                st.session_state.original_cv_image.copy(), **current_spot_params
            )
            if st.session_state.ground_truths_obb:
                temp_ooaps = []
                k_sigma = float(APP_CONFIG.get('metrics_params', {}).get('k_sigma_for_ooap', 2.0))
                for i, gt_verts in enumerate(st.session_state.ground_truths_obb):
                    ooap_val = calculate_single_ooap(gt_verts, current_spot_params, k_sigma)
                    temp_ooaps.append({'gt_index': i, 'ooap': ooap_val, 'vertices': gt_verts})
                st.session_state.ooap_results = temp_ooaps
            else: st.session_state.ooap_results = []
        else:
            st.session_state.image_with_spot_preview = st.session_state.original_cv_image.copy()
            st.session_state.ooap_results = []
    elif st.session_state.original_cv_image is not None and not enable_spot:
        st.session_state.image_with_spot_preview = st.session_state.original_cv_image.copy()
        st.session_state.ooap_results = []


    # 4. 检测参数 (与之前类似)
    st.subheader("4. 检测参数")
    conf_threshold = st.slider("置信度阈值:", 0.01, 0.99, DEFAULT_UI_PARAMS.get('confidence_threshold', 0.25), 0.01, key="slider_conf")
    iou_threshold = st.slider("IoU阈值 (NMS):", 0.01, 0.99, DEFAULT_UI_PARAMS.get('iou_threshold', 0.45), 0.01, key="slider_iou")

    # 5. 执行按钮
    st.markdown("---")
    if st.button("🚀 执行多模型对比检测", use_container_width=True, type="primary", key="btn_detect_multi"):
        if st.session_state.original_cv_image is None:
            st.warning("请先上传图像。")
        elif not selected_model_names_for_run:
            st.warning("请至少选择一个模型进行检测。")
        else:
            image_to_process_base = None # 基础图像（带或不带光斑）
            if enable_spot and st.session_state.image_with_spot_preview is not None:
                image_to_process_base = st.session_state.image_with_spot_preview.copy()
            elif st.session_state.original_cv_image is not None:
                image_to_process_base = st.session_state.original_cv_image.copy()
            
            if image_to_process_base is None:
                st.error("无法获取处理图像。")
            else:
                st.session_state.multi_model_detection_images = {} # 清空旧结果
                
                # --- 准备基础绘制图 (GT + OOAP) ---
                # 这张图将作为每个模型绘制检测框的起点
                base_image_with_gt_ooap = image_to_process_base.copy()
                if st.session_state.ground_truths_obb:
                    gt_color_cfg = APP_CONFIG.get('visualization', {}).get('ground_truth_color_bgr', [0,0,255])
                    gt_color = tuple(gt_color_cfg) if isinstance(gt_color_cfg, list) else (0,0,255)
                    
                    ooap_map = {res['gt_index']: res['ooap'] for res in st.session_state.ooap_results}
                    for i, gt_verts in enumerate(st.session_state.ground_truths_obb):
                        cv2.polylines(base_image_with_gt_ooap, [gt_verts.reshape((-1,1,2))], True, gt_color, 2)
                        # 计算GT中心点
                        gt_center_x = int(np.mean(gt_verts[:, 0]))
                        gt_center_y = int(np.mean(gt_verts[:, 1]))
                        cv2.circle(base_image_with_gt_ooap, (gt_center_x, gt_center_y), radius=5, color=gt_color, thickness=-1)
                        if enable_spot:
                            ooap_val = ooap_map.get(i, 0.0)
                            if ooap_val >= 0: # 只显示有效OOAP
                                ooap_text = f"OOAP: {ooap_val:.1f}%"
                                # 文本定位逻辑 (与之前类似，可能需要改进)
                                x_coords = gt_verts[:, 0]
                                y_coords = gt_verts[:, 1]
                                text_x = int(np.min(x_coords)) 
                                text_y = int(np.min(y_coords)) - 10
                                if text_y < 10: text_y = int(np.max(y_coords)) + 20 # 如果太靠上，放到下面
                                cv2.putText(base_image_with_gt_ooap, ooap_text, (text_x, text_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 1, cv2.LINE_AA)
                
                # --- 对每个选定的模型进行检测 ---
                total_models = len(selected_model_names_for_run)
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, model_name in enumerate(selected_model_names_for_run):
                    model_path = AVAILABLE_MODELS_CONFIG.get(model_name)
                    if not model_path:
                        st.warning(f"无法找到模型 '{model_name}' 的路径，跳过。")
                        continue

                    status_text.text(f"正在处理模型: {model_name} ({i+1}/{total_models})...")
                    detector_instance = get_detector_instance(model_name, model_path)
                    
                    if detector_instance:
                        detections = detector_instance.detect(
                            image_to_process_base, # 检测的是带光斑（如果启用）的图
                            conf_threshold=conf_threshold,
                            iou_threshold=iou_threshold
                        )
                        
                        # 为当前模型的结果创建一个新的图像副本
                        current_model_result_img = base_image_with_gt_ooap.copy()
                        
                        # 为该模型的检测结果选择一个颜色
                        # 可以从config.yaml为每个模型配置颜色，或动态生成
                        model_color_cfg = APP_CONFIG.get('visualization', {}).get('model_colors', [])
                        model_idx_for_color = model_names_options.index(model_name) % len(model_color_cfg) if model_color_cfg else i
                        default_det_color = tuple(model_color_cfg[model_idx_for_color]) if model_color_cfg and model_idx_for_color < len(model_color_cfg) else None # None会让draw_all使用随机色

                        # 绘制该模型的检测结果
                        # draw_all_detections 需要能接受一个固定的颜色，或者我们修改它
                        # 暂时让 draw_all_detections 使用其内部颜色逻辑，或传递特定颜色
                        # 我们还需要在标签中加入模型名称
                        
                        # 修改detections列表，加入模型名称信息，并为该模型指定颜色
                        detections_with_model_info = []
                        for det in detections:
                            det_copy = det.copy()
                            det_copy['model_name'] = model_name # 添加模型名到检测信息
                            # 自动补充 center_xy 字段
                            if 'center_xy' not in det_copy or det_copy['center_xy'] is None:
                                obb = det_copy.get('obb_vertices')
                                if obb is not None and isinstance(obb, np.ndarray) and obb.shape == (4, 2):
                                    det_copy['center_xy'] = (float(np.mean(obb[:, 0])), float(np.mean(obb[:, 1])))
                            detections_with_model_info.append(det_copy)
                        
                        # 修改 draw_all_detections 或 draw_single_detection 以便使用 'model_name' 和特定颜色
                        # 假设 draw_all_detections 内部的 draw_single_detection 可以处理 'model_name' 并在标签中显示
                        # 并且可以接受一个 color_override 参数
                        
                        # 简化：我们直接用特定颜色绘制，并在外部添加模型名标签
                        final_img_for_model = draw_all_detections(
                            current_model_result_img,
                            detections_with_model_info, # 传递带有模型信息的检测结果
                            show_center=True
                        )
                        
                        st.session_state.multi_model_detection_images[model_name] = final_img_for_model
                    progress_bar.progress((i + 1) / total_models)
                status_text.text("所有选定模型处理完成！")
                st.success("多模型对比检测完成！")


with col2: # 图像显示区域
    st.header("🖼️ 图像与结果显示")

    # --- 优先显示多模型对比结果 ---
    if st.session_state.multi_model_detection_images:
        st.subheader("模型对比检测结果:")
        num_models_processed = len(st.session_state.multi_model_detection_images)
        cols_per_row = min(num_models_processed, APP_CONFIG.get('visualization', {}).get('max_comparison_cols', 2))
        image_cols = st.columns(cols_per_row)
        model_idx = 0
        for model_name, img_cv in st.session_state.multi_model_detection_images.items():
            col_to_use = image_cols[model_idx % cols_per_row]
            with col_to_use:
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption=f"结果: {model_name}", use_container_width=True)
            model_idx += 1
        st.markdown("---")
    # 只有在没有检测结果时才显示实时预览/原图
    elif not st.session_state.multi_model_detection_images:
        st.subheader("图像预览与光斑设置:")
        display_image_cv_preview = None
        caption_text_preview = "请上传图像"
        if st.session_state.image_with_spot_preview is not None:
            display_image_cv_preview = st.session_state.image_with_spot_preview
            ooap_preview_text = ""
            if enable_spot and st.session_state.ooap_results:
                ooaps = [res['ooap'] for res in st.session_state.ooap_results if res['ooap'] >= 0]
                if ooaps: ooap_preview_text = f" (Avg GT OOAP: {np.mean(ooaps):.1f}%)"
            if enable_spot and st.session_state.spot_center_abs_xy is None:
                caption_text_preview = "预览 (请点击设置光斑中心)"
            elif enable_spot:
                caption_text_preview = f"实时光斑预览{ooap_preview_text}"
            else:
                caption_text_preview = "图像预览"
        elif st.session_state.original_cv_image is not None:
            display_image_cv_preview = st.session_state.original_cv_image
            caption_text_preview = "原始图像 (点击设置光斑中心)"
        if display_image_cv_preview is not None:
            img_rgb_preview = cv2.cvtColor(display_image_cv_preview, cv2.COLOR_BGR2RGB)
            info_text = "点击下方图像以选择光斑中心。" if enable_spot else "启用光斑以设置中心。"
            st.write(info_text)
            coords_value = streamlit_image_coordinates(img_rgb_preview, key="coords_preview_selector")
            if coords_value and enable_spot:
                new_center = (coords_value['x'], coords_value['y'])
                if st.session_state.spot_center_abs_xy != new_center:
                    st.session_state.spot_center_abs_xy = new_center
                    st.rerun()
            st.caption(caption_text_preview)
        else:
            st.info("请在左侧上传图像以开始。")

st.markdown("---")
st.markdown(APP_CONFIG.get("app_footer", "遥感飞机检测APP v1.3 (多模型对比)"))