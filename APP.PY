from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import json
import os
import cv2
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_DIR = 'artifacts_cnn'
PREFERRED_MODELS = ['cnn_best.keras', 'cnn_best.h5', 'cnn_final.keras', 'cnn_final.h5']
CLASSES_PATH = os.path.join(MODEL_DIR, 'cnn_classes.txt')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and food information
model = None
_loaded_model_path = None

# User data storage (in-memory for demo)
# User data storage (persistent)
USERS_DATA_FILE = 'users_data.json'

def load_users_data():
    """Load user data from file"""
    try:
        if os.path.exists(USERS_DATA_FILE):
            with open(USERS_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading users data: {e}")
    return {}

def save_users_data():
    """Save user data to file"""
    try:
        with open(USERS_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(users_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving users data: {e}")

users_data = load_users_data()
# Dynamically load class labels from artifacts to avoid index mismatch
def load_classes_from_file(file_path: str):
    labels = None
    try:
        if os.path.exists(file_path):
            parsed = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f.read().splitlines():
                    if not line.strip():
                        continue
                    parts = line.split('\t', 1)
                    if len(parts) == 2 and parts[0].strip().isdigit():
                        idx = int(parts[0].strip())
                        name = parts[1].strip()
                        parsed.append((idx, name))
                    else:
                        parsed.append((len(parsed), line.strip()))
            parsed.sort(key=lambda x: x[0])
            labels = [name for _, name in parsed]
    except Exception as e:
        print(f"Warning: failed to load classes from '{file_path}': {e}")
    return labels

food_classes = load_classes_from_file(CLASSES_PATH) or [
    'Canh chua có cá',
    'Canh chua không cá', 
    'Canh rau cải',
    'Canh rau muống',
    'Cá hú kho',
    'Cơm trắng',
    'Rau củ săn xào',
    'Rau đậu que xào',
    'Rau đậu đũa xào',
    'Sườn nướng',
    'Thịt Kho Trứng',
    'Thịt kho',
    'Trứng chiên',
    'Đậu hũ sốt cà',
    'Không có món ăn nào cả'
]

# Load food information from JSON
with open('person_info.json', 'r', encoding='utf-8') as f:
    food_info = json.load(f)

def _resolve_model_path() -> str:
    """Pick first existing model from preferred list in MODEL_DIR."""
    for fname in PREFERRED_MODELS:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            return path
    return None


def load_model():
    """Load the trained model"""
    global model, _loaded_model_path
    try:
        model_path = _resolve_model_path()
        if not model_path:
            print(f"No model file found in {MODEL_DIR}. Tried: {PREFERRED_MODELS}")
            return False
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        _loaded_model_path = model_path
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def _get_model_num_classes() -> int:
    try:
        if model is None:
            return None
        out_shape = getattr(model, 'output_shape', None)
        if isinstance(out_shape, (list, tuple)):
            return int(out_shape[-1]) if out_shape and isinstance(out_shape[-1], int) else None
        return None
    except Exception:
        return None


def _get_model_input_size() -> tuple:
    try:
        if model is None:
            return None
        in_shape = getattr(model, 'input_shape', None)
        if isinstance(in_shape, (list, tuple)) and len(in_shape) >= 4:
            h, w = int(in_shape[1]), int(in_shape[2])
            return (w, h)
        return None
    except Exception:
        return None

# -------------------- Tray utilities (no detector) --------------------
def _decode_base64_to_cv2(image_data: str):
    try:
        if isinstance(image_data, str) and image_data.startswith('data:'):
            image_data = image_data.split(',', 1)[1]
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return img_bgr
    except Exception:
        return None


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _get_five_compartment_boxes(width: int, height: int, padding_ratio: float = 0.02):
    grid = [
        (0.14, 0.04, 0.38, 0.41),
        (0.35, 0.04, 0.58, 0.41),
        (0.55, 0.04, 0.78, 0.41),
        (0.16, 0.42, 0.41, 0.92),
        (0.45, 0.42, 0.78, 0.92),
    ]

    boxes = []
    px = padding_ratio * width
    py = padding_ratio * height
    for (x1r, y1r, x2r, y2r) in grid:
        x1 = int(_clamp(x1r * width + px, 0, width - 1))
        y1 = int(_clamp(y1r * height + py, 0, height - 1))
        x2 = int(_clamp(x2r * width - px, 0, width - 1))
        y2 = int(_clamp(y2r * height - py, 0, height - 1))
        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)
        boxes.append((x1, y1, x2, y2))
    return boxes


def _preprocess_cv_crop_for_model(crop_bgr: np.ndarray) -> np.ndarray:
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    try:
        input_size = _get_model_input_size() or (224, 224)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, input_size, interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32) / 255.0
        return arr
    except Exception:
        return None


def _apply_position_priors_to_scores(scores: np.ndarray, slot_index: int) -> np.ndarray:
    priors = _build_position_priors()
    rice_idx = priors.get('rice_idx')
    soup_indices = priors.get('soup_indices', set())
    adjusted = scores.copy()
    if slot_index in (0, 1, 2):
        for si in soup_indices:
            if 0 <= si < adjusted.shape[0]:
                adjusted[si] *= 1.10
    if slot_index in (3, 4) and rice_idx is not None and 0 <= rice_idx < adjusted.shape[0]:
        adjusted[rice_idx] *= 1.15
    s = adjusted.sum()
    if s > 0:
        adjusted = adjusted / s
    return adjusted


def _find_empty_class_index() -> int:
    for idx, name in enumerate(food_classes):
        n = (name or '').lower()
        if 'không có món' in n or 'khong co mon' in n or 'empty' in n or 'none' in n:
            return idx
    return -1


def _predict_tray_slots(
    image_data: str,
    padding_ratio: float = 0.05,
    min_confidence: float = 0.6,
    empty_margin: float = 0.1,
    empty_var_thresh: float = 12.0,
    empty_sat_thresh: float = 15.0,
):
    if model is None:
        return {'error': 'Model not loaded'}
    img = _decode_base64_to_cv2(image_data)
    if img is None:
        return {'error': 'Invalid image data'}
    h, w = img.shape[:2]
    boxes = _get_five_compartment_boxes(w, h, padding_ratio=padding_ratio)

    crops = []
    raw_crops_bgr = []
    empty_hints = {}
    yellow_ratios = {}
    red_ratios = {}
    valid_indices = []
    
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        crop = img[y1:y2, x1:x2]
        try:
            if crop is not None and crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, stddev = cv2.meanStdDev(gray)
                std_val = float(stddev[0][0])
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                sat_mean = float(cv2.mean(hsv[:, :, 1])[0])
                empty_hints[idx] = (std_val < empty_var_thresh) and (sat_mean < empty_sat_thresh)
                
                try:
                    lower1 = np.array([8, 50, 80], dtype=np.uint8)
                    upper1 = np.array([55, 255, 255], dtype=np.uint8)
                    mask1 = cv2.inRange(hsv, lower1, upper1)
                    
                    lower2 = np.array([12, 80, 120], dtype=np.uint8)
                    upper2 = np.array([40, 255, 255], dtype=np.uint8)
                    mask2 = cv2.inRange(hsv, lower2, upper2)
                    
                    lower3 = np.array([15, 30, 200], dtype=np.uint8)
                    upper3 = np.array([35, 100, 255], dtype=np.uint8)
                    mask3 = cv2.inRange(hsv, lower3, upper3)
                    
                    combined_mask = mask1 | mask2 | mask3
                    ratio = float(cv2.countNonZero(combined_mask)) / float(combined_mask.size)
                    yellow_ratios[idx] = ratio
                except Exception:
                    yellow_ratios[idx] = 0.0
                    
                try:
                    lower1 = np.array([0, 90, 110], dtype=np.uint8)
                    upper1 = np.array([10, 255, 255], dtype=np.uint8)
                    lower2 = np.array([170, 90, 110], dtype=np.uint8)
                    upper2 = np.array([179, 255, 255], dtype=np.uint8)
                    mask1 = cv2.inRange(hsv, lower1, upper1)
                    mask2 = cv2.inRange(hsv, lower2, upper2)
                    rratio = float(cv2.countNonZero(mask1 | mask2)) / float(hsv.shape[0] * hsv.shape[1])
                    red_ratios[idx] = rratio
                except Exception:
                    red_ratios[idx] = 0.0
            else:
                empty_hints[idx] = False
                yellow_ratios[idx] = 0.0
                red_ratios[idx] = 0.0
        except Exception:
            empty_hints[idx] = False
            yellow_ratios[idx] = 0.0
            red_ratios[idx] = 0.0
            
        arr = _preprocess_cv_crop_for_model(crop)
        if arr is not None:
            crops.append(arr)
            raw_crops_bgr.append(crop)
            valid_indices.append(idx)

    if not crops:
        return {'error': 'Failed to prepare tray crops'}

    batch = np.stack(crops, axis=0)
    preds = model.predict(batch)
    slot_results = [None] * len(boxes)
    empty_idx = _find_empty_class_index()
    adjusted_scores_per_slot = {}
    raw_scores_per_slot = {}
    
    for i, slot_idx in enumerate(valid_indices):
        scores = preds[i]
        scores = _apply_position_priors_to_scores(scores, slot_idx)
        
        try:
            yellow_ratio = float(yellow_ratios.get(slot_idx, 0.0))
            red_ratio = float(red_ratios.get(slot_idx, 0.0))
            
            if yellow_ratio > 0.02:
                tkt_idx, tofu_idx, egg_idx, rice_idx2 = None, None, None, None
                for ci, name in enumerate(food_classes):
                    n = (name or '').lower()
                    if tkt_idx is None and ('thịt kho trứng' in n or ('trứng' in n and 'kho' in n)):
                        tkt_idx = ci
                    if egg_idx is None and ('trứng chiên' in n or 'trung chien' in n):
                        egg_idx = ci
                    if tofu_idx is None and ('đậu hũ sốt cà' in n or 'dau hu sot ca' in n or 'dau hu' in n):
                        tofu_idx = ci
                    if rice_idx2 is None and ('cơm trắng' in n or 'com trang' in n):
                        rice_idx2 = ci
                        
                scores = scores.copy()
                
                if egg_idx is not None and 0 <= egg_idx < scores.shape[0]:
                    if yellow_ratio > 0.1:
                        scores[egg_idx] *= 4.0
                    elif yellow_ratio > 0.05:
                        scores[egg_idx] *= 3.0
                    elif yellow_ratio > 0.02:
                        scores[egg_idx] *= 1.8
                    
                    if rice_idx2 is not None and 0 <= rice_idx2 < scores.shape[0]:
                        scores[rice_idx2] *= 0.1
                
                if slot_idx == 1:
                    if egg_idx is not None and 0 <= egg_idx < scores.shape[0]:
                        scores[egg_idx] *= 2.0
                    
                    veg_indices = []
                    for ci, name in enumerate(food_classes):
                        n = (name or '').lower()
                        if any(veg in n for veg in ['rau', 'củ', 'săn', 'đậu', 'que', 'đũa']):
                            veg_indices.append(ci)
                    
                    for veg_idx in veg_indices:
                        if 0 <= veg_idx < scores.shape[0]:
                            scores[veg_idx] *= 0.2
                
                if slot_idx in (0, 1, 2) and egg_idx is not None and 0 <= egg_idx < scores.shape[0]:
                    scores[egg_idx] *= 1.3
                
                if tkt_idx is not None and 0 <= tkt_idx < scores.shape[0]:
                    if yellow_ratio > 0.1 and red_ratio > 0.1:
                        scores[tkt_idx] *= 1.8
                    elif yellow_ratio > 0.05:
                        scores[tkt_idx] *= 1.4
                
                if tofu_idx is not None and 0 <= tofu_idx < scores.shape[0]:
                    scores[tofu_idx] *= 0.6
                
                soup_indices = []
                for ci, name in enumerate(food_classes):
                    n = (name or '').lower()
                    if 'canh' in n:
                        soup_indices.append(ci)
                
                if slot_idx in (3, 4):
                    for soup_idx in soup_indices:
                        if 0 <= soup_idx < scores.shape[0]:
                            scores[soup_idx] *= 1.4
                
                canh_rau_cai_idx = None
                lagim_sau_idx = None
                for ci, name in enumerate(food_classes):
                    n = (name or '').lower()
                    if 'canh rau cải' in n or 'canh rau cai' in n:
                        canh_rau_cai_idx = ci
                    if 'lagim sàu' in n or 'lagim sau' in n:
                        lagim_sau_idx = ci
                
                if slot_idx == 4 and canh_rau_cai_idx is not None and lagim_sau_idx is not None:
                    scores[canh_rau_cai_idx] *= 3.0
                    scores[lagim_sau_idx] *= 0.1
                
                if canh_rau_cai_idx is not None and lagim_sau_idx is not None:
                    if empty_hints.get(slot_idx, False) == False:
                        try:
                            crop = img[boxes[slot_idx][1]:boxes[slot_idx][3], boxes[slot_idx][0]:boxes[slot_idx][2]]
                            if crop is not None and crop.size > 0:
                                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                _, stddev = cv2.meanStdDev(gray)
                                std_val = float(stddev[0][0])
                                
                                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                                
                                lower_green1 = np.array([35, 30, 30], dtype=np.uint8)
                                upper_green1 = np.array([85, 255, 255], dtype=np.uint8)
                                green_mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
                                
                                lower_green2 = np.array([40, 50, 20], dtype=np.uint8)
                                upper_green2 = np.array([80, 255, 150], dtype=np.uint8)
                                green_mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
                                
                                lower_green3 = np.array([25, 40, 40], dtype=np.uint8)
                                upper_green3 = np.array([45, 255, 255], dtype=np.uint8)
                                green_mask3 = cv2.inRange(hsv, lower_green3, upper_green3)
                                
                                combined_green_mask = green_mask1 | green_mask2 | green_mask3
                                green_ratio = float(cv2.countNonZero(combined_green_mask)) / float(combined_green_mask.size)
                                
                                if std_val < 25 and green_ratio > 0.1:
                                    scores[canh_rau_cai_idx] *= 5.0
                                    scores[lagim_sau_idx] *= 0.1
                                elif std_val < 20:
                                    scores[canh_rau_cai_idx] *= 3.0
                                    scores[lagim_sau_idx] *= 0.2
                                elif green_ratio > 0.05:
                                    scores[canh_rau_cai_idx] *= 2.5
                                    scores[lagim_sau_idx] *= 0.3
                                elif green_ratio > 0.02:
                                    scores[canh_rau_cai_idx] *= 1.8
                                    scores[lagim_sau_idx] *= 0.5
                        except Exception:
                            pass
                
                s = scores.sum()
                if s > 0:
                    scores = scores / s
        except Exception:
            pass
            
        raw_scores_per_slot[slot_idx] = preds[i]
        adjusted_scores_per_slot[slot_idx] = scores
        max_index = int(np.argmax(scores))
        confidence = float(scores[max_index])
        
        if empty_idx >= 0 and confidence < float(min_confidence):
            empty_prob = float(scores[empty_idx]) if empty_idx < scores.shape[0] else 0.0
            if empty_prob >= confidence - float(empty_margin):
                max_index = empty_idx
                confidence = empty_prob
                
        if empty_idx >= 0 and empty_hints.get(slot_idx, False):
            max_index = empty_idx
            confidence = max(confidence, float(scores[empty_idx]) if empty_idx < scores.shape[0] else 0.95)
            
        try:
            top_indices = np.argsort(scores)[-3:][::-1]
            top3 = [
                {
                    'class': (food_classes[int(ci)] if int(ci) < len(food_classes) else str(int(ci))),
                    'confidence': float(scores[int(ci)])
                }
                for ci in top_indices
            ]
        except Exception:
            top3 = []
        
        debug_info = {
            'yellow_ratio': float(yellow_ratios.get(slot_idx, 0.0)),
            'red_ratio': float(red_ratios.get(slot_idx, 0.0)),
            'empty_hint': empty_hints.get(slot_idx, False)
        }
        
        try:
            if not empty_hints.get(slot_idx, False):
                crop = img[boxes[slot_idx][1]:boxes[slot_idx][3], boxes[slot_idx][0]:boxes[slot_idx][2]]
                if crop is not None and crop.size > 0:
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    
                    lower_green1 = np.array([35, 30, 30], dtype=np.uint8)
                    upper_green1 = np.array([85, 255, 255], dtype=np.uint8)
                    green_mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
                    
                    lower_green2 = np.array([40, 50, 20], dtype=np.uint8)
                    upper_green2 = np.array([80, 255, 150], dtype=np.uint8)
                    green_mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
                    
                    lower_green3 = np.array([25, 40, 40], dtype=np.uint8)
                    upper_green3 = np.array([45, 255, 255], dtype=np.uint8)
                    green_mask3 = cv2.inRange(hsv, lower_green3, upper_green3)
                    
                    combined_green_mask = green_mask1 | green_mask2 | green_mask3
                    green_ratio = float(cv2.countNonZero(combined_green_mask)) / float(combined_green_mask.size)
                    debug_info['green_ratio'] = green_ratio
                    
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    _, stddev = cv2.meanStdDev(gray)
                    std_val = float(stddev[0][0])
                    debug_info['texture_variance'] = std_val
        except Exception:
            debug_info['green_ratio'] = 0.0
            debug_info['texture_variance'] = 0.0
            
        label = food_classes[max_index] if max_index < len(food_classes) else str(max_index)
        slot_results[slot_idx] = {
            'slot': slot_idx + 1,
            'class': label,
            'confidence': confidence,
            'box': boxes[slot_idx],
            'top3': top3,
            'debug': debug_info,
        }

    # Global assignment to reduce duplicates
    try:
        top_k = 3
        num_slots = len(valid_indices)
        slot_to_candidates = {}
        for slot_idx in valid_indices:
            scores = adjusted_scores_per_slot[slot_idx]
            top_indices = np.argsort(scores)[-top_k:][::-1]
            slot_to_candidates[slot_idx] = [(int(ci), float(scores[ci])) for ci in top_indices]

        pri = _build_position_priors()
        rice_idx = pri.get('rice_idx')
        soup_set = pri.get('soup_indices', set())

        def pos_penalty(slot_idx: int, cls_idx: int) -> float:
            if slot_idx in (0, 1, 2) and cls_idx in soup_set:
                return 0.0
            if slot_idx in (3, 4) and rice_idx is not None and cls_idx == rice_idx:
                return 0.0
            return 0.2

        slots_order = sorted(valid_indices)
        best = {'cost': float('inf'), 'choice': None}

        def dfs(idx: int, choice: list, used_counts: dict, total_cost: float):
            if idx == len(slots_order):
                if total_cost < best['cost']:
                    best['cost'] = total_cost
                    best['choice'] = list(choice)
                return
            slot_idx = slots_order[idx]
            for cls_idx, prob in slot_to_candidates[slot_idx]:
                p = max(prob, 1e-6)
                cost = -float(np.log(p)) + pos_penalty(slot_idx, cls_idx)
                cnt = used_counts.get(cls_idx, 0)
                if cnt >= 1:
                    cost += 0.7 * cnt
                new_total = total_cost + cost
                if new_total >= best['cost']:
                    continue
                used_counts[cls_idx] = cnt + 1
                choice.append((slot_idx, cls_idx, prob))
                dfs(idx + 1, choice, used_counts, new_total)
                choice.pop()
                if cnt == 0:
                    del used_counts[cls_idx]
                else:
                    used_counts[cls_idx] = cnt

        dfs(0, [], {}, 0.0)

        if best['choice']:
            for slot_idx, cls_idx, prob in best['choice']:
                label = food_classes[cls_idx] if cls_idx < len(food_classes) else str(cls_idx)
                prev = slot_results[slot_idx] or {}
                slot_results[slot_idx] = {
                    'slot': slot_idx + 1,
                    'class': label,
                    'confidence': float(prob),
                    'box': boxes[slot_idx],
                    'top3': prev.get('top3', []),
                    'debug': prev.get('debug', {}),
                }
    except Exception as _:
        pass

    return {
        'slots': slot_results,
        'image_size': {'width': w, 'height': h},
        'raw_crops': raw_crops_bgr
    }

_PRIOR_CACHE = None
def _build_position_priors():
    global _PRIOR_CACHE
    if _PRIOR_CACHE is not None:
        return _PRIOR_CACHE
    rice_idx = None
    soup_indices = []
    try:
        for idx, name in enumerate(food_classes):
            n = (name or '').lower()
            if 'cơm trắng' in n or 'com trắng' in n or 'com trang' in n or 'cơm trang' in n:
                rice_idx = idx
            if 'canh' in n:
                soup_indices.append(idx)
    except Exception:
        pass
    _PRIOR_CACHE = {
        'rice_idx': rice_idx,
        'soup_indices': set(soup_indices),
    }
    return _PRIOR_CACHE

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_data):
    try:
        if isinstance(image_data, str) and image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_food(image_array):
    try:
        if model is None:
            return None, 0.0
        
        predictions = model.predict(image_array)
        max_index = np.argmax(predictions[0])
        confidence = float(predictions[0][max_index])
        
        if max_index >= len(food_classes):
            print(f"Warning: predicted index {max_index} out of range for labels {len(food_classes)}")
            predicted_class = str(max_index)
        else:
            predicted_class = food_classes[max_index]
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, 0.0


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

@app.route('/model/<path:filename>')
def serve_model(filename):
    """Serve model files"""
    return send_from_directory('artifacts_cnn', filename)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for food prediction"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_array = preprocess_image(data['image'])
        if image_array is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        predicted_class, confidence = predict_food(image_array)
        
        if predicted_class is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        food_data = food_info.get(predicted_class, {})
        
        response = {
            'success': True,
            'prediction': {
                'class': predicted_class,
                'confidence': confidence,
                'price': food_data.get('Giá', 'N/A'),
                'calories': food_data.get('Calo', 'N/A'),
                'type': food_data.get('Loại', 'Món ăn'),
                'health_score': food_data.get('HealthyScore', 5),
                'features': food_data.get('Đặc điểm', [])
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# ============ NEW ENDPOINT FOR FRONTEND ============
@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main endpoint for analyzing food tray - matches frontend expectations"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        user_id = data.get('user_id', 'anonymous')
        
        # Predict tray contents using advanced algorithm
        result = _predict_tray_slots(data['image'])
        
        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']}), 400
        
        # Convert slots to segments format for frontend
        segments = []
        foods = []
        empty_idx = _find_empty_class_index()
        
        for i, slot in enumerate(result['slots']):
            if slot is None:
                continue
                
            # Encode crop as base64 for frontend
            if i < len(result.get('raw_crops', [])):
                _, buffer = cv2.imencode('.jpg', result['raw_crops'][i])
                crop_base64 = base64.b64encode(buffer).decode('utf-8')
            else:
                crop_base64 = ""
            
            # Check if detected (not empty)
            is_detected = slot['class'] != food_classes[empty_idx] if empty_idx >= 0 else True
            
            # Get food info
            info = food_info.get(slot['class'], {})
            price = info.get('Giá', 0)
            calories = info.get('Calo', 0)
            
            # Handle string values
            if isinstance(price, str):
                try:
                    price = int(price.replace('đ', '').replace(',', '').strip())
                except:
                    price = 0
            
            if isinstance(calories, str):
                try:
                    calories = int(calories.replace('cal', '').replace(',', '').strip())
                except:
                    calories = 0
            
            segments.append({
                'position': slot['slot'],
                'image': crop_base64,
                'prediction': {
                    'name': slot['class'],
                    'confidence': round(slot['confidence'] * 100, 1),
                    'detected': is_detected,
                    'price': price,
                    'calories': calories
                },
                'box': slot.get('box', [])  # Thêm thông tin bounding box
            })
            
            # Only add to foods list if detected
            if is_detected:
                foods.append({
                    'position': slot['slot'],
                    'name': slot['class'],
                    'confidence': f"{round(slot['confidence'] * 100, 1)}%",
                    'price': price,
                    'calories': calories
                })
        
        # Calculate totals
        total_price = sum(food['price'] for food in foods)
        total_calories = sum(food['calories'] for food in foods)
        points_earned = int(total_price / 10000)  # 1 point per 10000 VND
        
        # Store transaction in user history
        if user_id not in users_data:
            users_data[user_id] = {'points': 0, 'history': []}
        
        users_data[user_id]['points'] += points_earned
        users_data[user_id]['history'].append({
            'timestamp': datetime.now().isoformat(),
            'foods': foods,
            'total_price': total_price,
            'total_calories': total_calories,
            'points_earned': points_earned
        })
        save_users_data()
        
        return jsonify({
            'success': True,
            'segments': segments,  # Đã bao gồm box information
            'foods': foods,
            'total_price': total_price,
            'total_calories': total_calories,
            'points_earned': points_earned,
            'total_points': users_data[user_id]['points'],
            'image_size': result.get('image_size', {})  # Thêm thông tin kích thước ảnh
        })
        
    except Exception as e:
        print(f"Error in analyze endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route('/api/user/<user_id>/points')
def get_user_points(user_id):
    """Get user's total points"""
    if user_id not in users_data:
        users_data[user_id] = {'points': 0, 'history': []}
        save_users_data() 
    
    return jsonify({
        'success': True,
        'total_points': users_data[user_id]['points']
    })

@app.route('/api/user/<user_id>/history')
def get_user_history(user_id):
    """Get user's transaction history"""
    if user_id not in users_data:
        users_data[user_id] = {'points': 0, 'history': []}
        save_users_data()
    
    return jsonify({
        'success': True,
        'history': users_data[user_id]['history']
    })

@app.route('/api/redeem', methods=['POST'])
def redeem_reward():
    """Redeem a reward using points"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        points = data.get('points', 0)
        
        if user_id not in users_data:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        
        if users_data[user_id]['points'] < points:
            return jsonify({'success': False, 'error': 'Insufficient points'}), 400
        
        users_data[user_id]['points'] -= points
        save_users_data() 
        return jsonify({
            'success': True,
            'message': 'Đổi thưởng thành công!',
            'remaining_points': users_data[user_id]['points']
        })
        
    except Exception as e:
        print(f"Error in redeem endpoint: {e}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    model_path = None
    try:
        model_path = _loaded_model_path or _resolve_model_path()
    except Exception:
        model_path = None
    try:
        classes_mtime = os.path.getmtime(CLASSES_PATH) if os.path.exists(CLASSES_PATH) else None
    except Exception:
        classes_mtime = None
    model_num_classes = _get_model_num_classes()
    labels_mismatch = (model_num_classes is not None and model_num_classes != len(food_classes))
    model_input_size = _get_model_input_size()
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': model_path,
        'food_classes_count': len(food_classes),
        'classes_path': CLASSES_PATH,
        'classes_mtime': classes_mtime,
        'model_num_classes': model_num_classes,
        'labels_mismatch': labels_mismatch,
        'model_input_size': model_input_size,
    })

@app.route('/api/food-info')
def get_food_info():
    """Get all food information"""
    return jsonify(food_info)

@app.route('/api/classes')
def get_classes():
    """Get all food classes"""
    return jsonify(food_classes)

@app.route('/api/predict-tray', methods=['POST'])
def predict_tray():
    """Original tray prediction endpoint"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        padding_ratio = float(data.get('padding_ratio', 0.02))
        min_conf = float(data.get('min_confidence', 0.6))
        empty_margin = float(data.get('empty_margin', 0.1))
        empty_var_thresh = float(data.get('empty_var_thresh', 12.0))
        empty_sat_thresh = float(data.get('empty_sat_thresh', 15.0))
        result = _predict_tray_slots(
            data['image'], padding_ratio=padding_ratio, min_confidence=min_conf, empty_margin=empty_margin,
            empty_var_thresh=empty_var_thresh, empty_sat_thresh=empty_sat_thresh
        )
        if 'error' in result:
            return jsonify(result), 400
        enriched = []
        for item in result['slots']:
            if not item:
                enriched.append(None)
                continue
            info = food_info.get(item['class'], {})
            enriched.append({
                **item,
                'price': info.get('Giá', 'N/A'),
                'calories': info.get('Calo', 'N/A'),
                'type': info.get('Loại', 'Món ăn'),
                'health_score': info.get('HealthyScore', 5),
                'features': info.get('Đặc điểm', []),
            })
        return jsonify({
            'success': True,
            'image_size': result['image_size'],
            'predictions': enriched,
        })
    except Exception as e:
        print(f"Error in predict-tray endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("Starting AI Food Recognition Server...")
    
    if load_model():
        print("✅ Model loaded successfully!")
    else:
        print("❌ Failed to load model!")
        exit(1)
    
    print(f"📊 Food classes: {len(food_classes)}")
    print(f"🍽️ Supported foods: {', '.join(food_classes[:3])}...")
    print("🚀 Server starting on http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
