# -*- coding: utf-8 -*-
import os
import numpy as np
from transformers import pipeline
from PIL import Image
# ───────────────────────────────────────
# 1. 경사도 계산 함수
# ───────────────────────────────────────
def calculate_ramp_angle(depth_array, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    depth_np = depth_array.numpy()
    roi = depth_np[y1:y2, x1:x2]
    
    h = roi.shape[0]
    if h < 10:
        return None, "측정 불가 (bbox 너무 작음)"
    
    n = max(3, h // 5)
    top_depth    = roi[:n, :].mean()
    bottom_depth = roi[-n:, :].mean()
    
    if bottom_depth < 0.01:
        return None, "측정 불가"
    
    # top/bottom depth 비율로 경사 등급 판단
    depth_ratio = top_depth / bottom_depth
    
    if depth_ratio >= 2.5:
        grade = "상 (통행 어려움)"
    elif depth_ratio >= 1.5:
        grade = "중 (주의 필요)"
    else:
        grade = "하 (통행 용이)"
    
    return round(depth_ratio, 2), grade
# ───────────────────────────────────────
# 2. 모델 로드 (CPU 사용)
# ───────────────────────────────────────
pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    device=-1  # CPU 사용 (GPU 없는 환경)
    # device=0  # GPU 사용 시 활성화
)

def run_depth(image_path, bbox=None):
    image = Image.open(image_path)
    result = pipe(image)
    depth_array = result['predicted_depth']
    
    center_dist = depth_array[depth_array.shape[0]//2, depth_array.shape[1]//2].item()
    
    # ramp bbox가 없으면 측정 불가 처리
    if bbox is None:
        return {
            "center_dist": center_dist,
            "angle": None,
            "grade": "측정 불가 (경사로 미탐지)"
        }

    angle, grade = calculate_ramp_angle(depth_array, bbox)
    
    return {
        "center_dist": center_dist,
        "angle": angle,
        "grade": grade
    }
