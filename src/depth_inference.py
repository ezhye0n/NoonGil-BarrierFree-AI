# -*- coding: utf-8 -*-
import os
import numpy as np
from transformers import pipeline
from PIL import Image
# ───────────────────────────────────────
# 1. 경사도 계산 함수
# ───────────────────────────────────────
def calculate_ramp_angle(depth_array, bbox, img_h=None):
    x1, y1, x2, y2 = map(int, bbox)
    depth_np = depth_array.numpy()
    roi = depth_np[y1:y2, x1:x2]
    
    h = roi.shape[0]
    if h < 10:
        return None, "측정 불가 (bbox 너무 작음)"
    
    # 상하 30%로 샘플링
    n = max(3, h // 3)
    top_depth    = roi[:n, :].mean()
    bottom_depth = roi[-n:, :].mean()
    
    if bottom_depth < 0.01:
        return None, "측정 불가"
    
    depth_ratio = top_depth / bottom_depth

    bbox_h = y2 - y1
    bbox_w = x2 - x1
    aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 1.0
    print(f"[DEBUG] top_depth={top_depth:.4f}, bottom_depth={bottom_depth:.4f}, depth_ratio={depth_ratio:.4f}, aspect_ratio={aspect_ratio:.2f}")

    # 방법 1: depth ratio 기반 등급
    if depth_ratio >= 2.0:
        grade_depth = 2  # 상
    elif depth_ratio >= 1.3:
        grade_depth = 1  # 중
    else:
        grade_depth = 0  # 하

    # 방법 2: bbox 세로 비율 기반 등급 (img_h 있을 때만)
    grade_bbox = 0
    if img_h and img_h > 0:
        bbox_ratio = bbox_h / img_h
        print(f"[DEBUG] bbox_h={bbox_h}, img_h={img_h}, bbox_ratio={bbox_ratio:.4f}")
        if bbox_ratio >= 0.4:
            grade_bbox = 2  # 상
        elif bbox_ratio >= 0.25:
            grade_bbox = 1  # 중
        else:
            grade_bbox = 0  # 하

    # 둘 중 더 높은 등급 채택
    final_grade = max(grade_depth, grade_bbox)
    grade_map = {2: "상 (통행 어려움)", 1: "중 (주의 필요)", 0: "하 (통행 용이)"}
    grade = grade_map[final_grade]

    return round(depth_ratio, 2), grade

# ───────────────────────────────────────
# 2. 모델 로드 (CPU 사용)
# ───────────────────────────────────────
pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    device=-1
)

def run_depth(image_path, bbox=None, img_h=None):
    image = Image.open(image_path)
    result = pipe(image)
    depth_array = result['predicted_depth']
    
    center_dist = depth_array[depth_array.shape[0]//2, depth_array.shape[1]//2].item()
    
    if bbox is None:
        return {
            "center_dist": center_dist,
            "angle": None,
            "grade": "측정 불가 (경사로 미탐지)"
        }

    angle, grade = calculate_ramp_angle(depth_array, bbox, img_h=img_h)
    
    return {
        "center_dist": center_dist,
        "angle": angle,
        "grade": grade
    }