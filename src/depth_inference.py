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

    # ── 상하 30% 샘플링 ──────────────────────────────
    n = max(3, h // 3)
    top_depth    = roi[:n, :].mean()
    bottom_depth = roi[-n:, :].mean()

    if bottom_depth < 0.01:
        return None, "측정 불가"

    depth_ratio = top_depth / bottom_depth

    # ── 실제 각도 추정 (metric depth 활용) ───────────
    # bbox 세로 픽셀 거리를 실제 거리로 환산
    # Depth-Anything-V2-Metric은 미터 단위 출력
    depth_diff_m = top_depth - bottom_depth          # 수평 거리 차이 (m)
    
    # bbox 세로 길이 → 실세계 y축 거리 근사
    # focal_length_px는 카메라 내부 파라미터, 없으면 실험적 추정값 사용
    FOCAL_LENGTH_PX = 600   # 스마트폰 기준 근사값, 실측 후 조정 권장
    pixel_height = y2 - y1
    avg_depth = (top_depth + bottom_depth) / 2
    real_height_m = (pixel_height / FOCAL_LENGTH_PX) * avg_depth  # 핀홀 카메라 모델
    
    if real_height_m > 0.01:
        angle_rad = np.arctan2(depth_diff_m, real_height_m)
        angle_deg = np.degrees(angle_rad)
    else:
        angle_deg = 0.0

    print(f"[DEBUG] top={top_depth:.3f}m, bottom={bottom_depth:.3f}m, "
          f"depth_ratio={depth_ratio:.3f}, angle≈{angle_deg:.1f}°")

    # ── 방법 1: 추정 각도 기반 등급 ──────────────────
    if angle_deg >= 5.0:
        grade_angle = 2   # 상
    elif angle_deg >= 2.0:
        grade_angle = 1   # 중
    else:
        grade_angle = 0   # 하

    # ── 방법 2: depth_ratio 기반 등급 (보정된 임계값) ─
    # metric depth 기준으로 재설정
    if depth_ratio >= 1.4:
        grade_depth = 2
    elif depth_ratio >= 1.15:
        grade_depth = 1
    else:
        grade_depth = 0

    # ── 방법 3: bbox 세로 비율 기반 등급 ─────────────
    grade_bbox = 0
    if img_h and img_h > 0:
        bbox_ratio = (y2 - y1) / img_h
        print(f"[DEBUG] bbox_ratio={bbox_ratio:.3f}")
        if bbox_ratio >= 0.4:
            grade_bbox = 2
        elif bbox_ratio >= 0.25:
            grade_bbox = 1

    # ── 최종: 셋 중 최댓값 (보수적 판단) ─────────────
    final_grade = max(grade_angle, grade_depth, grade_bbox)
    grade_map = {2: "상 (통행 어려움)", 1: "중 (주의 필요)", 0: "하 (통행 용이)"}

    return round(angle_deg, 2), grade_map[final_grade]

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
