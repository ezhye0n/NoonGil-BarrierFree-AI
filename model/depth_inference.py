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
    
    delta_h = abs(bottom_depth - top_depth)
    dist    = roi.mean()
    
    if dist < 0.01:
        return None, "측정 불가"
    
    angle_deg = np.degrees(np.arctan(delta_h / dist))
    
    if angle_deg >= 5:
        grade = "상 (통행 어려움)"
    elif angle_deg >= 2:
        grade = "중 (주의 필요)"
    else:
        grade = "하 (통행 용이)"
    
    return round(angle_deg, 2), grade

# ───────────────────────────────────────
# 2. 모델 로드
# ───────────────────────────────────────
pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf", device=0)

# ───────────────────────────────────────
# 3. 경로 설정
# ───────────────────────────────────────
input_folder  = "my_images"
output_folder = "output_results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ───────────────────────────────────────
# 4. 이미지 처리
# ───────────────────────────────────────
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"📂 '{input_folder}'에서 {len(image_files)}장의 사진을 찾았습니다.")

for filename in image_files:
    img_path = os.path.join(input_folder, filename)
    image = Image.open(img_path)
    
    result = pipe(image)
    depth_array = result['predicted_depth']
    
    # 중앙 거리
    center_dist = depth_array[depth_array.shape[0]//2, depth_array.shape[1]//2].item()
    
    # ── ramp bbox가 있을 경우 경사도 계산 ──
    # YOLO 연동 전 임시 테스트: 이미지 전체를 bbox로 사용
    H, W = depth_array.shape
    test_bbox = (0, 0, W, H)
    angle, grade = calculate_ramp_angle(depth_array, test_bbox)
    print(f"✅ {filename} | 중앙거리: {center_dist:.2f}m | 경사각: {angle}° | 등급: {grade}")
    
    # depth map 저장
    save_name = f"result_{center_dist:.2f}m_{filename}"
    save_path = os.path.join(output_folder, save_name)
    result['depth'].save(save_path)
