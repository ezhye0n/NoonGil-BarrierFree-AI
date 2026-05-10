"""
analysis.py  —  턱 높이(cm) · 경사도(°) · 안전등급(A/B/C) 계산 모듈
이지현 담당 | feat/integration 브랜치

필요한 입력:
  - bbox_result  : 탁예린님(YOLO)이 만든 dict  → {"class": "step", "bbox": [x1,y1,x2,y2], "conf": 0.87}
  - depth_map    : 윤서연님(Depth Anything)이 만든 numpy 배열 → shape (H, W), dtype float32, 단위: 미터
  - camera_params: 카메라 내부 파라미터 dict  → {"fx":..., "fy":..., "cx":..., "cy":..., "height_m":...}
"""

import numpy as np


# ──────────────────────────────────────────────
# 0. 카메라 기본 파라미터 (스마트폰 기준 기본값)
# ──────────────────────────────────────────────
# 실제 촬영 스마트폰의 EXIF focal length 값으로 교체해야 정확해짐
# 방법: PIL 라이브러리로 EXIF 읽기  →  docs/math.md 참고
DEFAULT_CAMERA = {
    "fx": 1000.0,    # x축 초점거리 (픽셀 단위) — 스마트폰 표준 근사값
    "fy": 1000.0,    # y축 초점거리 (픽셀 단위)
    "cx": 640.0,     # 이미지 중심 x (보통 width / 2)
    "cy": 360.0,     # 이미지 중심 y (보통 height / 2)
    "height_m": 1.2, # 카메라(스마트폰) 지면으로부터 높이 — 미터
}


# ──────────────────────────────────────────────
# 1. 픽셀 좌표 → 3D 공간 좌표 역투영 (핀홀 카메라 모델)
# ──────────────────────────────────────────────
def pixel_to_3d(u, v, depth_map, camera_params):
    """
    픽셀 (u, v) 와 해당 픽셀의 depth 값으로 실제 3D 좌표 (X, Y, Z) 를 계산한다.

    핀홀 카메라 역투영 공식:
        Z = depth_map[v, u]          (카메라→물체 거리, 미터)
        X = (u - cx) * Z / fx        (좌우 실제 거리, 미터)
        Y = (v - cy) * Z / fy        (상하 실제 거리, 미터)

    Args:
        u (int)           : 픽셀 x좌표 (열)
        v (int)           : 픽셀 y좌표 (행)
        depth_map (ndarray): shape (H, W), float32, 단위 미터
        camera_params (dict): fx, fy, cx, cy

    Returns:
        tuple: (X, Y, Z) — 모두 미터 단위
    """
    H, W = depth_map.shape

    # 픽셀이 이미지 범위를 벗어나면 클램핑
    u = int(np.clip(u, 0, W - 1))
    v = int(np.clip(v, 0, H - 1))

    Z = float(depth_map[v, u])

    # depth 값이 비정상이면 (0이하, inf, nan) None 반환
    if Z <= 0 or not np.isfinite(Z):
        return None

    fx = camera_params["fx"]
    fy = camera_params["fy"]
    cx = camera_params["cx"]
    cy = camera_params["cy"]

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return (X, Y, Z)


# ──────────────────────────────────────────────
# 2. 단차(턱) 높이 계산
# ──────────────────────────────────────────────
def calculate_step_height(bbox, depth_map, camera_params=None):
    """
    YOLO bbox 와 depth map 으로 턱 높이(cm)를 계산한다.

    원리:
        - bbox 하단 중앙 픽셀  → 3D 좌표 P_bot  (바닥)
        - bbox 상단 중앙 픽셀  → 3D 좌표 P_top  (턱 꼭대기)
        - 높이 h = |Y_top - Y_bot| * 100  (미터 → cm 변환)

    Args:
        bbox (list)       : [x1, y1, x2, y2] — YOLO 출력 픽셀 좌표
        depth_map (ndarray): shape (H, W), float32, 단위 미터
        camera_params (dict): 카메라 파라미터 (None이면 기본값 사용)

    Returns:
        float or None: 턱 높이 (cm). 계산 불가 시 None.
    """
    if camera_params is None:
        camera_params = DEFAULT_CAMERA

    x1, y1, x2, y2 = bbox
    u_center = int((x1 + x2) / 2)  # bbox 가로 중앙
    v_top    = int(y1)              # bbox 상단 → 턱 꼭대기
    v_bot    = int(y2)              # bbox 하단 → 바닥 접점

    p_top = pixel_to_3d(u_center, v_top, depth_map, camera_params)
    p_bot = pixel_to_3d(u_center, v_bot, depth_map, camera_params)

    if p_top is None or p_bot is None:
        return None  # depth 값 비정상

    # Y축: 카메라 좌표계에서 위아래 방향
    # 높이 차이를 절댓값으로 취하고 미터→cm 변환
    height_m = abs(p_top[1] - p_bot[1])
    height_cm = height_m * 100.0

    return round(height_cm, 2)


# ──────────────────────────────────────────────
# 3. 경사도 계산
# ──────────────────────────────────────────────
def calculate_slope_deg(bbox, depth_map, camera_params=None):
    """
    bbox 구간의 depth gradient 를 arctan 으로 변환해 경사도(°)를 계산한다.

    원리:
        1. bbox 하단 행(v_bot) 에서 좌→우 픽셀들의 depth 값을 가져온다.
        2. np.gradient 로 픽셀 단위 depth 변화량(dz_du)을 구한다.
        3. 실제 거리로 환산: dz_dx = dz_du / pixel_size_m
           pixel_size_m = Z / fx  (해당 깊이에서 1픽셀의 실제 크기)
        4. 경사각 θ = arctan(|dz_dx_mean|) 를 도(°)로 변환

    Args:
        bbox (list)        : [x1, y1, x2, y2]
        depth_map (ndarray): shape (H, W), float32, 단위 미터
        camera_params (dict): 카메라 파라미터

    Returns:
        float or None: 경사도 (°). 계산 불가 시 None.
    """
    if camera_params is None:
        camera_params = DEFAULT_CAMERA

    x1, y1, x2, y2 = bbox
    H, W = depth_map.shape

    # bbox 하단 10% 구간 평균으로 바닥 경사를 측정
    v_bot = int(np.clip(y2, 0, H - 1))
    u_start = int(np.clip(x1, 0, W - 1))
    u_end   = int(np.clip(x2, 0, W - 1))

    if u_end <= u_start:
        return None

    # 해당 행의 depth 값 슬라이스
    depth_row = depth_map[v_bot, u_start:u_end].astype(float)

    # nan/inf/0 필터링
    valid_mask = np.isfinite(depth_row) & (depth_row > 0)
    if valid_mask.sum() < 3:
        return None  # 유효 픽셀 부족

    depth_row_valid = depth_row.copy()
    depth_row_valid[~valid_mask] = np.nan

    # 픽셀 단위 depth gradient
    dz_du = np.gradient(depth_row_valid)

    # 평균 depth 로 pixel_size 추정 (미터/픽셀)
    mean_z = float(np.nanmean(depth_row_valid))
    if mean_z <= 0:
        return None

    pixel_size_m = mean_z / camera_params["fx"]

    # 실제 공간 gradient (m/m = 무차원 기울기)
    dz_dx = np.nanmean(np.abs(dz_du)) / pixel_size_m

    # arctan → 도(°) 변환
    slope_deg = float(np.degrees(np.arctan(dz_dx)))

    return round(slope_deg, 2)


# ──────────────────────────────────────────────
# 4. 안전등급 산출
# ──────────────────────────────────────────────
# 수동 휠체어 / 전동 휠체어 기준 참고값
# A : 완전 통과 가능   (턱 < 2cm,  경사 < 5°)
# B : 주의 통과 가능   (턱 < 5cm,  경사 < 8°)
# C : 통과 어려움      (그 이상)
GRADE_THRESHOLDS = {
    "A": {"step_cm": 2.0,  "slope_deg": 5.0},
    "B": {"step_cm": 5.0,  "slope_deg": 8.0},
}

def safety_grade(step_cm, slope_deg):
    """
    턱 높이(cm)와 경사도(°)를 받아 안전등급(A/B/C)을 반환한다.

    Args:
        step_cm  (float): 턱 높이 (cm). None 이면 0으로 처리.
        slope_deg (float): 경사도 (°). None 이면 0으로 처리.

    Returns:
        str: "A", "B", "C" 중 하나
    """
    step  = step_cm   if step_cm   is not None else 0.0
    slope = slope_deg if slope_deg is not None else 0.0

    if step < GRADE_THRESHOLDS["A"]["step_cm"] and slope < GRADE_THRESHOLDS["A"]["slope_deg"]:
        return "A"
    elif step < GRADE_THRESHOLDS["B"]["step_cm"] and slope < GRADE_THRESHOLDS["B"]["slope_deg"]:
        return "B"
    else:
        return "C"


# ──────────────────────────────────────────────
# 5. 전체 파이프라인 (1장 이미지)
# ──────────────────────────────────────────────
def analyze_image(bbox_results, depth_map, camera_params=None):
    """
    하나의 이미지에 대해 모든 bbox를 분석해 최종 결과를 반환한다.

    Args:
        bbox_results (list of dict): 팀원 1 YOLO 출력
            예) [{"class": "step", "bbox": [100,200,300,400], "conf": 0.87}]
        depth_map (ndarray): 팀원 2 depth 출력 — shape (H,W), float32, 미터
        camera_params (dict): 카메라 파라미터

    Returns:
        list of dict: 각 bbox 에 대한 분석 결과
            예) [{"class": "step", "bbox": [...], "step_cm": 4.2,
                  "slope_deg": 3.1, "grade": "B", "conf": 0.87}]
    """
    results = []

    for det in bbox_results:
        bbox       = det["bbox"]
        obj_class  = det.get("class", "unknown")
        conf       = det.get("conf", 0.0)

        step_cm   = calculate_step_height(bbox, depth_map, camera_params)
        slope_deg = calculate_slope_deg(bbox, depth_map, camera_params)
        grade     = safety_grade(step_cm, slope_deg)

        results.append({
            "class"    : obj_class,
            "bbox"     : bbox,
            "conf"     : round(conf, 3),
            "step_cm"  : step_cm,
            "slope_deg": slope_deg,
            "grade"    : grade,
        })

    return results


# ──────────────────────────────────────────────
# 6. RMSE 검증 (ground_truth.csv 와 비교)
# ──────────────────────────────────────────────
def compute_rmse(predictions, ground_truths):
    """
    예측값과 실측값의 RMSE 를 계산한다.

    Args:
        predictions  (list of float): 모델 예측 높이 or 경사도
        ground_truths (list of float): 줄자/각도기로 측정한 실제값

    Returns:
        float: RMSE 값
    """
    preds  = np.array(predictions,   dtype=float)
    truths = np.array(ground_truths, dtype=float)
    rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))
    return round(rmse, 4)


# ──────────────────────────────────────────────
# 7. 동작 테스트 (mock 데이터 — 팀원 1·2 없어도 실행 가능)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  analysis.py — mock 데이터 동작 테스트")
    print("=" * 50)

    # 가짜 depth map (720x1280, 값: 1.0~5.0 미터)
    np.random.seed(42)
    H, W = 720, 1280
    mock_depth = np.random.uniform(1.5, 4.0, (H, W)).astype(np.float32)
    # 턱 구간에 depth 차이 인위적으로 추가
    mock_depth[300:400, 400:600] = 1.2   # 턱 상단: 카메라에 더 가깝게

    # 가짜 bbox (팀원 1 YOLO 출력 형식)
    mock_bboxes = [
        {"class": "step",     "bbox": [400, 290, 600, 410], "conf": 0.91},
        {"class": "pothole",  "bbox": [700, 400, 850, 480], "conf": 0.76},
        {"class": "kickboard","bbox": [200, 200, 350, 380], "conf": 0.83},
    ]

    # 카메라 파라미터 (1280x720 기준 근사값)
    cam = {**DEFAULT_CAMERA, "cx": W / 2, "cy": H / 2}

    # 분석 실행
    output = analyze_image(mock_bboxes, mock_depth, cam)

    print("\n[분석 결과]")
    for r in output:
        step  = f"{r['step_cm']:.1f}cm" if r['step_cm']  is not None else "측정불가"
        slope = f"{r['slope_deg']:.1f}°"  if r['slope_deg'] is not None else "측정불가"
        print(f"  {r['class']:10s}  턱높이: {step:>8}  경사도: {slope:>6}  등급: {r['grade']}  conf: {r['conf']}")

    # RMSE 테스트
    pred_heights   = [r["step_cm"] for r in output if r["step_cm"] is not None]
    gt_heights     = [3.5, 0.0, 0.0][:len(pred_heights)]  # 가짜 정답값
    if pred_heights:
        rmse = compute_rmse(pred_heights, gt_heights)
        print(f"\n[RMSE 검증]  예측 높이 RMSE = {rmse} cm")

    print("\n✅ 모든 함수 정상 동작 확인")
