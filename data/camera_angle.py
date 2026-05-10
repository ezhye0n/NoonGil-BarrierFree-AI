import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def estimate_camera_angle(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=50, maxLineGap=10)
    if lines is None:
        return None
    horizon_candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(abs(y2-y1), abs(x2-x1)))
        if angle < 15:
            horizon_candidates.append((y1 + y2) / 2)
    if not horizon_candidates:
        return None
    horizon_y = np.median(horizon_candidates)
    horizon_ratio = horizon_y / h
    estimated_angle = (horizon_ratio - 0.3) * 90
    return {"horizon_ratio": round(horizon_ratio, 3),
            "estimated_tilt_deg": round(estimated_angle, 1)}

def analyze_bbox_position(label_path):
    if not Path(label_path).exists():
        return None
    bboxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cy = float(parts[2])
                bboxes.append(cy)
    if not bboxes:
        return None
    mean_cy = np.mean(bboxes)
    return {
        "bbox_mean_cy": round(mean_cy, 3),
        "interpretation": "눈높이(정면)" if mean_cy < 0.5 else "고각도(내려다봄)"
    }

def analyze_dataset(dataset_root, dataset_name):
    root = Path(dataset_root)
    results = []
    for split in ["train", "valid", "test"]:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            angle_info = estimate_camera_angle(img_path)
            bbox_info = analyze_bbox_position(lbl_path)
            results.append({
                "dataset": dataset_name,
                "file": img_path.name,
                "split": split,
                "horizon_ratio": angle_info["horizon_ratio"] if angle_info else None,
                "tilt_deg": angle_info["estimated_tilt_deg"] if angle_info else None,
                "bbox_mean_cy": bbox_info["bbox_mean_cy"] if bbox_info else None,
                "interpretation": bbox_info["interpretation"] if bbox_info else None,
            })
    return pd.DataFrame(results)

# ▼ 3개 데이터셋 분석
datasets = [
    ("./Pothole-detect-yolo-1", "pothole"),
    ("./curb-1",                "curb"),
    ("./stair-1",               "stair"),
]

all_dfs = []
for path, name in datasets:
    df = analyze_dataset(path, name)
    all_dfs.append(df)
    print(f"\n=== {name} ===")
    print(f"총 이미지 수: {len(df)}")
    print(f"평균 horizon ratio: {df['horizon_ratio'].mean():.3f}")
    print(f"평균 추정 앙각: {df['tilt_deg'].mean():.1f}°")
    if df['interpretation'].notna().any():
        print(df['interpretation'].value_counts())

final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_csv("angle_analysis.csv", index=False)
print("\n✅ angle_analysis.csv 저장 완료")