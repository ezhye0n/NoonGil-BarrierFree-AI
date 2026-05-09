# Model Output Interface (v1.0)

## 1. Team A (YOLO - Obstacle Detection)
- **Format:** JSON
- **Fields:**
  - `box_2d`: [xmin, ymin, xmax, ymax] (픽셀 좌표)
  - `label`: "curb", "pothole", "kickboard"
  - `conf`: 인식 신뢰도 (0~1)

## 2. Team B (Depth - Distance Mapping)
- **Format:** .npy (Numpy Array) 또는 .png (Depth Map)
- **Description:** 이미지의 각 픽셀에 대응하는 깊이값(0~255 또는 실제 거리) 배열