def analyze_bbox_position(label_path, img_height=640):
    """
    YOLO 포맷: class cx cy w h (모두 0~1 정규화)
    cy가 낮을수록(0에 가까울수록) 이미지 위쪽 → 눈높이 촬영
    cy가 높을수록(1에 가까울수록) 이미지 아래쪽 → 고각도 내려봄
    """
    bboxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cy = float(parts[2])  # normalized center y
                bboxes.append(cy)
    
    if not bboxes:
        return None
    
    return {
        "mean_cy": round(np.mean(bboxes), 3),
        "interpretation": (
            "눈높이 촬영 (정면)" if np.mean(bboxes) < 0.5 
            else "고각도 (위에서 내려다봄)"
        )
    }