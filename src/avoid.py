def get_zone(cx, img_w):
    """
    바운딩박스 중심점 cx가 이미지의 어느 구역인지 반환
    
    Args:
        cx (float): 바운딩박스 중심 x좌표
        img_w (int): 이미지 전체 너비
    
    Returns:
        str: "left" / "center" / "right"
    """
    left_bound  = img_w / 3
    right_bound = img_w / 3 * 2

    if cx < left_bound:
        return "left"
    elif cx < right_bound:
        return "center"
    else:
        return "right"
    

def get_avoid_direction(detections, img_w):
    """
    탐지 결과를 받아 회피 방향 결정
    
    Args:
        detections (list): [{"cx": float, "confidence": float, "class": str}, ...]
        img_w (int): 이미지 전체 너비
    
    Returns:
        str: "우측으로 우회하세요" / "좌측으로 우회하세요" / "장애물 없음"
    """
    if not detections:
        return "장애물 없음"

    # confidence 가장 높은 장애물 기준
    primary = max(detections, key=lambda d: d["confidence"])
    zone = get_zone(primary["cx"], img_w)

    if zone == "left":
        return "우측으로 우회하세요"
    elif zone == "right":
        return "좌측으로 우회하세요"
    else:
        # center인 경우: 나머지 박스 중 confidence 낮은 쪽으로 회피
        others = [d for d in detections if d != primary]
        if not others:
            return "우측으로 우회하세요"  # 기본값

        # 좌/우 구역 장애물 confidence 합산
        left_conf  = sum(d["confidence"] for d in others if get_zone(d["cx"], img_w) == "left")
        right_conf = sum(d["confidence"] for d in others if get_zone(d["cx"], img_w) == "right")

        # confidence 낮은 쪽(더 안전한 쪽)으로 회피
        if right_conf <= left_conf:
            return "우측으로 우회하세요"
        else:
            return "좌측으로 우회하세요"