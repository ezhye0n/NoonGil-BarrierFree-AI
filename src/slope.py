def get_slope_grade(angle):
    """
    경사도 각도를 받아 등급 반환

    Args:
        angle (float): 경사도 각도 (도 단위)

    Returns:
        str: "ramp_high" / "ramp_mid" / "ramp_low"
    """
    if angle >= 5.0:
        return "ramp_high"
    elif angle >= 2.0:
        return "ramp_mid"
    else:
        return "ramp_low"


def get_slope_label(grade):
    """
    등급을 한국어 표시용 텍스트로 변환

    Args:
        grade (str): "ramp_high" / "ramp_mid" / "ramp_low"

    Returns:
        str: 화면 출력용 텍스트
    """
    labels = {
        "ramp_high": "경사도: 상 ⚠ (5° 이상, 주의 필요)",
        "ramp_mid":  "경사도: 중 (2°~5°, 서행 권장)",
        "ramp_low":  "경사도: 하 (2° 미만, 통행 용이)",
    }
    return labels.get(grade, "경사도: 측정 불가")


# 사용 예시
if __name__ == "__main__":
    for angle in [1.5, 3.2, 6.8]:
        grade = get_slope_grade(angle)
        label = get_slope_label(grade)
        print(f"{angle}° → {label}")