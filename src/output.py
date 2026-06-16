import sys
import os
sys.path.append(os.path.dirname(__file__))
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO
from avoid import get_avoid_direction
from tts import speak, get_tts_message
from depth_inference import pipe as depth_pipe, calculate_ramp_angle
from slope import get_slope_label

CLASS_NAMES = {
    0: "bench", 1: "bicycle", 2: "bollard", 3: "clothing_bin",
    4: "cone", 5: "electric_scooter", 6: "fire_hydrant", 7: "motorcycle",
    8: "pavement_damage", 9: "ramp", 10: "step", 11: "street_light",
    12: "trash", 13: "tree"
}

CLASS_KO = {
    "bench": "벤치", "bicycle": "자전거", "bollard": "볼라드",
    "clothing_bin": "의류수거함", "cone": "라바콘", "electric_scooter": "전동킥보드",
    "fire_hydrant": "소화전", "motorcycle": "오토바이", "pavement_damage": "노면 파손",
    "ramp": "경사로", "step": "단차", "street_light": "가로등",
    "trash": "쓰레기통", "tree": "가로수"
}

import platform

if platform.system() == "Windows":
    FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
elif platform.system() == "Darwin":
    FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
else:
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def put_text_kr(image, text, position, font_size=20, color=(0, 200, 255)):
    """OpenCV 이미지에 한글 텍스트 추가"""
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)    
    font = ImageFont.truetype(FONT_PATH, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_depth_array(image_path: str):
    """depth_inference.pipe로 depth array 반환"""
    from PIL import Image as PILImage
    pil_img = PILImage.open(image_path)
    result = depth_pipe(pil_img)
    return result["predicted_depth"]


def draw_output(image_path: str, model, last_message: str = "", result_dir: str = None, tts_enabled: bool = True) -> tuple:
    """
    이미지에 탐지 결과 시각화 및 TTS 출력

    Args:
        image_path (str): 입력 이미지 경로
        model: YOLOv12 모델 객체
        last_message (str): 직전 TTS 발화 메시지
        result_dir (str): 결과 이미지 저장 폴더 (None이면 원본 폴더에 저장)
        tts_enabled (bool): TTS 출력 여부 (기본값 True)

    Returns:
        tuple: (last_message: str, result: dict)
    """
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    results = model.predict(source=image_path, conf=0.25, verbose=False)
    boxes = results[0].boxes

    # 저장 경로 설정
    if result_dir:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(result_dir, base + "_result.jpg")
    else:
        base, ext = os.path.splitext(image_path)
        output_path = base + "_result" + ext

    if len(boxes) == 0:
        image = put_text_kr(image, "장애물 없음", (10, 40),
                            font_size=24, color=(255, 255, 255))
        cv2.imwrite(output_path, image)
        print(f"저장 완료: {output_path}")
        result = {
            "detections": [],
            "avoid_direction": "장애물 없음",
            "tts_message": None,
            "output_image": output_path,
            "slope": None,
        }
        return last_message, result

    # ramp 탐지 시 depth 추론 (한 번만 실행)
    depth_array = None

    # detections 리스트 구성
    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        class_name = CLASS_NAMES.get(cls_id, "unknown")
        detections.append({
            "cx": cx,
            "confidence": conf,
            "class": class_name
        })

    # 회피 방향
    avoid_direction = get_avoid_direction(detections, w)

    # 바운딩박스 시각화
    best_conf = 0.0
    best_class = None
    detection_results = []
    slope_info = None

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_name = CLASS_NAMES.get(cls_id, "unknown")
        class_ko = CLASS_KO.get(class_name, class_name)
        print(f"탐지: {class_name} ({conf:.2f})")

        detection_results.append({
            "class": class_name,
            "class_ko": class_ko,
            "confidence": round(conf, 4),
            "bbox": [x1, y1, x2, y2],
        })

        # 바운딩박스
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 255), 2)

        # 클래스명 + 신뢰도 (한글)
        label = f"{class_ko} {conf:.2f}"
        image = put_text_kr(image, label, (x1, max(y1 - 25, 0)),
                            font_size=20, color=(0, 200, 255))

        # 경사도 표시 — ramp 탐지 시 depth 연동
        if class_name == "ramp":
            try:
                if depth_array is None:
                    depth_array = get_depth_array(image_path)
                angle, grade = calculate_ramp_angle(depth_array, (x1, y1, x2, y2), img_h=h)
                if angle is not None:
                    slope_label = get_slope_label(
                        "ramp_high" if angle >= 5 else "ramp_mid" if angle >= 2 else "ramp_low"
                    )
                    image = put_text_kr(image, slope_label, (x1, y2 + 25),
                                        font_size=20, color=(255, 100, 0))
                    slope_info = {"angle": angle, "grade": grade}
                    print(f"경사도: {angle}° / {grade}")
            except Exception as e:
                print(f"경사도 계산 오류: {e}")

        if conf > best_conf:
            best_conf = conf
            best_class = class_name

    # 회피 방향 하단 표시 (한글)
    image = put_text_kr(image, avoid_direction, (10, h - 50),
                        font_size=24, color=(0, 255, 0))

    # TTS 출력
    tts_msg = None
    if best_class and tts_enabled:
        tts_msg = get_tts_message(best_class, avoid_direction)
        if tts_msg:
            last_message = speak(tts_msg, last_message)

    # 결과 이미지 저장
    cv2.imwrite(output_path, image)
    print(f"저장 완료: {output_path}")

    result = {
        "detections": detection_results,
        "avoid_direction": avoid_direction,
        "tts_message": tts_msg,
        "output_image": output_path,
        "slope": slope_info,
    }
    return last_message, result

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "noongil_v4_best.pt")
    image_path = os.path.join(BASE_DIR, "../data/raw/images/경사로_1.jpg")
    model = YOLO(model_path)
    last = ""
    last, result = draw_output(image_path, model, last)
    print(result)