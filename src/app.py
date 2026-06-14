# src/app.py
import os
import sys
import json
import base64
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from output import draw_output
from avoid import get_avoid_direction
from slope import get_slope_grade, get_slope_label

app = Flask(__name__, static_folder="../frontend", static_url_path="")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "noongil_v4_best.pt")
RESULT_DIR = os.path.join(BASE_DIR, "../results/test_results")
os.makedirs(RESULT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

CLASS_NAMES = {
    0:"bench", 1:"bicycle", 2:"bollard", 3:"clothing_bin",
    4:"cone", 5:"electric_scooter", 6:"fire_hydrant", 7:"motorcycle",
    8:"pavement_damage", 9:"ramp", 10:"step", 11:"street_light",
    12:"trash", 13:"tree"
}
CLASS_KO = {
    "bench":"벤치", "bicycle":"자전거", "bollard":"볼라드",
    "clothing_bin":"의류수거함", "cone":"라바콘",
    "electric_scooter":"전동킥보드", "fire_hydrant":"소화전",
    "motorcycle":"오토바이", "pavement_damage":"노면 파손",
    "ramp":"경사로", "step":"단차", "street_light":"가로등",
    "trash":"쓰레기통", "tree":"가로수"
}

@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    # 1. 이미지 받기
    if "image" not in request.files:
        return jsonify({"error": "이미지가 없습니다"}), 400

    file = request.files["image"]
    upload_path = os.path.join(RESULT_DIR, "upload_temp.jpg")
    file.save(upload_path)

    # 2. YOLOv8 탐지
    results = model.predict(source=upload_path, conf=0.25, verbose=False)
    boxes = results[0].boxes
    img_h, img_w = results[0].orig_shape

    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        class_name = CLASS_NAMES.get(cls_id, "unknown")
        detections.append({
            "label":      CLASS_KO.get(class_name, class_name),
            "label_en":   class_name,
            "confidence": round(conf, 2),
            "bbox": {
                "x": round(x1 / img_w, 4),
                "y": round(y1 / img_h, 4),
                "w": round((x2 - x1) / img_w, 4),
                "h": round((y2 - y1) / img_h, 4)
            },
            "cx": cx
        })

    # 3. 회피 방향
    avoid_input = [{"cx": d["cx"], "confidence": d["confidence"],
                    "class": d["label_en"]} for d in detections]
    avoid_direction = get_avoid_direction(avoid_input, img_w) if detections else "장애물 없음"

    # 4. 경사도 (TODO: depth 연동 후 실제 각도로 교체)
    ramp_detected = any(d["label_en"] == "ramp" for d in detections)
    slope_grade   = "하"
    slope_angle   = 0.0
    slope_warning = "경사도 측정 불가 (depth 연동 예정)"
    if ramp_detected:
        slope_warning = "경사로 감지됨 — 실제 경사도 측정은 depth 연동 후 제공 예정"

    # 5. 휠체어 안내
    danger_classes = {"pavement_damage", "ramp", "step", "bollard", "cone"}
    has_danger = any(d["label_en"] in danger_classes for d in detections)
    wheelchair = {
        "needsAssistance":    has_danger,
        "canEnter":           "주의 진입" if has_danger else "가능",
        "hazardDescription":  avoid_direction + " — 위험 구간 우회 후 진입 권장" if has_danger else "안전한 경로입니다"
    }

    # 6. 결과 이미지 생성
    result_img_path = os.path.join(RESULT_DIR, "result.jpg")
    draw_output(upload_path, model, result_dir=RESULT_DIR, tts_enabled=False)

    # 결과 이미지 base64 인코딩
    result_img_b64 = ""
    candidate = os.path.join(RESULT_DIR, "upload_temp_result.jpg")
    if os.path.exists(candidate):
        with open(candidate, "rb") as f:
            result_img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 7. result.json 저장
    result = {
        "detections": detections,
        "avoidPath": {
            "direction":    avoid_direction,
            "description":  avoid_direction,
            "pathPoints":   []
        },
        "slope": {
            "grade":   slope_grade,
            "angle":   slope_angle,
            "warning": slope_warning
        },
        "wheelchair": wheelchair,
        "resultImage": result_img_b64
    }

    json_path = os.path.join(RESULT_DIR, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return jsonify(result)

if __name__ == "__main__":
    print("눈길 서버 시작: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)