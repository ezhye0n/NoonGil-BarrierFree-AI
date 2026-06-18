import sys
import os
import argparse
import json
sys.path.append(os.path.dirname(__file__))
from ultralytics import YOLO
from output import draw_output
from depth_inference import run_depth


def main():
    """전체 파이프라인 실행"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="입력 이미지 경로")
    parser.add_argument("--output_json", default=None, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    image_path = args.image
    print(f"📂 입력 이미지: {image_path}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "v12_no_curb_ep100_best.pt")
    print(f"🤖 모델 로드 중: {model_path}")

    result_dir = os.path.join(BASE_DIR, "../results/test_results/")
    os.makedirs(result_dir, exist_ok=True)

    model = YOLO(model_path)
    last_message = ""
    last_message, result = draw_output(image_path, model, last_message, result_dir=result_dir, tts_enabled=False)

    if not result.get("tts_message") and result.get("detections"):
        from tts import get_tts_message
        best = max(result["detections"], key=lambda d: d["confidence"])
        result["tts_message"] = get_tts_message(best["class"], result["avoid_direction"])

    # draw_output() 호출 후 추가
    # ramp 클래스 bbox 추출
    ramp_bbox = None
    for det in result.get("detections", []):
        if det["class"] == "ramp":
            ramp_bbox = det["bbox"]
            break
    
    # ramp bbox 전달 (없으면 None → 측정 불가 처리)
    try:
        depth_result = run_depth(image_path, bbox=ramp_bbox)
        result["slope"] = {
            "center_dist": float(depth_result["center_dist"]),
            "angle": float(depth_result["angle"]) if depth_result["angle"] is not None else None,
            "grade": depth_result["grade"]
        }
        print(f"[DEBUG] slope 계산 완료: angle={depth_result['angle']}, grade={depth_result['grade']}")
    except Exception as e:
        print(f"[ERROR] depth 추론 실패: {e}")
        result["slope"] = {
            "center_dist": None,
            "angle": None,
            "grade": "측정 불가"
        }

    if not result.get("tts_message"):
        result["tts_message"] = last_message

    if args.output_json:
        out_dir = os.path.dirname(os.path.abspath(args.output_json))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"📄 결과 JSON 저장: {args.output_json}")

    print("✅ 파이프라인 완료")


if __name__ == "__main__":
    main()
