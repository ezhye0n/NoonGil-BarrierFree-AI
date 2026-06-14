import sys
import os
import argparse
import json
sys.path.append(os.path.dirname(__file__))
from ultralytics import YOLO
from output import draw_output


def main():
    """전체 파이프라인 실행"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="입력 이미지 경로")
    parser.add_argument("--output_json", default=None, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    image_path = args.image
    print(f"📂 입력 이미지: {image_path}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "noongil_v4_best.pt")
    print(f"🤖 모델 로드 중: {model_path}")

    result_dir = os.path.join(BASE_DIR, "../results/test_results/")
    os.makedirs(result_dir, exist_ok=True)

    model = YOLO(model_path)
    last_message = ""
    last_message, result = draw_output(image_path, model, last_message, result_dir=result_dir)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"📄 결과 JSON 저장: {args.output_json}")

    print("✅ 파이프라인 완료")


if __name__ == "__main__":
    main()