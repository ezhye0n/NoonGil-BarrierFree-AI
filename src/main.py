"""
main.py
눈길(NoonGil) 전체 파이프라인 진입점
이미지 입력 → YOLOv8 탐지 → 회피 경로 → UI 출력
"""
import argparse
from ultralytics import YOLO
from output import draw_output

# TODO: depth 연동 확정 후 추가
# from depth_inference import get_depth_angle
# from slope import get_slope_grade, get_slope_label


def main(image_path: str, model_path: str = "../models/noongil_v4_best.pt"):
    """
    전체 파이프라인 실행

    Args:
        image_path (str): 입력 이미지 경로
        model_path (str): YOLOv8 모델 가중치 경로
    """
    print(f"📂 입력 이미지: {image_path}")
    print(f"🤖 모델 로드 중: {model_path}")

    model = YOLO(model_path)
    last_message = ""

    # 이미지 입력 → YOLOv8 탐지 → 회피 경로 → UI 출력
    last_message = draw_output(image_path, model, last_message)

    # TODO: depth 연동 후 경사도 파이프라인 추가
    # angle = get_depth_angle(image_path)
    # slope_grade = get_slope_grade(angle)
    # slope_label = get_slope_label(slope_grade)

    print("✅ 파이프라인 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="눈길 장애물 탐지 시스템")
    parser.add_argument("--image", type=str, required=True, help="입력 이미지 경로")
    parser.add_argument("--model", type=str,
                        default="../models/noongil_v4_best.pt",
                        help="모델 가중치 경로")
    args = parser.parse_args()

    main(args.image, args.model)