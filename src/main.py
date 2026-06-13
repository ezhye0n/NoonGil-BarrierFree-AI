import sys
import os
sys.path.append(os.path.dirname(__file__))
import argparse
from ultralytics import YOLO
from output import draw_output

# TODO: depth 연동 확정 후 추가
# from depth_inference import get_depth_angle
# from slope import get_slope_grade, get_slope_label


def main(image_path: str, model_path: str):
    """
    전체 파이프라인 실행

    Args:
        image_path (str): 입력 이미지 경로
        model_path (str): YOLOv8 모델 가중치 경로
    """
    print(f"📂 입력 이미지: {image_path}")
    print(f"🤖 모델 로드 중: {model_path}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(BASE_DIR, "../results/test_results/")
    os.makedirs(result_dir, exist_ok=True)

    model = YOLO(model_path)
    last_message = ""
    last_message = draw_output(image_path, model, last_message, result_dir=result_dir)

    # TODO: depth 연동 후 경사도 파이프라인 추가
    # angle = get_depth_angle(image_path)
    # slope_grade = get_slope_grade(angle)
    # slope_label = get_slope_label(slope_grade)

    print("✅ 파이프라인 완료")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "noongil_v4_best.pt")
    image_path = os.path.join(BASE_DIR, "../data/raw/images/경사로_1.jpg")

    result_dir = os.path.join(BASE_DIR, "../results/test_results/")
    os.makedirs(result_dir, exist_ok=True)

    model = YOLO(model_path)
    last_message = ""
    last_message = draw_output(image_path, model, last_message, result_dir=result_dir)

    print("✅ 탐지 완료")