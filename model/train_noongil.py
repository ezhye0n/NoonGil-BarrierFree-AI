"""
NoonGil 프로젝트 - 공개 데이터 사전학습 → Fine-tuning 파이프라인
YOLOv8 기반 인도 장애물 탐지


[학습 전략]
Stage 1: COCO pretrained weights 로드 (ultralytics 제공)
Stage 2: AI Hub 인도보행영상 전체 데이터로 fine-tuning
Stage 3: NoonGil 커스텀 데이터로 최종 fine-tuning

Requirements:
    pip install ultralytics
"""

from ultralytics import YOLO
import os
import yaml
from pathlib import Path

# COCO_OVERLAP 교체
COCO_OVERLAP = {
    "bicycle":      1,   # COCO class 1  → NoonGil class 1
    "motorcycle":   3,   # COCO class 3  → NoonGil class 7
    "fire_hydrant": 10,  # COCO class 10 → NoonGil class 6
    "bench":        13,  # COCO class 13 → NoonGil class 0
}

# CONFIG 주석 업데이트
CONFIG = {
    ...
    # 14 classes (Roboflow v1 기준)
    "aihub_data_yaml":   "./aihub_yolo/data.yaml",
    "noongil_data_yaml": "./noongil_yolo/data.yaml",
    ...
}


def stage1_load_pretrained(model_size: str = "s") -> YOLO:
    """
    Stage 1: COCO pretrained YOLOv8 로드
    - ultralytics가 자동으로 다운로드
    - COCO 80 classes로 학습된 backbone 사용
    """
    print("=" * 60)
    print("Stage 1: COCO Pretrained YOLOv8 로드")
    print("=" * 60)

    model_name = f"yolov8{model_size}.pt"
    model = YOLO(model_name)

    print(f"✅ {model_name} 로드 완료 (COCO 80 classes pretrained)")
    print(f"   - Backbone: CSPDarknet53")
    print(f"   - NoonGil 클래스와 겹치는 COCO 클래스 (전이학습 기대):")
    for noongil_cls, coco_id in COCO_OVERLAP.items():
        if coco_id is not None:
            print(f"     → {noongil_cls} (COCO #{coco_id})")
    print(f"   - 직접 수집 필요 클래스 (COCO 없음):")
    print(f"     → trash_can, pothole, uneven_block, curb, ramp")

    return model


def stage2_aihub_finetuning(model: YOLO, config: dict) -> YOLO:
    """
    Stage 2: AI Hub 인도보행영상 전체 데이터로 Fine-tuning
    - 한국 인도 환경에 도메인 적응
    - 학습 가능 클래스: kickboard(0), bicycle(1), motorcycle(2),
                       fire_hydrant(3), bollard(4), street_light(5),
                       bench(6), tree(8)
    """
    print("\n" + "=" * 60)
    print("Stage 2: AI Hub 데이터로 Fine-tuning")
    print("=" * 60)

    if not Path(config["aihub_data_yaml"]).exists():
        print(f"⚠️  AI Hub data.yaml을 찾을 수 없음: {config['aihub_data_yaml']}")
        print("   → AI Hub 전체 데이터 없이 Stage 3으로 진행합니다.")
        return model

    results = model.train(
        data=config["aihub_data_yaml"],
        epochs=config["stage2_epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        project=config["output_dir"],
        name="stage2_aihub",

        # 학습률 설정 (pretrained weight 보존)
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,

        # 데이터 증강
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,      # 상하 반전 X (인도 영상 특성상)
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,

        # 체크포인트
        save=True,
        save_period=10,

        # 조기 종료
        patience=20,

        # 전이학습: backbone freeze (처음 10 에폭)
        freeze=10,
    )

    best_model_path = Path(config["output_dir"]) / "stage2_aihub" / "weights" / "best.pt"
    if best_model_path.exists():
        model = YOLO(str(best_model_path))
        print(f"✅ Stage 2 완료. Best model: {best_model_path}")
    else:
        print("⚠️  best.pt를 찾을 수 없음. 현재 모델 유지.")

    return model


def stage3_noongil_finetuning(model: YOLO, config: dict) -> YOLO:
    """
    Stage 3: NoonGil 커스텀 데이터로 최종 Fine-tuning
    - 13개 클래스 전체 학습 (직접 수집 데이터 포함 시)
    - 낮은 학습률로 세밀한 조정
    """
    print("\n" + "=" * 60)
    print("Stage 3: NoonGil 커스텀 데이터로 최종 Fine-tuning")
    print("=" * 60)

    results = model.train(
        data=config["noongil_data_yaml"],
        epochs=config["stage3_epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        project=config["output_dir"],
        name="stage3_noongil_final",

        # 낮은 학습률 (fine-tuning)
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=1,

        # 증강은 덜 적극적으로
        mosaic=0.5,
        mixup=0.0,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,

        # 저장
        save=True,
        patience=15,
    )

    best_model_path = Path(config["output_dir"]) / "stage3_noongil_final" / "weights" / "best.pt"
    if best_model_path.exists():
        model = YOLO(str(best_model_path))
        print(f"✅ Stage 3 완료. 최종 모델: {best_model_path}")

    return model


def evaluate_model(model: YOLO, data_yaml: str, imgsz: int = 640):
    """최종 모델 평가"""
    print("\n" + "=" * 60)
    print("최종 모델 평가")
    print("=" * 60)

    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=0.25,
        iou=0.6,
    )

    print(f"\n📊 평가 결과:")
    print(f"   mAP@0.5     : {results.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"   Precision   : {results.box.mp:.4f}")
    print(f"   Recall      : {results.box.mr:.4f}")

    return results


def export_model(model: YOLO, format: str = "onnx"):
    """모델 내보내기 (실시간 추론용)"""
    print(f"\n모델 내보내기 ({format})...")
    model.export(format=format, imgsz=CONFIG["imgsz"])
    print(f"✅ {format} 내보내기 완료")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0,
                        help="시작 단계 (0=처음부터, 2=Stage2부터, 3=Stage3만)")
    parser.add_argument("--weights", type=str, default=None,
                        help="기존 가중치 경로 (stage 2,3 시작 시)")
    parser.add_argument("--model_size", type=str, default="s",
                        choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--device", type=str, default=None,
                        help="디바이스 지정 (0=GPU, mps=Apple Silicon, cpu=CPU만)")
    args = parser.parse_args()

    CONFIG["model_size"] = args.model_size
    if args.device is not None:
        CONFIG["device"] = args.device if args.device == "cpu" or args.device == "mps" \
                           else int(args.device)

    # ── Stage 1: Pretrained 로드 ──
    if args.weights and args.stage >= 2:
        print(f"기존 가중치 로드: {args.weights}")
        model = YOLO(args.weights)
    else:
        model = stage1_load_pretrained(CONFIG["model_size"])

    # ── Stage 2: AI Hub Fine-tuning ──
    if args.stage <= 2:
        model = stage2_aihub_finetuning(model, CONFIG)

    # ── Stage 3: NoonGil Fine-tuning ──
    if args.stage <= 3:
        model = stage3_noongil_finetuning(model, CONFIG)

    # ── 평가 ──
    evaluate_model(model, CONFIG["noongil_data_yaml"])

    # ── 내보내기 (실시간 추론용, 필요 시 주석 해제) ──
    # export_model(model, format="onnx")
