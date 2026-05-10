from ultralytics import YOLO

# 세 모델 비교 실험
models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]

datasets = [
    "./Pothole-detect-yolo-1/data.yaml",
    "./curb-1/data.yaml",
    "./stair-1/data.yaml",
]

for model_name in models:
    model = YOLO(model_name)
    for dataset_path in datasets:
        results = model.val(data=dataset_path)
        print(f"{model_name} | {dataset_path}: mAP={results.box.map:.3f}")
