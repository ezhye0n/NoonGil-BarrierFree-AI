from ultralytics import YOLO

# 모델 로드 (없으면 자동 다운로드)
model = YOLO("yolov8n.pt")

# 예제 이미지로 추론 (ultralytics 제공 샘플 이미지 사용)
results = model.predict(
    source="/Users/takyerin/takyerin/OSSP/TeamProject/curb-1/test/images/0a89796292c7e341de943cf8a632d41e_jpg.rf.65e653296ea687f55ef97c5e25903bce.jpg",  # 예제 이미지
    save=True,       # 결과 이미지 저장
    conf=0.5         # confidence threshold
)

print("추론 완료! runs/detect/predict/ 폴더에 결과 저장됨")