# Hyochang-BarrierFree-AI
[프로그램 구동 흐름]

Data Input: 실시간 촬영 이미지 1장 입력

Object Detection (YOLO): 장애물 Bbox 좌표 추출

Depth Estimation (Depth Anything): 거리 데이터(Depth Map) 생성

Integration & Calculation: 추출된 좌표와 거리값을 수식에 대입하여 실제 물리량 계산

Output: 통과 가능 여부 사용자 알림


Project Overview: 사회적 이슈(효창공원 언덕길 경험)와 서비스 목적.

Environment Setup: pip install -r requirements.txt, WSL2/GPU 설정법.

Dataset: 직접 수집한 데이터 구조 설명 및 정답값(Ground Truth) 측정 방식.

Model Architecture: YOLOv8과 Depth 모델이 어떻게 결합되는지 도식화.

How to Run: 교수님이 복사해서 바로 실행해볼 수 있는 명령어.

Results & Failure Analysis: 성공적인 시각화 결과와 솔직한 오작동 사례 분석 표.
수치적 결과 그래프와 '솔직한 실패 사례(오작동 사진 및 원인 분석)' 포함하기git
