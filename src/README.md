# ⚙️ Source Code Folder (Integration)
이 폴더는 각 모델의 결과물을 하나로 합치고, 실제 수치를 계산하는 핵심 알고리즘을 담습니다.

- **utils/**: 공통으로 사용되는 수학 공식 및 OpenCV 전처리 함수
- **integration.py**: YOLO와 Depth 데이터를 결합하는 메인 파이프라인 코드
- **validation.py**: 정답값(Ground Truth)과 비교하여 오차(RMSE)를 계산하는 검증 코드