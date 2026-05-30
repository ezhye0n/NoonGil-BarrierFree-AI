## 흐름 설계 (의사코드)

[입력]
  이미지 파일 경로를 받는다

[YOLOv8 추론]
  모델에 이미지를 넣어 추론한다
  결과로 바운딩박스 목록을 받는다
  각 박스는 (x1, y1, x2, y2, confidence, class_id) 형태

[바운딩박스 파싱]
  탐지된 박스가 없으면 "장애물 없음" 출력 후 종료
  박스가 있으면 confidence가 가장 높은 박스를 선택

[위치 판별]
  선택된 박스의 중심점 cx = (x1 + x2) / 2 계산
  이미지 너비를 3등분하여 cx가 어느 구역인지 판별
  → left / center / right 반환

[회피 방향 결정]
  zone이 left  → 우측으로 회피
  zone이 right → 좌측으로 회피
  zone이 center → 나머지 박스 중 confidence 낮은 쪽으로 회피
                  박스가 하나뿐이면 우측 기본값

[출력]
  원본 이미지에 바운딩박스 그리기
  클래스명, 회피 방향 텍스트 화면에 표시
  경사도 클래스(ramp_high/mid/low)이면 등급도 표시
  위험 클래스(pothole, ramp_high)이면 TTS 음성 출력

[저장]
  결과 이미지를 results/output_images/ 에 저장