# Model Output Interface (v1.0)

이지현 작성 | 최종 수정: 2026-06-14

---

## 1. 탁예린 (YOLO - Obstacle Detection)

- **Format:** JSON
- **Fields:**

| 필드 | 타입 | 설명 | 예시 |
|------|------|------|------|
| `class` | string | 장애물 클래스명 (영문) | `"tree"`, `"bicycle"`, `"electric_scooter"` |
| `class_ko` | string | 장애물 클래스명 (한글) | `"가로수"`, `"자전거"`, `"전동킥보드"` |
| `confidence` | float (0~1) | 인식 신뢰도 | `0.89` |
| `bbox` | int[4] | 바운딩박스 픽셀 좌표 `[x1, y1, x2, y2]` | `[463, 7, 650, 902]` |

- **전체 응답 구조 예시:**

```json
{
  "detections": [
    {
      "class": "tree",
      "class_ko": "가로수",
      "confidence": 0.8952,
      "bbox": [463, 7, 650, 902]
    },
    {
      "class": "electric_scooter",
      "class_ko": "전동킥보드",
      "confidence": 0.7296,
      "bbox": [633, 402, 901, 1347]
    }
  ],
  "avoid_direction": "좌측으로 우회하세요",
  "tts_message": "전방에 가로수 감지. 좌측으로 우회하세요",
  "output_image": "results/test_results/test_image1_result.jpg",
  "slope": {
    "center_dist": 13.46,
    "angle": 46.84,
    "grade": "상 (통행 어려움)"
  }
}
```

- **`slope` 필드 상세:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `center_dist` | float | Depth Map 기반 중앙 거리 추정값 |
| `angle` | float | 추정 경사 각도 (°) |
| `grade` | string | 경사도 등급 |

- **경사도 등급 기준:**

| 등급 | 조건 | 의미 |
|------|------|------|
| `"상 (통행 어려움)"` | 5° 이상 | 휠체어 단독 통행 어려움 |
| `"중 (주의 필요)"` | 2° 이상 5° 미만 | 주의 필요 |
| `"하 (통행 용이)"` | 2° 미만 | 통행 용이 |

- **지원 클래스 목록:**

| 영문 | 한글 |
|------|------|
| `bench` | 벤치 |
| `bicycle` | 자전거 |
| `bollard` | 볼라드 |
| `clothing_bin` | 의류수거함 |
| `cone` | 라바콘 |
| `electric_scooter` | 전동킥보드 |
| `fire_hydrant` | 소화전 |
| `motorcycle` | 오토바이 |
| `pavement_damage` | 노면 파손 |
| `ramp` | 경사로 |
| `step` | 단차 |
| `street_light` | 가로등 |
| `trash` | 쓰레기통 |
| `tree` | 가로수 |

---

## 2. 윤서연 (Depth - Distance Mapping)

- **Format:** `.npy` (Numpy Array) 또는 `.png` (Depth Map)
- **Description:** 이미지의 각 픽셀에 대응하는 깊이값 (0~255 또는 실제 거리) 배열
- **연동:** `slope.py`의 `get_slope_grade(angle)` 입력값으로 사용 (연동 완료)

---

## 3. 이지현 (Output - TTS & UI)

- **Input:** YOLO 탐지 결과 JSON (위 1번 형식)
- **Output:**
  - TTS 음성 출력 (`edge-tts`, `ko-KR-SunHiNeural`)
  - 결과 이미지 저장 (`results/test_results/xxx_result.jpg`)
  - 웹 UI 표시 (`index.html` → Flask `/analyze` 엔드포인트)