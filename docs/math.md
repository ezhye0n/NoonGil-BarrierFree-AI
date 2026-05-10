# 수식 정리 — 단차 높이 · 경사도 계산 원리

> 이지현 작성 | `docs/math.md`  
> 이 문서는 `analysis.py` 의 모든 계산 함수의 수학적 근거를 설명합니다.  
> 다른 팀원이 자신의 출력값이 어떻게 쓰이는지 이해하는 데도 참고.

---

## 목차

1. [핀홀 카메라 모델](#1-핀홀-카메라-모델)
2. [픽셀 → 3D 좌표 역투영](#2-픽셀--3d-좌표-역투영)
3. [단차(턱) 높이 계산](#3-단차턱-높이-계산)
4. [경사도 계산](#4-경사도-계산)
5. [안전등급 기준](#5-안전등급-기준)
6. [카메라 파라미터 추출 방법](#6-카메라-파라미터-추출-방법)
7. [오차 및 한계](#7-오차-및-한계)

---

## 1. 핀홀 카메라 모델

스마트폰 카메라는 **핀홀 카메라(Pinhole Camera)** 로 근사할 수 있습니다.  
3D 공간의 한 점 `P = (X, Y, Z)` 가 이미지의 픽셀 `(u, v)` 에 다음과 같이 투영됩니다.

```
u = fx * (X / Z) + cx
v = fy * (Y / Z) + cy
```

| 기호 | 의미 | 단위 |
|------|------|------|
| `fx`, `fy` | 초점거리 (x축, y축) | 픽셀 |
| `cx`, `cy` | 주점 (이미지 중심) | 픽셀 |
| `X`, `Y`, `Z` | 카메라 좌표계의 3D 좌표 | 미터 |
| `u`, `v` | 픽셀 좌표 (열, 행) | 픽셀 |

> **카메라 좌표계 방향**  
> - Z축: 카메라 전방 (피사체 방향)  
> - Y축: 아래 방향 (이미지 행 증가 방향)  
> - X축: 오른쪽 방향 (이미지 열 증가 방향)

---

## 2. 픽셀 → 3D 좌표 역투영

Depth Anything 모델이 출력한 `depth_map[v, u]` 값을 `Z` 로 사용하면,  
위 투영 공식을 역으로 풀어 3D 좌표를 복원할 수 있습니다.

```
Z = depth_map[v, u]          (단위: 미터)

X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

### 구현 함수: `pixel_to_3d(u, v, depth_map, camera_params)`

```python
Z = float(depth_map[v, u])
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
return (X, Y, Z)
```

> **주의**: `depth_map` 값이 **절대 거리(미터)** 인지 **상대값(0~1)** 인지 반드시 확인해야 합니다.  
> Depth Anything V2 기본 출력은 상대값일 수 있으므로, 팀원 2와 단위 합의 필수!  
> → `docs/interface.md` 참고

---

## 3. 단차(턱) 높이 계산

### 원리

YOLO가 검출한 bbox 의 **상단 픽셀**은 턱 꼭대기, **하단 픽셀**은 바닥 접점입니다.  
두 픽셀을 3D 좌표로 역투영한 뒤 Y축 차이를 구하면 실제 높이가 됩니다.

```
P_top = pixel_to_3d(u_center, y1, depth_map, cam)   # 턱 상단
P_bot = pixel_to_3d(u_center, y2, depth_map, cam)   # 바닥

h = |Y_top - Y_bot| * 100    (미터 → cm)
```

여기서 `u_center = (x1 + x2) / 2` (bbox 가로 중앙)

### 그림

```
카메라 (0,0,0)
    |
    |  광선 A (y1, bbox 상단)
    | /
    |/___________  ← 턱 꼭대기   (Y_top)
    |             |
    |             |  h = |Y_top - Y_bot|
    |  광선 B     |
    | (y2, 하단)  |
    |_____________|  ← 바닥       (Y_bot)
```

### 구현 함수: `calculate_step_height(bbox, depth_map, camera_params)`

```python
u_center = (x1 + x2) / 2
p_top = pixel_to_3d(u_center, y1, depth_map, cam)
p_bot = pixel_to_3d(u_center, y2, depth_map, cam)
height_cm = abs(p_top[1] - p_bot[1]) * 100
```

---

## 4. 경사도 계산

### 원리

경사면에서 depth 값은 수평으로 진행할수록 점점 변합니다.  
이 변화율(gradient)을 구한 뒤 `arctan` 으로 각도로 변환합니다.

**Step 1. depth gradient 계산 (픽셀 단위)**

```
dz_du[i] = depth_map[v, u+1] - depth_map[v, u]
```

NumPy 에서는 `np.gradient(depth_row)` 로 한 번에 계산합니다.

**Step 2. 픽셀 크기를 실제 거리(m)로 환산**

깊이 Z 에서 1픽셀이 실제로 몇 미터인지:

```
pixel_size_m = Z / fx    (단위: m/pixel)
```

**Step 3. 실제 공간 gradient**

```
dz_dx = dz_du / pixel_size_m    (무차원: m/m)
```

**Step 4. arctan → 각도**

```
θ = arctan( |dz_dx| )    (라디안)
θ_deg = θ * 180 / π      (도, °)
```

### 그림

```
     /|
    / |  Δz (depth 차이, 미터)
   /  |
  /θ  |
 /----+
   Δx (수평거리, 미터)

θ = arctan(Δz / Δx)
```

### 구현 함수: `calculate_slope_deg(bbox, depth_map, camera_params)`

```python
depth_row = depth_map[v_bot, x1:x2]
dz_du     = np.gradient(depth_row)
pixel_size_m = mean_z / fx
dz_dx     = np.mean(np.abs(dz_du)) / pixel_size_m
slope_deg = np.degrees(np.arctan(dz_dx))
```

---

## 5. 안전등급 기준

휠체어 관련 국내외 기준(BF 인증, ADA 기준)을 참고한 값입니다.

| 등급 | 턱 높이 | 경사도 | 의미 |
|------|---------|--------|------|
| **A** | < 2 cm | < 5°  | 수동·전동 휠체어 모두 통과 가능 |
| **B** | < 5 cm | < 8°  | 전동 휠체어 통과 가능, 수동은 주의 |
| **C** | 5 cm 이상 | 8° 이상 | 통과 어려움, 우회 경로 필요 |

> **근거 자료**  
> - ADA (미국 장애인법): 경사로 최대 경사 1:12 ≈ 4.76°  
> - 국내 BF 인증 기준: 단차 2cm 이하, 경사 1/18 ≈ 3.18°  
> - 전동 휠체어 제조사 권장: 단차 5cm 이하, 경사 8° 이하

---

## 6. 카메라 파라미터 추출 방법

### 방법 A: EXIF 자동 추출 (권장)

```bash
python extract_camera_params.py --image data/raw/IMG_0001.jpg
python extract_camera_params.py --folder data/raw/   # 여러 장 평균
```

출력된 `camera_params.json` 의 값을 `analysis.py` 의 `DEFAULT_CAMERA` 에 붙여넣으면 됩니다.

### 방법 B: 수동 계산

스마트폰 앱(EXIF Viewer 등)으로 확인한 focal length(mm)와 이미지 크기를 사용합니다.

```
fx = focal_length_mm / sensor_width_mm * image_width_px
fy = focal_length_mm / sensor_height_mm * image_height_px
cx = image_width_px / 2
cy = image_height_px / 2
```

### 일반적인 스마트폰 기본값 (EXIF 없을 때 폴백)

| 기종 | fx (1280px 기준) | 비고 |
|------|-----------------|------|
| iPhone 14/15 광각 | ~1400 | EXIF 확인 권장 |
| Samsung S23 광각 | ~1300 | |
| 일반 Android | max(W,H) 근사 | 오차 큼 |

> `height_m` (카메라 지면 높이) 는 EXIF 에 없습니다.  
> **현장 답사 시 줄자로 직접 측정**해서 `camera_params.json` 에 기입하세요.  
> 보통 스마트폰을 가슴 높이로 들고 촬영: 약 1.1 ~ 1.3 m

---

## 7. 오차 및 한계

### 예상 오차 원인

| 원인 | 영향 | 대응 |
|------|------|------|
| Depth Anything 상대값 문제 | 스케일 오차 큼 | 실측 거리로 스케일 보정 |
| EXIF focal length 부정확 | fx 오차 → 높이 오차 | 여러 장 평균, 방법 B 병행 |
| bbox 하단이 바닥이 아닌 경우 | 높이 과대 추정 | confidence 임계값 0.7 이상만 사용 |
| 그림자·역광 | depth 오인식 | 실패 사례로 문서화 |
| 카메라 기울어짐 | Y축 왜곡 | 수평 촬영 가이드 준수 |

### 실패 사례 (docs/failure_cases.md 에 상세 기록)

- 그림자가 턱처럼 bbox 쳐지는 경우 → 거짓 양성
- 젖은 바닥 반사로 depth 오인식
- bbox 가 턱을 일부만 포함하는 경우 → 높이 과소 추정
- 역광 구간에서 depth map 평탄화

### RMSE 목표

| 항목 | 목표 RMSE |
|------|----------|
| 턱 높이 | ≤ 2 cm |
| 경사도 | ≤ 3°   |

> 현장 수집 데이터(`data/ground_truth.csv`)와 비교해 `compute_rmse()` 로 측정합니다.

---

*최종 수정: 이지현 | feat/integration*
