# NoonGil-BarrierFree-AI 👁️ : Barrier-Free Navigation Assistant for Wheelchair Users

> 휠체어 사용자의 보행 안전을 위한 AI 기반 장애물 탐지 및 경사도 추정 시스템

---

## 📌 Project Overview

본 프로젝트는 휠체어 사용자가 보행 중 마주치는 **장애물 탐지**와 **경사도 추정**을 통해
통과 가능 여부를 실시간으로 알려주는 AI 기반 보행 안전 보조 시스템입니다.

효창공원 일대 보행로에서의 실제 경험을 바탕으로, 볼라드·전동킥보드 등 장애물로 인한
휠체어 사용자의 이동권 문제를 해결하고자 시작되었습니다.
정면 촬영된 이미지 한 장을 입력받아 장애물 탐지, 경사도 추정, 회피 경로 제시, TTS 알림까지
하나의 파이프라인으로 처리합니다.

---

## 🔄 Program Pipeline

Data Input         : 정면 촬영 이미지 1장 입력

↓

Object Detection   : YOLOv12 기반 장애물 Bounding Box 좌표 추출

↓

Depth Estimation   : Depth Anything 기반 거리 데이터(Depth Map) 생성

↓

Slope Estimation   : Depth Map 기반 경사도 추정

↓

Integration        : 좌표 + 거리 + 경사도 값 통합 계산

↓

Output             : 통과 가능 여부 알림 (UI / TTS) + 회피 경로 제시


---

## ⚙️ Environment Setup

### Requirements

- Python >= 3.9
- CUDA (NVIDIA GPU 권장)
- WSL2 환경에서 테스트됨

```bash
pip install -r requirements.txt
```

### GPU 설정 (WSL2 기준)

WSL2 환경에서 CUDA를 사용하려면 NVIDIA CUDA Toolkit 및 드라이버가 설치되어 있어야 합니다.
자세한 설정 방법은 [NVIDIA WSL2 가이드](https://docs.nvidia.com/cuda/wsl-user-guide/)를 참고하세요.

---

## 📦 Dataset

본 프로젝트는 **직접 수집 데이터**와 **공개 데이터셋**을 결합하여 학습 데이터를 구성하였습니다.

### 직접 수집 데이터
- 수집 장소: 효창공원 일대 보행로
- 수집 방법: 휠체어 사용자 시점(정면)에서 직접 촬영
- 주요 클래스: 볼라드(bollard), 전동킥보드(scooter) 등

### 공개 데이터셋
- 출처: [Roboflow Universe](https://universe.roboflow.com/) 공개 데이터셋 활용
- 직접 수집 데이터와 병합 후 Augmentation 적용

### 데이터 버전

| 버전 | 설명 |
|------|------|
| v6_augmented-3 | bollard + scooter augmentation 적용, crack 데이터셋 제외 |

### Augmentation 설정 (Roboflow 기준)
- Flip (좌우 반전)
- Rotation (회전)
- Brightness / Contrast 조정

---

## 🏗️ Model Architecture

| 모듈 | 모델 | 역할 |
|------|------|------|
| Object Detection | YOLOv12 | 장애물 탐지 및 Bounding Box 추출 |
| Depth Estimation | Depth Anything | Depth Map 생성 |
| Slope Estimation | Depth Map 기반 연산 | 경사도 추정 |

---

## 🤖 Model Training

### 학습 모델
- **YOLOv12** (Object Detection)

### 학습 환경
- Google Colab (GPU 사용)

### 하이퍼파라미터

| 항목 | 값 |
|------|----|
| Epochs | _(직접 입력)_ |
| Batch Size | _(직접 입력)_ |
| Image Size | _(직접 입력)_ |
| Optimizer | _(직접 입력)_ |

### 학습 결과

| 버전 | mAP@50 |
|------|--------|
| YOLOv12 (v6_augmented-3) | ~0.979 |

### 학습 가중치 위치

- models/best.pt

### How to Run: 복사해서 바로 실행해볼 수 있는 명령어.

### Results & Failure Analysis: 성공적인 시각화 결과와 솔직한 오작동 사례 분석 표.
수치적 결과 그래프와 '솔직한 실패 사례(오작동 사진 및 원인 분석)' 포함하기git
