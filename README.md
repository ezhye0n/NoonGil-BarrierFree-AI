# NoonGil-BarrierFree-AI 👁️ : Barrier-Free Navigation Assistant for Wheelchair Users

> 휠체어 사용자의 보행 안전을 위한 AI 기반 장애물 탐지 및 경사도 추정 시스템

Team NoonGil: 이지현(팀장), 윤서연, 탁예린 | Sookmyung Women's University, Seoul, Republic of Korea

---

## 📌 Project Overview

본 프로젝트는 휠체어 사용자가 보행 중 마주치는 **장애물 탐지**와 **경사도 추정**을 통해
통과 가능 여부를 실시간으로 알려주는 AI 기반 보행 안전 보조 시스템입니다.

효창공원 일대 보행로에서의 실제 경험을 바탕으로, 볼라드·전동킥보드 등 장애물로 인한
휠체어 사용자의 이동권 문제를 해결하고자 시작되었습니다.
정면 촬영된 이미지 한 장을 입력받아 장애물 탐지, 경사도 추정, 회피 경로 제시, TTS 알림까지
하나의 파이프라인으로 처리합니다.

---

## 🌍 Social Background

대한민국의 장애인 이동권 문제는 오랫동안 해결되지 않은 사회적 과제입니다.

2023년 보건복지부 통계에 따르면 국내 등록 장애인 수는 약 264만 명이며, 이 중 지체장애인은 전체의 44.3%로 가장 높은 비율을 차지합니다. 

그러나 도심 보도 환경은 여전히 이들의 이동을 가로막는 구조적 장벽으로 가득합니다.
불법 주정차 오토바이, 전동킥보드·자전거 방치, 보도블록 파손, 가파른 경사로 등은 휠체어 사용자의 실질적인 이동을 위협하는 요소들입니다. 

특히 이러한 장애물들은 사전에 예측하거나 우회 경로를 계획하기 어려워, 단순한 불편을 넘어 낙상 사고 및 고립 으로 이어질 수 있습니다.

본 프로젝트는 효창공원 일대에서의 직접적인 현장 관찰을 계기로 시작되었습니다. 
보행 환경이 충분히 정비되어 있다고 여겨지는 도심 속에서도, 휠체어 사용자가 혼자서는 안전하게 이동하기 어렵다는 현실을 목격하며 기술적 해결책의 필요성을 절감하였습니다.

> 🇺🇳 UN 장애인권리협약(CRPD) 제9조는 장애인의 물리적 환경에 대한 접근성을 기본권으로 명시하고 있으며, 대한민국 또한 2009년 이를 비준하였습니다.

NoonGil-BarrierFree-AI는 이러한 사회적 공백을 AI 기술로 메우고자 합니다. 
실시간 장애물 탐지와 경사도 분석을 통해, 휠체어 사용자가 보다 안전하고 자율적으로 도시를 이동할 수 있는 환경을 만드는 것이 이 프로젝트의 궁극적인 목표입니다.

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

## ⚙️ Getting Started

  ### Prerequisites
  
  - Python 3.9+
  - CUDA 지원 GPU (권장)
  - YOLOv12 별도 설치 필요
  
  ### GPU 설정 (WSL2 기준)
  
  WSL2 환경에서 CUDA를 사용하려면 NVIDIA CUDA Toolkit 및 드라이버가 설치되어 있어야 합니다.
  자세한 설정 방법은 [NVIDIA WSL2 가이드](https://docs.nvidia.com/cuda/wsl-user-guide/)를 참고하세요.

  ---

  ### How to Run: 복사해서 바로 실행해볼 수 있는 명령어.
  
  ```bash
  # 1. 저장소 클론
  git clone https://github.com/ezhye0n/NoonGil-BarrierFree-AI.git
  cd NoonGil-BarrierFree-AI
  
  # 2. YOLOv12 설치
  git clone https://github.com/sunsmarterjie/yolov12.git
  cd yolov12
  pip install -e .
  cd ..
  
  # 3. 의존성 설치
  pip install -r requirements.txt
  
  # 4. 웹 인터페이스 실행 (Flask)
  python src/app.py
  ```

---

## 📦 Dataset

본 프로젝트는 **직접 수집 데이터**와 **공개 데이터셋**을 결합하여 학습 데이터를 구성하였습니다.

### 직접 수집 데이터: 총 892장 (train 780 / val 74 / test 38)
- 수집 장소: 숙대입구역 ~ 효창공원역 일대의 보행로
- 수집 방법: 휠체어 사용자 시점(정면)에서 직접 촬영
- 주요 클래스: 볼라드(bollard), 전동킥보드(scooter) 등
- 데이터 버전
  | 버전 | 설명 |
  |------|------|
  | v6_augmented-3 | bollard + scooter augmentation 적용, crack 데이터셋 제외 |

### Augmentation 설정 (Roboflow 기준)
- Flip (좌우 반전)
- Rotation (회전)
- Brightness / Contrast 조정

### 공개 데이터셋: 총 685장
- 출처: [Roboflow Universe](https://universe.roboflow.com/) 공개 데이터셋 활용
- 직접 수집 데이터와 병합 후 Augmentation 적용
  - step_new v1 (273장)
  - pothole v3 (412장)

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
| Epochs | 100 |
| Batch Size | 16 |
| Image Size | 640 |

### 학습 결과

| 버전 | mAP@50 |
|------|--------|
| YOLOv12 (v12_no_curb_finetune_v10_best.pt) | ~0.883 |

### 학습 가중치 위치

- `src/v12_no_curb_finetune_v10_best.pt`

### Results & Failure Analysis: 성공적인 시각화 결과와 솔직한 오작동 사례 분석 표.
수치적 결과 그래프와 '솔직한 실패 사례(오작동 사진 및 원인 분석)' 포함하기
