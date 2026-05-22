# 개발 환경 설정 가이드

## 권장 환경
- Python 3.10 이상
- CUDA 11.8 이상 (GPU 사용 시)
- OS: Windows 10/11, macOS, Ubuntu 20.04+

## 설치 방법

### 1. 레포지토리 클론
git clone https://github.com/ezhye0n/NoonGil-BarrierFree-AI.git
cd NoonGil-BarrierFree-AI

### 2. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

### 3. 패키지 설치
pip install -r requirements.txt

## CUDA 버전 확인
nvidia-smi

## 설치 확인
python -c "from ultralytics import YOLO; print('OK')"