# NoonGil-BarrierFree-AI 👁️ : Barrier-Free Navigation Assistant for Wheelchair Users

> An AI-based obstacle detection and slope estimation system for safe wheelchair navigation
 
> **Team NoonGil** | Lee Jihyun (Lead) · Yun Seoyeon · Tak Yerin | Sookmyung Women's University, Seoul, Republic of Korea

---

## 📌 Project Overview

This project is an AI-powered pedestrian safety assistance system that provides real-time passability judgments for wheelchair users by detecting **obstacles** and **estimating slope grades** encountered during navigation.
 
Motivated by first-hand observations of accessibility barriers — such as bollards and illegally parked electric scooters — on sidewalks around Hyochang Park, we set out to build a technical solution for the mobility challenges faced by wheelchair users.
 
The system accepts a single front-facing image as input and processes it through a unified pipeline covering obstacle detection, slope estimation, avoidance path suggestion, and TTS alerts.

---

## 🌍 Social Background

The mobility rights of people with disabilities in South Korea remain a longstanding and unresolved social challenge.
 
According to 2023 statistics from the Ministry of Health and Welfare, approximately 2.64 million people are registered as having disabilities in Korea, of whom those with physical disabilities account for the highest proportion at 44.3%.
 
Yet urban sidewalk environments remain filled with structural barriers that obstruct their movement.
Illegally parked motorcycles, abandoned electric scooters and bicycles, damaged pavement, and steep ramps are among the factors that pose real threats to the safe mobility of wheelchair users.
 
These obstacles are particularly difficult to anticipate or route around in advance, meaning they represent more than mere inconvenience — they can lead to fall accidents and isolation.
 
This project was sparked by direct field observation in the Hyochang Park area.
Even in parts of the city considered well-maintained, we witnessed firsthand how difficult it is for wheelchair users to navigate safely on their own, and felt the urgent need for a technological solution.
 
> 🇺🇳 Article 9 of the UN Convention on the Rights of Persons with Disabilities (CRPD) enshrines access to the physical environment as a fundamental right. The Republic of Korea ratified this convention in 2009.
 
NoonGil-BarrierFree-AI aims to bridge this social gap through AI technology.
By enabling real-time obstacle detection and slope analysis, our ultimate goal is to create an environment where wheelchair users can navigate cities more safely and autonomously.

---

## 🔄 Program Pipeline

```
Data Input         : Single front-facing image
      ↓
Object Detection   : Obstacle bounding box extraction via YOLOv12
      ↓
Depth Estimation   : Depth map generation via Depth Anything
      ↓
Slope Estimation   : Slope grade estimation from depth map
      ↓
Integration        : Unified computation of coordinates + distance + slope
      ↓
Output             : Passability alert (UI / TTS) + Avoidance path suggestion
```

---

## 🏗️ Model Architecture

| Module | Model | Role |
|--------|-------|------|
| Object Detection | YOLOv12 | Obstacle detection and bounding box extraction |
| Depth Estimation | Depth Anything | Depth map generation |
| Slope Estimation | Depth map-based computation | Slope grade estimation |

---

### ⚙️ Prerequisites

| Item | Details |
|------|---------|
| **Language** | Python 3.9+ |
| **Web Framework** | Flask |
| **Object Detection** | YOLOv12 |
| **Depth Estimation** | Depth Anything |
| **Training Environment** | Google Colab (GPU) |
| **Runtime Environment** | WSL2 + CUDA (local) or GPU server recommended |
| **Data Management** | Roboflow (labeling, augmentation, version control) |
| **Version Control** | Git / GitHub |
 
> **GPU Setup (WSL2):** NVIDIA CUDA Toolkit and driver installation required.
> See the [NVIDIA WSL2 Guide](https://docs.nvidia.com/cuda/wsl-user-guide/) for details.

---

## 🚀 Getting Started

### Dependencies and Installation

* [Anaconda3](https://www.anaconda.com/download)
* Python == 3.9
* NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads) (recommended)
  * For WSL2 environments, refer to the [NVIDIA WSL2 Guide](https://docs.nvidia.com/cuda/wsl-user-guide/)

**1. Create conda environment**

```bash
conda create --name noongil python=3.9
conda activate noongil
```

**2. Clone the repository**

```bash
git clone https://github.com/ezhye0n/NoonGil-BarrierFree-AI.git
cd NoonGil-BarrierFree-AI
```

**3. Install [YOLOv12](https://github.com/sunsmarterjie/yolov12)**

```bash
git clone https://github.com/sunsmarterjie/yolov12.git
cd yolov12
pip install -e .
cd ..
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

---

### Pretrained Model

Download the pretrained weights from [**GitHub Releases**](https://github.com/ezhye0n/NoonGil-BarrierFree-AI/releases/tag/v12_no_curb_finetune) and place the file at `src/v12_no_curb_finetune_v10_best.pt`.

```
src/
└── v12_no_curb_finetune_v10_best.pt   ← place here
```

---

### Run

```bash
python src/app.py
```

Open your browser and go to `http://localhost:5000`.

---

### Testing

**Option 1 — Web UI (Recommended)**

1. Go to `http://localhost:5000`
2. Upload a front-facing sidewalk image (`.jpg` / `.png`)
3. Click **분석하기**
4. View detected obstacles with bounding boxes (class name and confidence score shown at top-left of each box), slope grade, avoidance path, and TTS alert

**Option 2 — Direct API call**

```bash
curl -X POST http://localhost:5000/analyze \
     -F "image=@test_images/sample_bollard.jpg"
```

---

### Dataset Preparation *(for training only)*

> ⚠️ **You do not need this to run the demo.** Only required if you want to retrain from scratch.

We used our own collected dataset combined with public datasets from [Roboflow Universe](https://universe.roboflow.com/).
The final preprocessed dataset (v4, YOLOv12 format) can be downloaded via the Roboflow API:

```python
pip install roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("s-workspace-6sd3n").project("noongil-barrierfree-ai")
version = project.version(4)
dataset = version.download("yolov12")
```

> Get your API key at [roboflow.com](https://roboflow.com) → Settings → API Key.

---

## 📦 Dataset

We combined our **own collected dataset** with **public datasets** from [Roboflow Universe](https://universe.roboflow.com/).

### Collected Dataset — 892 images (train 780 / val 74 / test 38)

- **Location:** Sidewalks around Sookmyung Women's University (효창공원앞역 ~ 숙명여대 정문)
- **Viewpoint:** Frontal, from wheelchair user's eye-level (90~100cm above ground)
- **Split ratio:** Train : Val : Test = **7 : 2 : 1** (random shuffle)

| Session | Date | Condition | Details |
|---------|------|-----------|---------|
| 1st | 05.19 | Cloudy, 16:40~17:38 | Hyochang Park / Hyochang Park Stn. ~ Sookmyung rear gate — base data |
| 2nd | 05.23 | Sunny, 12:00~13:00 | 44 images — cones, bicycles, street lights, clothing bins |
| 3rd | 05.27 | Cloudy, 13:40~14:20 | 84 images — bollards, scooters, motorcycles, steps, pavement damage |
| 4th | 05.28 | Cloudy + Night, 12:00/19:00/20:30~ | 116 images — night low-light data included |

**Collection Standard:**
- Single photographer (Yun Seoyeon), same device, fixed 3:4 aspect ratio
- Auto HDR & AI enhancement: **OFF**
- Camera height: **90~100cm from ground** (wheelchair user eye-level)

### Public Datasets

| Dataset | Images |
|---------|--------|
| step_new v1 (Roboflow Universe) | 273 |
| pothole v3 (Roboflow Universe) | 412 |
| **Total** | **685** |

### Labeling Classes

| Class | Instances |
|-------|-----------|
| `electric_scooter` | 31 |
| `bicycle` | 57 |
| `motorcycle` | 31 |
| `fire_hydrant` | 12 |
| `cone` | 32 |
| `bollard` | 43 |
| `street_light` | 114 |
| `bench` | 16 |
| `trash` | 20 |
| `clothing_bin` | 16 |
| `tree` | 118 |
| `pavement_damage` | 16 |
| `step` | 50 |
| `ramp` | 28 |

> **Step ground truth (tape measure, cm):** 3, 4, 4, 4, 6, 7, 9, 10, 10, 11, 12, 12  
> **Ramp ground truth (level app, °):** 4°, 6°, 8°, 10°, 13°, 14°, 16°  
> *(Wheelchair passage threshold: step ≥ 3cm, slope ≥ 4°)*

### Data Augmentation (via Roboflow)

| Augmentation | Value | Purpose |
|--------------|-------|---------|
| Horizontal Flip | Applied | Reduce directional bias |
| Rotation | ±15° | Handle device tilt and varied road angles |
| Brightness | ±15% | Adapt to low-light and high-contrast conditions |
| Resize | Fit (Letterbox) 512×512 | Preserve aspect ratio for model input |

---

## 🤖 Model Training

- **Model:** YOLOv12 (Object Detection)
- **Environment:** Google Colab (GPU)

### Training Pipeline

| Stage | Description | Hyperparameters |
|-------|-------------|-----------------|
| Base training | Initial training on collected + public dataset | Epochs 100, Batch 16, Imgsz 640 |
| Fine-tuning | Added step_new v1 (273 images), merged and retrained | Epochs 50, Freeze 10 |

### Final Results

| Metric | Value |
|--------|-------|
| mAP@50 | **0.883** |
| mAP@50-95 | **0.544** |
| Precision | **0.872** |
| Recall | **0.840** |
| F1 Score | **0.84** (at confidence 0.501) |

### Model Zoo

Please download the pretrained weights from [**GitHub Releases**](https://github.com/ezhye0n/NoonGil-BarrierFree-AI/releases/tag/v12_no_curb_finetune) and place the file at `src/v12_no_curb_finetune_v10_best.pt`.

| Release | Model | mAP@50 | mAP@50-95 | Download |
|---------|-------|--------|-----------|----------|
| **v12_no_curb_finetune** *(Latest)* | YOLOv12s | **0.883** | **0.544** | [Download](https://github.com/ezhye0n/NoonGil-BarrierFree-AI/releases/tag/v12_no_curb_finetune) |
| v12_no_curb_ep100 | YOLOv12s | 0.855 | 0.525 | [Download](https://github.com/ezhye0n/NoonGil-BarrierFree-AI/releases/tag/v12-no-curb-ep100) |

---

## 📊 Results & Failure Analysis

### Training Metrics

| Metric | Value |
|--------|-------|
| mAP@50 | **0.883** |
| mAP@50-95 | **0.544** |
| Precision | **0.872** |
| Recall | **0.840** |
| F1 Score | **0.84** (at confidence 0.501) |

*Evaluated on 178 validation images across all classes.*

### Per-Class AP (mAP@50)

| Class | AP@50 | Note |
|-------|-------|------|
| `clothing_bin` | 0.995 | |
| `cone` | 0.995 | |
| `electric_scooter` | 0.995 | |
| `fire_hydrant` | 0.995 | |
| `trash` | 0.995 | |
| `street_light` | 0.978 | |
| `bollard` | 0.926 | |
| `bicycle` | 0.911 | |
| `ramp` | 0.897 | ↑ +0.280 after fine-tuning (0.617 → 0.897) |
| `tree` | 0.778 | |
| `pavement_damage` | 0.697 | |
| `step` | 0.675 | ↑ +0.143 after fine-tuning (0.532 → 0.675) |
| `motorcycle` | 0.645 | Finalized as-is (see Failure Analysis) |

### Slope Estimation Logic

경사도 추정은 3가지 방법 중 max 값을 채택하는 방식으로 개선되었습니다.

| Method | Grade: 상 (≥5°) | Grade: 중 (≥2°) |
|--------|----------------|----------------|
| Method 1: Pinhole camera model angle | angle ≥ 5.0° | angle ≥ 2.0° |
| Method 2: Depth ratio | ratio ≥ 1.4 | ratio ≥ 1.15 |
| Method 3: BBox vertical ratio | ratio ≥ 0.4 | ratio ≥ 0.25 |

### Training Curves

![Training Results](results/metrics/results.png)

All three training losses (box, cls, dfl) show consistent downward trends. Validation losses also decrease steadily, with mAP@50 stabilizing around **0.87~0.89** and mAP@50-95 around **0.54**.

### Precision-Recall Curve

![PR Curve](results/metrics/PR_curve.png)

Overall **mAP@0.5 = 0.883**. Most classes achieve high precision across the full recall range. `step` and `motorcycle` show relatively lower area under curve.

### F1-Confidence Curve

![F1 Curve](results/metrics/F1_curve.png)

Best F1 score of **0.84** is achieved at confidence threshold **0.501**, which is used as the default inference threshold.

### Precision-Confidence Curve

![P Curve](results/metrics/P_curve.png)

### Confusion Matrix

![Confusion Matrix](results/metrics/confusion_matrix.png)

### Failure Analysis
📹 Demo Video: https://drive.google.com/file/d/14RINbrw1OEfcKJ9n8pw_ofYHYh5kEN_n/view?usp=sharing

| Target | Issue | Root Cause | Decision |
|--------|-------|------------|----------|
| `motorcycle` | Low AP (0.645), no improvement planned | Only 31 training instances; most peripheral to core wheelchair-safety use case | **Finalized as-is** — not a primary obstacle for wheelchair users |
| `step` / `background` | Background misclassified as `step` (10 cases) | Ambiguous boundary regions in low-contrast ground textures | Refine bounding box annotation standards in future iterations |
| Slope grade (중 → 상) | Gentle slopes (2°~5°) occasionally over-estimated as 상 | 3-method max strategy can amplify outlier estimates on mild gradients | Remaining known limitation; under-estimation preferred over over-estimation for safety |

### Detection Success Cases
📹 Demo Video: https://drive.google.com/file/d/1PwsMYNm2k2lIrOudPEKCAPo1fN90e1Ik/view?usp=sharing

| Case | Description |
|------|-------------|
| ![success_1](results/output_images/success_1.png) | **Ramp + Tree Detection** — Ramp (90%) and tree (81%) simultaneously detected. Slope grade judged as **High (≥5°)**; left-side avoidance path suggested. |
| ![success_2](results/output_images/success_2.png) | **Ramp Detection** — Ramp (87%) detected. Slope grade judged as **High**; right-side avoidance path suggested. Assistance recommended / caution entry warning issued. |
| ![success_3](results/output_images/success_3.png) | **Multiple Step Detection** — Two steps detected (32%, 40%). Slope grade judged as **Low (< 2°)**; left-side avoidance path suggested. |
| ![success_4](results/output_images/success_4.png) | **Cone + Step Detection** — Cone (61%) and step (36%) detected. Slope grade judged as **Low**; left-side avoidance path suggested. |


---

## 📁 Project Structure

```
NoonGil-BarrierFree-AI/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── ISSUE
│   └── PULL_REQUEST_TEMPLATE
├── data/
├── docs/
├── frontend/
│   ├── index.html
│   └── result.json
├── model/
│   ├── notebooks/
│   └── data.yaml
├── results/
│   ├── metrics/                            # Training graphs (PR curve, F1 curve, confusion matrix, etc.)
│   ├── output_images/                      # Detection success case screenshots
│   ├── failure_cases/
│   └── test_results/
├── src/
│   ├── app.py                              # Flask web server (entry point)
│   ├── main.py                             # Pipeline entry point
│   ├── avoid.py                            # Avoidance path logic
│   ├── depth_inference.py                  # Depth Anything inference
│   ├── slope.py                            # Slope estimation
│   ├── output.py                           # Result output
│   ├── tts.py                              # TTS alert
│   ├── test_pipeline.py                    # Pipeline test script
│   ├── test_result.json                    # Test output sample
│   └── v12_no_curb_finetune_v10_best.pt    # Pretrained weights (download from Releases)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📜 License

```
GPL-3.0 License

Copyright (c) 2026 Team NoonGil
(Lee Jihyun, Yun Seoyeon, Tak Yerin — Sookmyung Women's University)

This project is licensed under the GNU General Public License v3.0.
See https://www.gnu.org/licenses/gpl-3.0.html for details.
```

> **Third-party Library Licenses:**
> - YOLOv12: [GPL-3.0](https://github.com/sunsmarterjie/yolov12/blob/main/LICENSE)
> - Depth Anything: [Apache 2.0](https://github.com/LiheYoung/Depth-Anything/blob/main/LICENSE)
> - Flask: [BSD-3-Clause](https://flask.palletsprojects.com/en/stable/license/)

> ⚠️ This project incorporates YOLOv12 (GPL-3.0). Redistribution of this project must comply with the GPL-3.0 terms.
