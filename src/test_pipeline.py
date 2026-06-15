import os
from ultralytics import YOLO
from output import draw_output

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "noongil_v4_best.pt")

IMAGE_DIRS = [
    "../data/raw/images/",
    "../data/raw/id 0,1/",
    "../data/raw/id 2,3/",
    "../data/raw/id 4/",
    "../data/raw/id 5,6/",
    "../data/raw/id 7,8/",
]

RESULT_DIR = os.path.join(BASE_DIR, "../results/test_results/")
os.makedirs(RESULT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
last = ""

for image_dir in IMAGE_DIRS:
    full_dir = os.path.join(BASE_DIR, image_dir)
    if not os.path.exists(full_dir):
        print(f"폴더 없음: {full_dir}")
        continue

    image_files = [f for f in os.listdir(full_dir)
                   if f.endswith(".jpg") and "_result" not in f]

    print(f"\n📂 {image_dir} — {len(image_files)}장")

    for img_file in image_files:
        image_path = os.path.join(full_dir, img_file)
        print(f"\n--- {img_file} ---")
        last = draw_output(image_path, model, last, result_dir=RESULT_DIR, tts_enabled=False)