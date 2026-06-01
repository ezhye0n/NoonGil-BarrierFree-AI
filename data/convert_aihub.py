"""
NoonGil 프로젝트 - AI Hub 인도보행영상 → YOLO 포맷 변환기
bbox XML + polygon XML 동시 처리

폴더 구조 가정:
    aihub_data/
    ├── bbox/
    │   ├── MP_SEL_B027xxx.jpg
    │   └── bbox_sample.xml
    ├── polygon/
    │   ├── MP_SEL_PN000xxx.jpg
    │   └── polygon_sample.xml
    └── depth/  (스킵)

Usage:
    python convert_aihub.py --root ./aihub_data --output ./noongil_yolo
"""

import xml.etree.ElementTree as ET
import os
import shutil
import random
import argparse
from pathlib import Path


# NOONGIL_CLASSES 전체 교체
NOONGIL_CLASSES = [
    "bench",            # 0
    "bicycle",          # 1
    "bollard",          # 2
    "clothing_bin",     # 3: 의류수거함 (직접 수집 필요)
    "cone",             # 4: 공사용 콘
    "electric_scooter", # 5: 전동킥보드
    "fire_hydrant",     # 6: 소화전
    "motorcycle",       # 7: 오토바이
    "pavement_damage",  # 8: 노면 파손/포트홀 (직접 수집 필요)
    "ramp",             # 9: 경사로 (직접 수집 필요)
    "step",             # 10: 턱 (직접 수집 필요)
    "street_light",     # 11: 가로수/가로등
    "trash",            # 12: 쓰레기통 (직접 수집 필요)
    "tree",             # 13: 나무
]

# AIHUB_LABEL_MAP 전체 교체
AIHUB_LABEL_MAP = {
    # 전동킥보드
    "scooter":                  5,

    # 자전거
    "bicycle":                  1,

    # 오토바이
    "motorcycle":               7,

    # 소화전
    "fire_hydrant":             6,

    # 볼라드
    "bollard":                  2,
    "barricade":                2,

    # 공사용 콘
    "cone":                     4,

    # 가로수/가로등
    "pole":                     11,
    "traffic_light_controller": 11,
    "power_controller":         11,

    # 벤치
    "bench":                    0,

    # 나무
    "tree_trunk":               13,
}


# ── 파서 ─────────────────────────────────────────────────────

def parse_bbox_xml(xml_path: str) -> dict:
    """
    CVAT bbox XML 파싱
    반환: {img_name: (width, height, [(cls_id, cx, cy, w, h), ...])}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = {}

    for image in root.findall("image"):
        img_name = image.attrib["name"]
        W = int(image.attrib["width"])
        H = int(image.attrib["height"])
        boxes = []

        for box in image.findall("box"):
            label = box.attrib["label"]
            if label not in AIHUB_LABEL_MAP:
                continue  # 매핑 없는 라벨 스킵

            cls_id = AIHUB_LABEL_MAP[label]
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])

            cx = max(0, min(1, (xtl + xbr) / 2 / W))
            cy = max(0, min(1, (ytl + ybr) / 2 / H))
            w  = max(0, min(1, (xbr - xtl) / W))
            h  = max(0, min(1, (ybr - ytl) / H))

            if w > 0 and h > 0:
                boxes.append((cls_id, cx, cy, w, h))

        if boxes:
            result[img_name] = (W, H, boxes)

    return result


def parse_polygon_xml(xml_path: str) -> dict:
    """
    CVAT polygon XML 파싱 → bbox로 변환 (외접 사각형)
    반환: {img_name: (width, height, [(cls_id, cx, cy, w, h), ...])}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result = {}

    for image in root.findall("image"):
        img_name = image.attrib["name"]
        W = int(image.attrib["width"])
        H = int(image.attrib["height"])
        boxes = []

        for polygon in image.findall("polygon"):
            label = polygon.attrib["label"]
            if label not in AIHUB_LABEL_MAP:
                continue  # 매핑 없는 라벨 스킵

            cls_id = AIHUB_LABEL_MAP[label]

            # "x1,y1;x2,y2;..." 형식 파싱
            points_str = polygon.attrib["points"]
            try:
                pts = [
                    (float(p.split(",")[0]), float(p.split(",")[1]))
                    for p in points_str.strip().split(";")
                    if "," in p
                ]
            except (ValueError, IndexError):
                continue

            if len(pts) < 3:
                continue

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]

            # 외접 사각형으로 변환
            xtl, ytl = min(xs), min(ys)
            xbr, ybr = max(xs), max(ys)

            cx = max(0, min(1, (xtl + xbr) / 2 / W))
            cy = max(0, min(1, (ytl + ybr) / 2 / H))
            w  = max(0, min(1, (xbr - xtl) / W))
            h  = max(0, min(1, (ybr - ytl) / H))

            if w > 0 and h > 0:
                boxes.append((cls_id, cx, cy, w, h))

        if boxes:
            result[img_name] = (W, H, boxes)

    return result


# ── 변환 메인 ────────────────────────────────────────────────

def find_image(img_name: str, search_dirs: list) -> Path | None:
    """여러 디렉터리에서 이미지 탐색"""
    for d in search_dirs:
        candidate = d / img_name
        if candidate.exists():
            return candidate
        found = list(d.rglob(img_name))
        if found:
            return found[0]
    return None


def convert_aihub_dataset(
    root_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    root = Path(root_dir)
    output = Path(output_dir)
    random.seed(seed)

    # 출력 폴더 생성
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── 1. bbox XML 수집 ──
    print("📂 bbox XML 파싱 중...")
    all_annotations = {}

    bbox_dir = root / "bbox"
    if bbox_dir.exists():
        for xml_file in bbox_dir.glob("*.xml"):
            ann = parse_bbox_xml(str(xml_file))
            all_annotations.update(ann)
            print(f"   {xml_file.name}: {len(ann)}개 이미지")
    else:
        print("   ⚠️  bbox/ 폴더 없음")

    # ── 2. polygon XML 수집 ──
    print("📂 polygon XML 파싱 중...")

    polygon_dir = root / "polygon"
    if polygon_dir.exists():
        for xml_file in polygon_dir.glob("*.xml"):
            ann = parse_polygon_xml(str(xml_file))
            for k, v in ann.items():
                if k not in all_annotations:
                    all_annotations[k] = v
            print(f"   {xml_file.name}: {len(ann)}개 이미지")
    else:
        print("   ⚠️  polygon/ 폴더 없음")

    total = len(all_annotations)
    print(f"\n✅ 총 어노테이션: {total}개 이미지")

    if total == 0:
        print("❌ 어노테이션을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # ── 클래스별 통계 출력 ──
    class_counts = {i: 0 for i in range(len(NOONGIL_CLASSES))}
    for _, (_, _, boxes) in all_annotations.items():
        for cls_id, *_ in boxes:
            class_counts[cls_id] += 1

    print("\n📊 클래스별 어노테이션 수:")
    for cls_id, count in class_counts.items():
        status = "✅" if count > 0 else "⚠️  (직접 수집 필요)"
        print(f"   {cls_id:2d} {NOONGIL_CLASSES[cls_id]:<20} {count:>5}개  {status}")

    # ── 3. Train/Val 분리 ──
    img_names = list(all_annotations.keys())
    random.shuffle(img_names)
    n_train = int(total * train_ratio)
    splits = {
        "train": img_names[:n_train],
        "val":   img_names[n_train:],
    }

    search_dirs = [bbox_dir, polygon_dir, root]
    stats = {"train": 0, "val": 0, "missing": 0}

    for split, names in splits.items():
        print(f"\n{split} 처리 중 ({len(names)}개)...")
        for img_name in names:
            src = find_image(img_name, search_dirs)
            if src is None:
                stats["missing"] += 1
                continue

            dst_img = output / "images" / split / img_name
            shutil.copy2(str(src), str(dst_img))

            stem = Path(img_name).stem
            label_path = output / "labels" / split / f"{stem}.txt"
            _, _, boxes = all_annotations[img_name]

            with open(label_path, "w") as f:
                for cls_id, cx, cy, w, h in boxes:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            stats[split] += 1

    # ── 4. data.yaml 생성 ──
    yaml_lines = [
        f"# NoonGil - AI Hub 인도보행영상 (클래스 v2)",
        f"path: {output.resolve()}",
        f"train: images/train",
        f"val: images/val",
        f"",
        f"nc: {len(NOONGIL_CLASSES)}",
        f"names:",
    ]
    for cls in NOONGIL_CLASSES:
        yaml_lines.append(f"  - {cls}")

    yaml_path = output / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))

    # ── 5. 결과 출력 ──
    print(f"""
╔══════════════════════════════════════╗
║          변환 완료                   ║
╠══════════════════════════════════════╣
║  Train  : {stats['train']:>5}개                   ║
║  Val    : {stats['val']:>5}개                   ║
║  Missing: {stats['missing']:>5}개 (이미지 없음)       ║
╠══════════════════════════════════════╣
║  저장 위치: {str(output):<25}║
╚══════════════════════════════════════╝

⚠️  직접 수집이 필요한 클래스 (AI Hub 데이터 없음):
   - trash_can (7): 쓰레기통/의류 수거함
   - pothole (9): 노면 파손/포트홀
   - uneven_block (10): 보도블록 들뜸/단차
   - curb (11): 턱
   - ramp (12): 경사로

다음 명령어로 학습 시작:
  python train_noongil.py
""")

    return str(yaml_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, required=True,
        help="AI Hub 데이터 루트 (bbox/, polygon/, depth/ 포함)"
    )
    parser.add_argument(
        "--output", type=str, default="./noongil_yolo",
        help="YOLO 포맷 출력 경로"
    )
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convert_aihub_dataset(
        root_dir=args.root,
        output_dir=args.output,
        train_ratio=args.split,
        seed=args.seed,
    )
