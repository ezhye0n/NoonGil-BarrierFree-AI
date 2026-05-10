"""
extract_camera_params.py  —  스마트폰 사진에서 카메라 파라미터 추출
이지현 담당 | feat/integration 브랜치

사용법:
    python extract_camera_params.py --image 사진경로.jpg
    python extract_camera_params.py --folder data/raw/       # 폴더 전체 평균

출력:
    camera_params.json  →  analysis.py 의 DEFAULT_CAMERA 에 붙여넣기
"""

import json
import argparse
from pathlib import Path

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
except ImportError:
    raise ImportError("pip install Pillow 로 설치 후 재실행하세요.")


# ──────────────────────────────────────────────
# 1. EXIF에서 focal length 읽기
# ──────────────────────────────────────────────
def read_exif(image_path):
    """
    이미지 파일의 EXIF 데이터를 dict로 반환한다.

    Returns:
        dict: 태그명 → 값. EXIF 없으면 빈 dict.
    """
    try:
        img  = Image.open(image_path)
        raw  = img._getexif()
        if raw is None:
            return {}
        return {TAGS.get(k, k): v for k, v in raw.items()}
    except Exception:
        return {}


def get_focal_length_mm(exif):
    """
    EXIF에서 실제 초점거리(mm)를 읽는다.

    EXIF 태그 우선순위:
        1. FocalLengthIn35mmFilm  → 35mm 환산 focal length (가장 신뢰도 높음)
        2. FocalLength            → 실제 렌즈 focal length (IFDRational)

    Returns:
        float or None: 초점거리 (mm)
    """
    # 방법 1: 35mm 환산값 (정수 또는 IFDRational)
    fl_35 = exif.get("FocalLengthIn35mmFilm")
    if fl_35 and float(fl_35) > 0:
        return float(fl_35)

    # 방법 2: 실제 초점거리
    fl = exif.get("FocalLength")
    if fl:
        try:
            return float(fl)
        except Exception:
            pass

    return None


def get_sensor_size_mm(exif, image_width_px):
    """
    센서 폭(mm)을 추정한다.

    FocalPlaneXResolution / FocalPlaneResolutionUnit 을 활용하거나,
    없으면 35mm 환산 기준으로 역산한다.

    스마트폰 센서 폭 일반적 범위: 4.8mm (저가) ~ 9.6mm (플래그십)
    기본 폴백: 6.17mm (iPhone 표준값)

    Returns:
        float: 센서 폭 (mm)
    """
    FALLBACK_SENSOR_MM = 6.17  # iPhone/Android 중간값

    res   = exif.get("FocalPlaneXResolution")
    unit  = exif.get("FocalPlaneResolutionUnit")  # 2=inch, 3=cm

    if res and unit:
        try:
            res_val  = float(res)
            unit_val = int(unit)
            if res_val > 0:
                # pixels/inch 또는 pixels/cm → mm/pixel → 센서폭(mm)
                if unit_val == 2:    # inch
                    sensor_mm = image_width_px / res_val * 25.4
                elif unit_val == 3:  # cm
                    sensor_mm = image_width_px / res_val * 10.0
                else:
                    sensor_mm = FALLBACK_SENSOR_MM
                return sensor_mm
        except Exception:
            pass

    return FALLBACK_SENSOR_MM


# ──────────────────────────────────────────────
# 2. 픽셀 단위 초점거리(fx, fy) 계산
# ──────────────────────────────────────────────
def focal_px_from_mm(focal_mm, sensor_mm, image_width_px, image_height_px):
    """
    mm 단위 초점거리를 픽셀 단위로 변환한다.

    공식:
        fx = focal_mm / sensor_width_mm  * image_width_px
        fy = focal_mm / sensor_height_mm * image_height_px

    스마트폰은 픽셀이 정사각형이므로 보통 fx ≈ fy

    Returns:
        tuple: (fx, fy)
    """
    # 센서 높이는 폭의 3/4 비율로 근사 (4:3 센서 일반적)
    aspect = image_height_px / image_width_px
    sensor_height_mm = sensor_mm * aspect

    fx = (focal_mm / sensor_mm)        * image_width_px
    fy = (focal_mm / sensor_height_mm) * image_height_px

    return round(fx, 2), round(fy, 2)


# ──────────────────────────────────────────────
# 3. 단일 이미지에서 파라미터 추출
# ──────────────────────────────────────────────
def extract_from_image(image_path):
    """
    이미지 1장에서 카메라 파라미터를 추출한다.

    Returns:
        dict: {fx, fy, cx, cy, focal_mm, sensor_mm, width, height, source}
              추출 실패 시 기본값(DEFAULT_CAMERA)과 함께 반환
    """
    path = Path(image_path)
    img  = Image.open(path)
    W, H = img.size   # PIL은 (width, height) 순

    exif       = read_exif(path)
    focal_mm   = get_focal_length_mm(exif)
    sensor_mm  = get_sensor_size_mm(exif, W)

    if focal_mm and focal_mm > 0:
        fx, fy = focal_px_from_mm(focal_mm, sensor_mm, W, H)
        source = "EXIF"
    else:
        # EXIF 없을 때: fx ≈ max(W, H) 로 근사 (핀홀 카메라 기본 추정)
        fx = fy = float(max(W, H))
        focal_mm  = None
        sensor_mm = None
        source    = "fallback (EXIF 없음, fx=max(W,H) 근사값 사용)"

    return {
        "fx"       : fx,
        "fy"       : fy,
        "cx"       : round(W / 2, 2),
        "cy"       : round(H / 2, 2),
        "height_m" : 1.2,          # ← 촬영 시 카메라 높이(m)를 직접 측정해서 바꿀 것!
        "focal_mm" : focal_mm,
        "sensor_mm": round(sensor_mm, 4) if sensor_mm else None,
        "img_width": W,
        "img_height": H,
        "source"   : source,
        "file"     : path.name,
    }


# ──────────────────────────────────────────────
# 4. 폴더 전체 처리 → 평균 파라미터
# ──────────────────────────────────────────────
def extract_from_folder(folder_path, exts=(".jpg", ".jpeg", ".png")):
    """
    폴더 안의 이미지 전체를 처리하고 fx, fy 평균을 반환한다.
    같은 스마트폰으로 찍은 경우 여러 장 평균이 더 안정적이다.

    Returns:
        dict: 평균 카메라 파라미터
    """
    folder = Path(folder_path)
    images = [p for p in folder.iterdir() if p.suffix.lower() in exts]

    if not images:
        raise FileNotFoundError(f"{folder_path} 에서 이미지를 찾을 수 없습니다.")

    results = []
    for img_path in images:
        try:
            params = extract_from_image(img_path)
            results.append(params)
            print(f"  ✓ {params['file']:30s}  fx={params['fx']:8.1f}  source={params['source']}")
        except Exception as e:
            print(f"  ✗ {img_path.name}: {e}")

    if not results:
        raise RuntimeError("파라미터 추출에 성공한 이미지가 없습니다.")

    # EXIF 기반 결과만 평균에 포함 (fallback 제외)
    exif_results = [r for r in results if r["source"] == "EXIF"]
    base = exif_results if exif_results else results

    avg_fx = round(sum(r["fx"] for r in base) / len(base), 2)
    avg_fy = round(sum(r["fy"] for r in base) / len(base), 2)
    avg_cx = round(sum(r["cx"] for r in base) / len(base), 2)
    avg_cy = round(sum(r["cy"] for r in base) / len(base), 2)

    sample = base[0]
    return {
        "fx"        : avg_fx,
        "fy"        : avg_fy,
        "cx"        : avg_cx,
        "cy"        : avg_cy,
        "height_m"  : 1.2,
        "source"    : f"EXIF 평균 ({len(base)}장)",
        "img_width" : sample["img_width"],
        "img_height": sample["img_height"],
    }


# ──────────────────────────────────────────────
# 5. 결과 출력 및 저장
# ──────────────────────────────────────────────
def print_result(params):
    print("\n" + "=" * 50)
    print("  카메라 파라미터 추출 결과")
    print("=" * 50)
    print(f"  fx       = {params['fx']}")
    print(f"  fy       = {params['fy']}")
    print(f"  cx       = {params['cx']}")
    print(f"  cy       = {params['cy']}")
    print(f"  height_m = {params['height_m']}  ← 실제 촬영 높이로 수정 필요!")
    print(f"  출처     = {params['source']}")
    print()
    print("  ▼ analysis.py의 DEFAULT_CAMERA 에 이 값을 붙여넣으세요:")
    print()
    print("  DEFAULT_CAMERA = {")
    print(f"      'fx'      : {params['fx']},")
    print(f"      'fy'      : {params['fy']},")
    print(f"      'cx'      : {params['cx']},")
    print(f"      'cy'      : {params['cy']},")
    print(f"      'height_m': {params['height_m']},")
    print("  }")


def save_json(params, out_path="camera_params.json"):
    save = {k: v for k, v in params.items()
            if k in ("fx", "fy", "cx", "cy", "height_m", "source")}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 저장 완료: {out_path}")


# ──────────────────────────────────────────────
# 6. CLI 진입점
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="스마트폰 사진 EXIF → 카메라 파라미터 추출")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="단일 이미지 파일 경로 (예: data/raw/IMG_0001.jpg)")
    group.add_argument("--folder", help="이미지 폴더 경로 (예: data/raw/)")
    parser.add_argument("--out", default="camera_params.json", help="저장할 JSON 파일명")
    args = parser.parse_args()

    if args.image:
        params = extract_from_image(args.image)
    else:
        print(f"\n폴더 스캔 중: {args.folder}")
        params = extract_from_folder(args.folder)

    print_result(params)
    save_json(params, args.out)
