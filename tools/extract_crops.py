"""
extract_crops.py — Extrae 50 crops green + 50 white para ground truth manual
=============================================================================
USO:
    python extract_crops.py

OUTPUT:
    crops/green/green_000.jpg ... green_049.jpg
    crops/white/white_000.jpg ... white_049.jpg
    ground_truth.csv  (tú rellenas la columna 'gt_value')
"""

import cv2
import csv
import random
import os
import sys
from pathlib import Path
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio

BASE_DIR   = ROOT
BEST_PT    = BASE_DIR / "models/yolo/roi_detector_v1.pt"
IMAGES_DIR = BASE_DIR / "dataset/images"
OUT_DIR    = BASE_DIR / "crops"
GT_CSV     = BASE_DIR / "ground_truth.csv"

N_PER_CLASS = 50
CONF        = 0.3
SEED        = 42

def main():
    if not BEST_PT.exists():
        print(f"[ERROR] Modelo no encontrado: {BEST_PT}"); sys.exit(1)

    # Crear carpetas de salida
    (OUT_DIR / "green").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "white").mkdir(parents=True, exist_ok=True)

    print("  Cargando YOLO...")
    model = YOLO(str(BEST_PT))

    # Recolectar todas las imágenes del dataset
    exts   = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = []
    for split in ("train", "val", ""):
        d = IMAGES_DIR / split if split else IMAGES_DIR
        if d.exists():
            images += [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts]

    random.seed(SEED)
    random.shuffle(images)

    crops_green = []
    crops_white = []

    print(f"  Buscando crops en {len(images)} imágenes...")
    for img_path in images:
        if len(crops_green) >= N_PER_CLASS and len(crops_white) >= N_PER_CLASS:
            break

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        results = model(img_path, conf=CONF, verbose=False)[0]
        for box in results.boxes:
            cls  = int(box.cls)
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if cls == 1 and len(crops_green) < N_PER_CLASS:
                crops_green.append((crop, img_path.name))
            elif cls == 0 and len(crops_white) < N_PER_CLASS:
                crops_white.append((crop, img_path.name))

    print(f"  Encontrados: {len(crops_green)} green  |  {len(crops_white)} white")

    # Guardar crops y construir CSV
    rows = []
    for i, (crop, src) in enumerate(crops_green):
        fname = f"green_{i:03d}.jpg"
        cv2.imwrite(str(OUT_DIR / "green" / fname), crop)
        rows.append({"idx": i, "class": "green", "file": f"crops/green/{fname}",
                     "source_img": src, "gt_value": ""})

    for i, (crop, src) in enumerate(crops_white):
        fname = f"white_{i:03d}.jpg"
        cv2.imwrite(str(OUT_DIR / "white" / fname), crop)
        rows.append({"idx": i, "class": "white", "file": f"crops/white/{fname}",
                     "source_img": src, "gt_value": ""})

    # Escribir CSV
    with open(GT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx","class","file","source_img","gt_value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  [OK] Crops guardados en: {OUT_DIR}")
    print(f"  [OK] CSV creado en:      {GT_CSV}")
    print(f"\n  SIGUIENTE PASO:")
    print(f"    1. Abre el CSV con LibreOffice/Excel o un editor")
    print(f"    2. Rellena la columna 'gt_value' con el número real de cada imagen")
    print(f"    3. Guarda el CSV y avisa para correr el OCR + NSGA")

if __name__ == "__main__":
    main()
