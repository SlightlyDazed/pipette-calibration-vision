"""
regen_and_eval.py — Regenera crops con expand_box + evalúa original vs fine-tuned
==================================================================================
1. Corre YOLO sobre el dataset
2. Aplica expand_box para que las cajas white tengan mínimo 200px
3. Guarda los nuevos crops
4. Corre PaddleOCR original y fine-tuned sobre cada crop
5. Compara con ground truth

Uso:
    python regen_and_eval.py
"""

import cv2
import csv
import numpy as np
import re
from pathlib import Path
from ultralytics import YOLO
from paddleocr import PaddleOCR
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio

# ── RUTAS ─────────────────────────────────────────────────────────────────────
DATASET_DIR = (ROOT / "data/photos")
MODEL_PT    = (ROOT / "models/yolo/roi_detector_v1.pt")
CROPS_DIR   = (ROOT / "data/brio_results/crops")
CSV_PATH    = (ROOT / "data/brio_results/ground_truth_white.csv")
FINETUNED   = str((ROOT / "models/ocr_white"))

YOLO_CONF   = 0.4
SCALE       = 3
WHITE_MIN_W = 200
WHITE_MIN_H = 88


def expand_box(x1, y1, x2, y2, img_h, img_w, cls_name):
    if cls_name != "roi_white":
        return x1, y1, x2, y2
    bw, bh = x2 - x1, y2 - y1
    if bw < WHITE_MIN_W:
        delta = (WHITE_MIN_W - bw) // 2
        x1 = max(0, x1 - delta)
        x2 = min(img_w, x2 + delta)
    if bh < WHITE_MIN_H:
        delta = (WHITE_MIN_H - bh) // 2
        y1 = max(0, y1 - delta)
        y2 = min(img_h, y2 + delta)
    return x1, y1, x2, y2


def extract_number(raw: str) -> str:
    raw = raw.replace(',', '.').strip()
    match = re.search(r'-?\d+\.\d+|-?\d+', raw)
    return match.group() if match else "—"


def run_ocr(paddle, crop_bgr):
    h, w = crop_bgr.shape[:2]
    prep = cv2.resize(crop_bgr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_LANCZOS4)
    try:
        result = paddle.ocr(prep, cls=False)
        if result and result[0]:
            parts = [line[1][0] for line in result[0]]
            confs = [float(line[1][1]) * 100 for line in result[0]]
            return ' '.join(parts).strip(), extract_number(' '.join(parts)), sum(confs)/len(confs)
    except:
        pass
    return "", "—", 0.0


def main():
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar ground truth como lookup: crop_file → gt_value
    gt_lookup = {}
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            gt_lookup[r['crop_file']] = r['gt_value'].strip()

    # Cargar modelos
    print("Cargando YOLO...")
    yolo = YOLO(str(MODEL_PT))
    print("Cargando PaddleOCR ORIGINAL...")
    ocr_orig = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)
    print("Cargando PaddleOCR FINE-TUNED...")
    ocr_ft = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False,
                       rec_model_dir=FINETUNED)
    print("[✓] Todos listos\n")

    images = sorted([p for p in DATASET_DIR.iterdir()
                     if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])

    # Regenerar crops y evaluar
    orig_exact, ft_exact, total = 0, 0, 0
    errors_ft = []

    print(f"{'Crop':55s}  {'GT':>8s}  {'Original':>10s}  {'FineTuned':>10s}")
    print("-" * 95)

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        results = yolo.predict(source=img, conf=YOLO_CONF, verbose=False)
        boxes = results[0].boxes
        if not boxes or len(boxes) == 0:
            continue

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            cls_name = yolo.model.names[cls_id]

            if cls_name != "roi_white":
                continue

            # APLICAR EXPAND_BOX
            x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, img_h, img_w, cls_name)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_fname = f"{img_path.stem}_roi{idx:02d}_{cls_name}.jpg"
            cv2.imwrite(str(CROPS_DIR / crop_fname), crop)

            # Buscar GT
            gt = gt_lookup.get(crop_fname)
            if not gt or gt == '—':
                continue

            total += 1
            _, orig_val, _ = run_ocr(ocr_orig, crop)
            _, ft_val, _   = run_ocr(ocr_ft, crop)

            orig_ok = (orig_val == gt)
            ft_ok   = (ft_val == gt)
            if orig_ok: orig_exact += 1
            if ft_ok:   ft_exact += 1

            if not orig_ok or not ft_ok:
                om = "✓" if orig_ok else "✗"
                fm = "✓" if ft_ok else "✗"
                print(f"  {crop_fname:53s}  {gt:>8s}  {om} {orig_val:>8s}  {fm} {ft_val:>8s}")

            if not ft_ok:
                errors_ft.append((crop_fname, gt, ft_val))

    # Resumen
    print("\n" + "=" * 70)
    print("  RESUMEN (con expand_box aplicado)")
    print("=" * 70)
    print(f"  Total crops evaluados  : {total}")
    print(f"")
    print(f"  {'':30s}  {'ORIGINAL':>12s}  {'FINE-TUNED':>12s}")
    print(f"  {'Accuracy exacta':30s}  {orig_exact/max(1,total)*100:>11.1f}%  {ft_exact/max(1,total)*100:>11.1f}%")
    print(f"  {'Errores':30s}  {total-orig_exact:>12d}  {len(errors_ft):>12d}")
    print()

    if errors_ft:
        print(f"  Errores FINE-TUNED restantes ({len(errors_ft)}):")
        for crop, gt, pred in errors_ft[:20]:
            print(f"    {crop:50s}  GT={gt:>8s}  Pred={pred:>8s}")
    else:
        print("  🎯 FINE-TUNED: ¡CERO ERRORES!")
    print("=" * 70)


if __name__ == "__main__":
    main()
