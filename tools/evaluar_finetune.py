"""
evaluar_finetune.py — Compara modelo original vs fine-tuned en los crops white
===============================================================================
Usa el ground_truth_white.csv como referencia y mide:
  - Accuracy exacta (string completo)
  - Accuracy numérica (valor float)
  - Errores específicos

Uso:
    python evaluar_finetune.py
"""

import csv
import cv2
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR
import re
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio

CROPS_DIR = (ROOT / "data/brio_results/crops")
CSV_PATH  = (ROOT / "data/brio_results/ground_truth_white.csv")
FINETUNED = str((ROOT / "models/ocr_white"))
SCALE     = 3

def extract_number(raw: str) -> str:
    raw = raw.replace(',', '.').strip()
    match = re.search(r'-?\d+\.\d+|-?\d+', raw)
    return match.group() if match else "—"

def raw_upscale(crop_bgr):
    h, w = crop_bgr.shape[:2]
    return cv2.resize(crop_bgr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_LANCZOS4)

def run_ocr(paddle, crop_bgr):
    prep = raw_upscale(crop_bgr)
    try:
        result = paddle.ocr(prep, cls=False)
        if result and result[0]:
            parts = [line[1][0] for line in result[0]]
            confs = [float(line[1][1]) * 100 for line in result[0]]
            raw_text = ' '.join(parts).strip()
            return raw_text, extract_number(raw_text), sum(confs)/len(confs)
    except Exception:
        pass
    return "", "—", 0.0

def main():
    # Cargar ground truth
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['gt_value'].strip() and r['gt_value'].strip() != '—']
    print(f"Ground truth: {len(rows)} crops con etiqueta\n")

    # Cargar modelos
    print("[1/2] Cargando modelo ORIGINAL...")
    ocr_orig = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)

    print("[2/2] Cargando modelo FINE-TUNED...")
    ocr_ft = PaddleOCR(
        use_angle_cls=False, lang='en', use_gpu=False, show_log=False,
        rec_model_dir=FINETUNED
    )
    print("[✓] Ambos modelos listos\n")

    # Evaluar
    orig_exact = 0
    orig_numeric = 0
    ft_exact = 0
    ft_numeric = 0
    errors_orig = []
    errors_ft = []
    total = 0

    print(f"{'Crop':55s}  {'GT':>8s}  {'Original':>10s}  {'FineTuned':>10s}")
    print("-" * 95)

    for r in rows:
        crop_path = CROPS_DIR / r['crop_file']
        if not crop_path.exists():
            continue

        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue

        gt = r['gt_value'].strip()
        total += 1

        # Modelo original
        _, orig_val, orig_conf = run_ocr(ocr_orig, crop)
        # Modelo fine-tuned
        _, ft_val, ft_conf = run_ocr(ocr_ft, crop)

        # Exact match
        orig_ok = (orig_val == gt)
        ft_ok = (ft_val == gt)
        if orig_ok: orig_exact += 1
        if ft_ok: ft_exact += 1

        # Numeric match (float comparison)
        try:
            if abs(float(orig_val) - float(gt)) < 0.01: orig_numeric += 1
        except: pass
        try:
            if abs(float(ft_val) - float(gt)) < 0.01: ft_numeric += 1
        except: pass

        # Marcar errores
        orig_mark = "✓" if orig_ok else "✗"
        ft_mark   = "✓" if ft_ok else "✗"

        if not orig_ok or not ft_ok:
            print(f"  {r['crop_file']:53s}  {gt:>8s}  {orig_mark} {orig_val:>8s}  {ft_mark} {ft_val:>8s}")

        if not orig_ok:
            errors_orig.append((r['crop_file'], gt, orig_val))
        if not ft_ok:
            errors_ft.append((r['crop_file'], gt, ft_val))

    # Resumen
    print("\n" + "=" * 70)
    print("  RESUMEN DE EVALUACIÓN")
    print("=" * 70)
    print(f"  Total crops evaluados  : {total}")
    print(f"")
    print(f"  {'':30s}  {'ORIGINAL':>12s}  {'FINE-TUNED':>12s}")
    print(f"  {'Accuracy exacta':30s}  {orig_exact/total*100:>11.1f}%  {ft_exact/total*100:>11.1f}%")
    print(f"  {'Accuracy numérica':30s}  {orig_numeric/total*100:>11.1f}%  {ft_numeric/total*100:>11.1f}%")
    print(f"  {'Errores':30s}  {len(errors_orig):>12d}  {len(errors_ft):>12d}")
    print(f"")

    if errors_ft:
        print(f"  Errores del FINE-TUNED ({len(errors_ft)}):")
        for crop, gt, pred in errors_ft[:15]:
            print(f"    {crop:50s}  GT={gt:>8s}  Pred={pred:>8s}")
    else:
        print(f"  🎯 FINE-TUNED: ¡CERO ERRORES!")

    print("=" * 70)

if __name__ == "__main__":
    main()
