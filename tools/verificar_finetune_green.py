"""
verificar_finetune_green.py — Evalúa el modelo fine-tuned verde imagen por imagen
==================================================================================
Corre el modelo fine-tuned en val_list.txt y train_list.txt y muestra cada
predicción vs ground truth, con resumen de accuracy final.

Uso:
    cd <repo>
    python verificar_finetune_green.py
"""

import cv2
import csv
from pathlib import Path
from paddleocr import PaddleOCR
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio

FINETUNED_DIR = str((ROOT / "models/ocr_green"))
BASE_DIR = (ROOT / "data/finetune_green")


def evaluate_list(paddle_model, list_file, name, show_all=False):
    print(f"\n{'='*70}")
    print(f"  EVALUANDO: {name}")
    print(f"{'='*70}")

    with open(list_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    total, correct = 0, 0
    errors = []

    print(f"{'Archivo':52s} | {'GT':^8s} | {'Pred':^8s} | {'Conf':^6s} | Estado")
    print("-" * 95)

    for line in lines:
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        img_rel, gt = parts[0], parts[1].strip()
        img_path = BASE_DIR / img_rel

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        res = paddle_model.ocr(img, det=False, cls=False)

        pred_text = ""
        conf = 0.0
        if res and res[0] and len(res[0]) > 0:
            pred_text = res[0][0][0].strip()
            conf = res[0][0][1]

        is_correct = (pred_text == gt)
        total += 1
        if is_correct:
            correct += 1

        if not is_correct or show_all:
            status = "✓ OK" if is_correct else "✗ ERROR"
            fname = img_rel.replace("images/", "")
            print(f"  {fname:50s} | {gt:^8s} | {pred_text:^8s} | {conf*100:>5.1f}% | {status}")

        if not is_correct:
            errors.append((img_rel, gt, pred_text, conf))

    acc = correct / max(1, total) * 100
    print(f"\n  Resumen {name}:")
    print(f"  Total: {total} | Correctos: {correct} | Errores: {len(errors)} | Accuracy: {acc:.2f}%")
    return acc, errors


def main():
    print("Cargando modelo fine-tuned VERDE...")
    ocr_ft = PaddleOCR(
        use_angle_cls=False,
        lang='en',
        use_gpu=False,
        show_log=False,
        rec_model_dir=FINETUNED_DIR
    )
    print("[✓] Modelo cargado\n")

    val_acc,   val_errors   = evaluate_list(ocr_ft, BASE_DIR / "val_list.txt",   "VAL SET  (no visto en train)", show_all=True)
    train_acc, train_errors = evaluate_list(ocr_ft, BASE_DIR / "train_list.txt", "TRAIN SET (solo errores)",     show_all=False)

    print(f"\n{'='*70}")
    print(f"  RESUMEN GLOBAL")
    print(f"{'='*70}")
    print(f"  Val accuracy   : {val_acc:.2f}%  ({len(val_errors)} errores)")
    print(f"  Train accuracy : {train_acc:.2f}%  ({len(train_errors)} errores)")
    print(f"{'='*70}")

    if val_errors:
        print(f"\n  Errores en VAL ({len(val_errors)}):")
        for img_rel, gt, pred, conf in val_errors:
            fname = img_rel.replace("images/", "")
            print(f"    {fname:50s}  GT={gt:>8s}  Pred={pred:>8s}  ({conf*100:.1f}%)")


if __name__ == "__main__":
    main()
