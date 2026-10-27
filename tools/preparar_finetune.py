"""
preparar_finetune.py — Prepara el dataset para fine-tuning de PaddleOCR rec
===========================================================================
Lee el ground_truth_white.csv y genera:
  1. Copia los crops a una carpeta limpia (finetune_data/)
  2. train_list.txt y val_list.txt en formato PaddleOCR:
     ruta_relativa_imagen\tetiqueta

  Split: 80% train / 20% val

Uso:
    python ocr_yolo/preparar_finetune.py
"""

import csv
import os
import shutil
import random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio

CSV_PATH    = (ROOT / "data/brio_results/ground_truth_white.csv")
CROPS_DIR   = (ROOT / "data/brio_results/crops")
OUT_DIR     = (ROOT / "data/finetune_white")
TRAIN_RATIO = 0.8
SEED        = 42

def main():
    # Leer CSV
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # Filtrar solo los que tienen gt_value válido (numérico)
    valid = []
    for r in rows:
        gt = r['gt_value'].strip()
        if gt and gt != '—':
            valid.append(r)

    print(f"Total en CSV: {len(rows)}")
    print(f"Con ground truth válido: {len(valid)}")

    # Crear carpeta de imágenes
    img_dir = OUT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Copiar crops y construir lista
    entries = []
    missing = 0
    for r in valid:
        src = CROPS_DIR / r['crop_file']
        if not src.exists():
            missing += 1
            continue
        dst = img_dir / r['crop_file']
        shutil.copy2(str(src), str(dst))
        # PaddleOCR format: ruta_relativa \t etiqueta
        entries.append((f"images/{r['crop_file']}", r['gt_value'].strip()))

    if missing:
        print(f"  [!] {missing} crops no encontrados (saltados)")

    # Shuffle y split
    random.seed(SEED)
    random.shuffle(entries)
    split = int(len(entries) * TRAIN_RATIO)
    train_entries = entries[:split]
    val_entries   = entries[split:]

    # Escribir listas
    train_file = OUT_DIR / "train_list.txt"
    val_file   = OUT_DIR / "val_list.txt"

    with open(train_file, 'w', encoding='utf-8') as f:
        for path, label in train_entries:
            f.write(f"{path}\t{label}\n")

    with open(val_file, 'w', encoding='utf-8') as f:
        for path, label in val_entries:
            f.write(f"{path}\t{label}\n")

    # Generar diccionario de caracteres
    all_chars = set()
    for _, label in entries:
        all_chars.update(label)
    chars = sorted(all_chars)

    dict_file = OUT_DIR / "dict.txt"
    with open(dict_file, 'w', encoding='utf-8') as f:
        for c in chars:
            f.write(f"{c}\n")

    print(f"\n{'='*50}")
    print(f"  DATASET LISTO PARA FINE-TUNING")
    print(f"{'='*50}")
    print(f"  Imágenes copiadas : {len(entries)}")
    print(f"  Train             : {len(train_entries)}")
    print(f"  Val               : {len(val_entries)}")
    print(f"  Diccionario       : {len(chars)} caracteres → {chars}")
    print(f"  Carpeta           : {OUT_DIR}")
    print(f"  train_list.txt    : {train_file}")
    print(f"  val_list.txt      : {val_file}")
    print(f"  dict.txt          : {dict_file}")
    print(f"{'='*50}")

    # Preview
    print("\n  Primeras 5 entradas del train:")
    for p, l in train_entries[:5]:
        print(f"    {p:50s}  →  '{l}'")


if __name__ == "__main__":
    main()
