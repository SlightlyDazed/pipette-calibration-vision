"""
preparar_finetune_green.py — Prepara el dataset para fine-tuning del modelo verde
==================================================================================
Lee ground_truth_green.csv, copia los crops a finetune_data_green/images/
y genera train_list.txt / val_list.txt en formato PaddleOCR.

Excluye entradas con gt_value == 'SKIP'.
Split: 80% train / 20% val (aleatorio con semilla fija).

Uso:
    cd <repo>
    python preparar_finetune_green.py
"""

import csv
import random
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio

# ── RUTAS ─────────────────────────────────────────────────────────────────────
CROPS_DIR   = (ROOT / "data/brio_results/crops")
CSV_PATH    = (ROOT / "data/brio_results/ground_truth_green.csv")
OUT_DIR     = (ROOT / "data/finetune_green")
IMAGES_DIR  = OUT_DIR / "images"

TRAIN_RATIO = 0.80
SEED        = 42

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)

    # Leer CSV
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # Filtrar válidos (excluir SKIP y vacíos)
    valid = []
    skipped = 0
    missing = 0
    for row in rows:
        gt = row['gt_value'].strip()
        if gt in ('SKIP', '', '—'):
            skipped += 1
            continue
        crop_path = CROPS_DIR / row['crop_file']
        if not crop_path.exists():
            print(f"  [!] Crop no encontrado: {row['crop_file']}")
            missing += 1
            continue
        valid.append((row['crop_file'], gt, crop_path))

    print(f"Entradas totales : {len(rows)}")
    print(f"SKIPs excluidos  : {skipped}")
    print(f"Faltantes        : {missing}")
    print(f"Válidos          : {len(valid)}")

    # Shuffle y split
    random.seed(SEED)
    random.shuffle(valid)
    n_train = int(len(valid) * TRAIN_RATIO)
    train_set = valid[:n_train]
    val_set   = valid[n_train:]
    print(f"Train            : {len(train_set)}")
    print(f"Val              : {len(val_set)}")

    # Copiar imágenes y escribir listas
    def write_list(subset, list_path):
        lines = []
        for crop_file, gt, src_path in subset:
            dst = IMAGES_DIR / crop_file
            if not dst.exists():
                shutil.copy2(src_path, dst)
            lines.append(f"images/{crop_file}\t{gt}\n")
        with open(list_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"  Escrito: {list_path} ({len(lines)} líneas)")

    write_list(train_set, OUT_DIR / "train_list.txt")
    write_list(val_set,   OUT_DIR / "val_list.txt")

    print(f"\n[✓] Dataset listo en: {OUT_DIR}")
    print(f"    Imágenes       : {len(list(IMAGES_DIR.iterdir()))}")


if __name__ == "__main__":
    main()
