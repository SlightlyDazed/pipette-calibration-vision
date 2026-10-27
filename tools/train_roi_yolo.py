"""
train_roi_yolo.py
=================
Pipeline completo de entrenamiento YOLO11 para detección de ROIs en imágenes de medidores.

Clases:
    0 → roi_white  (columna derecha, displays blancos, cajas ~pequeñas)
    1 → roi_green  (columna izquierda, displays verdes, cajas ~grandes)

Estructura esperada en disco ANTES de correr este script:
    ocr_yolo/
    ├── yolo11m.pt
    ├── database/          ← imágenes originales (todas)
    ├── dataset/
    │   ├── images/        ← imágenes YA copiadas por roi_labeler.py
    │   └── labels/        ← archivos .txt con labels en formato YOLO
    └── train_roi_yolo.py  ← este script

Estructura que este script CREA:
    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml

GPU target: AMD RX 9070 (ROCm) — usa device=0 como CUDA vía HIP
"""

import os
import shutil
import random
import yaml
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — ajusta solo estos valores si necesitas
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR    = ROOT
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH  = BASE_DIR / "yolo11m.pt"

# Clases del proyecto
CLASS_NAMES = {0: "roi_white", 1: "roi_green"}

# Split train/val
VAL_RATIO   = 0.20   # 20% para validación
RANDOM_SEED = 42

# Hiperparámetros de entrenamiento
EPOCHS      = 100
IMG_SIZE    = 640
BATCH       = 8      # Ajusta si tu GPU se queda sin VRAM (baja a 4 o sube a 16)
DEVICE      = 0      # 0 = GPU (RX 9070 via ROCm/HIP). Usa 'cpu' si hay problemas
LR0         = 0.01   # Learning rate inicial
LRF         = 0.01   # Factor de decay final (lr_final = LR0 * LRF)
PATIENCE    = 20     # Early stopping si no mejora en N epochs
PROJECT     = str(BASE_DIR / "runs" / "detect")
RUN_NAME    = "roi_detector_v1"

# Augmentación (valores conservadores para imágenes de medidores)
AUGMENT_CFG = dict(
    degrees     = 5.0,    # rotación máxima (los medidores son casi siempre rectos)
    translate   = 0.05,   # traslación leve
    scale       = 0.15,   # zoom
    shear       = 2.0,    # inclinación mínima
    perspective = 0.0,    # sin distorsión de perspectiva
    flipud      = 0.0,    # sin flip vertical (medidores tienen orientación fija)
    fliplr      = 0.0,    # sin flip horizontal
    mosaic      = 0.5,    # mosaic activo al 50%
    mixup       = 0.0,    # sin mixup (pocas clases, no necesario)
    hsv_h       = 0.015,  # variación de matiz
    hsv_s       = 0.4,    # variación de saturación
    hsv_v       = 0.3,    # variación de brillo
)

# ──────────────────────────────────────────────────────────────────────────────
# PASO 1 — Verificar que el dataset existe y está completo
# ──────────────────────────────────────────────────────────────────────────────

def verify_dataset():
    images_dir = DATASET_DIR / "images"
    labels_dir = DATASET_DIR / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"No se encontraron las carpetas 'images' o 'labels' dentro de:\n{DATASET_DIR}\n"
            "Asegúrate de haber corrido roi_labeler.py primero."
        )

    # Buscar imágenes (descartamos subcarpetas train/ val/ si ya existen)
    img_files = [
        f for f in images_dir.glob("*")
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        and f.parent.name == "images"   # solo las del nivel raíz
    ]

    # Verificar que cada imagen tenga su label
    paired, missing_labels = [], []
    for img in img_files:
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists():
            paired.append((img, lbl))
        else:
            missing_labels.append(img.name)

    print(f"[✓] Imágenes encontradas con label: {len(paired)}")
    if missing_labels:
        print(f"[!] Imágenes SIN label ({len(missing_labels)}): {missing_labels[:5]}{'...' if len(missing_labels)>5 else ''}")

    if len(paired) == 0:
        raise ValueError("No hay pares imagen+label. Verifica la carpeta dataset/.")

    return paired

# ──────────────────────────────────────────────────────────────────────────────
# PASO 2 — Inspeccionar los labels y reportar estadísticas
# ──────────────────────────────────────────────────────────────────────────────

def inspect_labels(paired):
    from collections import Counter
    class_counts = Counter()
    total_boxes  = 0

    for _, lbl in paired:
        with open(lbl) as f:
            for line in f:
                line = line.strip()
                if line:
                    cls = int(line.split()[0])
                    class_counts[cls] += 1
                    total_boxes += 1

    print(f"\n[ESTADÍSTICAS DEL DATASET]")
    print(f"  Imágenes totales   : {len(paired)}")
    print(f"  Boxes totales      : {total_boxes}")
    print(f"  Promedio boxes/img : {total_boxes/len(paired):.1f}")
    for cls_id, name in CLASS_NAMES.items():
        n = class_counts.get(cls_id, 0)
        print(f"  Clase {cls_id} ({name:12s}): {n} boxes ({100*n/total_boxes:.1f}%)")

# ──────────────────────────────────────────────────────────────────────────────
# PASO 3 — Split train / val y reorganizar carpetas
# ──────────────────────────────────────────────────────────────────────────────

def split_dataset(paired):
    random.seed(RANDOM_SEED)
    random.shuffle(paired)

    n_val   = max(1, int(len(paired) * VAL_RATIO))
    val_set = paired[:n_val]
    trn_set = paired[n_val:]

    print(f"\n[SPLIT]  train={len(trn_set)}  |  val={len(val_set)}")

    # Crear carpetas destino
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    def move_pair(pairs, split):
        for img, lbl in pairs:
            dst_img = DATASET_DIR / "images" / split / img.name
            dst_lbl = DATASET_DIR / "labels" / split / lbl.name
            if not dst_img.exists():
                shutil.copy2(img, dst_img)
            if not dst_lbl.exists():
                shutil.copy2(lbl, dst_lbl)

    move_pair(trn_set, "train")
    move_pair(val_set, "val")

    print(f"[✓] Archivos copiados a dataset/images/{{train,val}} y dataset/labels/{{train,val}}")
    return trn_set, val_set

# ──────────────────────────────────────────────────────────────────────────────
# PASO 4 — Crear data.yaml
# ──────────────────────────────────────────────────────────────────────────────

def create_yaml():
    yaml_path = DATASET_DIR / "data.yaml"

    data = {
        "path"  : str(DATASET_DIR),
        "train" : "images/train",
        "val"   : "images/val",
        "nc"    : len(CLASS_NAMES),
        "names" : [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"\n[✓] data.yaml creado en: {yaml_path}")
    print(f"    Clases: {data['names']}")
    return yaml_path

# ──────────────────────────────────────────────────────────────────────────────
# PASO 5 — Entrenamiento
# ──────────────────────────────────────────────────────────────────────────────

def train(yaml_path):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Instala ultralytics primero: pip install ultralytics")

    print(f"\n{'='*60}")
    print(f"  INICIANDO ENTRENAMIENTO")
    print(f"  Modelo base : {MODEL_PATH}")
    print(f"  Dataset     : {yaml_path}")
    print(f"  Epochs      : {EPOCHS}  |  Batch: {BATCH}  |  imgSz: {IMG_SIZE}")
    print(f"  Device      : {DEVICE}  (AMD RX 9070 via ROCm/HIP)")
    print(f"{'='*60}\n")

    model = YOLO(str(MODEL_PATH))

    results = model.train(
        data      = str(yaml_path),
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH,
        device    = DEVICE,
        project   = PROJECT,
        name      = RUN_NAME,
        patience  = PATIENCE,
        lr0       = LR0,
        lrf       = LRF,
        save      = True,
        plots     = True,
        verbose   = True,

        # Augmentación conservadora para medidores
        **AUGMENT_CFG,
    )

    best_weights = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    print(f"\n{'='*60}")
    print(f"  ENTRENAMIENTO FINALIZADO")
    print(f"  Mejor modelo guardado en:")
    print(f"  {best_weights}")
    print(f"{'='*60}\n")

    return results, best_weights

# ──────────────────────────────────────────────────────────────────────────────
# PASO 6 — Validación rápida post-entrenamiento
# ──────────────────────────────────────────────────────────────────────────────

def validate(best_weights, yaml_path):
    from ultralytics import YOLO

    print("\n[VALIDACIÓN POST-ENTRENAMIENTO]")
    model = YOLO(str(best_weights))
    metrics = model.val(data=str(yaml_path), device=DEVICE, imgsz=IMG_SIZE)

    print(f"\n  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    for i, name in enumerate(CLASS_NAMES.values()):
        if i < len(metrics.box.ap_class_index):
            idx = list(metrics.box.ap_class_index).index(i) if i in metrics.box.ap_class_index else None
            if idx is not None:
                print(f"  AP50 {name:12s}: {metrics.box.ap50[idx]:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[ROI YOLO TRAINER — ocr_yolo]\n")

    # 1. Verificar dataset
    paired = verify_dataset()

    # 2. Inspeccionar estadísticas
    inspect_labels(paired)

    # 3. Split y reorganizar
    trn_set, val_set = split_dataset(paired)

    # 4. Crear data.yaml
    yaml_path = create_yaml()

    # 5. Entrenar
    results, best_weights = train(yaml_path)

    # 6. Validar
    if best_weights.exists():
        validate(best_weights, yaml_path)

    print("\n[DONE] Proceso completo. Revisa runs/detect/roi_detector_v1/")
