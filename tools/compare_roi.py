"""
compare_roi.py
==============
Compara las ROIs predichas por best.pt contra las etiquetas manuales (GT)
para 10 imágenes aleatorias del dataset.

Genera una grilla PNG con cada imagen mostrando:
  - Cajas GT     → verde sólido   (lo que etiquetaste tú)
  - Cajas PRED   → naranja dashed (lo que predice el modelo)
  - IoU por caja → texto encima de cada predicción

Salida: compare_roi_output.png  (en el mismo directorio que este script)
"""

import cv2
import numpy as np
import random
import math
from pathlib import Path
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR    = ROOT
BEST_PT     = BASE_DIR / "models/yolo/roi_detector_v1.pt"
IMAGES_DIR  = BASE_DIR / "dataset/images"
LABELS_DIR  = BASE_DIR / "dataset/labels"

N_SAMPLES   = 10
CONF_THRESH = 0.3
RANDOM_SEED = 7
OUTPUT_PATH = Path(__file__).parent / "compare_roi_output.png"

CLASS_NAMES = {0: "roi_white", 1: "roi_green"}

# Colores BGR
COLOR_GT   = (50,  200,  50)   # verde
COLOR_PRED = (30,  140, 255)   # naranja
COLOR_TXT  = (255, 255, 255)   # blanco

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Normalizado xywh → píxeles xyxy"""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def compute_iou(boxA, boxB):
    """IoU entre dos cajas [x1,y1,x2,y2]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def best_iou_for_pred(pred_box, gt_boxes):
    """Encuentra el IoU máximo de una predicción contra todas las GT del mismo cls"""
    if not gt_boxes:
        return 0.0
    return max(compute_iou(pred_box, g) for g in gt_boxes)


def load_gt(label_path, img_w, img_h):
    """Lee archivo .txt y devuelve dict {cls: [xyxy, xyxy, ...]}"""
    gt = {0: [], 1: []}
    if not label_path.exists():
        return gt
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                box = yolo_to_xyxy(*map(float, parts[1:]), img_w, img_h)
                gt[cls].append(box)
    return gt


def draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_len=10):
    """Dibuja un rectángulo con línea punteada"""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash_len * 2):
        cv2.line(img, (x, y1), (min(x + dash_len, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash_len, x2), y2), color, thickness)
    for y in range(y1, y2, dash_len * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_len, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash_len, y2)), color, thickness)


def put_label(img, text, x, y, bg_color, font_scale=0.42, thickness=1):
    """Texto con fondo de color"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x, y - th - bl - 2), (x + tw + 4, y + 2), bg_color, -1)
    cv2.putText(img, text, (x + 2, y - bl), font, font_scale, COLOR_TXT, thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# RECOLECTAR IMÁGENES DISPONIBLES (train + val)
# ──────────────────────────────────────────────────────────────────────────────

def collect_pairs():
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    pairs = []
    for split in ("train", "val", ""):           # busca en subcarpetas y raíz
        img_dir = IMAGES_DIR / split if split else IMAGES_DIR
        lbl_dir = LABELS_DIR / split if split else LABELS_DIR
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() in exts and img_path.is_file():
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if lbl_path.exists():
                    pairs.append((img_path, lbl_path))
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO DE UNA IMAGEN
# ──────────────────────────────────────────────────────────────────────────────

def process_image(model, img_path, lbl_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {img_path}")
    h, w = img.shape[:2]
    canvas = img.copy()

    # ── Ground Truth (verde sólido) ──
    gt = load_gt(lbl_path, w, h)
    for cls, boxes in gt.items():
        for box in boxes:
            cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), COLOR_GT, 2)
            put_label(canvas, f"GT {CLASS_NAMES[cls]}", box[0], box[1], (30, 160, 30))

    # ── Predicciones (naranja punteado) ──
    results = model(img_path, conf=CONF_THRESH, verbose=False)[0]
    ious_per_img = []

    for box in results.boxes:
        cls  = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        pred_box = (x1, y1, x2, y2)

        iou = best_iou_for_pred(pred_box, gt.get(cls, []))
        ious_per_img.append(iou)

        draw_dashed_rect(canvas, (x1, y1), (x2, y2), COLOR_PRED, thickness=2)
        label = f"PRED {CLASS_NAMES[cls]} {conf:.2f} IoU:{iou:.2f}"
        put_label(canvas, label, x1, y2 + 14, (180, 90, 20))

    mean_iou = np.mean(ious_per_img) if ious_per_img else 0.0

    # ── Leyenda en esquina ──
    legend_y = h - 10
    put_label(canvas, f"mean IoU: {mean_iou:.3f}  |  GT=verde  PRED=naranja",
              4, legend_y, (40, 40, 40), font_scale=0.45)

    # ── Nombre archivo arriba ──
    put_label(canvas, img_path.name, 4, 18, (40, 40, 40), font_scale=0.45)

    return canvas, mean_iou


# ──────────────────────────────────────────────────────────────────────────────
# GRILLA FINAL
# ──────────────────────────────────────────────────────────────────────────────

def build_grid(images, cols=2):
    """Construye una grilla de imágenes redimensionadas"""
    target_w, target_h = 640, 480
    resized = [cv2.resize(im, (target_w, target_h)) for im in images]

    rows = math.ceil(len(resized) / cols)
    grid_h = rows * target_h
    grid_w = cols * target_w
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx, im in enumerate(resized):
        r, c = divmod(idx, cols)
        grid[r*target_h:(r+1)*target_h, c*target_w:(c+1)*target_w] = im

    return grid


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n[compare_roi] Cargando modelo: {BEST_PT}")
    model = YOLO(str(BEST_PT))

    print(f"[compare_roi] Buscando pares imagen+label...")
    pairs = collect_pairs()

    if not pairs:
        raise RuntimeError("No se encontraron pares imagen+label. Revisa las rutas.")

    random.seed(RANDOM_SEED)
    sample = random.sample(pairs, min(N_SAMPLES, len(pairs)))
    print(f"[compare_roi] Seleccionadas {len(sample)} imágenes de {len(pairs)} disponibles.\n")

    rendered = []
    all_ious = []

    for i, (img_path, lbl_path) in enumerate(sample):
        print(f"  [{i+1:02d}/{len(sample)}] {img_path.name}", end=" ... ")
        canvas, mean_iou = process_image(model, img_path, lbl_path)
        rendered.append(canvas)
        all_ious.append(mean_iou)
        print(f"mean IoU = {mean_iou:.3f}")

    print(f"\n[compare_roi] IoU promedio global: {np.mean(all_ious):.4f}")
    print(f"[compare_roi] IoU mín / máx: {min(all_ious):.4f} / {max(all_ious):.4f}")

    grid = build_grid(rendered, cols=2)
    cv2.imwrite(str(OUTPUT_PATH), grid)
    print(f"\n[✓] Grilla guardada en: {OUTPUT_PATH}")
    print(f"    Resolución: {grid.shape[1]}x{grid.shape[0]} px")
