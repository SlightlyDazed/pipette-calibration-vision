"""
verifica_expand_box.py — Comprueba que expand_box funciona correctamente
=========================================================================
Toma 10 imágenes del dataset, corre YOLO, y muestra:
- Tamaño original de la caja YOLO
- Tamaño después de expand_box
- Guarda las comparativas en brio_results/expand_test/
  left = crop original (YOLO tal cual)
  right = crop expandido (con enforce mínimo)
"""

import cv2
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
import numpy as np

# Importar la función que acabamos de agregar
from pipeline_brio import expand_box, WHITE_MIN_W, WHITE_MIN_H

DATASET_DIR = (ROOT / "data/photos")
MODEL_PT    = (ROOT / "models/yolo/roi_detector_v1.pt")
OUT_DIR     = (ROOT / "data/brio_results/expand_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YOLO_CONF = 0.4

print("Cargando YOLO...")
yolo = YOLO(str(MODEL_PT))
print("Listo\n")

images = sorted([p for p in DATASET_DIR.iterdir()
                 if p.suffix.lower() in {'.jpg','.jpeg','.png'}])[:10]

print(f"{'Imagen':40s}  {'ROI':5s}  {'Clase':12s}  {'Original':15s}  {'Expandido':15s}  {'¿Expandió?'}")
print("-" * 110)

for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    img_h, img_w = img.shape[:2]
    results = yolo.predict(source=img, conf=YOLO_CONF, verbose=False)
    boxes = results[0].boxes

    if not boxes or len(boxes) == 0:
        print(f"  {img_path.name[:38]:38s}  — sin detecciones")
        continue

    for idx, box in enumerate(boxes):
        x1o, y1o, x2o, y2o = map(int, box.xyxy[0].tolist())
        cls_id   = int(box.cls[0].item())
        cls_name = yolo.model.names[cls_id]

        if cls_name != "roi_white":
            continue   # solo verificamos blancos

        # Tamaño original
        orig_w = x2o - x1o
        orig_h = y2o - y1o

        # Tamaño expandido
        x1e, y1e, x2e, y2e = expand_box(x1o, y1o, x2o, y2o, img_h, img_w, cls_name)
        exp_w = x2e - x1e
        exp_h = y2e - y1e

        expandio = "✓ SÍ" if (x1e != x1o or x2e != x2o or y1e != y1o or y2e != y2o) else "— no"

        print(f"  {img_path.name[:38]:38s}  {idx:3d}  {cls_name:12s}  "
              f"{orig_w:4d}×{orig_h:<4d}  →   {exp_w:4d}×{exp_h:<4d}      {expandio}")

        # Guardar comparativa visual side-by-side
        crop_orig = img[y1o:y2o, x1o:x2o]
        crop_exp  = img[y1e:y2e, x1e:x2e]

        # Normalizar ambos al mismo alto para ponerlos lado a lado
        target_h = max(crop_orig.shape[0], crop_exp.shape[0], 80)
        scale_o = target_h / crop_orig.shape[0]
        scale_e = target_h / crop_exp.shape[0]
        orig_disp = cv2.resize(crop_orig, (int(crop_orig.shape[1]*scale_o*2), target_h*2))
        exp_disp  = cv2.resize(crop_exp,  (int(crop_exp.shape[1]*scale_e*2),  target_h*2))

        # Añadir etiquetas
        cv2.putText(orig_disp, f"ORIGINAL {orig_w}x{orig_h}", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(exp_disp,  f"EXPANDIDO {exp_w}x{exp_h}", (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        # Separador vertical
        sep = np.ones((target_h*2, 5, 3), dtype=np.uint8) * 128

        # Si tienen distinto alto, padear el más bajo
        if orig_disp.shape[0] != exp_disp.shape[0]:
            max_h = max(orig_disp.shape[0], exp_disp.shape[0])
            orig_disp = cv2.copyMakeBorder(orig_disp, 0, max_h-orig_disp.shape[0], 0, 0,
                                           cv2.BORDER_CONSTANT, value=(40,40,40))
            exp_disp  = cv2.copyMakeBorder(exp_disp,  0, max_h-exp_disp.shape[0],  0, 0,
                                           cv2.BORDER_CONSTANT, value=(40,40,40))

        side_by_side = np.hstack([orig_disp, sep, exp_disp])
        fname = f"{img_path.stem}_roi{idx:02d}_compare.jpg"
        cv2.imwrite(str(OUT_DIR / fname), side_by_side)

print(f"\n[✓] Comparativas guardadas en: {OUT_DIR}")
print(f"    Mínimo esperado para roi_white: {WHITE_MIN_W}×{WHITE_MIN_H}px")
