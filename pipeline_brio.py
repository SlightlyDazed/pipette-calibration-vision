"""
pipeline_brio.py — Pipeline completo con dataset de la Logitech BRIO
=====================================================================
FASE 1: YOLO detecta ROIs automáticamente en cada imagen del dataset
FASE 2: PaddleOCR evalúa los crops en BRUTO (sin preprocesamiento)
FASE 3: PaddleOCR evalúa los crops con FILTRO (proc_crop v2)

Rutas hardcodeadas para el entorno del usuario:
    Dataset:    data/photos/
    Modelo:     models/yolo/roi_detector_v1.pt
    Salida:     data/brio_results/
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re
ROOT = Path(__file__).resolve().parents[0]  # raíz del repositorio

# ── RUTAS ─────────────────────────────────────────────────────────────────────
DATASET_DIR  = (ROOT / "data/photos")
MODEL_PT     = (ROOT / "models/yolo/roi_detector_v1.pt")
OUT_DIR      = (ROOT / "data/brio_results")
YOLO_CONF    = 0.4
SCALE        = 3

# ── TAMAÑO MÍNIMO DE ROI (basado en estadísticas de crops) ────────────────────────────────────
# Crops white: avg 200×112px. Expansión simmétrica para que quepan 4 dígitos.
WHITE_MIN_W  = 200   # ancho mínimo para roi_white
WHITE_MIN_H  = 88    # alto mínimo para roi_white

# Crops green (EXTECH): 3 filas de números — TOP/MID: 3 dígitos, BOT: 4 dígitos
# Estadística real: min=134px ancho, min=126px alto, avg 298×178px
GREEN_MIN_W  = 150   # ancho mínimo para roi_green (garantiza 3 dígitos)
GREEN_MIN_H  = 130   # alto mínimo para roi_green

def expand_box(x1, y1, x2, y2, img_h, img_w, cls_name):
    """
    Expande la caja YOLO si es más angosta que el mínimo establecido.
    La expansión es SIMÉTRICA (igual hacia ambos lados) para no sesgar el crop.
    """
    bw = x2 - x1
    bh = y2 - y1

    if cls_name == "roi_white":
        min_w, min_h = WHITE_MIN_W, WHITE_MIN_H
    elif cls_name == "roi_green":
        min_w, min_h = GREEN_MIN_W, GREEN_MIN_H
    else:
        return x1, y1, x2, y2

    if bw < min_w:
        delta = (min_w - bw) // 2
        x1 = max(0, x1 - delta)
        x2 = min(img_w, x2 + delta)

    if bh < min_h:
        delta = (min_h - bh) // 2
        y1 = max(0, y1 - delta)
        y2 = min(img_h, y2 + delta)

    return x1, y1, x2, y2

# ── PREPROCESAMIENTO (proc_crop v2) ───────────────────────────────────────────
def proc_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """PROC v2 — CLAHE suave + threshold adaptativo local (no destruye decimales)."""
    h, w = crop_bgr.shape[:2]
    up   = cv2.resize(crop_bgr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    tile  = max(2, gray.shape[0] // 4)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(tile, tile))
    enh   = clahe.apply(gray)
    bsize = max(11, (gray.shape[0] // 6) | 1)
    thresh = cv2.adaptiveThreshold(
        enh, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=bsize,
        C=4
    )
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def raw_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Solo upscale ×3."""
    h, w = crop_bgr.shape[:2]
    return cv2.resize(crop_bgr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_LANCZOS4)


# ── OCR ───────────────────────────────────────────────────────────────────────
def extract_number(raw: str) -> str:
    raw = raw.replace(',', '.').strip()
    match = re.search(r'-?\d+\.\d+|-?\d+', raw)
    return match.group() if match else "—"


def run_paddle_on_crop(paddle, crop_bgr: np.ndarray, use_filter: bool):
    """Corre PaddleOCR sobre un crop (raw o filtrado). Retorna (texto_crudo, número, confianza)."""
    prep = proc_crop(crop_bgr) if use_filter else raw_crop(crop_bgr)
    try:
        result = paddle.ocr(prep, cls=False)
        if result and result[0]:
            parts, confs = [], []
            for line in result[0]:
                parts.append(line[1][0])
                confs.append(float(line[1][1]) * 100)
            raw_text = ' '.join(parts).strip()
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            return raw_text, extract_number(raw_text), avg_conf
    except Exception:
        pass
    return "", "—", 0.0


# ── PIPELINE PRINCIPAL ────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    crops_dir = OUT_DIR / "crops"
    crops_dir.mkdir(exist_ok=True)

    # Cargar modelos
    print("=" * 62)
    print("  CARGANDO MODELOS")
    print("=" * 62)
    print(f"  [YOLO]      {MODEL_PT}")
    yolo = YOLO(str(MODEL_PT))
    print(f"  [PaddleOCR] cargando...")
    paddle = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=True, show_log=False)
    print(f"  [✓] Listos\n")

    # Imágenes del dataset
    exts   = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = sorted([p for p in DATASET_DIR.iterdir()
                     if p.is_file() and p.suffix.lower() in exts])
    print(f"  Dataset: {len(images)} imágenes en {DATASET_DIR}\n")

    # Acumuladores para el resumen final
    summary = []

    print("=" * 62)
    print("  FASE 1+2+3: YOLO → OCR BRUTO → OCR FILTRADO")
    print("=" * 62)

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # ── FASE 1: YOLO ──────────────────────────────────────────────────────
        results = yolo.predict(source=img, conf=YOLO_CONF, verbose=False)
        boxes   = results[0].boxes

        if not boxes or len(boxes) == 0:
            print(f"  {img_path.name[:40]:40s} → sin detecciones YOLO")
            continue

        img_results = []

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id   = int(box.cls[0].item())
            cls_conf = float(box.conf[0].item())
            cls_name = yolo.model.names[cls_id]

            # Expandir la caja si es más estrecha que el mínimo para 4 dígitos
            img_h, img_w = img.shape[:2]
            x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, img_h, img_w, cls_name)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Guardar crop original
            crop_fname = f"{img_path.stem}_roi{idx:02d}_{cls_name}.jpg"
            cv2.imwrite(str(crops_dir / crop_fname), crop)

            # ── FASE 2: OCR BRUTO ────────────────────────────────────────────
            raw_text, raw_val, raw_conf = run_paddle_on_crop(paddle, crop, use_filter=False)

            # ── FASE 3: OCR FILTRADO ─────────────────────────────────────────
            flt_text, flt_val, flt_conf = run_paddle_on_crop(paddle, crop, use_filter=True)

            img_results.append({
                "roi_idx":   idx,
                "class":     cls_name,
                "yolo_conf": round(cls_conf * 100, 1),
                "bbox":      [x1, y1, x2, y2],
                "raw":  {"text": raw_text, "value": raw_val,  "conf": round(raw_conf, 1)},
                "filt": {"text": flt_text, "value": flt_val, "conf": round(flt_conf, 1)},
            })

        summary.append({"image": img_path.name, "rois": img_results})

        # Imprimir resumen por imagen
        print(f"\n  ▶ {img_path.name}")
        for r in img_results:
            raw_v  = r["raw"]["value"]
            flt_v  = r["filt"]["value"]
            raw_c  = r["raw"]["conf"]
            flt_c  = r["filt"]["conf"]
            y_conf = r["yolo_conf"]
            cls    = r["class"]
            print(f"    ROI{r['roi_idx']:02d} [{cls:10s}] YOLO:{y_conf:.0f}%  "
                  f"| RAW: {raw_v:>10s} ({raw_c:.0f}%)  "
                  f"| FILTRO: {flt_v:>10s} ({flt_c:.0f}%)")

    # ── GUARDAR JSON DE RESULTADOS ────────────────────────────────────────────
    json_path = OUT_DIR / "resultados.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── RESUMEN FINAL ─────────────────────────────────────────────────────────
    total_rois = sum(len(e["rois"]) for e in summary)
    raw_detected  = sum(1 for e in summary for r in e["rois"] if r["raw"]["value"] != "—")
    filt_detected = sum(1 for e in summary for r in e["rois"] if r["filt"]["value"] != "—")

    print("\n" + "=" * 62)
    print("  RESUMEN FINAL")
    print("=" * 62)
    print(f"  Imágenes procesadas : {len(summary)}")
    print(f"  ROIs detectados     : {total_rois}")
    print(f"  OCR BRUTO detectó   : {raw_detected}/{total_rois} ({raw_detected/max(1,total_rois)*100:.1f}%)")
    print(f"  OCR FILTRADO detectó: {filt_detected}/{total_rois} ({filt_detected/max(1,total_rois)*100:.1f}%)")
    print(f"\n  JSON guardado en    : {json_path}")
    print(f"  Crops guardados en  : {crops_dir}")
    print("=" * 62)


if __name__ == "__main__":
    main()
