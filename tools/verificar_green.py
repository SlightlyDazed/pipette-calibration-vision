"""
verificar_green.py — Genera, revisa y corrige OCR en ROIs verdes (Logitech BRIO)
=================================================================================
Display verde EXTECH tiene SIEMPRE 3 ROIs por imagen:
    - roi_top    → 3 dígitos  (ej: 47.3 %RH)
    - roi_medio  → 3 dígitos  (ej: 25.0 °C)
    - roi_bottom → 4 dígitos  (ej: 987.8)

Resultados guardados en: ocr_yolo/brio_results/ground_truth_green.csv

Uso:
    cd <repo>
    python verificar_green.py

Controles:
    [r]      → La lectura RAW acertó
    [f]      → La lectura con FILTRO acertó
    [valor]  → Ninguna acertó, escribe el valor real (ej: 47.3)
    [s]      → Saltar esta imagen (sin guardar)
    [z]      → Deshacer última entrada guardada
    [q]      → Salir y guardar lo que llevas
"""

import cv2
import csv
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR
from ultralytics import YOLO
import re
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio

# ── RUTAS ─────────────────────────────────────────────────────────────────────
DATASET_DIR = (ROOT / "data/photos")
MODEL_PT    = (ROOT / "models/yolo/roi_detector_v1.pt")
CROPS_DIR   = (ROOT / "data/brio_results/crops")
OUT_CSV     = (ROOT / "data/brio_results/ground_truth_green.csv")

YOLO_CONF   = 0.4
SCALE       = 3
GREEN_MIN_W = 150   # ancho mínimo para garantizar 3 dígitos
GREEN_MIN_H = 130   # alto mínimo

# ── UTILIDADES ────────────────────────────────────────────────────────────────
def extract_number(raw: str) -> str:
    raw = raw.replace(',', '.').strip()
    match = re.search(r'-?\d+\.\d+|-?\d+', raw)
    return match.group() if match else "—"


def raw_crop(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    return cv2.resize(crop_bgr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_LANCZOS4)


def proc_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """PROC v2 — CLAHE suave + threshold adaptativo local."""
    h, w = crop_bgr.shape[:2]
    up    = cv2.resize(crop_bgr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_LANCZOS4)
    gray  = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
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


def run_ocr(paddle, crop_bgr, use_filter=False):
    prep = proc_crop(crop_bgr) if use_filter else raw_crop(crop_bgr)
    try:
        result = paddle.ocr(prep, cls=False)
        if result and result[0]:
            parts, confs = [], []
            for line in result[0]:
                parts.append(line[1][0])
                confs.append(float(line[1][1]) * 100)
            raw_text = ' '.join(parts).strip()
            return raw_text, extract_number(raw_text), sum(confs) / len(confs)
    except Exception:
        pass
    return "", "—", 0.0


def resize_to_h(img_in, th):
    ih, iw = img_in.shape[:2]
    nw = max(1, int(iw * (th / ih)))
    return cv2.resize(img_in, (nw, th))


def undo_last(csv_path: Path):
    """Elimina la última línea del CSV."""
    if not csv_path.exists():
        return None
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if len(lines) <= 1:  # solo header
        print("    ⚠️  Nada que deshacer.")
        return None
    removed = lines[-1].strip()
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.writelines(lines[:-1])
    return removed


def load_done(csv_path: Path) -> set:
    done = set()
    if csv_path.exists():
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                done.add(row['crop_file'])
    return done


def append_row(csv_path: Path, row: dict):
    fieldnames = ["crop_file", "raw_value", "flt_value", "gt_value", "source"]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    already_done = load_done(OUT_CSV)
    print(f"[{len(already_done)} crops verdes ya verificados]\n")

    print("Cargando YOLO...")
    yolo = YOLO(str(MODEL_PT))
    print("Cargando PaddleOCR...")
    paddle = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)
    print("[✓] Modelos listos\n")

    print("=" * 70)
    print("  VERIFICADOR DE ROIs VERDES — Display EXTECH")
    print("  Estructura por imagen: ROI-top (3 dig) | ROI-mid (3 dig) | ROI-bot (4 dig)")
    print("  [r]=Raw OK  [f]=Filtro OK  [valor]=Corregir  [s]=Saltar  [z]=Deshacer  [q]=Salir")
    print("=" * 70 + "\n")

    images = sorted([p for p in DATASET_DIR.iterdir()
                     if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])

    stats = {"raw": 0, "flt": 0, "manual": 0, "skip": 0}

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = yolo.predict(source=img, conf=YOLO_CONF, verbose=False)
        boxes = results[0].boxes
        if not boxes:
            continue

        # Recoger solo los roi_green de esta imagen, ordenados por posición vertical (y1)
        green_boxes = []
        for idx, box in enumerate(boxes):
            cls_name = yolo.model.names[int(box.cls[0].item())]
            if cls_name != "roi_green":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # expand_box: forzar tamaño mínimo para que quepan los dígitos
            img_h_full, img_w_full = img.shape[:2]
            bw, bh = x2 - x1, y2 - y1
            if bw < GREEN_MIN_W:
                delta = (GREEN_MIN_W - bw) // 2
                x1 = max(0, x1 - delta); x2 = min(img_w_full, x2 + delta)
            if bh < GREEN_MIN_H:
                delta = (GREEN_MIN_H - bh) // 2
                y1 = max(0, y1 - delta); y2 = min(img_h_full, y2 + delta)

            green_boxes.append((y1, idx, x1, y1, x2, y2))

        if not green_boxes:
            continue

        green_boxes.sort()  # orden top→bottom por y1
        n_green = len(green_boxes)

        for pos, (_, idx, x1, y1, x2, y2) in enumerate(green_boxes):
            fname = f"{img_path.stem}_roi{idx:02d}_roi_green.jpg"

            if fname in already_done:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Auto-skip crops casi negros (YOLO detectó fondo sin display)
            if crop.mean() < 25:
                print(f"  [AUTO-SKIP] {fname} — imagen muy oscura (media={crop.mean():.1f})")
                append_row(OUT_CSV, {"crop_file": fname, "raw_value": "—",
                                     "flt_value": "—", "gt_value": "SKIP", "source": "auto"})
                already_done.add(fname)
                continue

            # Guardar crop
            crop_path = CROPS_DIR / fname
            if not crop_path.exists():
                cv2.imwrite(str(crop_path), crop)

            # OCR raw y filtrado
            _, raw_v, raw_c = run_ocr(paddle, crop, use_filter=False)
            _, flt_v, flt_c = run_ocr(paddle, crop, use_filter=True)

            # Etiqueta de posición dentro del grupo de 3
            pos_label = ["TOP (3 dig)", "MID (3 dig)", "BOT (4 dig)"][pos] if n_green == 3 else f"ROI {pos+1}/{n_green}"

            # ── Visualización ─────────────────────────────────────────────
            target_h = max(120, crop.shape[0] * SCALE)
            d_orig = resize_to_h(crop, target_h)
            d_raw  = resize_to_h(raw_crop(crop), target_h)
            d_flt  = resize_to_h(proc_crop(crop), target_h)

            sep = np.full((target_h, 4, 3), 100, dtype=np.uint8)
            display = np.hstack([d_orig, sep, d_raw, sep, d_flt])

            # Panel de info
            pw = display.shape[1]
            panel = np.zeros((90, pw, 3), dtype=np.uint8)
            cv2.putText(panel, f"{fname}  [{pos_label}]", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1)
            cv2.putText(panel, f"[r] RAW:    {raw_v}  ({raw_c:.0f}%)", (8, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
            cv2.putText(panel, f"[f] FILTRO: {flt_v}  ({flt_c:.0f}%)", (8, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 80), 2)

            display = np.vstack([panel, display])

            cv2.imshow("Verificador Green", display)
            cv2.waitKey(1)

            # ── Input terminal ─────────────────────────────────────────────
            print(f"\n  [{img_path.stem}] [{pos_label}] → {fname}")
            print(f"    [r] RAW   : {raw_v:>8s}  ({raw_c:.0f}%)")
            print(f"    [f] FILTRO: {flt_v:>8s}  ({flt_c:.0f}%)")

            answer = input(f"    ¿Correcto? [r / f / valor / s / z=deshacer / q]: ").strip()

            if answer.lower() == 'q':
                cv2.destroyAllWindows()
                print("\n  [q] Guardado y saliendo...")
                _print_stats(stats)
                return

            elif answer.lower() == 'z':
                removed = undo_last(OUT_CSV)
                if removed:
                    print(f"    ↩  Deshecha: {removed}")
                    # Quitar del set también para que se pueda re-evaluar
                    already_done = load_done(OUT_CSV)
                # Re-evaluar este mismo crop (no avanzar)
                # Truco: decrementar pos no es directo en for, así que simplemente
                # reponemos el fname en el set y reiniciamos el loop de imágenes
                # con un continue que no guarda nada
                continue

            elif answer.lower() == 's':
                stats["skip"] += 1
                print(f"    → Saltado (guardado para no repetir)")
                append_row(OUT_CSV, {"crop_file": fname, "raw_value": raw_v,
                                     "flt_value": flt_v, "gt_value": "SKIP", "source": "skip"})
                already_done.add(fname)
                continue

            elif answer.lower() == 'r':
                gt_value = raw_v
                stats["raw"] += 1
                print(f"    ✓ RAW seleccionado: {gt_value}")

            elif answer.lower() == 'f':
                gt_value = flt_v
                stats["flt"] += 1
                print(f"    ✓ FILTRO seleccionado: {gt_value}")

            else:
                gt_value = answer
                stats["manual"] += 1
                print(f"    ✎ Corregido a: {gt_value}")

            row = {
                "crop_file": fname,
                "raw_value": raw_v,
                "flt_value": flt_v,
                "gt_value":  gt_value,
                "source":    answer.lower() if answer.lower() in ['r', 'f'] else 'manual'
            }
            append_row(OUT_CSV, row)
            already_done.add(fname)

    cv2.destroyAllWindows()
    _print_stats(stats)


def _print_stats(stats):
    total = stats["raw"] + stats["flt"] + stats["manual"]
    print("\n" + "=" * 60)
    print("  RESUMEN DE SESIÓN GREEN")
    print("=" * 60)
    print(f"  Guardados     : {total}")
    print(f"  Acertó RAW    : {stats['raw']}")
    print(f"  Acertó FILTRO : {stats['flt']}")
    print(f"  Manuales      : {stats['manual']}")
    print(f"  Saltados      : {stats['skip']}")
    print(f"  CSV           : {OUT_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
