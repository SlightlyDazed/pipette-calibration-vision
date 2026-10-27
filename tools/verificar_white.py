"""
verificar_white.py — Revisar y corregir OCR en ROIs blancos (Logitech BRIO)
============================================================================
Muestra cada crop roi_white del dataset, ejecuta PaddleOCR en RAW,
y te pide confirmar o corregir el valor.

Resultados guardados en: ocr_yolo/brio_results/ground_truth_white.csv

Uso:
    python ocr_yolo/verificar_white.py

Controles:
    [Enter]       → El OCR acertó, guardar como correcto
    [número]      → Escribir el valor real (ej: 22.92)
    [s]           → Saltar esta imagen (sin guardar)
    [q]           → Salir y guardar lo que llevas
"""

import cv2
import csv
import os
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR
import re
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio

CROPS_DIR  = (ROOT / "data/brio_results/crops")
OUT_CSV    = (ROOT / "data/brio_results/ground_truth_white.csv")
SCALE      = 3


def extract_number(raw: str) -> str:
    raw = raw.replace(',', '.').strip()
    match = re.search(r'-?\d+\.\d+|-?\d+', raw)
    return match.group() if match else "—"


def raw_crop(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    return cv2.resize(crop_bgr, (w * SCALE, h * SCALE), interpolation=cv2.INTER_LANCZOS4)


def run_ocr(paddle, crop_bgr):
    prep = raw_crop(crop_bgr)
    try:
        result = paddle.ocr(prep, cls=False)
        if result and result[0]:
            parts, confs = [], []
            for line in result[0]:
                parts.append(line[1][0])
                confs.append(float(line[1][1]) * 100)
            raw_text = ' '.join(parts).strip()
            return raw_text, extract_number(raw_text), sum(confs)/len(confs)
    except Exception:
        pass
    return "", "—", 0.0


def main():
    # Buscar todos los crops blancos
    crops = sorted([
        p for p in CROPS_DIR.iterdir()
        if p.is_file() and "roi_white" in p.name and p.suffix.lower() == ".jpg"
    ])

    if not crops:
        print(f"[!] No se encontraron crops en {CROPS_DIR}")
        return

    print(f"\nEncontrados {len(crops)} crops roi_white\n")
    print("[Cargando PaddleOCR...]")
    paddle = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=True, show_log=False)
    print("[✓] Listo\n")

    print("=" * 60)
    print("  VERIFICADOR DE ROIs BLANCOS")
    print("  [Enter]=Correcto  [valor]=Corregir  [s]=Saltar  [q]=Salir")
    print("=" * 60 + "\n")

    # Cargar ya guardados para no repetir
    already_done = set()
    if OUT_CSV.exists():
        with open(OUT_CSV, newline='') as f:
            for row in csv.DictReader(f):
                already_done.add(row['crop_file'])
        print(f"  [{len(already_done)} crops ya verificados, continuando...]\n")

    rows = []
    corrects = 0
    wrongs   = 0
    skipped  = 0

    for i, crop_path in enumerate(crops):
        fname = crop_path.name

        # Saltar ya verificados
        if fname in already_done:
            continue

        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue

        # Correr OCR
        raw_text, ocr_val, conf = run_ocr(paddle, crop)

        # Mostrar la imagen en ventana
        display = cv2.resize(crop, (crop.shape[1]*4, crop.shape[0]*4),
                             interpolation=cv2.INTER_NEAREST)
        
        # Dibujar texto de predicción sobre la imagen
        label = f"OCR: {ocr_val}  ({conf:.0f}%)"
        cv2.putText(display, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        
        cv2.imshow(f"Crop {i+1}/{len(crops)}", display)
        cv2.waitKey(1)

        # Pedir confirmación en terminal
        progress = f"[{i+1}/{len(crops)}]"
        print(f"  {progress} {fname}")
        print(f"    OCR dice  : '{ocr_val}'  (raw: '{raw_text}', conf: {conf:.0f}%)")

        # ⚠️ Alerta: los ROI blancos deben tener siempre 4 dígitos (ej: 13.68)
        # Si YOLO recortó mal (ej: 1 izquierda estrecho), se leerán solo 3 dígitos
        digits_only = re.sub(r'[^\d]', '', ocr_val)  # solo números, sin punto/guion
        if ocr_val != '—' and len(digits_only) < 4:
            print(f"    ⚠️  ALERTA: solo {len(digits_only)} dígito(s) detectado(s) — YOLO pudo haber cortado mal el ROI")
            # Pintar la alerta en la ventana también (borde rojo)
            cv2.rectangle(display, (0, 0), (display.shape[1]-1, display.shape[0]-1),
                          (0, 0, 255), 8)
            cv2.putText(display, "ALERTA: <4 digitos", (10, display.shape[0]-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(f"Crop {i+1}/{len(crops)}", display)
            cv2.waitKey(1)

        answer = input(f"    ¿Correcto? [Enter=sí / valor real / s=saltar / q=salir]: ").strip()
        
        cv2.destroyAllWindows()

        if answer.lower() == 'q':
            print("\n  Guardando y saliendo...")
            break
        elif answer.lower() == 's':
            skipped += 1
            print(f"    → Saltado\n")
            continue
        elif answer == '':
            # OCR correcto
            gt_value = ocr_val
            corrects += 1
            print(f"    ✓ Guardado como: {gt_value}\n")
        else:
            # Usuario corrigió
            gt_value = answer
            wrongs += 1
            print(f"    ✗ Corregido a: {gt_value}\n")

        rows.append({
            "crop_file": fname,
            "ocr_raw":   raw_text,
            "ocr_value": ocr_val,
            "ocr_conf":  round(conf, 1),
            "gt_value":  gt_value,
            "correcto":  "si" if answer == '' else "no"
        })

    # Guardar CSV (modo append para continuar donde se dejó)
    fieldnames = ["crop_file", "ocr_raw", "ocr_value", "ocr_conf", "gt_value", "correcto"]
    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    # Resumen
    total = corrects + wrongs
    pct   = (corrects / total * 100) if total > 0 else 0
    print("\n" + "=" * 60)
    print("  RESUMEN DE SESIÓN")
    print("=" * 60)
    print(f"  Verificados : {total}")
    print(f"  Correctos   : {corrects}  ({pct:.1f}%)")
    print(f"  Corregidos  : {wrongs}")
    print(f"  Saltados    : {skipped}")
    print(f"  CSV guardado: {OUT_CSV}")
    print("=" * 60)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
