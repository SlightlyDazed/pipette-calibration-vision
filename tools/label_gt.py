"""
label_gt.py — Etiquetado visual de ground truth para crops
===========================================================
Muestra cada imagen en pantalla y pide el número real por consola.
Guarda en ground_truth.csv después de cada entrada.

CONTROLES (en consola):
    <número>  → guarda el valor y avanza
    s / ENTER → salta (deja vacío)
    b         → vuelve a la imagen anterior
    q         → guarda y sale

La imagen se muestra ampliada para facilitar la lectura.
"""

import cv2
import csv
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio

BASE_DIR = ROOT
GT_CSV   = BASE_DIR / "ground_truth.csv"

ZOOM      = 4      # Factor de ampliación de la imagen
MAX_W     = 1200   # Ancho máximo de la ventana
MAX_H     = 700    # Alto máximo
WIN       = "Ground Truth Labeler  |  Escribe el número en la consola"
FONT      = cv2.FONT_HERSHEY_SIMPLEX

COL_GREEN  = (50,  220,  50)
COL_ORANGE = (30,  140, 255)
COL_YELLOW = (0,  220, 220)
COL_BG     = (20,  20,  20)


def load_csv():
    rows = []
    with open(GT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_csv(rows):
    with open(GT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx","class","file","source_img","gt_value"])
        writer.writeheader()
        writer.writerows(rows)


def make_display(img, row, current, total, zoom):
    """Construye el frame para mostrar: imagen ampliada + panel de info."""
    h, w = img.shape[:2]

    # Escalar imagen manteniendo aspecto
    scale = min(zoom, MAX_W / w, (MAX_H - 80) / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    interp = cv2.INTER_NEAREST if scale >= 2 else cv2.INTER_LINEAR
    zoomed = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # Canvas negro
    canvas_w = max(new_w + 20, 600)
    canvas_h = new_h + 90
    canvas = __import__('numpy').zeros((canvas_h, canvas_w, 3), dtype='uint8')
    canvas[:] = COL_BG

    # Centrar imagen
    ox = (canvas_w - new_w) // 2
    canvas[10:10 + new_h, ox:ox + new_w] = zoomed

    # Panel inferior
    cls_name = row["class"]
    color    = COL_GREEN if cls_name == "green" else COL_ORANGE
    fname    = Path(row["file"]).name
    src      = row["source_img"]
    gt_now   = row["gt_value"] or "(vacío)"

    # Barra de progreso
    bar_w = int(canvas_w * (current / total))
    canvas[new_h + 10:new_h + 16, 0:bar_w] = color

    info = f"[{current}/{total}]  {fname}  ({cls_name})  src: {src}  actual: {gt_now}"
    cv2.putText(canvas, info, (8, new_h + 35),
                FONT, 0.46, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Escribe en la CONSOLA:  numero=valor  s=saltar  b=atras  q=salir",
                (8, new_h + 60), FONT, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

    return canvas


def main():
    rows = load_csv()
    total = len(rows)

    if total == 0:
        print("[ERROR] CSV vacío.")
        sys.exit(1)

    # Empezar desde la primera sin gt_value
    filled   = sum(1 for r in rows if r["gt_value"].strip())
    start_at = next((i for i, r in enumerate(rows) if not r["gt_value"].strip()), 0)

    print("=" * 60)
    print("  GROUND TRUTH LABELER")
    print("=" * 60)
    print(f"  Total crops : {total}")
    print(f"  Ya llenados : {filled}")
    print(f"  Pendientes  : {total - filled}")
    print()
    print("  Escribe el número que ves en la imagen y pulsa ENTER.")
    print("  s / ENTER vacío → saltar")
    print("  b               → volver atrás")
    print("  q               → guardar y salir")
    print("=" * 60)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    i = start_at
    while 0 <= i < total:
        row = rows[i]
        img_path = BASE_DIR / row["file"]

        # Cargar imagen
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [!] No se pudo leer: {img_path}")
            i += 1
            continue

        frame = make_display(img, row, i + 1, total, ZOOM)
        cv2.imshow(WIN, frame)
        cv2.waitKey(1)   # Forzar render

        # Leer input del usuario en consola
        gt_current = row["gt_value"] or ""
        prefix     = f"\n  [{i+1}/{total}] {Path(row['file']).name}"
        if gt_current:
            prefix += f"  (actual: {gt_current})"
        prefix += "\n  → "

        try:
            user_input = input(prefix).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  [Guardando y saliendo...]")
            break

        if user_input.lower() == 'q':
            print("  [Guardando y saliendo...]")
            break
        elif user_input.lower() == 'b':
            i = max(0, i - 1)
            continue
        elif user_input == '' or user_input.lower() == 's':
            # Saltar sin modificar
            i += 1
            continue
        else:
            # Guardar el valor
            rows[i]["gt_value"] = user_input
            save_csv(rows)

            filled_now = sum(1 for r in rows if r["gt_value"].strip())
            print(f"  ✔ Guardado '{user_input}'  ({filled_now}/{total} completados)")
            i += 1

    cv2.destroyAllWindows()

    # Resumen final
    rows = load_csv()
    filled   = sum(1 for r in rows if r["gt_value"].strip())
    n_green  = sum(1 for r in rows if r["class"] == "green" and r["gt_value"].strip())
    n_white  = sum(1 for r in rows if r["class"] == "white" and r["gt_value"].strip())

    print()
    print("=" * 60)
    print(f"  Sesión terminada")
    print(f"  Llenados : {filled}/{total}")
    print(f"    green  : {n_green}/50")
    print(f"    white  : {n_white}/50")
    print(f"  CSV      : {GT_CSV}")
    print()
    if filled >= 30:
        print("  ✅ Suficientes datos para correr OCR + NSGA")
    else:
        print(f"  ⚠ Faltan {30 - filled} más para tener señal de fitness suficiente")
    print("=" * 60)


if __name__ == "__main__":
    main()
