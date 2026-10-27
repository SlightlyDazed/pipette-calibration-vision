"""
compare_roi_viewer.py — Visor interactivo GT vs predicciones YOLO
==================================================================
Mismo estilo que roi_selector.py pero de solo lectura.
Muestra 10 imágenes aleatorias con sus ROIs comparados.

CONTROLES:
    N  /  →     → siguiente imagen
    P  /  ←     → imagen anterior
    G           → toggle: mostrar solo GT
    D           → toggle: mostrar solo PRED
    Q           → salir

USO:
    python compare_roi_viewer.py
"""

import cv2
import os
import sys
import random
from pathlib import Path
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR    = ROOT
BEST_PT     = BASE_DIR / "models/yolo/roi_detector_v1.pt"
IMAGES_DIR  = BASE_DIR / "dataset/images"
LABELS_DIR  = BASE_DIR / "dataset/labels"

N_SAMPLES    = 10
CONF_THRESH  = 0.3
RANDOM_SEED  = 7
DISPLAY_SCALE = 0.75

CLASS_NAMES = {0: "roi_white", 1: "roi_green"}

# Colores BGR — mismo esquema que roi_selector.py
COLOR_GT_W   = (50,  220,  50)    # verde      → GT roi_white
COLOR_GT_G   = (50,  220,  50)    # verde      → GT roi_green
COLOR_PRED_W = (30,  140, 255)    # naranja    → PRED roi_white
COLOR_PRED_G = (200,  80, 255)    # violeta    → PRED roi_green
COLOR_PANEL  = (25,   25,  25)    # fondo panel
COLOR_WHITE  = (240, 240, 240)
COLOR_YELLOW = (0,   220, 220)
COLOR_DIM    = (140, 140, 140)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
PANEL_H    = 90   # altura del panel inferior en píxeles display

# ── ESTADO ────────────────────────────────────────────────────────────────────
state = {
    "idx":       0,
    "show_gt":   True,
    "show_pred": True,
    "frames":    [],    # lista de dicts precalculados
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def yolo_to_xyxy(cx, cy, w, h, iw, ih):
    x1 = int((cx - w / 2) * iw)
    y1 = int((cy - h / 2) * ih)
    x2 = int((cx + w / 2) * iw)
    y2 = int((cy + h / 2) * ih)
    return x1, y1, x2, y2


def compute_iou(a, b):
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    aA = (a[2]-a[0]) * (a[3]-a[1])
    aB = (b[2]-b[0]) * (b[3]-b[1])
    union = aA + aB - inter
    return inter / union if union > 0 else 0.0


def best_iou(pred_box, gt_list):
    return max((compute_iou(pred_box, g) for g in gt_list), default=0.0)


def load_gt(label_path, iw, ih):
    gt = {0: [], 1: []}
    if not label_path.exists():
        return gt
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                gt[cls].append(yolo_to_xyxy(*map(float, parts[1:]), iw, ih))
    return gt


def draw_dashed_rect(img, pt1, pt2, color, thickness=2, gap=8):
    x1, y1 = pt1
    x2, y2 = pt2
    d = gap * 2
    for x in range(x1, x2, d):
        cv2.line(img, (x, y1), (min(x+gap, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x+gap, x2), y2), color, thickness)
    for y in range(y1, y2, d):
        cv2.line(img, (x1, y), (x1, min(y+gap, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y+gap, y2)), color, thickness)


def put_label(img, text, x, y, bg, fg=COLOR_WHITE, scale=0.42, thick=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (x-2, y-th-bl-2), (x+tw+4, y+2), bg, -1)
    cv2.putText(img, text, (x+2, y-bl), font, scale, fg, thick, cv2.LINE_AA)


def scale_box(box, scale):
    return tuple(int(v * scale) for v in box)


# ── PRECALCULAR FRAMES ────────────────────────────────────────────────────────

def precompute_frames(pairs, model):
    frames = []
    print()
    for i, (img_path, lbl_path) in enumerate(pairs):
        print(f"  Procesando [{i+1}/{len(pairs)}] {img_path.name} ...", end="\r")

        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            continue
        ih, iw = img_orig.shape[:2]

        gt  = load_gt(lbl_path, iw, ih)

        results = model(img_path, conf=CONF_THRESH, verbose=False)[0]
        preds   = []
        ious    = []
        for box in results.boxes:
            cls  = int(box.cls)
            conf = float(box.conf)
            xyxy = tuple(map(int, box.xyxy[0].tolist()))
            iou  = best_iou(xyxy, gt.get(cls, []))
            preds.append({"cls": cls, "conf": conf, "box": xyxy, "iou": iou})
            ious.append(iou)

        frames.append({
            "path":      img_path,
            "img_orig":  img_orig,
            "gt":        gt,
            "preds":     preds,
            "mean_iou":  sum(ious)/len(ious) if ious else 0.0,
        })

    print(f"\n  [✓] {len(frames)} imágenes listas.\n")
    return frames


# ── RENDERIZAR UNA IMAGEN ─────────────────────────────────────────────────────

def render(frame):
    img_orig = frame["img_orig"]
    ih, iw   = img_orig.shape[:2]
    scale    = DISPLAY_SCALE

    dw = int(iw * scale)
    dh = int(ih * scale)
    canvas = cv2.resize(img_orig, (dw, dh))

    gt    = frame["gt"]
    preds = frame["preds"]
    show_gt   = state["show_gt"]
    show_pred = state["show_pred"]

    # ── GT (verde sólido) ──
    if show_gt:
        for cls, boxes in gt.items():
            color = COLOR_GT_W if cls == 0 else COLOR_GT_G
            for box in boxes:
                sb = scale_box(box, scale)
                cv2.rectangle(canvas, (sb[0], sb[1]), (sb[2], sb[3]), color, 2)
                put_label(canvas, f"GT {CLASS_NAMES[cls]}",
                          sb[0], sb[1]-2, (20, 130, 20), scale=0.38)

    # ── PRED (punteado naranja/violeta) ──
    if show_pred:
        for p in preds:
            color = COLOR_PRED_W if p["cls"] == 0 else COLOR_PRED_G
            sb = scale_box(p["box"], scale)
            draw_dashed_rect(canvas, (sb[0], sb[1]), (sb[2], sb[3]), color, 2)
            lbl = f"PRED {CLASS_NAMES[p['cls']]} {p['conf']:.2f}  IoU:{p['iou']:.2f}"
            put_label(canvas, lbl, sb[0], sb[3]+14, (100, 60, 10), scale=0.38)

    # ── PANEL INFERIOR ──
    panel = _build_panel(dw, frame)
    composite = cv2.vconcat([canvas, panel])
    return composite


def _build_panel(width, frame):
    panel = _solid(width, PANEL_H, COLOR_PANEL)
    idx   = state["idx"]
    total = len(state["frames"])
    fname = frame["path"].name
    miou  = frame["mean_iou"]
    n_gt  = sum(len(v) for v in frame["gt"].values())
    n_pr  = len(frame["preds"])

    # Fila 1 — nombre e índice
    cv2.putText(panel, f"[{idx+1}/{total}]  {fname}",
                (10, 22), FONT, 0.52, COLOR_YELLOW, 1, cv2.LINE_AA)

    # Fila 2 — métricas
    iou_color = (50, 220, 50) if miou >= 0.8 else (30, 140, 255) if miou >= 0.6 else (0, 0, 220)
    cv2.putText(panel, f"GT={n_gt}  PRED={n_pr}  mean IoU={miou:.3f}",
                (10, 46), FONT, 0.50, iou_color, 1, cv2.LINE_AA)

    # Leyenda de colores
    _legend_box(panel,  10, 62, COLOR_GT_W,   "GT  (solido)")
    _legend_box(panel, 175, 62, COLOR_PRED_W, "PRED roi_white (punteado)")
    _legend_box(panel, 430, 62, COLOR_PRED_G, "PRED roi_green (punteado)")

    # Controles (esquina derecha)
    ctrl = "N/→ siguiente   P/← anterior   G GT   D PRED   Q salir"
    (tw, _), _ = cv2.getTextSize(ctrl, FONT, 0.40, 1)
    cv2.putText(panel, ctrl, (width - tw - 8, 80),
                FONT, 0.40, COLOR_DIM, 1, cv2.LINE_AA)

    return panel


def _solid(w, h, color):
    img = __import__("numpy").zeros((h, w, 3), dtype=__import__("numpy").uint8)
    img[:] = color
    return img


def _legend_box(panel, x, y, color, label):
    cv2.rectangle(panel, (x, y-10), (x+18, y+2), color, -1)
    cv2.putText(panel, label, (x+22, y), FONT, 0.40, COLOR_DIM, 1, cv2.LINE_AA)


# ── PRINT HELP TERMINAL ───────────────────────────────────────────────────────

def print_help(frame):
    os.system("clear")
    idx   = state["idx"]
    total = len(state["frames"])
    fname = frame["path"].name
    miou  = frame["mean_iou"]

    print("=" * 58)
    print(f"  ROI COMPARE VIEWER — [{idx+1}/{total}] {fname}")
    print("=" * 58)
    print(f"  mean IoU : {miou:.4f}  {'✓ EXCELENTE' if miou>=0.8 else '~ OK' if miou>=0.6 else '✗ REVISAR'}")
    print()

    gt = frame["gt"]
    preds = frame["preds"]
    print(f"  GT ({sum(len(v) for v in gt.values())} cajas):")
    for cls, boxes in gt.items():
        for b in boxes:
            print(f"    [{CLASS_NAMES[cls]:10s}]  {b}")

    print(f"\n  PRED ({len(preds)} cajas):")
    for p in preds:
        iou_tag = "✓" if p["iou"] >= 0.8 else "~" if p["iou"] >= 0.6 else "✗"
        print(f"    {iou_tag} [{CLASS_NAMES[p['cls']]:10s}]  conf={p['conf']:.3f}  IoU={p['iou']:.3f}  {p['box']}")

    print()
    print("  N / →  siguiente  |  P / ←  anterior")
    print("  G      toggle GT  |  D      toggle PRED")
    print("  Q      salir")
    print("=" * 58)


# ── RECOLECTAR PARES ──────────────────────────────────────────────────────────

def collect_pairs():
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    pairs = []
    for split in ("train", "val", ""):
        idir = IMAGES_DIR / split if split else IMAGES_DIR
        ldir = LABELS_DIR / split if split else LABELS_DIR
        if not idir.exists():
            continue
        for img in idir.iterdir():
            if img.suffix.lower() in exts and img.is_file():
                lbl = ldir / (img.stem + ".txt")
                if lbl.exists():
                    pairs.append((img, lbl))
    return pairs


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.system("clear")
    print("=" * 58)
    print("  ROI COMPARE VIEWER")
    print("=" * 58)
    print(f"  Modelo : {BEST_PT.name}")
    print(f"  Dataset: {IMAGES_DIR}")
    print(f"  Conf   : {CONF_THRESH}  |  Muestras: {N_SAMPLES}")
    print()

    if not BEST_PT.exists():
        print(f"[ERROR] No se encontró best.pt en:\n  {BEST_PT}")
        sys.exit(1)

    print("  Cargando modelo YOLO...")
    model = YOLO(str(BEST_PT))

    pairs = collect_pairs()
    if not pairs:
        print("[ERROR] No se encontraron pares imagen+label.")
        sys.exit(1)

    random.seed(RANDOM_SEED)
    sample = random.sample(pairs, min(N_SAMPLES, len(pairs)))
    print(f"  Seleccionadas {len(sample)} imágenes de {len(pairs)} disponibles.")

    state["frames"] = precompute_frames(sample, model)

    if not state["frames"]:
        print("[ERROR] No se pudo procesar ninguna imagen.")
        sys.exit(1)

    cv2.namedWindow("ROI Compare Viewer", cv2.WINDOW_AUTOSIZE)
    print_help(state["frames"][0])

    while True:
        idx   = state["idx"]
        frame = state["frames"][idx]
        img   = render(frame)
        cv2.imshow("ROI Compare Viewer", img)

        key = cv2.waitKey(20) & 0xFF

        # Siguiente
        if key in (ord('n'), ord('N'), 83):   # 83 = →
            if idx < len(state["frames"]) - 1:
                state["idx"] += 1
                print_help(state["frames"][state["idx"]])

        # Anterior
        elif key in (ord('p'), ord('P'), 81): # 81 = ←
            if idx > 0:
                state["idx"] -= 1
                print_help(state["frames"][state["idx"]])

        # Toggle GT
        elif key in (ord('g'), ord('G')):
            state["show_gt"] = not state["show_gt"]
            print(f"  [GT {'ON' if state['show_gt'] else 'OFF'}]")

        # Toggle PRED
        elif key in (ord('d'), ord('D')):
            state["show_pred"] = not state["show_pred"]
            print(f"  [PRED {'ON' if state['show_pred'] else 'OFF'}]")

        # Salir
        elif key in (ord('q'), ord('Q'), 27):
            break

        # Cerrar con X
        if cv2.getWindowProperty("ROI Compare Viewer", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print("\n  [Viewer cerrado]\n")


if __name__ == "__main__":
    main()
