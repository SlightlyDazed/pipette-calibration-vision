"""
filter_tuner.py — Ajuste interactivo de preprocesamiento + PaddleOCR
=====================================================================
CONTROLES:
    G / W     → cambiar clase activa (roi_green / roi_white)
    O         → ejecutar PaddleOCR sobre crops actuales
    N / →     → siguiente imagen
    P / ←     → imagen anterior
    R         → resetear params clase actual a defaults
    S         → guardar filter_config.json
    Q / ESC   → salir

TRACKBARS (aplican a la clase activa):
    Blur        → 0-5   (1,3,5,7,9,11 px kernel gaussiano)
    ThreshType  → 0=Binary  1=Inv_Binary  2=Otsu  3=Adapt_Gauss  4=Adapt_Mean
    ThreshVal   → 0-255
    AdaptBlock  → 1-24  (→ bloque impar: 3,5,7...51)
    AdaptC      → 0-40  (→ -20...+20)
    CLAHE       → 0=off  1=on
    CLAHEClip   → 1-40  (→ 0.5...20)
    MorphOp     → 0=none  1=erode  2=dilate  3=open  4=close
    MorphK      → 1-10
"""

import cv2
import json
import os
import re
import sys
import time
import numpy as np
import random
from pathlib import Path
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR    = ROOT
BEST_PT     = BASE_DIR / "models/yolo/roi_detector_v1.pt"
IMAGES_DIR  = BASE_DIR / "dataset/images"
LABELS_DIR  = BASE_DIR / "dataset/labels"
OUTPUT_JSON = Path(__file__).parent / "filter_config.json"

N_SAMPLES   = 10
CONF_THRESH = 0.3
RANDOM_SEED = 7
WINDOW      = "Filter Tuner"
CROP_W      = 220
CROP_H      = 110
OCR_H       = 22
FONT        = cv2.FONT_HERSHEY_SIMPLEX

CLASS_NAMES    = {0: "roi_white", 1: "roi_green"}
COL_GREEN      = (50,  220,  50)
COL_ORANGE     = (30,  140, 255)
COL_PANEL      = (20,   20,  20)
COL_WHITE      = (240, 240, 240)
COL_YELLOW     = (0,  220, 220)
COL_DIM        = (130, 130, 130)
COL_RED        = (50,   50, 220)
COL_GOOD       = (50,  200,  50)
THRESH_LABELS  = ["Binary", "Inv_Binary", "Otsu", "Adapt_Gauss", "Adapt_Mean"]
MORPH_LABELS   = ["none", "erode", "dilate", "open", "close"]
MORPH_OPS      = [None, cv2.MORPH_ERODE, cv2.MORPH_DILATE, cv2.MORPH_OPEN, cv2.MORPH_CLOSE]
BLUR_MAP       = [1, 3, 5, 7, 9, 11]

DEFAULT_PARAMS = {
    "roi_green": dict(blur=1, thresh_type=2, thresh_val=100, adapt_block=6,
                      adapt_C=22, clahe=1, clahe_clip=8, morph_op=4, morph_k=2),
    "roi_white": dict(blur=1, thresh_type=3, thresh_val=128, adapt_block=4,
                      adapt_C=22, clahe=0, clahe_clip=4, morph_op=0, morph_k=2),
}

state = {
    "cls_active":  "roi_green",
    "params":      {k: dict(v) for k, v in DEFAULT_PARAMS.items()},
    "img_idx":     0,
    "images":      [],
    "ocr_results": {},
    "ocr_engine":  None,
    "ocr_ready":   False,
}


# ── PADDLEOCR ─────────────────────────────────────────────────────────────────

def init_paddle():
    try:
        import os
        os.environ['FLAGS_use_mkldnn'] = '0'  # Deshabilitar oneDNN (Intel) en AMD
        from paddleocr import TextRecognition
        print("  Iniciando PaddleOCR v3 - TextRecognition (CPU)...")
        rec = TextRecognition(device='cpu')
        state["ocr_engine"] = rec
        state["ocr_ready"]  = True
        print("  [OK] PaddleOCR listo.")
    except ImportError:
        print("  [!] PaddleOCR no instalado: pip install paddlepaddle paddleocr")
        state["ocr_ready"] = False
    except Exception as e:
        print(f"  [!] Error PaddleOCR: {e}")
        state["ocr_ready"] = False


def clean_number(raw: str) -> str:
    t = raw.upper()
    for a, b in [('O','0'),('D','0'),('Q','0'),('I','1'),('L','1'),
                 ('|','1'),('S','5'),('B','8'),('G','6'),(',','.')]:
        t = t.replace(a, b)
    return re.sub(r'[^0-9.\-]', '', t)


def run_ocr(crops_by_cls: dict) -> dict:
    """Corre PaddleOCR v3 TextRecognition sobre los crops filtrados."""
    if not state["ocr_ready"]:
        return {}
    rec = state["ocr_engine"]
    results = {}
    for cls_id, crops in crops_by_cls.items():
        texts = []
        for crop in crops:
            filt = apply_filter(crop, state["params"][CLASS_NAMES[cls_id]])
            try:
                # API v3: predict() devuelve lista de dicts con rec_text y rec_score
                res = list(rec.predict(filt))
                if res:
                    raw  = res[0].get('rec_text', '')
                    conf = res[0].get('rec_score', 0.0)
                    texts.append((clean_number(str(raw)), float(conf)))
                else:
                    texts.append(("--", 0.0))
            except Exception:
                texts.append(("ERR", 0.0))
        results[cls_id] = texts
    return results


def blur_val(idx):   return BLUR_MAP[min(idx, len(BLUR_MAP)-1)]
def block_val(idx):  return (idx * 2 + 3) | 1
def C_val(idx):      return idx - 20
def clip_val(idx):   return max(0.5, idx / 2.0)


def apply_filter(bgr: np.ndarray, p: dict) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if p["clahe"]:
        cl   = cv2.createCLAHE(clipLimit=clip_val(p["clahe_clip"]), tileGridSize=(4,4))
        gray = cl.apply(gray)
    k = blur_val(p["blur"])
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    tt = p["thresh_type"]
    if tt == 0:
        _, binary = cv2.threshold(gray, p["thresh_val"], 255, cv2.THRESH_BINARY)
    elif tt == 1:
        _, binary = cv2.threshold(gray, p["thresh_val"], 255, cv2.THRESH_BINARY_INV)
    elif tt == 2:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif tt == 3:
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                     cv2.THRESH_BINARY, block_val(p["adapt_block"]), C_val(p["adapt_C"]))
    else:
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                     cv2.THRESH_BINARY, block_val(p["adapt_block"]), C_val(p["adapt_C"]))
    mo = p["morph_op"]
    if mo > 0:
        ker    = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1,p["morph_k"]),)*2)
        binary = cv2.morphologyEx(binary, MORPH_OPS[mo], ker)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _put(img, text, x, y, color, scale=0.40, thick=1):
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    cv2.rectangle(img, (x-1, y-th-bl-1), (x+tw+3, y+1), (0,0,0), -1)
    cv2.putText(img, text, (x+1, y-bl), FONT, scale, color, thick, cv2.LINE_AA)


def _build_panel(w, entry, active):
    h     = 78
    panel = np.full((h, w, 3), COL_PANEL, dtype=np.uint8)
    idx   = state["img_idx"]
    total = len(state["images"])
    p     = state["params"][active]
    tt    = THRESH_LABELS[p["thresh_type"]]
    mo    = MORPH_LABELS[p["morph_op"]]
    cl    = f"CLAHE clip={clip_val(p['clahe_clip'])}" if p["clahe"] else "CLAHE off"
    cv2.putText(panel, f"[{idx+1}/{total}]  {entry['path'].name}",
                (8, 20), FONT, 0.50, COL_YELLOW, 1, cv2.LINE_AA)
    summary = (f"[{active}]  Blur={blur_val(p['blur'])}px  Thresh={tt}"
               f"(val={p['thresh_val']})  Block={block_val(p['adapt_block'])}"
               f"  C={C_val(p['adapt_C'])}  {cl}  Morph={mo}({p['morph_k']})")
    c = COL_GREEN if active == "roi_green" else COL_ORANGE
    cv2.putText(panel, summary, (8, 42), FONT, 0.37, c, 1, cv2.LINE_AA)
    ocr_st = "[OK] PaddleOCR listo" if state["ocr_ready"] else "[!] pip install paddlepaddle paddleocr"
    cv2.putText(panel, ocr_st, (8, 60), FONT, 0.37,
                COL_GOOD if state["ocr_ready"] else COL_RED, 1, cv2.LINE_AA)
    ctrl = "G=verde  W=blanco  O=OCR  N/P=nav  R=reset  S=guardar  Q=salir"
    (tw, _), _ = cv2.getTextSize(ctrl, FONT, 0.37, 1)
    cv2.putText(panel, ctrl, (w - tw - 8, 60), FONT, 0.37, COL_DIM, 1, cv2.LINE_AA)
    return panel


# ── BUILD FRAME ───────────────────────────────────────────────────────────────

def build_frame() -> np.ndarray:
    images = state["images"]
    if not images:
        return np.zeros((300, 800, 3), dtype=np.uint8)
    entry    = images[state["img_idx"]]
    idx      = state["img_idx"]
    active   = state["cls_active"]
    ocr_r    = state["ocr_results"].get(idx, {})
    cell_h   = CROP_H + OCR_H
    rows     = []

    for cls_id, cls_name, p_cls in [(1, "roi_green", state["params"]["roi_green"]),
                                     (0, "roi_white", state["params"]["roi_white"])]:
        crops  = entry["crops"].get(cls_id, []) or [np.zeros((CROP_H, CROP_W, 3), dtype=np.uint8)] * 3
        ocr_texts = ocr_r.get(cls_id, [("", 0.0)] * len(crops))
        border_c  = COL_YELLOW if cls_name == active else COL_DIM
        row_cells = []

        for i, crop in enumerate(crops[:3]):
            orig = cv2.resize(crop, (CROP_W, CROP_H))
            filt = cv2.resize(apply_filter(crop, p_cls), (CROP_W, CROP_H))
            cv2.rectangle(orig, (0,0), (CROP_W-1, CROP_H-1), border_c, 2)
            cv2.rectangle(filt, (0,0), (CROP_W-1, CROP_H-1), border_c, 2)
            _put(orig, cls_name[:9], 3, 14, COL_GREEN if cls_id==1 else COL_ORANGE, 0.38)
            _put(filt, "FILTRADO", 3, 14, border_c, 0.38)

            # Banda OCR bajo el crop filtrado
            ocr_band = np.full((OCR_H, CROP_W, 3), (30, 30, 30), dtype=np.uint8)
            if i < len(ocr_texts) and ocr_texts[i][0]:
                txt, conf = ocr_texts[i]
                color = COL_GOOD if conf>0.7 else COL_YELLOW if conf>0.4 else COL_RED
                cv2.putText(ocr_band, f"{txt} ({conf:.2f})", (4, OCR_H-5),
                            FONT, 0.45, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(ocr_band, "O=OCR", (4, OCR_H-5),
                            FONT, 0.38, COL_DIM, 1, cv2.LINE_AA)

            orig_col = np.vstack([orig, np.full((OCR_H, CROP_W, 3), (20,20,20), dtype=np.uint8)])
            filt_col = np.vstack([filt, ocr_band])
            row_cells.append(np.hstack([orig_col, filt_col]))

        while len(row_cells) < 3:
            row_cells.append(np.zeros((cell_h, CROP_W*2, 3), dtype=np.uint8))

        rows.append(np.hstack(row_cells))

    sep  = np.full((3, rows[0].shape[1], 3), 50, dtype=np.uint8)
    grid = np.vstack([rows[0], sep, rows[1]])
    return np.vstack([grid, _build_panel(grid.shape[1], entry, active)])


# ── TRACKBARS ─────────────────────────────────────────────────────────────────

TB_KEYS = [
    ("Blur       ", "blur",        1, 5),
    ("ThreshType ", "thresh_type", 0, 4),
    ("ThreshVal  ", "thresh_val",  128, 255),
    ("AdaptBlock ", "adapt_block", 4, 24),
    ("AdaptC     ", "adapt_C",     22, 40),
    ("CLAHE      ", "clahe",       0, 1),
    ("CLAHEClip  ", "clahe_clip",  4, 40),
    ("MorphOp    ", "morph_op",    0, 4),
    ("MorphK     ", "morph_k",     2, 10),
]

def _noop(v): pass

def create_trackbars():
    for name, _, default, mx in TB_KEYS:
        cv2.createTrackbar(name, WINDOW, default, mx, _noop)

def read_trackbars() -> dict:
    return {key: cv2.getTrackbarPos(name, WINDOW) for name, key, _, _ in TB_KEYS}

def push_trackbars(p: dict):
    for name, key, _, _ in TB_KEYS:
        cv2.setTrackbarPos(name, WINDOW, p[key])


# ── PRECOMPUTE ────────────────────────────────────────────────────────────────

def collect_pairs():
    exts  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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


def precompute(pairs, model) -> list:
    images = []
    for i, (img_path, _) in enumerate(pairs):
        print(f"  Cargando [{i+1}/{len(pairs)}] {img_path.name}...", end="\r")
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        results = model(img_path, conf=CONF_THRESH, verbose=False)[0]
        by_cls  = {0: [], 1: []}
        for box in results.boxes:
            cls  = int(box.cls)
            xyxy = tuple(map(int, box.xyxy[0].tolist()))
            by_cls[cls].append(xyxy)
        crops = {0: [], 1: []}
        for cls in (0, 1):
            by_cls[cls].sort(key=lambda b: b[1])
            for b in by_cls[cls]:
                c = bgr[b[1]:b[3], b[0]:b[2]]
                if c.size > 0:
                    crops[cls].append(c)
        images.append({"path": img_path, "crops": crops})
    print(f"\n  [OK] {len(images)} imagenes listas.\n")
    return images


# ── GUARDAR CONFIG ────────────────────────────────────────────────────────────

def save_config():
    cfg = {}
    for cls_name, p in state["params"].items():
        cfg[cls_name] = {
            "blur_k":         blur_val(p["blur"]),
            "thresh_type":    THRESH_LABELS[p["thresh_type"]],
            "thresh_val":     p["thresh_val"],
            "adaptive_block": block_val(p["adapt_block"]),
            "adaptive_C":     C_val(p["adapt_C"]),
            "clahe":          bool(p["clahe"]),
            "clahe_clip":     clip_val(p["clahe_clip"]),
            "morph_op":       MORPH_LABELS[p["morph_op"]],
            "morph_k":        p["morph_k"],
        }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\n  [OK] Guardado -> {OUTPUT_JSON}")
    for cls_name, v in cfg.items():
        print(f"\n  {cls_name}:")
        for k, val in v.items():
            print(f"    {k:18s}: {val}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
# FIX: usamos waitKeyEx (no waitKey) para capturar flechas en Linux sin & 0xFF
# FIX: flechas usan sus códigos reales (65361-65364), sin conflicto con S/Q

def main():
    os.system("clear")
    print("=" * 60)
    print("  FILTER TUNER + PaddleOCR")
    print("=" * 60)

    if not BEST_PT.exists():
        print(f"  [ERROR] No se encontro:\n  {BEST_PT}")
        sys.exit(1)

    print("  Cargando YOLO...")
    model = YOLO(str(BEST_PT))
    init_paddle()

    pairs = collect_pairs()
    if not pairs:
        print("  [ERROR] Sin imagenes etiquetadas en dataset/")
        sys.exit(1)

    random.seed(RANDOM_SEED)
    sample           = random.sample(pairs, min(N_SAMPLES, len(pairs)))
    state["images"]  = precompute(sample, model)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    create_trackbars()
    push_trackbars(state["params"]["roi_green"])

    while True:
        # Leer trackbars y actualizar params de la clase activa
        state["params"][state["cls_active"]] = read_trackbars()
        cv2.imshow(WINDOW, build_frame())

        # FIX: waitKeyEx captura el código completo (necesario para flechas en Linux)
        key = cv2.waitKeyEx(30)
        k   = key & 0xFF   # byte bajo para teclas ASCII normales

        # Navegación: flechas (códigos Linux: 65361=izq, 65363=der)
        # Usamos el código completo (key, no k) para no confundir con ASCII
        if key in (65363, 65364):    # flecha der / abajo
            if state["img_idx"] < len(state["images"]) - 1:
                state["img_idx"] += 1

        elif key in (65361, 65362):  # flecha izq / arriba
            if state["img_idx"] > 0:
                state["img_idx"] -= 1

        # Teclas ASCII normales (sin conflicto con flechas)
        elif k in (ord('n'), ord('N')):
            if state["img_idx"] < len(state["images"]) - 1:
                state["img_idx"] += 1

        elif k in (ord('p'), ord('P')):
            if state["img_idx"] > 0:
                state["img_idx"] -= 1

        elif k in (ord('g'), ord('G')):
            state["cls_active"] = "roi_green"
            push_trackbars(state["params"]["roi_green"])

        elif k in (ord('w'), ord('W')):
            state["cls_active"] = "roi_white"
            push_trackbars(state["params"]["roi_white"])

        elif k in (ord('o'), ord('O')):
            idx   = state["img_idx"]
            crops = state["images"][idx]["crops"]
            print("\n  Ejecutando PaddleOCR...", end="", flush=True)
            t0 = time.time()
            state["ocr_results"][idx] = run_ocr(crops)
            print(f" {(time.time()-t0)*1000:.0f} ms")
            # Imprimir resultados en consola
            for cls_id, texts in state["ocr_results"][idx].items():
                print(f"  {CLASS_NAMES[cls_id]}:")
                for i, (txt, conf) in enumerate(texts):
                    tag = "OK" if conf>0.7 else "~" if conf>0.4 else "?"
                    print(f"    [{i}] {tag} '{txt}'  conf={conf:.3f}")

        elif k in (ord('r'), ord('R')):
            cls = state["cls_active"]
            state["params"][cls] = dict(DEFAULT_PARAMS[cls])
            push_trackbars(state["params"][cls])
            print(f"  [Reset {cls}]")

        elif k in (ord('s'), ord('S')):
            save_config()

        elif k in (ord('q'), ord('Q'), 27):   # Q o ESC
            break

        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cv2.destroyAllWindows()
    print("\n  [Cerrado]\n")


if __name__ == "__main__":
    main()
