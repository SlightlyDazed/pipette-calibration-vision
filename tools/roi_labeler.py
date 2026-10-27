"""
roi_labeler.py — Etiquetado manual de ROIs para entrenamiento YOLO
==================================================================
CONTROLES:
    Scroll ↑/↓          → zoom in/out (centrado en cursor)
    Flechas ←↑↓→         → pan (mover vista)
    Click izq + drag     → dibuja un ROI
    W                    → confirma ROI como fondo BLANCO
    G                    → confirma ROI como fondo VERDE
    Z                    → deshace el último ROI
    R                    → cancela el rect actual
    F                    → reset zoom (fit to screen)
    S                    → guarda y avanza
    Q                    → salta imagen
"""

import cv2
import signal
import numpy as np
import os, sys, shutil, json
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATABASE_DIR  = BASE_DIR / "database"
DATASET_DIR   = BASE_DIR / "dataset"
IMAGES_DIR    = DATASET_DIR / "images"
LABELS_DIR    = DATASET_DIR / "labels"
PROGRESS_FILE = BASE_DIR / "labeling_progress.json"

CLASS_NAMES = {"white": (0, "roi_white"), "green": (1, "roi_green")}

# Colores BGR
COL_WHITE   = (255, 255, 255)
COL_GREEN   = (0, 220, 80)
COL_PENDING = (0, 220, 255)
COL_DRAW    = (180, 180, 180)
COL_HUD_BG  = (30, 30, 30)
COL_HUD     = (200, 200, 200)

SCREEN_W = 1920
SCREEN_H = 1080

ZOOM_STEP = 1.20
PAN_STEP  = 80     # pixeles de imagen por pulsación de flecha

WIN = "ROI Labeler"

# Referencia global al progreso activo (para el handler de Ctrl+C)
_active_prog = {}
_active_img_path = None
_active_result = None

# Arrow key codes (OpenCV special keys)
KEY_LEFT  = 81
KEY_UP    = 82
KEY_RIGHT = 83
KEY_DOWN  = 84

# ── ESTADO GLOBAL ─────────────────────────────────────────────────────────────
st = {
    "img": None, "img_h": 0, "img_w": 0,
    "zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0,
    # Dibujo ROI (SOLO click izquierdo)
    "drawing": False,
    "draw_sx0": 0, "draw_sy0": 0,  # punto inicio en coords PANTALLA
    "draw_sx1": 0, "draw_sy1": 0,  # punto actual en coords PANTALLA
    "pending": None,               # (ix0,iy0,ix1,iy1) rect en coords IMAGEN
    "rois": [],
}


# ── TRANSFORMS ────────────────────────────────────────────────────────────────
def s2i(sx, sy):
    """Pantalla → imagen."""
    return st["pan_x"] + sx / st["zoom"], st["pan_y"] + sy / st["zoom"]

def i2s(ix, iy):
    """Imagen → pantalla."""
    return int((ix - st["pan_x"]) * st["zoom"]), int((iy - st["pan_y"]) * st["zoom"])


# ── VIEWPORT ─────────────────────────────────────────────────────────────────
def fit_to_screen():
    iw, ih = st["img_w"], st["img_h"]
    st["zoom"] = min(SCREEN_W / iw, SCREEN_H / ih, 1.0)
    view_w = SCREEN_W / st["zoom"]
    view_h = SCREEN_H / st["zoom"]
    st["pan_x"] = -(view_w - iw) / 2
    st["pan_y"] = -(view_h - ih) / 2


# ── MOUSE: solo scroll + click izquierdo ──────────────────────────────────────
def mouse_cb(event, x, y, flags, param):

    # ── SCROLL → zoom centrado en cursor ─────────────────────────────
    if event == cv2.EVENT_MOUSEWHEEL:
        factor = ZOOM_STEP if flags > 0 else 1.0 / ZOOM_STEP
        new_zoom = max(0.1, min(20.0, st["zoom"] * factor))
        if new_zoom == st["zoom"]:
            return
        ix, iy = s2i(x, y)
        st["zoom"] = new_zoom
        st["pan_x"] = ix - x / new_zoom
        st["pan_y"] = iy - y / new_zoom
        if not st["drawing"]:
            redraw()
        else:
            # Si estamos dibujando, recalcular el rect visual
            _update_draw_visual()
        return

    # ── CLICK IZQUIERDO → dibujar ROI ────────────────────────────────
    if event == cv2.EVENT_LBUTTONDOWN:
        st["drawing"]  = True
        st["pending"]  = None
        st["draw_sx0"] = x
        st["draw_sy0"] = y
        st["draw_sx1"] = x
        st["draw_sy1"] = y
        # Guardar coords de imagen del punto de inicio
        st["_draw_ix0"], st["_draw_iy0"] = s2i(x, y)

    elif event == cv2.EVENT_MOUSEMOVE and st["drawing"]:
        st["draw_sx1"] = x
        st["draw_sy1"] = y
        _update_draw_visual()

    elif event == cv2.EVENT_LBUTTONUP and st["drawing"]:
        st["drawing"] = False
        st["draw_sx1"] = x
        st["draw_sy1"] = y
        # Convertir ambos puntos a coords de imagen
        ix0, iy0 = st["_draw_ix0"], st["_draw_iy0"]
        ix1, iy1 = s2i(x, y)
        rx0 = min(ix0, ix1); ry0 = min(iy0, iy1)
        rx1 = max(ix0, ix1); ry1 = max(iy0, iy1)
        # Ignorar clicks accidentales
        if (rx1 - rx0) > 3 and (ry1 - ry0) > 3:
            st["pending"] = (rx0, ry0, rx1, ry1)
        redraw()


def _update_draw_visual():
    """Redibuja solo el rect de dibujo (rápido, sin re-renderizar todo)."""
    redraw()


# ── REDRAW ────────────────────────────────────────────────────────────────────
def redraw():
    iw, ih = st["img_w"], st["img_h"]
    zoom   = st["zoom"]

    vx0 = st["pan_x"]
    vy0 = st["pan_y"]
    vx1 = vx0 + SCREEN_W / zoom
    vy1 = vy0 + SCREEN_H / zoom

    cx0 = int(max(0, vx0));  cy0 = int(max(0, vy0))
    cx1 = int(min(iw, vx1)); cy1 = int(min(ih, vy1))

    canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

    if cx1 > cx0 and cy1 > cy0:
        crop = st["img"][cy0:cy1, cx0:cx1]
        out_w = max(1, min(int((cx1 - cx0) * zoom), SCREEN_W))
        out_h = max(1, min(int((cy1 - cy0) * zoom), SCREEN_H))
        interp = cv2.INTER_LINEAR if zoom >= 1.0 else cv2.INTER_AREA
        scaled = cv2.resize(crop, (out_w, out_h), interpolation=interp)
        dx = max(0, int((cx0 - vx0) * zoom))
        dy = max(0, int((cy0 - vy0) * zoom))
        ex = min(dx + out_w, SCREEN_W)
        ey = min(dy + out_h, SCREEN_H)
        canvas[dy:ey, dx:ex] = scaled[:ey-dy, :ex-dx]

    # ROIs confirmados
    for roi in st["rois"]:
        _draw_roi_box(canvas, roi)

    # ROI pendiente (amarillo)
    if st["pending"]:
        px0, py0, px1, py1 = st["pending"]
        sp0 = i2s(px0, py0); sp1 = i2s(px1, py1)
        cv2.rectangle(canvas, sp0, sp1, COL_PENDING, 2)
        cv2.putText(canvas, "W=blanco  G=verde", (sp0[0]+4, max(sp0[1]-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_PENDING, 2, cv2.LINE_AA)

    # ROI en construcción
    elif st["drawing"]:
        sx0, sy0 = st["draw_sx0"], st["draw_sy0"]
        sx1, sy1 = st["draw_sx1"], st["draw_sy1"]
        p0 = (min(sx0,sx1), min(sy0,sy1))
        p1 = (max(sx0,sx1), max(sy0,sy1))
        cv2.rectangle(canvas, p0, p1, COL_DRAW, 2)

    # HUD barra inferior
    cv2.rectangle(canvas, (0, SCREEN_H-30), (SCREEN_W, SCREEN_H), COL_HUD_BG, -1)
    hud = f"Zoom {st['zoom']:.1f}x | ROIs {len(st['rois'])} | Scroll=zoom  Flechas=pan  F=fit  W/G=clase  Z=deshacer  S=guardar  Q=saltar"
    cv2.putText(canvas, hud, (10, SCREEN_H-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COL_HUD, 1, cv2.LINE_AA)

    cv2.imshow(WIN, canvas)


def _draw_roi_box(canvas, roi):
    x1, y1, x2, y2 = roi["rect"]
    sp0 = i2s(x1, y1); sp1 = i2s(x2, y2)
    color = COL_WHITE if roi["class_id"] == 0 else COL_GREEN
    cv2.rectangle(canvas, sp0, sp1, color, 2)
    label = roi["name"]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    bx, by = sp0[0], max(sp0[1] - th - 6, 0)
    cv2.rectangle(canvas, (bx, by), (bx+tw+4, by+th+6), color, -1)
    cv2.putText(canvas, label, (bx+2, by+th+2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)


# ── CONFIRMAR ─────────────────────────────────────────────────────────────────
def confirm_roi(key):
    if st["pending"] is None:
        print("  [!] Primero dibuja un ROI")
        return
    cid, cname = CLASS_NAMES[key]
    n = sum(1 for r in st["rois"] if r["class_id"] == cid)
    name = f"{cname}_{n}"
    iw, ih = st["img_w"], st["img_h"]
    r = st["pending"]
    rect = (max(0,int(r[0])), max(0,int(r[1])), min(iw,int(r[2])), min(ih,int(r[3])))
    st["rois"].append({"class_id": cid, "name": name, "rect": rect})
    st["pending"] = None
    redraw()
    print(f"  ✔ [{name}]  {rect}")


# ── PROCESAR IMAGEN ───────────────────────────────────────────────────────────
def _sigint_handler(sig, frame):
    """Ctrl+C: guarda la imagen actual si tiene ROIs y sale limpiamente."""
    global _active_prog, _active_img_path
    print("\n\n  [Ctrl+C] Guardando progreso antes de salir...")
    # Guardar ROIs de la imagen actual si los hay
    if _active_img_path and st["rois"]:
        img_path = _active_img_path
        tmp = cv2.imread(str(img_path))
        if tmp is not None:
            ih, iw = tmp.shape[:2]
            lbl = LABELS_DIR / (img_path.stem + ".txt")
            lines = rois_to_yolo(st["rois"], iw, ih)
            lbl.write_text("\n".join(lines) + "\n")
            dest = IMAGES_DIR / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)
            _active_prog[img_path.name] = {
                "img_w": iw, "img_h": ih,
                "rois": [{"class_id": r["class_id"], "name": r["name"],
                          "rect": list(r["rect"])} for r in st["rois"]]
            }
            save_progress(_active_prog)
            print(f"  ✔ Guardados {len(st['rois'])} ROIs de {img_path.name}")
    elif _active_prog:
        save_progress(_active_prog)
    cv2.destroyAllWindows()
    print("  ✅ Progreso guardado. Puedes reanudar ejecutando el script de nuevo.")
    sys.exit(0)


def process_image(img_path, existing):
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    ih, iw = img.shape[:2]
    st["img"], st["img_h"], st["img_w"] = img, ih, iw
    st["rois"] = [dict(r) for r in existing]
    st["pending"] = None
    st["drawing"] = False
    fit_to_screen()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, SCREEN_W, SCREEN_H)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WIN, mouse_cb)
    redraw()
    show_status(img_path.name)

    while True:
        key = cv2.waitKeyEx(20)  # waitKeyEx para capturar flechas
        k = key & 0xFF

        if k in (ord('q'), ord('Q')):
            return None
        if k in (ord('s'), ord('S')):
            print(f"  → Guardados {len(st['rois'])} ROIs")
            return st["rois"]
        if k in (ord('w'), ord('W')):
            confirm_roi("white"); show_status(img_path.name)
        if k in (ord('g'), ord('G')):
            confirm_roi("green"); show_status(img_path.name)
        if k in (ord('z'), ord('Z')) and st["rois"]:
            rm = st["rois"].pop()
            print(f"  ↩ Deshecho: {rm['name']}")
            redraw(); show_status(img_path.name)
        if k in (ord('r'), ord('R')):
            st["pending"] = None; st["drawing"] = False; redraw()
        if k in (ord('f'), ord('F')):
            fit_to_screen(); redraw()

        # Flechas → pan
        if key in (KEY_LEFT, 65361):
            st["pan_x"] -= PAN_STEP; redraw()
        if key in (KEY_RIGHT, 65363):
            st["pan_x"] += PAN_STEP; redraw()
        if key in (KEY_UP, 65362):
            st["pan_y"] -= PAN_STEP; redraw()
        if key in (KEY_DOWN, 65364):
            st["pan_y"] += PAN_STEP; redraw()

        try:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                return st["rois"]
        except cv2.error:
            return st["rois"]


# ── STATUS ────────────────────────────────────────────────────────────────────
def show_status(fname):
    os.system("clear")
    print("=" * 62)
    print("  🎯 ROI LABELER — YOLO Dataset Builder")
    print("=" * 62)
    print(f"  Imagen : {fname}")
    print(f"  ROIs   : {len(st['rois'])}")
    for r in st["rois"]:
        icon = "⬜" if r["class_id"] == 0 else "🟩"
        print(f"    {icon} {r['name']}  {r['rect']}")
    print()
    print("  MOUSE:")
    print("    Scroll ↑↓       → zoom in/out")
    print("    Click + drag    → dibuja ROI")
    print("  TECLADO:")
    print("    Flechas          → mover vista")
    print("    W / G            → blanco / verde")
    print("    F                → reset zoom")
    print("    Z                → deshacer")
    print("    S                → guardar y avanzar")
    print("    Q                → saltar")
    print("=" * 62)


# ── YOLO ──────────────────────────────────────────────────────────────────────
def rois_to_yolo(rois, iw, ih):
    lines = []
    for r in rois:
        x1, y1, x2, y2 = r["rect"]
        lines.append(f"{r['class_id']} {(x1+x2)/2/iw:.6f} {(y1+y2)/2/ih:.6f} {(x2-x1)/iw:.6f} {(y2-y1)/ih:.6f}")
    return lines


# ── HELPERS ───────────────────────────────────────────────────────────────────
def setup_dirs():
    for d in [IMAGES_DIR, LABELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def write_data_yaml():
    (DATASET_DIR / "data.yaml").write_text(f"""path: {DATASET_DIR.resolve()}
train: images/train
val:   images/val
nc: 2
names:
  0: roi_white
  1: roi_green
""")

def save_progress(p):
    PROGRESS_FILE.write_text(json.dumps(p, indent=2))

def load_progress():
    return json.loads(PROGRESS_FILE.read_text()) if PROGRESS_FILE.exists() else {}


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    global _active_prog, _active_img_path
    signal.signal(signal.SIGINT, _sigint_handler)

    EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = sorted([p for p in DATABASE_DIR.iterdir()
                     if p.is_file() and p.suffix.lower() in EXTS])
    if not images:
        print(f"[!] No hay imágenes en {DATABASE_DIR}"); sys.exit(1)

    setup_dirs()
    prog = load_progress()
    done = set(prog.keys())
    pending = [i for i in images if i.name not in done]

    print(f"\n  🎯 Total: {len(images)}  Hechas: {len(done)}  Pendientes: {len(pending)}")
    if not pending:
        print("  ✅ ¡Todo etiquetado!"); _finalize(prog); return

    input("  ENTER para comenzar...")
    n = 0

    for img_path in pending:
        _active_img_path = img_path   # ← Ctrl+C sabe qué imagen es
        _active_prog = prog           # ← Ctrl+C sabe el progreso actual
        existing = [{"class_id": r["class_id"], "name": r["name"], "rect": tuple(r["rect"])}
                     for r in prog.get(img_path.name, {}).get("rois", [])]
        result = process_image(img_path, existing)
        if result is None:
            continue

        dest = IMAGES_DIR / img_path.name
        if not dest.exists(): shutil.copy2(img_path, dest)
        tmp = cv2.imread(str(img_path))
        if tmp is None: continue
        ih, iw = tmp.shape[:2]
        lbl = LABELS_DIR / (img_path.stem + ".txt")
        lines = rois_to_yolo(result, iw, ih)
        lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
        prog[img_path.name] = {"img_w": iw, "img_h": ih,
            "rois": [{"class_id": r["class_id"], "name": r["name"], "rect": list(r["rect"])} for r in result]}
        save_progress(prog)
        n += 1

    cv2.destroyAllWindows()
    print(f"\n  ✅ Etiquetadas: {n}")
    _finalize(prog)

def _finalize(prog):
    write_data_yaml()
    nw = sum(sum(1 for r in v.get("rois",[]) if r["class_id"]==0) for v in prog.values())
    ng = sum(sum(1 for r in v.get("rois",[]) if r["class_id"]==1) for v in prog.values())
    print(f"  📊 roi_white: {nw}  roi_green: {ng}")
    print(f"  📁 {DATASET_DIR}\n  ⚡ Siguiente: python train_yolo.py")

if __name__ == "__main__":
    main()
