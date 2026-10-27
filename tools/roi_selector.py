"""
roi_selector.py — Herramienta interactiva para marcar ROIs con el mouse
=======================================================================
USO:
    python ocr_work/roi_selector.py

CONTROLES:
    Click + drag  → dibuja un ROI en la zona que quieres leer
    ENTER         → confirma el ROI dibujado y te pide su nombre
    R             → rehace el último ROI (si te equivocaste al dibujar)
    Z             → deshace el último ROI guardado
    S             → guarda todos los ROIs del archivo actual y pasa al siguiente
    Q             → sale sin guardar el archivo actual

Al terminar con todas las imágenes, los ROIs se guardan en:
    ocr_work/rois_config.json

Ese JSON lo lee script_lcd.py automáticamente.
"""

import cv2
import json
import os
import sys

# ── CONFIG ────────────────────────────────────────────────────────────────────
WORK_TEST_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "work_test"
)
OUTPUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "rois_config.json")

# Escala de visualización (para que la imagen entre en pantalla)
DISPLAY_SCALE = 0.75

# Colores para los ROIs dibujados (BGR)
COLORS = [
    (0, 255, 0),    # verde
    (0, 200, 255),  # naranja
    (255, 100, 0),  # azul
    (0, 0, 255),    # rojo
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cian
    (100, 255, 100),
    (100, 100, 255),
]

# ── ESTADO GLOBAL del mouse ───────────────────────────────────────────────────
state = {
    "drawing": False,
    "x0": 0, "y0": 0,
    "x1": 0, "y1": 0,
    "confirmed_rect": None,   # (x0,y0,x1,y1) en coords de DISPLAY
    "saved_rois": [],         # lista de {"name":..., "roi":[x1,y1,x2,y2]}
    "base_img": None,         # imagen original sin anotaciones
    "display_img": None,      # imagen con anotaciones actuales
    "scale": DISPLAY_SCALE,
}


def mouse_cb(event, x, y, flags, param):
    s = state
    if event == cv2.EVENT_LBUTTONDOWN:
        s["drawing"] = True
        s["x0"], s["y0"] = x, y
        s["x1"], s["y1"] = x, y
        s["confirmed_rect"] = None

    elif event == cv2.EVENT_MOUSEMOVE and s["drawing"]:
        s["x1"], s["y1"] = x, y
        redraw()

    elif event == cv2.EVENT_LBUTTONUP:
        s["drawing"] = False
        s["x1"], s["y1"] = x, y
        s["confirmed_rect"] = (
            min(s["x0"], s["x1"]), min(s["y0"], s["y1"]),
            max(s["x0"], s["x1"]), max(s["y0"], s["y1"])
        )
        redraw()


def redraw():
    """Redibuja la imagen con todos los ROIs guardados + el actual en progreso."""
    s = state
    img = s["base_img"].copy()

    # ROIs ya guardados
    for i, roi_info in enumerate(s["saved_rois"]):
        rx1, ry1, rx2, ry2 = [int(v * s["scale"]) for v in roi_info["roi"]]
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, 2)
        cv2.putText(img, roi_info["name"], (rx1 + 4, ry1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # ROI en progreso (mientras se arrastra)
    if s["drawing"] or s["confirmed_rect"]:
        if s["drawing"]:
            x0, y0, x1, y1 = s["x0"], s["y0"], s["x1"], s["y1"]
        else:
            x0, y0, x1, y1 = s["confirmed_rect"]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)

    s["display_img"] = img
    cv2.imshow("ROI Selector", img)


def display_to_original(rect, scale):
    """Convierte coordenadas de display → coordenadas originales."""
    x0, y0, x1, y1 = rect
    return (
        int(x0 / scale), int(y0 / scale),
        int(x1 / scale), int(y1 / scale)
    )


def print_help(fname, n_saved):
    os.system("clear")
    print("=" * 58)
    print(f"  ROI SELECTOR — {fname}")
    print("=" * 58)
    print(f"  ROIs guardados hasta ahora: {n_saved}")
    for i, r in enumerate(state["saved_rois"]):
        print(f"    [{i}] {r['name']:15s} → {r['roi']}")
    print()
    print("  CONTROLES:")
    print("    Click + drag → dibuja un ROI")
    print("    ENTER        → confirma y nombra el ROI actual")
    print("    R            → rehace el rect actual (borra sin guardar)")
    print("    Z            → deshace el último ROI guardado")
    print("    S            → guarda y pasa a la siguiente imagen")
    print("    Q            → salta esta imagen sin guardar")
    print("=" * 58)


def ask_name(suggestion: str = "") -> tuple[str, int]:
    """Pide el nombre del ROI y sus decimales esperados en consola."""
    prompt = f"  Nombre del ROI [{suggestion}]: "
    name = input(prompt).strip()
    if not name:
        name = suggestion

    dec_prompt = f"  Decimales esperados para '{name}' (0/1/2) [2]: "
    dec_str = input(dec_prompt).strip()
    try:
        decimals = int(dec_str)
    except ValueError:
        decimals = 2

    return name, decimals


def process_image(image_path: str, existing_rois: list) -> list | None:
    """
    Muestra la imagen y permite al usuario marcar ROIs.

    Returns:
        Lista de {"name": str, "roi": [x1,y1,x2,y2]} o None si se saltó.
    """
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"[ERROR] No se pudo leer: {image_path}")
        return []

    h, w = img_orig.shape[:2]
    scale = DISPLAY_SCALE

    # Reducir si es muy grande
    disp_w = int(w * scale)
    disp_h = int(h * scale)
    base = cv2.resize(img_orig, (disp_w, disp_h))

    fname = os.path.basename(image_path)

    # Cargar ROIs previos si existen
    state["saved_rois"] = [dict(r) for r in existing_rois]
    state["base_img"] = base.copy()
    state["display_img"] = base.copy()
    state["confirmed_rect"] = None
    state["drawing"] = False
    state["scale"] = scale

    cv2.namedWindow("ROI Selector", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("ROI Selector", mouse_cb)

    # Dibujar ROIs previos al iniciar
    redraw()

    print_help(fname, len(state["saved_rois"]))

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("  [Saltando imagen]")
            return None

        elif key == ord('s') or key == ord('S'):
            print(f"  [Guardando {len(state['saved_rois'])} ROIs para {fname}]")
            return state["saved_rois"]

        elif key == ord('r') or key == ord('R'):
            state["confirmed_rect"] = None
            state["drawing"] = False
            redraw()
            print("  [ROI actual borrado]")

        elif key == ord('z') or key == ord('Z'):
            if state["saved_rois"]:
                removed = state["saved_rois"].pop()
                redraw()
                print(f"  [Deshecho: {removed['name']}]")
            else:
                print("  [No hay ROIs para deshacer]")

        elif key == 13:  # ENTER
            rect = state["confirmed_rect"]
            if rect is None:
                print("  [Primero dibuja un ROI con el mouse]")
                continue

            # Sugerir nombre basado en cuántos hay ya
            n = len(state["saved_rois"])
            suggestions = ["main", "min", "max", "hum", "temp", "pres",
                           "roi6", "roi7", "roi8"]
            suggestion = suggestions[n] if n < len(suggestions) else f"roi{n}"

            name, decimals = ask_name(suggestion)
            if not name:
                print("  [Nombre vacío, cancelado]")
                continue

            orig_rect = display_to_original(rect, scale)
            state["saved_rois"].append({
                "name":     name,
                "roi":      list(orig_rect),
                "decimals": decimals
            })
            state["confirmed_rect"] = None
            redraw()
            print_help(fname, len(state["saved_rois"]))

        # Cerrar ventana con X
        if cv2.getWindowProperty("ROI Selector", cv2.WND_PROP_VISIBLE) < 1:
            return state["saved_rois"]


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    folder = WORK_TEST_FOLDER
    exts = ('.jpg', '.jpeg', '.png', '.bmp')

    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ])

    if not files:
        print(f"[!] No hay imágenes en: {folder}")
        sys.exit(1)

    # Cargar config existente si la hay
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON) as f:
            config = json.load(f)
        print(f"[i] Cargando config existente: {OUTPUT_JSON}")
    else:
        config = {}

    print(f"\n  {len(files)} imágenes encontradas en work_test/")
    print(f"  Los ROIs se guardarán en: {OUTPUT_JSON}\n")
    input("  Presiona ENTER para empezar...")

    for img_path in files:
        fname = os.path.basename(img_path)
        existing = config.get(fname, [])
        result = process_image(img_path, existing)
        if result is not None:
            config[fname] = result
            # Guardar después de cada imagen por si se cierra el programa
            with open(OUTPUT_JSON, "w") as f:
                json.dump(config, f, indent=2)

    cv2.destroyAllWindows()

    print("\n" + "=" * 58)
    print("  ¡LISTO! ROIs guardados en:")
    print(f"  {OUTPUT_JSON}")
    print("=" * 58)

    # Mostrar resumen
    with open(OUTPUT_JSON) as f:
        final = json.load(f)
    for fname, rois in final.items():
        print(f"\n  {fname[:20]}")
        for r in rois:
            print(f"    {r['name']:10s} → {r['roi']}")


if __name__ == "__main__":
    main()
