"""
nsga_filter_optimizer.py — NSGA-II para optimizar filtros OCR (v2)
==================================================================
Cambios respecto a la versión original:
  ✓ API PaddleOCR corregida (TextRecognition no existe en 2.x)
  ✓ Parsing de resultado corregido (listas, no dicts)
  ✓ RoundingRepair en SBX + PM  (operadores continuos → vars enteras)
  ✓ adapt_block xl=0  (block_min=3, antes era 5)
  ✓ exact_match con tolerancia relativa (no absoluta)
  ✓ baseline_acc calculado dinámicamente
  ✓ CSV abierto con context manager
  ✓ pop_size=50  (era 30; regla: ≥ 4×n_vars)
  + Variable [9]  upscale    (1-4×) → mayor impacto en PaddleOCR
  + Variable [10] gamma      (0.5-3.0) → corrige sub/sobreexposición
  + Variable [11] channel    (gray/B/G/R/H/S/V) → canal óptimo por display
  + Variable [12] pad_px     (0-20 px) → dígitos en bordes

13 variables enteras · 2 objetivos (accuracy, confianza)

PREREQUISITO — ground_truth.csv:
    Columnas: class,file,gt_value
    class    → "green" (roi_green) o "white" (roi_white)
    file     → ruta relativa al crop desde BASE  (ej. crops/IMG_0050_g0.jpg)
    gt_value → valor real mostrado en el display  (ej. "572")

    Si no tienes el CSV, corre primero: python nsga_filter_optimizer.py --gen-csv
    Ese modo usa YOLO para extraer crops y te pide los gt_value por terminal.

INSTALACIÓN:
    pip install paddlepaddle paddleocr pymoo
"""

import os, csv, re, cv2, json, time, argparse
import numpy as np
from pathlib import Path
from functools import lru_cache
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio


# ── LEVENSHTEIN (sin dependencia externa) ──────────────────────────────────────────
def _levenshtein(a: str, b: str) -> int:
    """Distancia de edición simple entre dos strings."""
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j+1]+1, curr[j]+1,
                            prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def edit_similarity(pred: str, gt: str) -> float:
    """Similitud normalizada 0–1 basada en distancia de edición."""
    max_len = max(len(pred), len(gt), 1)
    return 1.0 - _levenshtein(pred, gt) / max_len

os.environ['FLAGS_use_mkldnn'] = '0'

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE        = ROOT
GT_CSV      = BASE / "ground_truth.csv"
CROPS_DIR   = BASE / "crops"
OUT_JSON    = BASE / "nsga_results.json"
BEST_PT     = BASE / "models/yolo/roi_detector_v1.pt"

TARGET_CLS  = "green"   # "green" | "white" | "both"
N_POP       = 50        # ≥ 4×n_vars = 4×13 = 52; usa 50 como mínimo práctico
N_GEN       = 60        # más generaciones → mejor convergencia
SEED        = 42
CONF_THRESH = 0.3       # para YOLO en modo gen-csv

# ── ESPACIO DE VARIABLES (13 enteras) ────────────────────────────────────────
#
#  idx  nombre       xl   xu   mapping
#  [0]  blur          0    5   BLUR_MAP[idx] = 1,3,5,7,9,11 px
#  [1]  thresh_type   0    4   0=Binary 1=InvBin 2=Otsu 3=AdGauss 4=AdMean
#  [2]  thresh_val    0  255   solo activo para tipos 0/1
#  [3]  adapt_block   0   24   (idx*2+3)|1 = 3,5,7...51 px  ← xl=0 (fix)
#  [4]  adapt_C       0   40   C = idx-20 = -20..+20
#  [5]  clahe         0    1   0=off 1=on
#  [6]  clahe_clip    1   40   clip = idx/2.0 = 0.5..20
#  [7]  morph_op      0    4   0=none 1=erode 2=dilate 3=open 4=close
#  [8]  morph_k       1   10   tamaño kernel morph
#  [9]  upscale       1    4   factor escala antes de OCR          ← NUEVO
# [10]  gamma_idx     5   30   gamma = idx/10.0 = 0.5..3.0        ← NUEVO
# [11]  channel       0    6   0=gray 1=B 2=G 3=R 4=H 5=S 6=V    ← NUEVO
# [12]  pad_px        0   20   px de padding alrededor del crop    ← NUEVO

XL = np.array([0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  5,  0,  0])
XU = np.array([5,  4, 255, 24, 40,  1, 40,  4, 10,  4, 30,  6, 20])

BLUR_MAP    = [1, 3, 5, 7, 9, 11]
MORPH_OPS   = [None, cv2.MORPH_ERODE, cv2.MORPH_DILATE,
               cv2.MORPH_OPEN, cv2.MORPH_CLOSE]
THRESH_LBLS = ["Binary", "InvBinary", "Otsu", "AdaptGauss", "AdaptMean"]
MORPH_LBLS  = ["none", "erode", "dilate", "open", "close"]
CH_NAMES    = ["gray", "B", "G", "R", "H", "S", "V"]


# ── LUT de gamma cacheada (no reconstruir en cada evaluación) ─────────────────
@lru_cache(maxsize=26)
def _gamma_lut(gamma_idx: int) -> np.ndarray:
    g = max(0.1, gamma_idx / 10.0)
    return np.array([((i / 255.0) ** g) * 255 for i in range(256)],
                    dtype=np.uint8)


# ── FILTRO PRINCIPAL (13 variables) ──────────────────────────────────────────
def apply_filter(bgr: np.ndarray, x) -> np.ndarray:
    (blur_k_idx, tt, tv, ab, ac,
     clahe_on, cc, mo, mk,
     upscale, gamma_idx, channel, pad_px) = (int(v) for v in x)

    # 1. Upscale (antes de todo para máxima calidad)
    if upscale > 1:
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (w * upscale, h * upscale),
                         interpolation=cv2.INTER_CUBIC)

    # 2. Extracción de canal
    if channel == 0:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    elif channel in (1, 2, 3):          # B=1, G=2, R=3
        gray = bgr[:, :, channel - 1].copy()
    else:                               # H=4, S=5, V=6
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = hsv[:, :, channel - 4].copy()

    # 3. Gamma
    gray = cv2.LUT(gray, _gamma_lut(gamma_idx))

    # 4. CLAHE
    if clahe_on:
        clip = max(0.5, cc / 2.0)
        cl   = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4))
        gray = cl.apply(gray)

    # 5. Blur gaussiano
    k = BLUR_MAP[min(blur_k_idx, len(BLUR_MAP) - 1)]
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # 6. Umbralización
    block = (ab * 2 + 3) | 1   # forzar impar (min=3 con xl=0)
    C     = ac - 20
    # FIX: thresh_val es inerte en Otsu/Adaptativo — fijar en 128 para
    # reducir dimensionalidad efectiva y no desperdiciar mutaciones.
    if tt >= 2:
        tv = 128

    if tt == 0:
        _, binary = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY)
    elif tt == 1:
        _, binary = cv2.threshold(gray, tv, 255, cv2.THRESH_BINARY_INV)
    elif tt == 2:
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif tt == 3:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block, C)
    else:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, block, C)

    # 7. Morfología
    if mo > 0:
        ker    = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (max(1, mk), max(1, mk)))
        binary = cv2.morphologyEx(binary, MORPH_OPS[mo], ker)

    # 8. Padding (PaddleOCR falla con texto en el borde del crop)
    if pad_px > 0:
        binary = cv2.copyMakeBorder(binary, pad_px, pad_px, pad_px, pad_px,
                                    cv2.BORDER_CONSTANT, value=0)

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ── OCR UTILITIES ─────────────────────────────────────────────────────────────
def init_ocr():
    """
    PaddleOCR v3.x — usamos TextRecognition directamente porque:
      1. YOLO ya nos da el crop exacto → no necesitamos detección.
      2. El pipeline completo usa oneDNN (Intel) que falla en AMD ROCm.
    """
    from paddleocr import TextRecognition
    print("  Iniciando PaddleOCR v3 — TextRecognition (CPU)...")
    rec = TextRecognition(device='cpu')
    print("  [OK] PaddleOCR listo.")
    return rec


def ocr_single(rec, bgr) -> tuple[str, float]:
    """Corre PaddleOCR v3 TextRecognition sobre un crop BGR. Retorna (texto, conf)."""
    try:
        # predict() devuelve lista de dicts: [{'rec_text': ..., 'rec_score': ...}]
        res = list(rec.predict(bgr))
        if not res:
            return '', 0.0
        raw  = res[0].get('rec_text', '')
        conf = res[0].get('rec_score', 0.0)
        return str(raw), float(conf)
    except Exception:
        return '', 0.0


def clean(raw: str) -> str:
    """Limpia texto OCR → solo dígitos, punto decimal y signo menos."""
    t = raw.upper()
    for a, b in [('O','0'),('D','0'),('Q','0'),('I','1'),
                 ('L','1'),('|','1'),('S','5'),('B','8'),
                 ('G','6'),(',','.')]:
        t = t.replace(a, b)
    return re.sub(r'[^0-9.\-]', '', t)


def exact_match(pred: str, gt: str, rel_tol: float = 0.01) -> bool:
    """Coincidencia con tolerancia relativa del 1% (no absoluta)."""
    try:
        p, g = float(pred), float(gt)
        if abs(g) < 1e-9:
            return abs(p) < 1e-6
        return abs(p - g) / abs(g) <= rel_tol
    except ValueError:
        return pred.strip() == gt.strip()


# ── PROBLEMA NSGA-II ──────────────────────────────────────────────────────────
class FilterProblem(Problem):
    def __init__(self, samples, ocr, n_gen_total, pop_size):
        super().__init__(
            n_var=13, n_obj=3, n_ieq_constr=0,   # 3 objetivos: acc, conf, edit_sim
            xl=XL, xu=XU, vtype=int
        )
        self.samples       = samples
        self.ocr           = ocr
        self.n_eval        = 0
        self.best_acc      = 0.0
        self.best_conf     = 0.0
        self.n_gen_total   = n_gen_total
        self.pop_size      = pop_size
        self.eval_times    = []        # segundos por individuo
        self.t_start       = time.time()

    def _evaluate(self, X, out, *args, **kwargs):
        F       = np.zeros((len(X), 3))   # acc, conf_correct, edit_sim
        n_indiv = len(X)
        n_samp  = len(self.samples)

        for i, x in enumerate(X):
            t_ind = time.time()
            acc, avg_conf, avg_edit = self._score(x)
            dt = time.time() - t_ind
            self.eval_times.append(dt)
            F[i, 0] = -acc
            F[i, 1] = -avg_conf
            F[i, 2] = -avg_edit

            # Barra de progreso del batch actual
            done  = int(20 * (i + 1) / n_indiv)
            bar   = '\u2588' * done + '\u2591' * (20 - done)
            spd   = n_samp / dt if dt > 0 else 0
            print(f"  \r  individuo [{i+1:3d}/{n_indiv}]  "
                  f"[{bar}]  "
                  f"acc={acc:.3f}  edit={avg_edit:.3f}  "
                  f"{spd:.1f} crops/s   ",
                  end='', flush=True)

        out["F"] = F
        self.n_eval += n_indiv

        best_batch = float(-F[:, 0].min())
        best_batch_conf = float(-F[F[:,0].argmin(), 1])
        if best_batch > self.best_acc:
            self.best_acc  = best_batch
            self.best_conf = best_batch_conf
        print()  # salto de linea tras la barra

    def _score(self, x) -> tuple[float, float, float]:
        """Devuelve (accuracy, conf_prom_solo_correctos, edit_sim_prom)."""
        ok_count  = 0
        conf_ok   = 0.0   # FIX: conf solo sobre correctos
        edit_sum  = 0.0
        for row in self.samples:
            filt = apply_filter(row['bgr'], x)
            raw, conf = ocr_single(self.ocr, filt)
            pred = clean(raw)
            is_ok = exact_match(pred, row['gt'])
            if is_ok:
                ok_count += 1
                conf_ok  += conf
            # senal suave continua (gradiente implicito para NSGA)
            edit_sum += edit_similarity(pred, row['gt'])
        n = max(len(self.samples), 1)
        avg_conf = conf_ok / max(ok_count, 1)   # conf media solo en correctos
        return ok_count / n, avg_conf, edit_sum / n


# ── MONITOR CALLBACK ──────────────────────────────────────────────────────────
class MonitorCallback(Callback):
    """
    Se ejecuta al final de CADA generación del NSGA-II.
    Imprime en consola:
      ─ Número de generación / total
      ─ Barra de progreso global
      ─ Tiempo de la generación y ETA restante
      ─ Mejor / media / peor accuracy del Pareto front
      ─ Flecha si se rompió el récord de accuracy
    """
    W = 62   # ancho de la caja

    def __init__(self, problem: FilterProblem, n_gen_total: int,
                 baseline_acc: float, cls_name: str):
        super().__init__()
        self.problem      = problem
        self.n_gen_total  = n_gen_total
        self.baseline_acc = baseline_acc
        self.cls_name     = cls_name
        self.t_gen_start  = time.time()
        self.t_run_start  = time.time()
        self.prev_best    = 0.0
        self.gen_accs     = []   # historia de best_acc por gen

    def notify(self, algorithm):
        g         = algorithm.n_gen
        F         = algorithm.pop.get("F")   # objetivos pob actual
        accs      = -F[:, 0]
        confs     = -F[:, 1]
        edits     = -F[:, 2]

        best_acc  = float(accs.max())
        mean_acc  = float(accs.mean())
        worst_acc = float(accs.min())
        best_conf = float(confs[accs.argmax()])
        mean_edit = float(edits.mean())
        best_edit = float(edits.max())
        n_samp    = len(self.problem.samples)

        # Timing
        now       = time.time()
        t_gen     = now - self.t_gen_start
        t_elapsed = now - self.t_run_start
        avg_gen_t = t_elapsed / g if g > 0 else t_gen
        t_remain  = avg_gen_t * (self.n_gen_total - g)

        new_record = best_acc > self.prev_best
        if new_record:
            self.prev_best = best_acc
        self.gen_accs.append(best_acc)
        self.t_gen_start = now

        # Helpers de formateo
        def fmt_t(sec):
            m, s = divmod(int(sec), 60)
            h, m = divmod(m, 60)
            return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s" if m else f"{s}s"

        pct    = g / self.n_gen_total
        filled = int(30 * pct)
        bar    = '\u2588' * filled + '\u2591' * (30 - filled)

        # Velocidad media OCR
        times  = self.problem.eval_times
        avg_sp = (n_samp / np.mean(times[-self.problem.pop_size:])
                  if times else 0)

        # ── Imprimir caja ───────────────────────────────────────────────────
        SEP  = '\u2550' * self.W
        SEP2 = '\u2500' * self.W
        star = '  \u2605 NUEVO RECORD' if new_record else ''
        print(f"\n{SEP}")
        print(f"  Gen {g:3d}/{self.n_gen_total}  [{bar}]  "
              f"{pct*100:5.1f}%  ETA {fmt_t(t_remain)}")
        print(SEP2)
        print(f"  [{self.cls_name.upper()}]  Mejor acc : "
              f"{best_acc:.3f}  ({best_acc*n_samp:.0f}/{n_samp})"
              f"  conf={best_conf:.3f}{star}")
        print(f"  Media pob  : {mean_acc:.3f}  ({mean_acc*n_samp:.1f}/{n_samp})"
              f"  edit_sim={mean_edit:.3f}")
        print(f"  Peor pob   : {worst_acc:.3f}  ({worst_acc*n_samp:.1f}/{n_samp})"
              f"  best_edit={best_edit:.3f}")
        print(SEP2)
        delta = best_acc - self.baseline_acc
        print(f"  Baseline   : {self.baseline_acc:.3f}  "
              f"Mejora acum: {delta:+.3f} ({delta*n_samp:+.1f} crops extra)")
        print(f"  Tiempo gen : {fmt_t(t_gen):>8s}   Total: {fmt_t(t_elapsed):>8s}")
        print(f"  Velocidad  : {avg_sp:.1f} crops/s  "
              f"  Evals: {self.problem.n_eval:,}/{self.n_gen_total*self.problem.pop_size:,}")

        # Mini grafico sparkline
        if len(self.gen_accs) > 1:
            hist   = self.gen_accs[-20:]
            lo, hi = min(hist), max(hist)
            rng    = hi - lo if hi > lo else 1
            BARS   = ' \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588'
            spark  = ''.join(BARS[min(8, int(8*(v-lo)/rng))] for v in hist)
            print(f"  Historial  : [{spark}] (\u00faltimas {len(hist)} gens)")

        # Checkpoint completo: guarda generación + población entera (X + F)
        if g % 10 == 0:
            chk = {
                "gen"      : g,
                "cls"      : self.cls_name,
                "best_acc" : float(self.prev_best),
                "gen_accs" : self.gen_accs,
                "pop_X"    : algorithm.pop.get("X").tolist(),   # población completa
                "pop_F"    : algorithm.pop.get("F").tolist(),
            }
            ckpt_path = BASE / f"checkpoint_{self.cls_name}.json"
            with open(ckpt_path, "w") as f:
                json.dump(chk, f)
            print(f"  [Checkpoint gen {g} guardado → {ckpt_path.name}]")

        print(SEP)


def compute_baseline(samples, ocr) -> tuple[float, float]:
    """Accuracy con parámetros neutros (sin optimización)."""
    # Otsu, sin CLAHE, sin morph, upscale=1, gamma=1.0, canal gray, pad=0
    x_neutral = np.array([0, 2, 128, 4, 22, 0, 4, 0, 2, 1, 10, 0, 0])
    ok, conf_sum = 0, 0.0
    for row in samples:
        filt = apply_filter(row['bgr'], x_neutral)
        raw, conf = ocr_single(ocr, filt)
        if exact_match(clean(raw), row['gt']):
            ok += 1
        conf_sum += conf
    n = max(len(samples), 1)
    return ok / n, conf_sum / n


# ── GENERAR GROUND TRUTH CSV (modo --gen-csv) ─────────────────────────────────
def generate_gt_csv():
    """
    Extrae crops con YOLO de todas las imágenes en dataset/images/ y
    pide al usuario el gt_value por terminal. Guarda ground_truth.csv.
    """
    from ultralytics import YOLO

    if not BEST_PT.exists():
        print(f"[ERROR] No se encontró best.pt en:\n  {BEST_PT}")
        return

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Cargando YOLO: {BEST_PT.name}")
    model = YOLO(str(BEST_PT))

    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for split in ("train", "val", ""):
        idir = BASE / "dataset" / "images" / split if split else BASE / "dataset" / "images"
        if idir.exists():
            images.extend(p for p in idir.iterdir()
                          if p.suffix.lower() in exts and p.is_file())

    if not images:
        print("[ERROR] No se encontraron imágenes en dataset/images/")
        return

    print(f"  {len(images)} imágenes. Extrayendo crops...\n")

    rows = []
    cls_map = {0: "white", 1: "green"}

    for img_path in sorted(images):
        bgr  = cv2.imread(str(img_path))
        if bgr is None:
            continue
        ih, iw = bgr.shape[:2]

        results = model(img_path, conf=CONF_THRESH, verbose=False)[0]
        by_cls  = {0: [], 1: []}
        for box in results.boxes:
            cls  = int(box.cls)
            xyxy = tuple(map(int, box.xyxy[0].tolist()))
            by_cls[cls].append(xyxy)

        os.system("clear")
        print(f"  Imagen: {img_path.name}")
        print("  Introduce el valor GT para cada crop (ENTER=omitir, q=saltar imagen):\n")

        skip_image = False
        for cls_id in (1, 0):
            boxes = sorted(by_cls[cls_id], key=lambda b: b[1])
            for i, box in enumerate(boxes):
                crop = bgr[box[1]:box[3], box[0]:box[2]]
                if crop.size == 0:
                    continue

                # Mostrar crop en ventana temporal
                show = cv2.resize(crop, (220, 110))
                cv2.imshow(f"{cls_map[cls_id]} #{i}", show)
                cv2.waitKey(1)

                gt_val = input(f"    {cls_map[cls_id]} #{i}: ").strip()
                cv2.destroyAllWindows()

                if gt_val.lower() == 'q':
                    skip_image = True
                    break
                if not gt_val:
                    continue

                # Guardar crop
                crop_name = f"{img_path.stem}_{cls_map[cls_id]}{i}.jpg"
                crop_path = CROPS_DIR / crop_name
                cv2.imwrite(str(crop_path), crop)

                rows.append({
                    "class"   : cls_map[cls_id],
                    "file"    : str(crop_path.relative_to(BASE)),
                    "gt_value": gt_val,
                })

            if skip_image:
                break

    cv2.destroyAllWindows()

    with open(GT_CSV, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["class", "file", "gt_value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  [✓] {len(rows)} crops guardados en: {GT_CSV}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-csv", action="store_true",
                        help="Generar ground_truth.csv con YOLO + input manual")
    parser.add_argument("--cls",    default=TARGET_CLS,
                        help="Clase a optimizar: green | white | both")
    parser.add_argument("--pop",    type=int, default=N_POP)
    parser.add_argument("--gen",    type=int, default=N_GEN)
    parser.add_argument("--resume", action="store_true",
                        help="Reanudar desde checkpoint_<cls>.json (la pob guardada)")
    args = parser.parse_args()

    if args.gen_csv:
        generate_gt_csv()
        return

    print("=" * 62)
    print(f"  NSGA-II — Filtros OCR  [{args.cls}]")
    print(f"  Variables: 13   Población: {args.pop}   Generaciones: {args.gen}")
    print("=" * 62)

    if not GT_CSV.exists():
        print(f"\n  [ERROR] No se encontró:\n  {GT_CSV}")
        print("  Genera el CSV primero:  python nsga_filter_optimizer.py --gen-csv")
        return

    # Cargar datos
    target_classes = ["green", "white"] if args.cls == "both" else [args.cls]
    all_results = {}

    with open(GT_CSV, newline='') as f:
        rows = list(csv.DictReader(f))

    ocr = init_ocr()

    for cls_name in target_classes:
        samples = []
        for r in rows:
            if r['class'] != cls_name or not r['gt_value'].strip():
                continue
            img_path = BASE / r['file']
            bgr      = cv2.imread(str(img_path))
            if bgr is not None:
                samples.append({
                    'bgr' : bgr,
                    'gt'  : r['gt_value'].strip(),
                    'file': r['file'],
                })

        if not samples:
            print(f"\n  [!] Sin muestras para clase '{cls_name}'")
            continue

        print(f"\n  [{cls_name.upper()}]  {len(samples)} muestras")

        # Baseline dinámico
        print("  Calculando baseline...", end="", flush=True)
        base_acc, base_conf = compute_baseline(samples, ocr)
        print(f"  acc={base_acc:.3f}  conf={base_conf:.3f}")

        # Problema y algoritmo
        problem = FilterProblem(samples, ocr, n_gen_total=args.gen, pop_size=args.pop)
        repair  = RoundingRepair()
        monitor = MonitorCallback(problem, args.gen, base_acc, cls_name)

        # ── Warm-start: cargar checkpoint si --resume ────────────────────────
        sampling_init = IntegerRandomSampling()
        if args.resume:
            ckpt_path = BASE / f"checkpoint_{cls_name}.json"
            if ckpt_path.exists():
                with open(ckpt_path) as f:
                    ckpt = json.load(f)
                if ckpt.get("cls") == cls_name:
                    X_init = np.array(ckpt["pop_X"], dtype=int)
                    # pymoo acepta ndarray directamente como población inicial
                    sampling_init      = X_init
                    monitor.gen_accs   = ckpt.get("gen_accs", [])
                    monitor.prev_best  = ckpt.get("best_acc", 0.0)
                    resumed_gen        = ckpt["gen"]
                    print(f"  [✓] Retomando desde gen {resumed_gen}  "
                          f"best_acc={ckpt['best_acc']:.3f}  "
                          f"corriendo {args.gen} gens más")
                else:
                    print(f"  [!] Checkpoint es para clase '{ckpt.get('cls')}', "
                          f"no para '{cls_name}'. Iniciando desde cero.")
            else:
                print(f"  [!] No se encontró {ckpt_path.name}. Iniciando desde cero.")

        algorithm = NSGA2(
            pop_size  = args.pop,
            sampling  = sampling_init,
            # FIX: eta bajo → más exploración en vars semi-categóricas (thresh_type, channel)
            crossover = SBX(prob=0.9, eta=3,  vtype=float, repair=repair),
            mutation  = PM(eta=5,     vtype=float, repair=repair),
            # FIX: False → preserva diversidad en espacio discreto con rangos pequeños
            eliminate_duplicates=False,
        )

        termination = get_termination("n_gen", args.gen)

        print(f"  Corriendo NSGA-II ({args.pop}×{args.gen}"
              f" = {args.pop*args.gen} evals máx)...\n")
        t0  = time.time()
        res = minimize(problem, algorithm, termination,
                       seed=SEED, verbose=False, callback=monitor)
        elapsed = time.time() - t0
        print(f"\n  Tiempo total: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

        # Pareto front
        F = res.F
        X = res.X.astype(int)
        order    = np.argsort(F[:, 0])
        X_sorted = X[order]
        F_sorted = F[order]

        print(f"\n{'='*62}")
        print(f"  TOP-5  [{cls_name.upper()}]  (por accuracy desc)")
        print(f"{'='*62}")

        best_params = None
        for rank, (x, f) in enumerate(zip(X_sorted[:5], F_sorted[:5])):
            acc  = -f[0]
            conf = -f[1]
            (blur_k_idx, tt, tv, ab, ac,
             clahe_on, cc, mo, mk,
             upscale, gamma_idx, channel, pad_px) = x

            params = {
                "blur_k"        : BLUR_MAP[min(int(blur_k_idx), 5)],
                "thresh_type"   : THRESH_LBLS[int(tt)],
                "thresh_val"    : int(tv),
                "adaptive_block": int((ab * 2 + 3) | 1),
                "adaptive_C"    : int(ac - 20),
                "clahe"         : bool(clahe_on),
                "clahe_clip"    : round(max(0.5, cc / 2.0), 1),
                "morph_op"      : MORPH_LBLS[int(mo)],
                "morph_k"       : int(mk),
                "upscale"       : int(upscale),
                "gamma"         : round(gamma_idx / 10.0, 1),
                "channel"       : CH_NAMES[int(channel)],
                "pad_px"        : int(pad_px),
            }
            if rank == 0:
                best_params = params

            n = len(samples)
            print(f"\n  #{rank+1}  acc={acc:.3f} ({acc*n:.0f}/{n})"
                  f"  conf={conf:.3f}"
                  f"  mejora={acc-base_acc:+.3f}")
            for k, v in params.items():
                print(f"    {k:18s}: {v}")

        # Guardar
        all_results[cls_name] = {
            "class"        : cls_name,
            "n_samples"    : len(samples),
            "n_pop"        : args.pop,
            "n_gen"        : args.gen,
            "baseline_acc" : float(base_acc),
            "baseline_conf": float(base_conf),
            "best_acc"     : float(-F_sorted[0, 0]),
            "best_conf"    : float(-F_sorted[0, 1]),
            "improvement"  : float(-F_sorted[0, 0] - base_acc),
            "best_params"  : best_params,
            "pareto_front" : [
                {"acc": float(-f[0]), "conf": float(-f[1]), "x": x.tolist()}
                for x, f in zip(X_sorted, F_sorted)
            ]
        }

    if all_results:
        with open(OUT_JSON, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n\n  [✓] Resultados guardados: {OUT_JSON}")
        for cls_name, r in all_results.items():
            print(f"  {cls_name:6s}: baseline={r['baseline_acc']:.3f}"
                  f"  → best={r['best_acc']:.3f}"
                  f"  (+{r['improvement']:.3f})")

    print("=" * 62)


if __name__ == "__main__":
    main()