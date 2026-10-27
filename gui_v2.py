"""
calibracion_gui.py — Sistema de Calibración Automatizada de Pipeta
===================================================================

Pipeline:
    COM (Mettler Toledo) → trigger → BRIO captura → YOLO ROIs →
    PaddleOCR White (Temperaturas blancas) + PaddleOCR Green (Humedad/Temp/Presión) →
    GUI muestra resultados + guarda en Excel

Uso:
    cd <repo>
    python calibracion_gui.py
"""

import customtkinter as ctk
import cv2
import numpy as np
import threading
import queue
import time
import re
import os
LAB_NAME = os.environ.get("CALIBRATION_LAB_NAME", "Laboratorio de Calibración")
import serial
import serial.tools.list_ports
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageTk
from openpyxl import Workbook, load_workbook

import wave
import math
import subprocess
import json
import logging
import traceback
import hashlib

from metrologia import (
    load_config_equipos, z_factor, volume_from_mass,
    calculate_expanded_uncertainty, evaluar_calibracion, iso8655_2_tolerancias
)

try:
    from pdf_reporte import generar_certificado_pdf
    _PDF_OK = True
except ImportError:
    _PDF_OK = False

LOG_FILE = Path(__file__).resolve().parent / "debug_ocr.log"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.DEBUG,
    format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("calibracion_gui")

# Usar Path relativo al script para mayor portabilidad
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent

CONFIG_FILE = _SCRIPT_DIR / "config.json"

DEFAULT_CONFIG = {
    "theme": "Light",
    "scaling": 1.0,
    "font_family": "Consolas",
    "font_size": 16,             # tamaño de las lecturas grandes
    "alarm_volume": 0.5,         # 0.0 (mudo) a 1.0
    "export_format": "xlsx",     # "xlsx" | "csv" | "txt"
    "show_roi_thumbnails": True, # mostrar/ocultar fila de recortes YOLO
    "cam_size": "normal",        # "compact" | "normal" | "large"
    "table_rows_height": 150,    # alto en px del scroll del historial
    # ── Configuración de hardware (editar sin tocar código fuente) ────────
    "com_port": "/dev/ttyUSB0",  # Puerto serie de la balanza; dejar vacío para auto-detectar
    "baudrate": 9600,
    "camera_index": 2,
}

CAM_SIZES = {
    "compact": (400, 225),
    "normal":  (560, 315),
    "large":   (720, 405),
}

def load_config():
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                cfg.update(json.load(f))
        except: pass
    return cfg

def save_config(cfg):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f)

# ── SONIDOS ───────────────────────────────────────────────────────────────────
_ALARM_VOLUME = 0.5  # 0.0 (mudo) a 1.0, actualizado desde Configuración

def set_alarm_volume(v):
    global _ALARM_VOLUME
    _ALARM_VOLUME = max(0.0, min(1.0, float(v)))

def _generate_beep(filename, freq, duration_ms, vol=0.5):
    if os.path.exists(filename): return
    sample_rate = 44100
    n_samples = int(sample_rate * (duration_ms / 1000.0))
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        for i in range(n_samples):
            t = float(i) / sample_rate
            sample = int(vol * 32767.0 * math.sin(2.0 * math.pi * freq * t))
            f.writeframesraw(sample.to_bytes(2, byteorder='little', signed=True))

def play_sound(sound_type):
    vol = _ALARM_VOLUME
    if vol <= 0.005:
        return  # usuario silenció las alarmas
    # El nombre del archivo cachea por nivel de volumen redondeado, así un
    # cambio de volumen en Configuración se escucha en el próximo beep sin
    # tener que borrar el caché de wavs generados a mano.
    tag = f"{vol:.2f}"
    try:
        if sound_type == "success":
            f1, f2 = f"/tmp/beep_s1_{tag}.wav", f"/tmp/beep_s2_{tag}.wav"
            _generate_beep(f1, 800, 100, vol)
            _generate_beep(f2, 1200, 150, vol)
            subprocess.Popen(f"aplay -q {f1} && aplay -q {f2}", shell=True)
        elif sound_type == "retry":
            f = f"/tmp/beep_retry_{tag}.wav"
            _generate_beep(f, 400, 200, vol)
            subprocess.Popen(f"aplay -q {f}", shell=True)
        elif sound_type == "error":
            f = f"/tmp/beep_error_{tag}.wav"
            _generate_beep(f, 150, 600, vol)
            subprocess.Popen(f"aplay -q {f}", shell=True)
    except:
        pass

# ── RUTAS DE MODELOS (relativas al directorio del script) ────────────────────
YOLO_MODEL    = _SCRIPT_DIR / "models" / "yolo" / "roi_detector_v1.pt"
WHITE_REC_DIR = str(_SCRIPT_DIR / "models" / "ocr_white")
GREEN_REC_DIR = str(_SCRIPT_DIR / "models" / "ocr_green")
EXCEL_DIR     = _SCRIPT_DIR / "resultados"
EXCEL_DIR.mkdir(exist_ok=True)

# ── PARÁMETROS (cargados desde config.json en __init__) ──────────────────────
# Valores por defecto — se sobreescriben al inicializar CalibracionApp
CAMERA_INDEX = 2
COM_PORT     = "/dev/ttyUSB0"
BAUDRATE     = 9600
YOLO_CONF    = 0.4
SCALE        = 3

def _resolve_com_port(port: str) -> str:
    """
    Intenta usar el puerto configurado. Si no existe, auto-detecta el primer
    puerto USB-serie disponible. Registra el resultado en el log.
    """
    import serial.tools.list_ports
    if port and Path(port).exists():
        return port
    # Auto-detección
    ports = [p.device for p in serial.tools.list_ports.comports()
             if 'USB' in (p.description or '') or 'ttyUSB' in p.device or 'ttyACM' in p.device]
    if ports:
        logger.warning(f"Puerto configurado '{port}' no encontrado. Auto-detectado: {ports[0]}")
        return ports[0]
    logger.error(f"No se encontró ningún puerto USB-serie. Puerto configurado: '{port}'")
    return port  # devolver el configurado; el hilo COM reportará el error

# Expand box mínimos
WHITE_MIN_W, WHITE_MIN_H = 200, 88
GREEN_MIN_W, GREEN_MIN_H = 150, 130

# ── COLORES CORPORATIVOS ──────────────────────────────────────────────────────
BG_WHITE     = ("#FFFFFF", "#121212")
BG_CARD      = ("#F0F4FA", "#1E1E1E")
BG_CARD_DARK = ("#E3EAF5", "#2A2A2A")
BLUE_DARK    = ("#1B3A5C", "#93C5FD")
BLUE_HEADER  = ("#1B3A5C", "#0F2036")
BLUE_MID     = ("#2563EB", "#3B82F6")
BLUE_LIGHT   = ("#3B82F6", "#60A5FA")
BLUE_ACCENT  = ("#60A5FA", "#60A5FA")
GREEN_OK     = ("#22C55E", "#22C55E")
RED_ERR      = ("#EF4444", "#EF4444")
GRAY_TEXT    = ("#6B7280", "#9CA3AF")
GRAY_BORDER  = ("#D1D5DB", "#333333")


def expand_box(x1, y1, x2, y2, img_h, img_w, cls_name):
    """Expande la caja YOLO si es más angosta que el mínimo."""
    bw, bh = x2 - x1, y2 - y1
    if cls_name == "roi_white":
        min_w, min_h = WHITE_MIN_W, WHITE_MIN_H
    elif cls_name == "roi_green":
        min_w, min_h = GREEN_MIN_W, GREEN_MIN_H
    else:
        return x1, y1, x2, y2
    if bw < min_w:
        d = (min_w - bw) // 2
        x1, x2 = max(0, x1 - d), min(img_w, x2 + d)
    if bh < min_h:
        d = (min_h - bh) // 2
        y1, y2 = max(0, y1 - d), min(img_h, y2 + d)
    return x1, y1, x2, y2


def extract_number(raw: str) -> str:
    raw = raw.replace(',', '.').strip()
    m = re.search(r'-?\d+\.\d+|-?\d+', raw)
    return m.group() if m else "—"


def validate_readings(entry):
    """Valida los rangos físicos y formato esperado de cada campo OCR."""
    valid = {}
    
    def check(val_str, v_min, v_max, decimals=None, force_decimals=False):
        if val_str == "—": return False
        try:
            v = float(val_str)
            if not (v_min <= v <= v_max): return False
            if decimals is not None:
                if force_decimals:
                    if "." not in val_str: return False
                    if len(val_str.split(".")[1]) != decimals: return False
                elif "." in val_str:
                    # Tolerar menos decimales de los esperados...
                    if len(val_str.split(".")[1]) > decimals: return False
            return True
        except:
            return False

    valid["humedad"]   = check(entry["humedad"], 45.0, 80.0, 1, force_decimals=True)   # ISO 8655-7
    valid["temp_amb"]  = check(entry["temp_amb"], 15.0, 30.0, 1, force_decimals=True)   # ISO 8655-7
    valid["presion"]   = check(entry["presion"], 900.0, 1100.0, 1, force_decimals=True)
    # t_agua: temperatura del agua del Excursion-Trac (display blanco, columna derecha)
    valid["t_agua"]    = check(entry["t_agua"], 15.0, 30.0, 2, force_decimals=True)     # ISO 8655-7: 20±3 °C
    valid["temp_min"]  = check(entry["temp_min"], 10.0, 35.0, 2, force_decimals=True)
    valid["temp_max"]  = check(entry["temp_max"], 15.0, 45.0, 2, force_decimals=True)
    valid["peso"]      = True  # El peso viene del COM, asumimos ok
    
    # Regla física: T.Min <= T.Agua <= T.Max (todas del Excursion-Trac)
    try:
        t_min = float(entry["temp_min"])
        t_agua = float(entry["t_agua"])
        t_max = float(entry["temp_max"])
        if not (t_min <= t_agua <= t_max):
            valid["t_agua"] = valid["temp_min"] = valid["temp_max"] = False
    except:
        pass
        
    return valid



# ══════════════════════════════════════════════════════════════════════════════
#  AUDIT TRAIL
# ══════════════════════════════════════════════════════════════════════════════
class AuditTrail:
    """
    Registro inmutable de TODAS las operaciones sobre los datos de medición.
    Tipos de acción:
      MEASURE_VALID_ISO  — medición válida dentro de sesión ISO activa
      MEASURE_VALID      — medición válida fuera de sesión (libre)
      MEASURE_ERROR      — medición guardada con errores de validación
      SESSION_COMPLETE   — lote de 10 réplicas completado
    Cumple: NTP-ISO/IEC 17025:2017 §7.5, FDA 21 CFR Part 11
    """
    def __init__(self, audit_path: Path):
        self.audit_path = audit_path

    def log(self, action: str, entry_num: int, operator: str, data: dict,
            valid_dict: dict = None):
        record = {
            "timestamp":  datetime.now().isoformat(),
            "action":     action,
            "entry_num":  entry_num,
            "operator":   operator,
            "data_hash":  hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()
        }
        if valid_dict:
            failed = [k for k, v in valid_dict.items() if not v]
            if failed:
                record["validation_errors"] = failed
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def verify_integrity(self, entry: dict, stored_hash: str) -> bool:
        computed = hashlib.sha256(
            json.dumps(entry, sort_keys=True).encode()
        ).hexdigest()
        return computed == stored_hash


# ══════════════════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════════════════
class CalibracionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.config = load_config()
        ctk.set_appearance_mode(self.config["theme"])
        ctk.set_widget_scaling(self.config["scaling"])
        set_alarm_volume(self.config["alarm_volume"])

        # ── Ventana ───────────────────────────────────────────────────────────
        self.title("Sistema de Calibración de Pipetas")
        self.geometry("1400x950")
        self.configure(fg_color=BG_WHITE)
        self.resizable(True, True)

        ctk.set_default_color_theme("blue")

        # ── Estado ────────────────────────────────────────────────────────────
        self.running = True
        self.com_connected = False
        self.camera_ok = False
        self.last_com_data = ""
        self.last_com_time = ""
        self.historial = []
        self.roi_images_tk = []      # Mantener referencia para evitar GC

        # ── Errores (UI) ──────────────────────────────────────────────────────
        self.error_logs = []
        self.error_window = None

        # ── Sesión y Audit Trail ──────────────────────────────────────────────
        self.session_data = {
            "operador": "Operador",
            "pipeta": "Pipeta",
            "v_nominal": 1000.0
        }
        self.session_measurements = []
        self.session_entries      = []   # copia completa de entries para el PDF
        EXCEL_DIR.mkdir(exist_ok=True)
        self.audit = AuditTrail(EXCEL_DIR / "audit_trail.jsonl")

        # ── Concurrencia (locks / flags anti race-condition) ────────────────────
        self._frame_lock = threading.Lock()   # protege self.latest_frame (hilo cámara vs pipeline)
        self._ocr_lock = threading.Lock()     # PaddleOCR no es thread-safe
        self._excel_lock = threading.Lock()   # protege escritura del archivo de datos (xlsx/csv/txt)
        self._hist_lock = threading.Lock()    # protege self.historial y el contador de medición
        self._processing = False              # evita pipelines paralelos disparados por el COM
        self._measure_counter = 0             # contador atómico para entry["num"]

        # ── Fuentes configurables (CTkFont es "vivo": .configure() propaga a
        #    todos los widgets que la usan, sin tener que reconstruir la GUI) ──
        self.font_readings = ctk.CTkFont(
            family=self.config["font_family"], size=self.config["font_size"], weight="bold"
        )
        self.font_table = ctk.CTkFont(
            family=self.config["font_family"], size=max(9, self.config["font_size"] - 5)
        )

        # ── Tamaño de vista de cámara (configurable) ─────────────────────────
        self.cam_display_size = CAM_SIZES.get(self.config["cam_size"], CAM_SIZES["normal"])

        # Bug 8: Paddle/oneDNN falla intermitentemente ("could not execute a
        # primitive") en la PRIMERA inferencia ejecutada por un hilo del SO
        # recién creado. Antes cada medición lanzaba un threading.Thread nuevo,
        # así que la primera llamada OCR de cada medición (canal verde) caía
        # en un hilo "virgen" para oneDNN. Solución: un único hilo trabajador
        # persistente que procesa todas las mediciones en cola, de modo que
        # predictor.run() siempre corre sobre el mismo hilo del SO.
        self._work_queue = queue.Queue()

        # ── Modelos (carga diferida) ──────────────────────────────────────────
        self.yolo = None
        self.ocr_white = None
        self.ocr_green = None
        self.models_loaded = False

        # ── Ruta de datos (según formato de exportación configurado) ────────────
        self._update_data_path()

        # ── Construir GUI ─────────────────────────────────────────────────────
        self._build_header()
        self._build_body()
        self._build_footer()

        # ── Iniciar hilos ─────────────────────────────────────────────────────
        self.after(100, self._load_models_async)
        self.after(200, self._start_camera_thread)
        self.after(300, self._start_com_thread)
        threading.Thread(target=self._measurement_worker, daemon=True).start()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════════════════════
    #  HEADER
    # ══════════════════════════════════════════════════════════════════════════
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=BLUE_HEADER, corner_radius=0, height=100)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        text_frame = ctk.CTkFrame(header, fg_color="transparent")
        text_frame.pack(side="left", padx=30, pady=15, fill="y")

        # Empresa
        ctk.CTkLabel(
            text_frame, text=f"🏥  {LAB_NAME}",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="normal"),
            text_color=BLUE_ACCENT
        ).pack(anchor="w")

        # Título
        ctk.CTkLabel(
            text_frame, text="Sistema de Calibración Automatizada de Pipeta",
            font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
            text_color="#FFFFFF"
        ).pack(anchor="w", pady=(2, 0))

        # Botón Configuración
        ctk.CTkButton(
            header, text="⚙️", width=40, height=40,
            font=ctk.CTkFont(size=20),
            fg_color="transparent", hover_color=BLUE_MID,
            command=self._open_settings
        ).pack(side="right", padx=30)

    # ══════════════════════════════════════════════════════════════════════════
    #  BODY
    # ══════════════════════════════════════════════════════════════════════════
    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color=BG_WHITE, corner_radius=0)
        body.pack(fill="both", expand=True, padx=20, pady=10)

        # ── Fila 1: Cámara + Lecturas ─────────────────────────────────────
        row1 = ctk.CTkFrame(body, fg_color="transparent")
        row1.pack(fill="x", pady=(0, 10))

        # Cámara
        cam_card = self._card(row1, "📷  Vista de Cámara")
        cam_card.pack(side="left", fill="both", expand=True, padx=(0, 10))

        cam_w, cam_h = self.cam_display_size
        self.cam_label = ctk.CTkLabel(
            cam_card, text="Iniciando cámara...",
            font=ctk.CTkFont(size=14), text_color=GRAY_TEXT,
            width=cam_w, height=cam_h
        )
        self.cam_label.pack(padx=10, pady=10)

        # Panel derecho: Estado COM + Lecturas
        right_panel = ctk.CTkFrame(row1, fg_color="transparent", width=420)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

        # Sesión de Calibración
        session_card = self._card(right_panel, "🎯  Sesión ISO 8655")
        session_card.pack(fill="x", pady=(0, 10))

        self.lbl_session_status = ctk.CTkLabel(
            session_card, text="Pipeta | 1000.0µL | 0/10",
            font=ctk.CTkFont(size=12), text_color=BLUE_DARK
        )
        self.lbl_session_status.pack(pady=5)

        self.btn_new_session = ctk.CTkButton(
            session_card, text="Configurar Sesión",
            command=self._open_new_session_dialog,
            height=28
        )
        self.btn_new_session.pack(pady=(0, 8))

        # Estado COM
        com_card = self._card(right_panel, "🔌  Puerto COM")
        com_card.pack(fill="x", pady=(0, 10))

        com_inner = ctk.CTkFrame(com_card, fg_color="transparent")
        com_inner.pack(fill="x", padx=15, pady=5)

        self.com_status_led = ctk.CTkLabel(
            com_inner, text="●", font=ctk.CTkFont(size=22),
            text_color=RED_ERR, width=30
        )
        self.com_status_led.pack(side="left")

        self.com_status_text = ctk.CTkLabel(
            com_inner, text="Desconectado",
            font=ctk.CTkFont(size=13), text_color=GRAY_TEXT
        )
        self.com_status_text.pack(side="left", padx=5)

        self.com_data_label = ctk.CTkLabel(
            com_card, text="Última lectura: —",
            font=ctk.CTkFont(size=12), text_color=GRAY_TEXT
        )
        self.com_data_label.pack(padx=15, pady=(0, 8), anchor="w")

        # Estado Modelos
        model_card = self._card(right_panel, "🧠  Modelos IA")
        model_card.pack(fill="x", pady=(0, 10))

        self.model_status = ctk.CTkLabel(
            model_card, text="⏳ Cargando modelos...",
            font=ctk.CTkFont(size=13), text_color=GRAY_TEXT
        )
        self.model_status.pack(padx=15, pady=8, anchor="w")

        # Lecturas actuales
        lect_card = self._card(right_panel, "📊  Lecturas Actuales")
        lect_card.pack(fill="both", expand=True)

        self.reading_labels = {}
        readings_data = [
            ("peso",        "Peso (COM)",                      "— mg",   "⚖️"),
            ("humedad",     "Humedad Relativa",                "— %",    "💧"),
            ("temp_amb",    "T. Amb. (Extech SD700)",          "— °C",   "🌡️"),
            ("presion",     "Presión Atmosférica",             "— hPa",  "📊"),
            ("t_agua",      "T. Agua (Excursion-Trac)",        "— °C",   "🌊"),
            ("temp_min",    "T. Agua Mínima (Excursion-Trac)", "— °C",   "📉"),
            ("temp_max",    "T. Agua Máxima (Excursion-Trac)", "— °C",   "🔥"),
        ]
        for key, label, default, icon in readings_data:
            row = ctk.CTkFrame(lect_card, fg_color="transparent")
            row.pack(fill="x", padx=15, pady=1)

            ctk.CTkLabel(
                row, text=f"{icon} {label}:",
                font=ctk.CTkFont(size=12), text_color=GRAY_TEXT, anchor="w"
            ).pack(side="left")

            val_lbl = ctk.CTkLabel(
                row, text=default,
                font=self.font_readings,
                text_color=BLUE_DARK, anchor="e"
            )
            val_lbl.pack(side="right")
            self.reading_labels[key] = val_lbl

        # ── Fila 2: ROIs detectados ───────────────────────────────────────
        self.roi_card = self._card(body, "🔍  ROIs Detectados por YOLO")
        if self.config.get("show_roi_thumbnails", True):
            self.roi_card.pack(fill="x", pady=(0, 10))

        self.roi_scroll = ctk.CTkScrollableFrame(self.roi_card, fg_color="transparent", height=120, orientation="horizontal")
        self.roi_scroll.pack(fill="x", padx=10, pady=5)

        self.roi_placeholder = ctk.CTkLabel(
            self.roi_scroll, text="Los recortes aparecerán aquí al capturar una imagen",
            font=ctk.CTkFont(size=12), text_color=GRAY_TEXT
        )
        self.roi_placeholder.pack(pady=40)

        # ── Fila 3: Historial ─────────────────────────────────────────────
        self.hist_card = self._card(body, "📋  Historial de Mediciones")
        hist_card = self.hist_card
        hist_card.pack(fill="both", expand=True)

        # Headers de tabla
        tbl_header = ctk.CTkFrame(hist_card, fg_color=BG_CARD_DARK, height=35, corner_radius=6)
        tbl_header.pack(fill="x", padx=10, pady=(5, 0))
        tbl_header.pack_propagate(False)

        cols = ["#", "Hora", "Peso (g)", "Hum %", "T.Amb °C", "Presión hPa", "T.Prom °C", "T.Mín °C", "T.Máx °C"]
        col_widths = [30, 70, 90, 80, 80, 100, 90, 90, 90]
        self.table_header_labels = []
        for i, (col, w) in enumerate(zip(cols, col_widths)):
            hdr_lbl = ctk.CTkLabel(
                tbl_header, text=col,
                font=self.font_table,
                text_color=BLUE_DARK, width=w, anchor="center"
            )
            hdr_lbl.pack(side="left", padx=2)
            self.table_header_labels.append(hdr_lbl)

        # Scrollable table body
        self.table_scroll = ctk.CTkScrollableFrame(
            hist_card, fg_color=BG_WHITE, height=self.config.get("table_rows_height", 150),
            scrollbar_button_color=BLUE_LIGHT
        )
        self.table_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # ══════════════════════════════════════════════════════════════════════════
    #  FOOTER
    # ══════════════════════════════════════════════════════════════════════════
    def _build_footer(self):
        footer = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=60)
        footer.pack(fill="x", padx=0, pady=0)
        footer.pack_propagate(False)

        btn_frame = ctk.CTkFrame(footer, fg_color="transparent")
        btn_frame.pack(expand=True)

        ctk.CTkButton(
            btn_frame, text="📷  Captura Manual", width=180, height=38,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=BLUE_MID, hover_color=BLUE_DARK,
            command=self._manual_capture
        ).pack(side="left", padx=10, pady=10)

        ctk.CTkButton(
            btn_frame, text="💾  Exportar Datos", width=180, height=38,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#059669", hover_color="#047857",
            command=self._export_excel
        ).pack(side="left", padx=10)

        self.btn_errors = ctk.CTkButton(
            btn_frame, text="⚠️ Errores (0)", width=140, height=38,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=RED_ERR, hover_color="#B91C1C",
            command=self._show_errors_window
        )
        self.btn_errors.pack(side="left", padx=10)

        self.status_bar = ctk.CTkLabel(
            footer, text="Listo",
            font=ctk.CTkFont(size=11), text_color=GRAY_TEXT
        )
        self.status_bar.pack(side="right", padx=20)

    # ══════════════════════════════════════════════════════════════════════════
    #  ERRORES GUI
    # ══════════════════════════════════════════════════════════════════════════
    def _log_error(self, title, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        err_str = f"[{timestamp}] {title}:\n{msg}\n{'-'*40}\n"
        self.error_logs.append(err_str)
        self.after(0, lambda: self.btn_errors.configure(text=f"⚠️ Errores ({len(self.error_logs)})"))

    def _show_errors_window(self):
        if self.error_window is None or not self.error_window.winfo_exists():
            self.error_window = ctk.CTkToplevel(self)
            self.error_window.title("Registro de Errores")
            self.error_window.geometry("700x500")
            
            textbox = ctk.CTkTextbox(self.error_window, font=ctk.CTkFont(family="Consolas", size=12))
            textbox.pack(fill="both", expand=True, padx=10, pady=10)
            
            if not self.error_logs:
                textbox.insert("0.0", "No hay errores registrados.")
            else:
                for err in self.error_logs:
                    textbox.insert("end", err)
            
            textbox.configure(state="disabled")
        else:
            self.error_window.focus()

    # ══════════════════════════════════════════════════════════════════════════
    #  UTILIDADES GUI
    # ══════════════════════════════════════════════════════════════════════════
    def _card(self, parent, title):
        """Crea una tarjeta con título azul y fondo claro."""
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=12,
                            border_width=1, border_color=GRAY_BORDER)

        title_bar = ctk.CTkFrame(card, fg_color="transparent", height=32)
        title_bar.pack(fill="x", padx=12, pady=(8, 2))
        title_bar.pack_propagate(False)

        ctk.CTkLabel(
            title_bar, text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=BLUE_MID
        ).pack(side="left")

        return card

    def _set_status(self, msg, color=GRAY_TEXT):
        self.status_bar.configure(text=msg, text_color=color)

    # ══════════════════════════════════════════════════════════════════════════
    #  CARGA DE MODELOS
    # ══════════════════════════════════════════════════════════════════════════
    def _load_models_async(self):
        def _load():
            try:
                self.after(0, lambda: self.model_status.configure(text="⏳ Cargando YOLO..."))
                from ultralytics import YOLO
                self.yolo = YOLO(str(YOLO_MODEL))

                self.after(0, lambda: self.model_status.configure(text="⏳ Cargando OCR blanco..."))
                from paddleocr import PaddleOCR
                # enable_mkldnn=False: red de seguridad adicional. El backend
                # oneDNN/MKL-DNN de Paddle es el origen de "could not execute
                # a primitive" cuando predictor.run() cambia de hilo del SO.
                # Ya lo resolvimos moviendo todo a un único hilo persistente
                # (_measurement_worker), pero desactivar MKLDNN quita el riesgo
                # por completo a costa de un poco de velocidad en CPU.
                self.ocr_white = PaddleOCR(
                    use_angle_cls=False, lang='en', use_gpu=False,
                    enable_mkldnn=False, cpu_threads=4,
                    show_log=False, rec_model_dir=WHITE_REC_DIR
                )

                self.after(0, lambda: self.model_status.configure(text="⏳ Cargando OCR verde..."))
                self.ocr_green = PaddleOCR(
                    use_angle_cls=False, lang='en', use_gpu=False,
                    enable_mkldnn=False, cpu_threads=4,
                    show_log=False, rec_model_dir=GREEN_REC_DIR
                )

                self.models_loaded = True
                self.after(0, lambda: self.model_status.configure(
                    text="✅ YOLO + OCR Blanco + OCR Verde listos",
                    text_color=GREEN_OK
                ))
                self.after(0, lambda: self._set_status("Modelos cargados — listo para operar"))
            except Exception as e:
                self.after(0, lambda err=e: self.model_status.configure(
                    text=f"❌ Error: {err}", text_color=RED_ERR
                ))

        threading.Thread(target=_load, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    #  CÁMARA
    # ══════════════════════════════════════════════════════════════════════════
    def _start_camera_thread(self):
        def _cam():
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                self.after(0, lambda: self.cam_label.configure(
                    text="❌ Cámara no detectada", text_color=RED_ERR
                ))
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.camera_ok = True
            self.cap = cap
            
            frame_skip = 2
            count = 0
            last_boxes = None

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                with self._frame_lock:
                    self.latest_frame = frame.copy()
                disp_frame = frame.copy()
                
                # YOLO en tiempo real sobre el feed
                if self.models_loaded and self.yolo is not None:
                    count += 1
                    if count % frame_skip == 0:
                        # Predecir sin bloquear demasiado (usa resize interno)
                        _t0 = time.perf_counter()
                        results = self.yolo.predict(source=disp_frame, conf=YOLO_CONF, verbose=False)
                        logger.debug(f"YOLO preview predict tiempo={time.perf_counter()-_t0:.3f}s")
                        last_boxes = results[0].boxes
                        
                    if last_boxes is not None:
                        box_list = []
                        for box in last_boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cx = (x1 + x2) / 2
                            conf = float(box.conf[0])
                            box_list.append({'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2, 'cx':cx, 'conf':conf})
                        
                        # Agrupar por mitad de pantalla (Izquierda vs Derecha)
                        img_w = disp_frame.shape[1]
                        left_cands = [b for b in box_list if b['cx'] < img_w / 2]
                        right_cands = [b for b in box_list if b['cx'] >= img_w / 2]
                        
                        # Tomar los 3 mejores de cada lado
                        left_boxes = sorted(left_cands, key=lambda b: b['conf'], reverse=True)[:3]
                        right_boxes = sorted(right_cands, key=lambda b: b['conf'], reverse=True)[:3]
                        
                        if len(left_boxes) == 3 and len(right_boxes) == 3:
                            for b in left_boxes:
                                cv2.rectangle(disp_frame, (b['x1'], b['y1']), (b['x2'], b['y2']), (0, 255, 0), 4)
                                cv2.putText(disp_frame, "Verde", (b['x1'], max(30, b['y1']-15)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                            
                            for b in right_boxes:
                                cv2.rectangle(disp_frame, (b['x1'], b['y1']), (b['x2'], b['y2']), (255, 255, 255), 4)
                                cv2.putText(disp_frame, "Blanco", (b['x1'], max(30, b['y1']-15)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                        else:
                            for b in box_list:
                                cv2.rectangle(disp_frame, (b['x1'], b['y1']), (b['x2'], b['y2']), (0, 255, 255), 4)
                                cv2.putText(disp_frame, f"Buscando (L:{len(left_boxes)} R:{len(right_boxes)})...", (b['x1'], max(30, b['y1']-15)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                # Bug 9 (config GUI): tamaño configurable desde Ajustes en vez
                # de fijo, se lee en cada frame para reflejar cambios en vivo.
                disp_size = self.cam_display_size
                disp = cv2.resize(disp_frame, disp_size)
                disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(disp)
                imgtk = ctk.CTkImage(light_image=img, size=disp_size)

                self.after(0, lambda i=imgtk: self.cam_label.configure(image=i, text=""))

                time.sleep(0.033)

            cap.release()

        self.latest_frame = None
        threading.Thread(target=_cam, daemon=True).start()

    def _open_new_session_dialog(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Configurar Sesión")
        dialog.geometry("350x400")
        dialog.attributes('-topmost', True)
        
        ctk.CTkLabel(dialog, text="Operador:").pack(pady=(15,0))
        op_entry = ctk.CTkEntry(dialog, width=200)
        op_entry.pack()
        op_entry.insert(0, self.session_data["operador"])
        
        ctk.CTkLabel(dialog, text="ID Pipeta:").pack(pady=(15,0))
        pip_entry = ctk.CTkEntry(dialog, width=200)
        pip_entry.pack()
        pip_entry.insert(0, self.session_data["pipeta"])
        
        ctk.CTkLabel(dialog, text="Volumen Nominal (µL):").pack(pady=(15,0))
        vol_entry = ctk.CTkEntry(dialog, width=200)
        vol_entry.pack()
        vol_entry.insert(0, str(self.session_data["v_nominal"]))
        
        def save():
            try:
                v = float(vol_entry.get())
                self.session_data = {
                    "operador": op_entry.get() or "Operador",
                    "pipeta": pip_entry.get() or "Pipeta",
                    "v_nominal": v
                }
                self.session_measurements = []
                self.session_entries      = []
                self.lbl_session_status.configure(text=f"{self.session_data['pipeta']} | {v}µL | 0/10", text_color=BLUE_DARK)
                dialog.destroy()
            except:
                pass
                
        ctk.CTkButton(dialog, text="Guardar y Reiniciar Lote", command=save).pack(pady=25)
        
    def _finish_session(self, unc_report, evaluacion):
        """Cierra el lote actual: guarda Excel, genera PDF y reinicia el contador."""
        meas_snapshot   = list(self.session_entries)   # copia antes de limpiar
        session_snapshot = dict(self.session_data)
        self.session_measurements = []
        self.session_entries      = []
        v_nom = session_snapshot["v_nominal"]
        pip   = session_snapshot["pipeta"]
        self.lbl_session_status.configure(text=f"{pip} | {v_nom}µL | 0/10", text_color=BLUE_DARK)

        # ── Excel: agregar resumen con APROBADO/RECHAZADO ──────────────────────
        if getattr(self, "data_path_session", None) and self.data_path_session.exists() and self.data_path_session.suffix == ".xlsx":
            try:
                from openpyxl import load_workbook
                from openpyxl.styles import Font, PatternFill
                with self._excel_lock:
                    wb = load_workbook(self.data_path_session)
                    ws = wb.active
                    ws.append([])
                    hdr_row = ws.max_row + 1
                    ws.append(["RESULTADOS METROLÓGICOS ISO 8655-6:2022 / GUM (JCGM 100:2008)"])
                    for cell in ws[hdr_row]:
                        cell.font = Font(bold=True, color="FFFFFF")
                        cell.fill = PatternFill(start_color="1B3A5C", fill_type="solid")
                    ws.append(["N° de réplicas válidas:",           unc_report["n_mediciones"]])
                    ws.append(["Volumen Medio V̄ (µL):",            unc_report["V_media_uL"]])
                    ws.append(["Error Sistemático Es (µL):",        unc_report["error_sistematico_uL"]])
                    ws.append(["CV (%):",                             unc_report["CV_pct"]])
                    ws.append(["Factor Z (µL/mg):",                  unc_report["Z_factor"]])
                    ws.append(["u combinada uc (µL):",               unc_report["u_combinada_uL"]])
                    ws.append(["  componente balanza (µL):",         unc_report["u_balanza_uL"]])
                    ws.append(["  componente repetibilidad (µL):",   unc_report["u_repetibilidad_uL"]])
                    ws.append(["  componente temperatura (µL):",     unc_report["u_temperatura_uL"]])
                    ws.append(["  componente evaporación (µL):",    unc_report["u_evaporacion_uL"]])
                    k   = unc_report["k_cobertura"]
                    nu  = unc_report["grados_libertad"]
                    ws.append([f"Factor de cobertura k (t-Student \u03bd={nu}, 95%):", k])
                    ws.append([f"Incertidumbre Expandida U = k\u00b7uc (µL):",  unc_report["U_expandida_k_uL"]])
                    ws.append([f"U (%):",                             unc_report["U_expandida_k_pct"]])
                    ws.append([])
                    # APROBADO / RECHAZADO
                    verd_row = ws.max_row + 1
                    ws.append([f"EVALUACIÓN ISO 8655-2:2022:  {evaluacion['resultado']}"])
                    color_verd = "22C55E" if evaluacion["aprobado"] else "EF4444"
                    for cell in ws[verd_row]:
                        cell.font = Font(bold=True, color="FFFFFF")
                        cell.fill = PatternFill(start_color=color_verd, fill_type="solid")
                    ws.append(["  Error sistém. |es| (µL):", evaluacion["es_uL"],
                               f"Límite: ≤ {evaluacion['max_es_pct']} %",
                               "PASA" if evaluacion["pasa_error_sistematico"] else "NO PASA"])
                    ws.append(["  Coef. variación CV (%):",   evaluacion["cv_pct"],
                               f"Límite: ≤ {evaluacion['max_cv_pct']} %",
                               "PASA" if evaluacion["pasa_coeficiente_variacion"] else "NO PASA"])
                    wb.save(self.data_path_session)
            except Exception as e:
                logger.error(f"Error guardando reporte metrológico: {e}")

        # ── PDF: generar certificado no editable ──────────────────────────────────
        pdf_path = None
        if _PDF_OK and getattr(self, "data_path_session", None):
            pdf_path = self.data_path_session.with_suffix(".pdf")
            try:
                ok = generar_certificado_pdf(
                    session_data=session_snapshot,
                    measurements=meas_snapshot,
                    unc_report=unc_report,
                    evaluacion=evaluacion,
                    output_path=pdf_path
                )
                if not ok:
                    pdf_path = None
            except Exception as e:
                logger.error(f"Error generando PDF: {e}")
                pdf_path = None

        # ── Popup de resultados ───────────────────────────────────────────────────
        k  = unc_report["k_cobertura"]
        nu = unc_report["grados_libertad"]
        resultado = evaluacion["resultado"]
        emoji_r   = "✅" if evaluacion["aprobado"] else "❌"
        pdf_msg   = f"\nPDF: {pdf_path.name}" if pdf_path else ("\n(instalar reportlab para PDF)" if not _PDF_OK else "")
        txt = (f"{emoji_r}  {resultado}\n\n"
               f"V̄ = {unc_report['V_media_uL']} µL   |   Es = {evaluacion['es_uL']} µL ({evaluacion['es_pct']}%)"
               f"\nCV = {evaluacion['cv_pct']}%"
               f"\nU (k={k}, ν={nu}, 95%) = ±{unc_report['U_expandida_k_uL']} µL"
               f"{pdf_msg}")
        self.after(0, lambda: self._show_error(txt, title=f"ISO 8655-6 — {resultado}"))

    # ══════════════════════════════════════════════════════════════════════════
    #  COM PORT
    # ══════════════════════════════════════════════════════════════════════════
    def _start_com_thread(self):
        # Puerto y baudrate leidos desde config.json (configurables sin editar código)
        _port    = _resolve_com_port(self.config.get("com_port", "/dev/ttyUSB0"))
        _baud    = int(self.config.get("baudrate", 9600))
        def _com():
            while self.running:
                try:
                    ser = serial.Serial(_port, _baud, timeout=1)
                    self.com_connected = True
                    self.after(0, lambda: self._update_com_status(True, f"{_port} @ {_baud}"))

                    while self.running:
                        if ser.in_waiting > 0:
                            raw = ser.readline()
                            line = raw.decode('utf-8', errors='ignore').strip()
                            if line:
                                now = datetime.now().strftime("%H:%M:%S")
                                self.last_com_data = line
                                self.last_com_time = now

                                self.after(0, lambda l=line, t=now:
                                    self.com_data_label.configure(
                                        text=f"Última: {l}  ({t})"
                                    ))

                                com_value = extract_number(line)
                                if com_value != "—":
                                    try:
                                        if float(com_value) == 0.0:
                                            continue  # Ignorar disparo si el peso es 0
                                    except ValueError:
                                        pass
                                        
                                    self.after(0, lambda: self._set_status(
                                        "📡 Dato COM recibido — procesando..."
                                    ))
                                    self._process_measurement(com_value)

                        time.sleep(0.05)

                except serial.SerialException:
                    self.com_connected = False
                    self.after(0, lambda: self._update_com_status(False, "Desconectado"))
                    time.sleep(3)
                except Exception:
                    time.sleep(3)

        threading.Thread(target=_com, daemon=True).start()

    def _update_com_status(self, connected, text):
        color = GREEN_OK if connected else RED_ERR
        self.com_status_led.configure(text_color=color)
        self.com_status_text.configure(text=text)

    # ══════════════════════════════════════════════════════════════════════════
    #  PIPELINE
    # ══════════════════════════════════════════════════════════════════════════
    def _process_measurement(self, com_value="—"):
        # Bug 3: si ya hay un pipeline en curso, ignorar este disparo en vez de
        # apilar otro pipeline en paralelo (evita corrupción de historial/Excel).
        if self._processing:
            return
        self._processing = True
        self._work_queue.put(com_value)

    def _measurement_worker(self):
        """
        Hilo persistente único que procesa TODAS las mediciones, una por una,
        durante toda la vida de la app.

        Bug 8: antes cada medición hacía threading.Thread(target=_run).start(),
        o sea un hilo del SO nuevo por cada medición. El backend de inferencia
        de Paddle (oneDNN/MKL-DNN) cachea sus "primitives" por hilo, y la
        PRIMERA llamada a predictor.run() en un hilo recién creado a veces
        fallaba con "RuntimeError: could not execute a primitive". Como el
        canal verde se procesa antes que el blanco, siempre era el que
        absorbía ese fallo. Al procesar todo desde este único hilo persistente,
        predictor.run() siempre corre sobre el mismo hilo del SO y el problema
        de raíz desaparece.
        """
        while True:
            com_value = self._work_queue.get()
            try:
                max_retries = 3
                entry = None
                roi_crops = []
                valid_dict = {}
                all_valid = False

                # Bug 2: reintentos como loop, no como recursión (la versión
                # anterior lanzaba un threading.Thread nuevo por cada reintento
                # y acumulaba hilos zombi compitiendo por cámara/OCR/Excel).
                for attempt in range(max_retries):
                    if not self.models_loaded or self.latest_frame is None:
                        self.after(0, lambda: self._set_status("⚠️ Modelos o cámara no listos", RED_ERR))
                        break

                    # Bug 1: leer el frame bajo el mismo lock con el que se escribe.
                    with self._frame_lock:
                        frame = self.latest_frame.copy()
                    img_h, img_w = frame.shape[:2]

                    _t0 = time.perf_counter()
                    results = self.yolo.predict(source=frame, conf=YOLO_CONF, verbose=False)
                    logger.debug(f"YOLO medición predict tiempo={time.perf_counter()-_t0:.3f}s attempt={attempt}")
                    boxes = results[0].boxes

                    box_list = []
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cx = (x1 + x2) / 2
                            conf = float(box.conf[0])
                            box_list.append({'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2, 'cx':cx, 'conf':conf})

                    # Agrupar por mitad de pantalla (Izquierda vs Derecha)
                    left_cands = [b for b in box_list if b['cx'] < img_w / 2]
                    right_cands = [b for b in box_list if b['cx'] >= img_w / 2]

                    # Tomar los 3 mejores de cada lado
                    left_boxes = sorted(left_cands, key=lambda b: b['conf'], reverse=True)[:3]
                    right_boxes = sorted(right_cands, key=lambda b: b['conf'], reverse=True)[:3]

                    if len(left_boxes) < 3 or len(right_boxes) < 3:
                        play_sound("retry")
                        self.after(0, lambda lb=len(left_boxes), rb=len(right_boxes), a=attempt:
                            self._set_status(f"⚠️ YOLO detectó L:{lb} R:{rb}, reintentando ({a+1}/{max_retries})...", RED_ERR))
                        time.sleep(1.5)
                        continue  # próximo intento del loop

                    # Ordenar de arriba a abajo en cada lado
                    left_boxes.sort(key=lambda b: b['y1'])
                    right_boxes.sort(key=lambda b: b['y1'])

                    roi_crops = []

                    # 1. OCR Green (Izquierda)
                    humedad_val, temp_amb_val, presion_val = "—", "—", "—"
                    labels_green = ["Hum. Relativa", "Temp. Amb.", "Pres. Atmos."]
                    for i, b in enumerate(left_boxes):
                        x1, y1, x2, y2 = expand_box(b['x1'], b['y1'], b['x2'], b['y2'], img_h, img_w, "roi_green")
                        crop = frame[y1:y2, x1:x2]
                        val = self._run_ocr(self.ocr_green, crop, label=f"green/{labels_green[i]}")
                        roi_crops.append((labels_green[i], crop))
                        if i == 0: humedad_val = val
                        elif i == 1: temp_amb_val = val
                        elif i == 2: presion_val = val

                    # 2. OCR White (Derecha) — Termómetro Traceable Excursion-Trac
                    # t_agua = temperatura actual del agua (lectura principal del probe)
                    # NOTA METROLÓGICA: este ROI es el del termómetro de agua (Excursion-Trac),
                    # NO el del termohigrómetro ambiental (Extech SD700). Se usa
                    # directamente como T_agua en z_factor() conforme ISO 8655-6.
                    t_agua_val, temp_min_val, temp_max_val = "—", "—", "—"
                    labels_white = ["T.Agua (Excursion-Trac)", "T.Agua Mín.", "T.Agua Máx."]
                    for i, b in enumerate(right_boxes):
                        x1, y1, x2, y2 = expand_box(b['x1'], b['y1'], b['x2'], b['y2'], img_h, img_w, "roi_white")
                        crop = frame[y1:y2, x1:x2]
                        val = self._run_ocr(self.ocr_white, crop, label=f"white/{labels_white[i]}")
                        roi_crops.append((labels_white[i], crop))
                        if i == 0: t_agua_val = val
                        elif i == 1: temp_min_val = val
                        elif i == 2: temp_max_val = val

                    now = datetime.now().strftime("%H:%M:%S")
                    entry = {
                        "num": None,  # se asigna atómicamente más abajo
                        "hora": now,
                        "peso": com_value,
                        "humedad": humedad_val,
                        "temp_amb": temp_amb_val,
                        "presion": presion_val,
                        # t_agua: temperatura del AGUA medida por Excursion-Trac (display blanco)
                        # — fuente directa para z_factor() conforme ISO 8655-6:2022 §7.2
                        "t_agua": t_agua_val,
                        "temp_min": temp_min_val,
                        "temp_max": temp_max_val,
                    }

                    valid_dict = validate_readings(entry)
                    all_valid = all(valid_dict.values())

                    if all_valid:
                        break

                    if attempt < max_retries - 1:
                        play_sound("retry")
                        self.after(0, lambda e=entry, c=roi_crops, v=valid_dict: self._update_readings(e, c, v, add_to_table=False))
                        failed = [k for k, v in valid_dict.items() if not v]
                        self.after(0, lambda a=attempt, fl=failed: self._set_status(
                            f"⚠️ Lectura inválida en {fl} — contramedida (Intento {a+1}/{max_retries})...", RED_ERR))
                        time.sleep(1.5)
                        # continúa al siguiente intento del loop

                if entry is None:
                    # Ningún intento logró detectar los 6 ROIs (o modelos/cámara no listos)
                    play_sound("error")
                    self.after(0, lambda: self._set_status("❌ Fallo: YOLO no encontró los 6 ROIs tras 3 intentos", RED_ERR))
                    continue

                # Bug 7: número de medición y append al historial bajo lock,
                # para que dos mediciones casi simultáneas no colisionen.
                with self._hist_lock:
                    self._measure_counter += 1
                    entry["num"] = self._measure_counter
                    self.historial.append(entry)

                self._save_measurement(entry)
                self.after(0, lambda e=entry, c=roi_crops, v=valid_dict: self._update_readings(e, c, v, add_to_table=True))

                # ── Audit Trail INTEGRAL: registrar TODA medición ────────────────────
                # NTP-ISO/IEC 17025:2017 §7.5: trazabilidad de TODOS los datos.
                # Incluye errores y mediciones libres (fuera de sesión ISO).
                _operator = (self.session_data["operador"]
                             if getattr(self, "session_data", None) else "libre")
                if all_valid:
                    _action = ("MEASURE_VALID_ISO" if self.session_data
                               else "MEASURE_VALID")
                else:
                    _action = "MEASURE_ERROR"
                self.audit.log(_action, entry["num"], _operator, entry,
                               valid_dict=valid_dict)

                if all_valid:
                    play_sound("success")
                    self.after(0, lambda e=entry: self._set_status(f"✅ Medición #{e['num']} registrada — {e['hora']}", GREEN_OK))
                    
                    # ISO 8655 Session Logic
                    if getattr(self, "session_data", None) is not None:
                        try:
                            # Temperatura del agua: campo t_agua (Excursion-Trac, display blanco)
                            # conforme ISO 8655-6:2022 §7.2 y la función z_factor() CIPM-2001
                            t_agua = float(entry["t_agua"])
                            mass_mg = float(entry["peso"])
                            v_real = volume_from_mass(mass_mg, t_agua)
                            entry["vol_real"] = round(v_real, 2)
                            
                            self.session_measurements.append(v_real)
                            # Guardar copia del entry para el PDF (necesita entry completo)
                            self.session_entries.append(entry.copy())
                            n_meas = len(self.session_measurements)
                            v_nom = self.session_data["v_nominal"]
                            pip = self.session_data["pipeta"]
                            self.after(0, lambda n=n_meas: self.lbl_session_status.configure(text=f"{pip} | {v_nom}µL | {n}/10"))
                            
                            if n_meas >= 10:
                                unc = calculate_expanded_uncertainty(v_nom, self.session_measurements, t_agua)
                                evaluacion = evaluar_calibracion(unc, v_nom)
                                # Audit Trail: registrar cierre de sesión
                                self.audit.log("SESSION_COMPLETE", entry["num"],
                                               self.session_data["operador"],
                                               {"veredicto": evaluacion["resultado"],
                                                "V_media": unc["V_media_uL"],
                                                "U_k": unc["U_expandida_k_uL"],
                                                "k": unc["k_cobertura"]})
                                self._finish_session(unc, evaluacion)
                        except Exception as ex:
                            logger.error(f"Error procesando sesión ISO: {ex}")
                else:
                    play_sound("error")
                    self.after(0, lambda e=entry: self._set_status(f"❌ Medición #{e['num']} guardada con errores (Máx reintentos)", RED_ERR))
            except Exception:
                err = traceback.format_exc()
                logger.error(f"Excepción no controlada en _measurement_worker:\n{err}")
                self._log_error("Pipeline Worker Error", err)
                self.after(0, lambda: self._set_status("❌ Error interno en el pipeline (ver debug_ocr.log)", RED_ERR))
            finally:
                # Bug 3: liberar el flag pase lo que pase (éxito, fallo o excepción)
                # para que el siguiente dato del COM pueda disparar un nuevo pipeline.
                self._processing = False
                self._work_queue.task_done()

    def _run_ocr(self, ocr_model, crop, label="?"):
        t0 = time.perf_counter()
        try:
            with self._ocr_lock:
                res = ocr_model.ocr(crop, det=False, cls=False)
            elapsed = time.perf_counter() - t0

            if res and res[0] and len(res[0]) > 0:
                raw_text = res[0][0][0]
                confidence = res[0][0][1] if len(res[0][0]) > 1 else None
                text = raw_text.strip()
                num = extract_number(text)
                logger.debug(
                    f"OCR OK [{label}] crop={crop.shape} tiempo={elapsed:.3f}s "
                    f"raw='{raw_text}' conf={confidence} -> '{num}'"
                )
                return num
            else:
                # Bug 8 (candidato): Paddle no lanzó excepción, pero devolvió
                # una estructura vacía/None. Antes esto caía silenciosamente
                # a "—" sin dejar rastro de qué se recibió realmente.
                logger.warning(
                    f"OCR VACÍO [{label}] crop={crop.shape} tiempo={elapsed:.3f}s "
                    f"res_repr={res!r}"
                )
        except Exception:
            elapsed = time.perf_counter() - t0
            err_trace = traceback.format_exc()
            logger.error(
                f"OCR EXCEPCIÓN [{label}] crop={crop.shape if crop is not None else None} "
                f"tiempo={elapsed:.3f}s\n{err_trace}"
            )
            self._log_error(f"OCR Exception [{label}]", err_trace)
        return "—"

    def _update_readings(self, entry, roi_crops, valid_dict, add_to_table=True):
        # Panel: Colorear de rojo si no es válido
        def get_color(key):
            return BLUE_DARK if valid_dict.get(key, False) else RED_ERR

        self.reading_labels["peso"].configure(text=f"{entry['peso']} mg", text_color=get_color("peso"))
        self.reading_labels["humedad"].configure(text=f"{entry['humedad']} %RH", text_color=get_color("humedad"))
        self.reading_labels["temp_amb"].configure(text=f"{entry['temp_amb']} °C", text_color=get_color("temp_amb"))
        self.reading_labels["presion"].configure(text=f"{entry['presion']} hPa", text_color=get_color("presion"))
        self.reading_labels["t_agua"].configure(text=f"{entry['t_agua']} °C", text_color=get_color("t_agua"))
        self.reading_labels["temp_min"].configure(text=f"{entry['temp_min']} °C", text_color=get_color("temp_min"))
        self.reading_labels["temp_max"].configure(text=f"{entry['temp_max']} °C", text_color=get_color("temp_max"))

        # Thumbnails
        if self.roi_placeholder.winfo_ismapped():
            self.roi_placeholder.pack_forget()

        for w in self.roi_scroll.winfo_children():
            if w != self.roi_placeholder:
                w.destroy()
        self.roi_images_tk.clear()

        for label, crop in roi_crops:
            roi_container = ctk.CTkFrame(self.roi_scroll, fg_color=BG_CARD_DARK, corner_radius=8)
            roi_container.pack(side="left", padx=5, pady=5)

            h, w_c = crop.shape[:2]
            scale = 80 / max(h, 1)
            thumb = cv2.resize(crop, (int(w_c * scale), 80))
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(thumb_rgb)
            imgtk = ctk.CTkImage(light_image=img, size=(int(w_c * scale), 80))
            self.roi_images_tk.append(imgtk)

            ctk.CTkLabel(roi_container, image=imgtk, text="").pack(padx=5, pady=(5, 0))
            ctk.CTkLabel(
                roi_container, text=label,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=BLUE_DARK
            ).pack(pady=(0, 5))

        if not add_to_table:
            return

        # Tabla
        col_widths = [30, 70, 90, 80, 80, 100, 90, 90, 90]
        row_frame = ctk.CTkFrame(self.table_scroll, fg_color="transparent", height=28)
        row_frame.pack(fill="x", pady=1)

        vals = [
            str(entry["num"]), entry["hora"], entry["peso"],
            entry["humedad"], entry["temp_amb"], entry["presion"],
            entry["t_agua"], entry["temp_min"], entry["temp_max"]
        ]
        
        # Resaltar fila entera si hubo error permanente, o celda específica
        for i, (val, w) in enumerate(zip(vals, col_widths)):
            keys = ["", "", "peso", "humedad", "temp_amb", "presion", "t_agua", "temp_min", "temp_max"]
            c = BLUE_DARK
            if i >= 2 and not valid_dict.get(keys[i], False):
                c = RED_ERR
                
            ctk.CTkLabel(
                row_frame, text=val,
                font=ctk.CTkFont(family=self.config["font_family"], size=max(9, self.config["font_size"] - 5), weight="bold" if c == RED_ERR else "normal"),
                text_color=c, width=w, anchor="center"
            ).pack(side="left", padx=2)

    # ══════════════════════════════════════════════════════════════════════════
    #  EXPORTACIÓN DE DATOS (xlsx / csv / txt)
    # ══════════════════════════════════════════════════════════════════════════
    _HEADERS = ["#", "Hora", "Peso (mg)", "Humedad %RH", "T.Amb.Extech °C",
                "Presión hPa", "T.Agua.ExcursionTrac °C", "T.Agua Mín °C", "T.Agua Máx °C", "Volumen Real (µL)"]

    def _update_data_path(self, last_time=None):
        """Recalcula las rutas según el formato de exportación elegido."""
        ext = self.config.get("export_format", "xlsx")
        if ext not in ("xlsx", "csv", "txt"):
            ext = "xlsx"
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 1. Reporte Diario
        self.data_path_daily = EXCEL_DIR / f"Reporte_{today}.{ext}"
        
        # 2. Reporte de Sesión (se renombra con cada última medición)
        sesiones_dir = EXCEL_DIR / "sesiones"
        sesiones_dir.mkdir(exist_ok=True)
        
        if last_time is None:
            last_time = getattr(self, "last_time_str", None)
        else:
            self.last_time_str = last_time
            
        if last_time:
            self.data_path_session = sesiones_dir / f"Sesion_{today}_{last_time}.{ext}"
        else:
            self.data_path_session = None

    def _save_measurement(self, entry):
        """Guarda una medición en el formato configurado (xlsx/csv/txt)."""
        if not getattr(self, "session_started", False):
            self.session_started = True

        last_time_clean = entry["hora"].replace(":", "-")
        old_session_path = getattr(self, "data_path_session", None)
        
        self._update_data_path(last_time=last_time_clean)

        # Si el archivo de sesión de este formato ya existe y cambió de hora, lo renombramos
        if old_session_path and old_session_path.exists() and old_session_path != self.data_path_session:
            if old_session_path.suffix == self.data_path_session.suffix:
                old_session_path.rename(self.data_path_session)

        fmt = self.config.get("export_format", "xlsx")
        try:
            paths_to_save = [p for p in (self.data_path_daily, self.data_path_session) if p]
            for p in paths_to_save:
                if fmt == "csv":
                    self._save_entry_csv(entry, p)
                elif fmt == "txt":
                    self._save_entry_txt(entry, p)
                else:
                    self._save_entry_xlsx(entry, p)
        except Exception as e:
            self.after(0, lambda err=e: self._set_status(
                f"❌ Error guardando datos (dato #{entry.get('num', '?')} no persistido): {err}", RED_ERR
            ))

    def _row_values(self, entry):
        return [entry["num"], entry["hora"], entry["peso"], entry["humedad"],
                entry["temp_amb"], entry["presion"], entry["t_agua"],
                entry["temp_min"], entry["temp_max"], entry.get("vol_real", "—")]

    def _save_entry_xlsx(self, entry, path):
        with self._excel_lock:
            if path.exists():
                wb = load_workbook(path)
                ws = wb.active
            else:
                wb = Workbook()
                ws = wb.active
                ws.title = "Calibración"
                ws.append(self._HEADERS)

                from openpyxl.styles import Font, PatternFill, Alignment
                for cell in ws[1]:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill(start_color="1B3A5C", fill_type="solid")
                    cell.alignment = Alignment(horizontal="center")

            ws.append(self._row_values(entry))
            wb.save(path)

    def _save_entry_csv(self, entry, path):
        import csv
        file_exists = path.exists()
        with self._excel_lock:
            with open(path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(self._HEADERS)
                writer.writerow(self._row_values(entry))

    def _save_entry_txt(self, entry, path):
        widths = [4, 10, 10, 10, 10, 10, 10, 10, 10]
        file_exists = path.exists()
        with self._excel_lock:
            with open(path, 'a', encoding='utf-8') as f:
                if not file_exists:
                    header_line = " ".join(h.rjust(w) for h, w in zip(self._HEADERS, widths))
                    f.write(header_line + "\n")
                    f.write("-" * len(header_line) + "\n")
                row_line = " ".join(str(v).rjust(w) for v, w in zip(self._row_values(entry), widths))
                f.write(row_line + "\n")

    def _export_excel(self):
        if not self.historial:
            self._set_status("⚠️ No hay datos para exportar")
            return
        fmt = self.config.get("export_format", "xlsx").upper()
        self._set_status(f"💾 Datos guardados ({fmt}): {self.data_path_daily.name}")

    # ══════════════════════════════════════════════════════════════════════════
    #  AJUSTES
    # ══════════════════════════════════════════════════════════════════════════
    def _open_settings(self):
        win = ctk.CTkToplevel(self)
        win.title("⚙️ Configuración GUI")
        win.geometry("480x560")
        win.attributes('-topmost', True)

        tabs = ctk.CTkTabview(win, width=440, height=520)
        tabs.pack(padx=15, pady=15, fill="both", expand=True)
        tab_apariencia = tabs.add("🎨 Apariencia")
        tab_sonido = tabs.add("🔊 Sonido")
        tab_diseno = tabs.add("🧩 Diseño")
        tab_export = tabs.add("💾 Exportación")

        # ── helper genérico: fila "label + widget a la derecha" ─────────────
        def _row(parent, text, pady=8):
            f = ctk.CTkFrame(parent, fg_color="transparent")
            f.pack(fill="x", padx=15, pady=pady)
            ctk.CTkLabel(f, text=text, anchor="w").pack(side="left")
            return f

        # ════════════════════════════ APARIENCIA ═══════════════════════════
        tema_frame = _row(tab_apariencia, "Tema:")
        tema_var = ctk.StringVar(value=self.config["theme"])
        def change_theme(val):
            ctk.set_appearance_mode(val)
            self.config["theme"] = val
            save_config(self.config)
        ctk.CTkOptionMenu(tema_frame, values=["Light", "Dark", "System"],
                           variable=tema_var, command=change_theme, width=140).pack(side="right")

        escala_frame = _row(tab_apariencia, "Tamaño de interfaz:")
        escala_lbl = ctk.CTkLabel(escala_frame, text=f"{self.config['scaling']:.1f}x")
        escala_lbl.pack(side="right", padx=(10, 0))
        def change_scaling(val):
            val = round(float(val), 1)
            escala_lbl.configure(text=f"{val:.1f}x")
            ctk.set_widget_scaling(val)
            self.config["scaling"] = val
            save_config(self.config)
        escala_slider = ctk.CTkSlider(tab_apariencia, from_=0.8, to=2.0, number_of_steps=12, command=change_scaling)
        escala_slider.set(self.config["scaling"])
        escala_slider.pack(fill="x", padx=15, pady=(0, 15))

        fuente_frame = _row(tab_apariencia, "Tipo de letra:")
        import tkinter.font as tkfont
        try:
            familias = sorted(set(tkfont.families(win)))
        except Exception:
            familias = ["Consolas", "Arial", "Helvetica", "Courier New", "Segoe UI"]
        fuente_var = ctk.StringVar(value=self.config["font_family"])
        def change_font_family(val):
            self.config["font_family"] = val
            save_config(self.config)
            self.font_readings.configure(family=val)
            self.font_table.configure(family=val)
        ctk.CTkComboBox(fuente_frame, values=familias, variable=fuente_var,
                         command=change_font_family, width=180).pack(side="right")

        tam_letra_frame = _row(tab_apariencia, "Tamaño de letra (lecturas):")
        tam_letra_lbl = ctk.CTkLabel(tam_letra_frame, text=str(self.config["font_size"]))
        tam_letra_lbl.pack(side="right", padx=(10, 0))
        def change_font_size(val):
            val = int(round(float(val)))
            tam_letra_lbl.configure(text=str(val))
            self.config["font_size"] = val
            save_config(self.config)
            self.font_readings.configure(size=val)
            self.font_table.configure(size=max(9, val - 5))
        tam_letra_slider = ctk.CTkSlider(tab_apariencia, from_=10, to=28, number_of_steps=18, command=change_font_size)
        tam_letra_slider.set(self.config["font_size"])
        tam_letra_slider.pack(fill="x", padx=15, pady=(0, 15))

        # ══════════════════════════════ SONIDO ══════════════════════════════
        vol_frame = _row(tab_sonido, "Volumen de alarmas:")
        vol_lbl = ctk.CTkLabel(vol_frame, text=f"{int(self.config['alarm_volume']*100)}%")
        vol_lbl.pack(side="right", padx=(10, 0))
        def change_volume(val):
            v = round(float(val), 2)
            vol_lbl.configure(text=f"{int(v*100)}%")
            self.config["alarm_volume"] = v
            save_config(self.config)
            set_alarm_volume(v)
        vol_slider = ctk.CTkSlider(tab_sonido, from_=0.0, to=1.0, number_of_steps=20, command=change_volume)
        vol_slider.set(self.config["alarm_volume"])
        vol_slider.pack(fill="x", padx=15, pady=(0, 10))

        ctk.CTkButton(
            tab_sonido, text="▶️  Probar sonido", width=160,
            command=lambda: play_sound("success")
        ).pack(padx=15, pady=10, anchor="w")

        # ══════════════════════════════ DISEÑO ══════════════════════════════
        cam_frame = _row(tab_diseno, "Tamaño de vista de cámara:")
        cam_labels = {"compact": "Compacta", "normal": "Normal", "large": "Grande"}
        cam_labels_rev = {v: k for k, v in cam_labels.items()}
        cam_var = ctk.StringVar(value=cam_labels.get(self.config["cam_size"], "Normal"))
        def change_cam_size(val):
            key = cam_labels_rev.get(val, "normal")
            self.config["cam_size"] = key
            save_config(self.config)
            self.cam_display_size = CAM_SIZES[key]
            w, h = self.cam_display_size
            self.cam_label.configure(width=w, height=h)
        ctk.CTkOptionMenu(cam_frame, values=list(cam_labels.values()),
                           variable=cam_var, command=change_cam_size, width=140).pack(side="right")

        roi_frame = _row(tab_diseno, "Mostrar miniaturas de ROIs:")
        roi_switch_var = ctk.BooleanVar(value=self.config.get("show_roi_thumbnails", True))
        def toggle_roi(*_):
            show = roi_switch_var.get()
            self.config["show_roi_thumbnails"] = show
            save_config(self.config)
            if show:
                self.roi_card.pack(fill="x", pady=(0, 10), before=self.hist_card)
            else:
                self.roi_card.pack_forget()
        ctk.CTkSwitch(roi_frame, text="", variable=roi_switch_var, command=toggle_roi).pack(side="right")

        tabla_frame = _row(tab_diseno, "Alto de tabla de historial:")
        tabla_lbl = ctk.CTkLabel(tabla_frame, text=f"{self.config['table_rows_height']}px")
        tabla_lbl.pack(side="right", padx=(10, 0))
        def change_table_height(val):
            val = int(round(float(val) / 10) * 10)
            tabla_lbl.configure(text=f"{val}px")
            self.config["table_rows_height"] = val
            save_config(self.config)
            self.table_scroll.configure(height=val)
        tabla_slider = ctk.CTkSlider(tab_diseno, from_=100, to=400, number_of_steps=30, command=change_table_height)
        tabla_slider.set(self.config["table_rows_height"])
        tabla_slider.pack(fill="x", padx=15, pady=(0, 15))

        # ═══════════════════════════ EXPORTACIÓN ════════════════════════════
        ctk.CTkLabel(
            tab_export, text="Formato de archivo de datos:",
            anchor="w"
        ).pack(fill="x", padx=15, pady=(15, 5))

        export_var = ctk.StringVar(value=self.config.get("export_format", "xlsx"))
        ruta_lbl = ctk.CTkLabel(
            tab_export, text=f"Archivo diario: {self.data_path_daily.name}",
            font=ctk.CTkFont(size=11), text_color=GRAY_TEXT, anchor="w",
            wraplength=400, justify="left"
        )
        def change_export_format(val):
            self.config["export_format"] = val
            save_config(self.config)
            self._update_data_path()
            ruta_lbl.configure(text=f"Archivo diario: {self.data_path_daily.name}")
            self._set_status(f"📄 Formato de exportación cambiado a {val.upper()}")

        ctk.CTkSegmentedButton(
            tab_export, values=["xlsx", "csv", "txt"],
            variable=export_var, command=change_export_format
        ).pack(fill="x", padx=15, pady=(0, 10))

        ruta_lbl.pack(fill="x", padx=15, pady=(10, 5))

        ctk.CTkLabel(
            tab_export,
            text="Nota: cambiar el formato crea/usa un archivo nuevo con esa\n"
                 "extensión; las mediciones anteriores no se migran.",
            font=ctk.CTkFont(size=10), text_color=GRAY_TEXT, anchor="w",
            justify="left"
        ).pack(fill="x", padx=15, pady=(0, 10))

        # ══════════════════════════════ PUERTO COM ═════════════════════════════
        tab_com = tabs.add("🔌 Puerto COM")

        def _get_ports():
            import serial.tools.list_ports
            return [p.device for p in serial.tools.list_ports.comports()] or ["/dev/ttyUSB0"]

        _row(tab_com, "Puerto serie de la balanza:", pady=4)
        port_var = ctk.StringVar(value=self.config.get("com_port", "/dev/ttyUSB0"))
        ports_now = _get_ports()
        port_menu = ctk.CTkOptionMenu(tab_com, values=ports_now, variable=port_var, width=200)
        port_menu.pack(padx=15, pady=(0,5), anchor="w")

        def refresh_ports():
            new_ports = _get_ports()
            port_menu.configure(values=new_ports)
            self._set_status(f"🔍 Puertos detectados: {new_ports}")
        ctk.CTkButton(tab_com, text="🔄  Refrescar puertos",
                      command=refresh_ports, width=180).pack(padx=15, pady=(0,10), anchor="w")

        _row(tab_com, "Baudrate:", pady=2)
        baud_var = ctk.StringVar(value=str(self.config.get("baudrate", 9600)))
        ctk.CTkOptionMenu(tab_com, values=["9600","19200","38400","57600","115200"],
                          variable=baud_var, width=120).pack(padx=15, pady=(0,15), anchor="w")

        def apply_com():
            self.config["com_port"] = port_var.get()
            self.config["baudrate"] = int(baud_var.get())
            save_config(self.config)
            self._set_status(f"⚠️  Puerto COM actualizado → {port_var.get()} @ {baud_var.get()} bd. Reinicia la app para aplicar.")

        ctk.CTkButton(tab_com, text="💾  Guardar configuración COM",
                      command=apply_com, width=220).pack(padx=15, pady=(0,8), anchor="w")
        ctk.CTkLabel(
            tab_com,
            text="El cambio aplica al reiniciar la aplicación.\n"
                 "Si la balanza no se detecta, usa 'Refrescar puertos'.",
            font=ctk.CTkFont(size=10), text_color=GRAY_TEXT, anchor="w", justify="left"
        ).pack(fill="x", padx=15, pady=(0,10))

    # ══════════════════════════════════════════════════════════════════════════
    #  CAPTURA MANUAL
    # ══════════════════════════════════════════════════════════════════════════
    def _manual_capture(self):
        if not self.models_loaded:
            self._set_status("⚠️ Espera a que los modelos terminen de cargar")
            return
        if self.latest_frame is None:
            self._set_status("⚠️ Cámara no disponible")
            return

        self._set_status("📷 Captura manual en proceso...")
        self._process_measurement(com_value="MANUAL")

    def _on_close(self):
        self.running = False
        time.sleep(0.2)
        self.destroy()

if __name__ == "__main__":
    app = CalibracionApp()
    app.mainloop()