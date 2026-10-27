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
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio

CONFIG_FILE = (ROOT / "config.json")

def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except: pass
    return {"theme": "Light", "scaling": 1.0}

def save_config(cfg):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f)

# ── SONIDOS ───────────────────────────────────────────────────────────────────
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
    try:
        if sound_type == "success":
            _generate_beep("/tmp/beep_s1.wav", 800, 100)
            _generate_beep("/tmp/beep_s2.wav", 1200, 150)
            subprocess.Popen("aplay -q /tmp/beep_s1.wav && aplay -q /tmp/beep_s2.wav", shell=True)
        elif sound_type == "retry":
            _generate_beep("/tmp/beep_retry.wav", 400, 200)
            subprocess.Popen("aplay -q /tmp/beep_retry.wav", shell=True)
        elif sound_type == "error":
            _generate_beep("/tmp/beep_error.wav", 150, 600)
            subprocess.Popen("aplay -q /tmp/beep_error.wav", shell=True)
    except:
        pass

# ── RUTAS DE MODELOS ──────────────────────────────────────────────────────────
YOLO_MODEL    = (ROOT / "models/yolo/roi_detector_v1.pt")
WHITE_REC_DIR = str((ROOT / "models/ocr_white"))
GREEN_REC_DIR = str((ROOT / "models/ocr_green"))
EXCEL_DIR     = (ROOT / "resultados")
EXCEL_DIR.mkdir(exist_ok=True)

# ── PARÁMETROS ────────────────────────────────────────────────────────────────
CAMERA_INDEX = 2
COM_PORT     = "/dev/ttyUSB0"
BAUDRATE     = 9600
YOLO_CONF    = 0.4
SCALE        = 3

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
    
    def check(val_str, v_min, v_max, decimals=None):
        if val_str == "—": return False
        try:
            v = float(val_str)
            if not (v_min <= v <= v_max): return False
            if decimals is not None and "." in val_str:
                # Tolerar menos decimales de los esperados (ej. display sin el
                # último dígito o redondeo del OCR); solo rechazar si hay MÁS
                # decimales de los que la magnitud física admite.
                if len(val_str.split(".")[1]) > decimals: return False
            return True
        except:
            return False

    valid["humedad"]   = check(entry["humedad"], 20.0, 80.0, 1)
    valid["temp_amb"]  = check(entry["temp_amb"], 15.0, 35.0, 1)
    valid["presion"]   = check(entry["presion"], 900.0, 1100.0, 1)
    valid["temp_prom"] = check(entry["temp_prom"], 15.0, 35.0, 2)
    valid["temp_min"]  = check(entry["temp_min"], 10.0, 35.0, 2)
    valid["temp_max"]  = check(entry["temp_max"], 15.0, 45.0, 2)
    valid["peso"]      = True # El peso viene del COM, asumimos ok
    
    # Regla física: T.Min <= T.Prom <= T.Max
    try:
        t_min = float(entry["temp_min"])
        t_prom = float(entry["temp_prom"])
        t_max = float(entry["temp_max"])
        if not (t_min <= t_prom <= t_max):
            valid["temp_prom"] = valid["temp_min"] = valid["temp_max"] = False
    except:
        pass
        
    return valid



# ══════════════════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════════════════
class CalibracionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.config = load_config()
        ctk.set_appearance_mode(self.config["theme"])
        ctk.set_widget_scaling(self.config["scaling"])

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

        # ── Concurrencia (locks / flags anti race-condition) ────────────────────
        self._frame_lock = threading.Lock()   # protege self.latest_frame (hilo cámara vs pipeline)
        self._ocr_lock = threading.Lock()     # PaddleOCR no es thread-safe
        self._excel_lock = threading.Lock()   # protege escritura del .xlsx
        self._hist_lock = threading.Lock()    # protege self.historial y el contador de medición
        self._processing = False              # evita pipelines paralelos disparados por el COM
        self._measure_counter = 0             # contador atómico para entry["num"]

        # ── Modelos (carga diferida) ──────────────────────────────────────────
        self.yolo = None
        self.ocr_white = None
        self.ocr_green = None
        self.models_loaded = False

        # ── Excel ─────────────────────────────────────────────────────────────
        today = datetime.now().strftime("%Y-%m-%d")
        self.excel_path = EXCEL_DIR / f"calibracion_{today}.xlsx"

        # ── Construir GUI ─────────────────────────────────────────────────────
        self._build_header()
        self._build_body()
        self._build_footer()

        # ── Iniciar hilos ─────────────────────────────────────────────────────
        self.after(100, self._load_models_async)
        self.after(200, self._start_camera_thread)
        self.after(300, self._start_com_thread)

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

        self.cam_label = ctk.CTkLabel(
            cam_card, text="Iniciando cámara...",
            font=ctk.CTkFont(size=14), text_color=GRAY_TEXT,
            width=560, height=315
        )
        self.cam_label.pack(padx=10, pady=10)

        # Panel derecho: Estado COM + Lecturas
        right_panel = ctk.CTkFrame(row1, fg_color="transparent", width=420)
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

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
            ("peso",        "Peso (COM)",             "— g",    "⚖️"),
            ("humedad",     "Humedad Relativa",       "— %",    "💧"),
            ("temp_amb",    "Temperatura Ambiental",  "— °C",   "🌡️"),
            ("presion",     "Presión Atmosférica",    "— hPa",  "📊"),
            ("temp_prom",   "Temperatura Promedio",   "— °C",   "📈"),
            ("temp_min",    "Temperatura Mínima",     "— °C",   "📉"),
            ("temp_max",    "Temperatura Máxima",     "— °C",   "🔥"),
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
                font=ctk.CTkFont(family="Consolas", size=16, weight="bold"),
                text_color=BLUE_DARK, anchor="e"
            )
            val_lbl.pack(side="right")
            self.reading_labels[key] = val_lbl

        # ── Fila 2: ROIs detectados ───────────────────────────────────────
        roi_card = self._card(body, "🔍  ROIs Detectados por YOLO")
        roi_card.pack(fill="x", pady=(0, 10))

        self.roi_scroll = ctk.CTkScrollableFrame(roi_card, fg_color="transparent", height=120, orientation="horizontal")
        self.roi_scroll.pack(fill="x", padx=10, pady=5)

        self.roi_placeholder = ctk.CTkLabel(
            self.roi_scroll, text="Los recortes aparecerán aquí al capturar una imagen",
            font=ctk.CTkFont(size=12), text_color=GRAY_TEXT
        )
        self.roi_placeholder.pack(pady=40)

        # ── Fila 3: Historial ─────────────────────────────────────────────
        hist_card = self._card(body, "📋  Historial de Mediciones")
        hist_card.pack(fill="both", expand=True)

        # Headers de tabla
        tbl_header = ctk.CTkFrame(hist_card, fg_color=BG_CARD_DARK, height=35, corner_radius=6)
        tbl_header.pack(fill="x", padx=10, pady=(5, 0))
        tbl_header.pack_propagate(False)

        cols = ["#", "Hora", "Peso (g)", "Hum %", "T.Amb °C", "Presión hPa", "T.Prom °C", "T.Mín °C", "T.Máx °C"]
        col_widths = [30, 70, 90, 80, 80, 100, 90, 90, 90]
        for i, (col, w) in enumerate(zip(cols, col_widths)):
            ctk.CTkLabel(
                tbl_header, text=col,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=BLUE_DARK, width=w, anchor="center"
            ).pack(side="left", padx=2)

        # Scrollable table body
        self.table_scroll = ctk.CTkScrollableFrame(
            hist_card, fg_color=BG_WHITE, height=150,
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
            btn_frame, text="💾  Exportar Excel", width=180, height=38,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#059669", hover_color="#047857",
            command=self._export_excel
        ).pack(side="left", padx=10)

        self.status_bar = ctk.CTkLabel(
            footer, text="Listo",
            font=ctk.CTkFont(size=11), text_color=GRAY_TEXT
        )
        self.status_bar.pack(side="right", padx=20)

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
                self.ocr_white = PaddleOCR(
                    use_angle_cls=False, lang='en', use_gpu=False,
                    show_log=False, rec_model_dir=WHITE_REC_DIR
                )

                self.after(0, lambda: self.model_status.configure(text="⏳ Cargando OCR verde..."))
                self.ocr_green = PaddleOCR(
                    use_angle_cls=False, lang='en', use_gpu=False,
                    show_log=False, rec_model_dir=GREEN_REC_DIR
                )

                self.models_loaded = True
                self.after(0, lambda: self.model_status.configure(
                    text="✅ YOLO + OCR Blanco + OCR Verde listos",
                    text_color=GREEN_OK
                ))
                self.after(0, lambda: self._set_status("Modelos cargados — listo para operar"))
            except Exception as e:
                self.after(0, lambda: self.model_status.configure(
                    text=f"❌ Error: {e}", text_color=RED_ERR
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
                        results = self.yolo.predict(source=disp_frame, conf=YOLO_CONF, verbose=False)
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

                disp = cv2.resize(disp_frame, (560, 315))
                disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(disp)
                imgtk = ctk.CTkImage(light_image=img, size=(560, 315))

                self.after(0, lambda i=imgtk: self.cam_label.configure(image=i, text=""))

                time.sleep(0.033)

            cap.release()

        self.latest_frame = None
        threading.Thread(target=_cam, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    #  COM PORT
    # ══════════════════════════════════════════════════════════════════════════
    def _start_com_thread(self):
        def _com():
            while self.running:
                try:
                    ser = serial.Serial(COM_PORT, BAUDRATE, timeout=1)
                    self.com_connected = True
                    self.after(0, lambda: self._update_com_status(True, f"{COM_PORT} @ {BAUDRATE}"))

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

        def _run():
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
                        return

                    # Bug 1: leer el frame bajo el mismo lock con el que se escribe.
                    with self._frame_lock:
                        frame = self.latest_frame.copy()
                    img_h, img_w = frame.shape[:2]

                    results = self.yolo.predict(source=frame, conf=YOLO_CONF, verbose=False)
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
                        val = self._run_ocr(self.ocr_green, crop)
                        roi_crops.append((labels_green[i], crop))
                        if i == 0: humedad_val = val
                        elif i == 1: temp_amb_val = val
                        elif i == 2: presion_val = val

                    # 2. OCR White (Derecha)
                    temp_prom_val, temp_min_val, temp_max_val = "—", "—", "—"
                    labels_white = ["Temp. Promedio", "Temp. Mínima", "Temp. Máxima"]
                    for i, b in enumerate(right_boxes):
                        x1, y1, x2, y2 = expand_box(b['x1'], b['y1'], b['x2'], b['y2'], img_h, img_w, "roi_white")
                        crop = frame[y1:y2, x1:x2]
                        val = self._run_ocr(self.ocr_white, crop)
                        roi_crops.append((labels_white[i], crop))
                        if i == 0: temp_prom_val = val
                        elif i == 1: temp_min_val = val
                        elif i == 2: temp_max_val = val

                    now = datetime.now().strftime("%H:%M:%S")
                    entry = {
                        "num": None,  # se asigna atómicamente más abajo (Bug 7)
                        "hora": now,
                        "peso": com_value,
                        "humedad": humedad_val,
                        "temp_amb": temp_amb_val,
                        "presion": presion_val,
                        "temp_prom": temp_prom_val,
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
                        self.after(0, lambda a=attempt: self._set_status(f"⚠️ Medida anómala detectada, iniciando contramedida (Intento {a+1}/{max_retries})...", RED_ERR))
                        time.sleep(1.5)
                        # continúa al siguiente intento del loop

                if entry is None:
                    # Ningún intento logró detectar los 6 ROIs
                    play_sound("error")
                    self.after(0, lambda: self._set_status("❌ Fallo: YOLO no encontró los 6 ROIs tras 3 intentos", RED_ERR))
                    return

                # Bug 7: número de medición y append al historial bajo lock,
                # para que dos mediciones casi simultáneas no colisionen.
                with self._hist_lock:
                    self._measure_counter += 1
                    entry["num"] = self._measure_counter
                    self.historial.append(entry)

                self._save_to_excel(entry)
                self.after(0, lambda e=entry, c=roi_crops, v=valid_dict: self._update_readings(e, c, v, add_to_table=True))

                if all_valid:
                    play_sound("success")
                    self.after(0, lambda e=entry: self._set_status(f"✅ Medición #{e['num']} registrada — {e['hora']}", GREEN_OK))
                else:
                    play_sound("error")
                    self.after(0, lambda e=entry: self._set_status(f"❌ Medición #{e['num']} guardada con errores (Máx reintentos)", RED_ERR))
            finally:
                # Bug 3: liberar el flag pase lo que pase (éxito, fallo o excepción)
                # para que el siguiente dato del COM pueda disparar un nuevo pipeline.
                self._processing = False

        threading.Thread(target=_run, daemon=True).start()

    def _run_ocr(self, ocr_model, crop):
        try:
            with self._ocr_lock:
                res = ocr_model.ocr(crop, det=False, cls=False)
            if res and res[0] and len(res[0]) > 0:
                text = res[0][0][0].strip()
                return extract_number(text)
        except Exception:
            pass
        return "—"

    def _update_readings(self, entry, roi_crops, valid_dict, add_to_table=True):
        # Panel: Colorear de rojo si no es válido
        def get_color(key):
            return BLUE_DARK if valid_dict.get(key, False) else RED_ERR

        self.reading_labels["peso"].configure(text=f"{entry['peso']} g", text_color=get_color("peso"))
        self.reading_labels["humedad"].configure(text=f"{entry['humedad']} %RH", text_color=get_color("humedad"))
        self.reading_labels["temp_amb"].configure(text=f"{entry['temp_amb']} °C", text_color=get_color("temp_amb"))
        self.reading_labels["presion"].configure(text=f"{entry['presion']} hPa", text_color=get_color("presion"))
        self.reading_labels["temp_prom"].configure(text=f"{entry['temp_prom']} °C", text_color=get_color("temp_prom"))
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
            entry["temp_prom"], entry["temp_min"], entry["temp_max"]
        ]
        
        # Resaltar fila entera si hubo error permanente, o celda específica
        for i, (val, w) in enumerate(zip(vals, col_widths)):
            # Determinar color de la celda según valid_dict (índices 3 al 8 corresponden a las variables)
            keys = ["", "", "peso", "humedad", "temp_amb", "presion", "temp_prom", "temp_min", "temp_max"]
            c = BLUE_DARK
            if i >= 2 and not valid_dict.get(keys[i], False):
                c = RED_ERR
                
            ctk.CTkLabel(
                row_frame, text=val,
                font=ctk.CTkFont(family="Consolas", size=11, weight="bold" if c == RED_ERR else "normal"),
                text_color=c, width=w, anchor="center"
            ).pack(side="left", padx=2)

    # ══════════════════════════════════════════════════════════════════════════
    #  EXCEL
    # ══════════════════════════════════════════════════════════════════════════
    def _save_to_excel(self, entry):
        headers = ["#", "Hora", "Peso (g)", "Humedad %RH", "Temp Amb °C",
                   "Presión hPa", "Temp Prom °C", "Temp Mín °C", "Temp Máx °C"]

        try:
            with self._excel_lock:
                if self.excel_path.exists():
                    wb = load_workbook(self.excel_path)
                    ws = wb.active
                else:
                    wb = Workbook()
                    ws = wb.active
                    ws.title = "Calibración"
                    ws.append(headers)

                    from openpyxl.styles import Font, PatternFill, Alignment
                    for cell in ws[1]:
                        cell.font = Font(bold=True, color="FFFFFF")
                        cell.fill = PatternFill(start_color="1B3A5C", fill_type="solid")
                        cell.alignment = Alignment(horizontal="center")

                ws.append([
                    entry["num"], entry["hora"], entry["peso"], entry["humedad"],
                    entry["temp_amb"], entry["presion"], entry["temp_prom"],
                    entry["temp_min"], entry["temp_max"]
                ])
                wb.save(self.excel_path)
        except Exception as e:
            self.after(0, lambda err=e: self._set_status(
                f"❌ Error guardando en Excel (dato #{entry.get('num', '?')} no persistido): {err}", RED_ERR
            ))

    def _export_excel(self):
        if not self.historial:
            self._set_status("⚠️ No hay datos para exportar")
            return
        self._set_status(f"💾 Excel guardado: {self.excel_path}")

    # ══════════════════════════════════════════════════════════════════════════
    #  AJUSTES
    # ══════════════════════════════════════════════════════════════════════════
    def _open_settings(self):
        win = ctk.CTkToplevel(self)
        win.title("⚙️ Configuración GUI")
        win.geometry("400x300")
        win.attributes('-topmost', True)
        
        ctk.CTkLabel(win, text="Apariencia", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 10))
        
        # Tema
        tema_frame = ctk.CTkFrame(win, fg_color="transparent")
        tema_frame.pack(fill="x", padx=40, pady=5)
        ctk.CTkLabel(tema_frame, text="Tema:").pack(side="left")
        
        tema_var = ctk.StringVar(value=self.config["theme"])
        def change_theme(val):
            ctk.set_appearance_mode(val)
            self.config["theme"] = val
            save_config(self.config)
            
        ctk.CTkOptionMenu(tema_frame, values=["Light", "Dark", "System"], variable=tema_var, command=change_theme).pack(side="right")
        
        # Escala UI
        escala_frame = ctk.CTkFrame(win, fg_color="transparent")
        escala_frame.pack(fill="x", padx=40, pady=20)
        ctk.CTkLabel(escala_frame, text="Tamaño de Interfaz:").pack(side="left")
        
        val_lbl = ctk.CTkLabel(escala_frame, text=f"{self.config['scaling']:.1f}x")
        val_lbl.pack(side="right", padx=(10, 0))
        
        def change_scaling(val):
            val = round(float(val), 1)
            val_lbl.configure(text=f"{val:.1f}x")
            ctk.set_widget_scaling(val)
            self.config["scaling"] = val
            save_config(self.config)
            
        slider = ctk.CTkSlider(win, from_=0.8, to=2.0, number_of_steps=12, command=change_scaling)
        slider.set(self.config["scaling"])
        slider.pack(fill="x", padx=40)

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