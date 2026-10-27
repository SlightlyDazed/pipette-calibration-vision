"""
auto_detect_filters.py — FASE 3: Inferencia Automática y Filtros OCR
======================================================================
Este script carga el modelo YOLO re-entrenado, detecta automáticamente 
los ROIs en las imágenes y aplica los filtros de preprocesamiento 
(OpenCV) específicos para cada clase (blanco o verde) para maximizar 
la precisión del OCR.

USO:
    python auto_detect_filters.py --model path/al/best.pt --source path/a/imagenes
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from ultralytics import YOLO

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("  [INFO] pytesseract no instalado. Se saltará la extracción de texto.")

# ── FILTROS OCR POR CLASE ─────────────────────────────────────────────────────

def filter_white_roi(crop):
    """
    Filtro para displays con fondo blanco y letras oscuras.
    Objetivo: Aumentar el contraste y binarizar limpiamente.
    """
    # 1. Convertir a escala de grises
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 2. Aumentar contraste (CLAHE - Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    
    # 3. Desenfoque suave para eliminar ruido (opcional, ayuda con el threshold)
    blur = cv2.GaussianBlur(contrast, (3, 3), 0)
    
    # 4. Binarización (Fondo blanco, texto negro)
    # Usamos Otsu para encontrar el umbral óptimo automáticamente
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def filter_green_roi(crop):
    """
    Filtro para displays con fondo verde (frecuente en monitores médicos) 
    y letras claras/oscuras.
    Objetivo: Aislar el texto usando el canal verde, invertir si es necesario.
    """
    # 1. Extraer canales B, G, R
    b, g, r = cv2.split(crop)
    
    # En displays verdes, el canal G suele tener el mejor contraste del texto.
    # Usaremos el canal G en lugar de convertir a escala de grises normal.
    base = g
    
    # 2. Mejorar el contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast = clahe.apply(base)
    
    # 3. Binarización Adaptativa (excelente para iluminación irregular en displays)
    thresh = cv2.adaptiveThreshold(
        contrast, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 4. Operación morfológica (opcional) para conectar partes rotas de los números
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph


# ── INFERENCIA Y PIPELINE ─────────────────────────────────────────────────────

def main(model_path, source_dir, output_dir):
    # Cargar el modelo YOLO
    if not os.path.exists(model_path):
        print(f"[ERROR] No se encontró el modelo en: {model_path}")
        return
        
    print(f"[*] Cargando modelo YOLO: {model_path}")
    model = YOLO(model_path)
    
    source_path = Path(source_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Extensiones de imagen soportadas
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = [p for p in source_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    
    print(f"[*] Procesando {len(images)} imágenes desde {source_dir}...")
    
    for img_file in images:
        img_cv = cv2.imread(str(img_file))
        if img_cv is None:
            continue
            
        # Ejecutar YOLO en la imagen
        # conf=0.5 -> Solo detecciones con >50% de confianza
        results = model.predict(source=img_cv, conf=0.5, verbose=False)
        
        for idx, box in enumerate(results[0].boxes):
            # Obtener coordenadas del ROI
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            # Nombre de la clase (0=roi_white, 1=roi_green)
            cls_name = model.names[cls_id]
            
            # Recortar el ROI
            crop = img_cv[y1:y2, x1:x2]
            if crop.size == 0:
                continue
                
            # APLICAR EL FILTRO ADECUADO
            if cls_name == "roi_white":
                filtered_img = filter_white_roi(crop)
            elif cls_name == "roi_green":
                filtered_img = filter_green_roi(crop)
            else:
                filtered_img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) # Fallback
                
            # Extraer texto con Tesseract si está disponible
            text = ""
            if HAS_TESSERACT:
                # psm 6 = Asume un solo bloque de texto uniforme. Ideal para números de displays.
                # Puedes cambiarlo a psm 7 si es una sola línea de texto.
                config = '--psm 6' 
                text = pytesseract.image_to_string(filtered_img, config=config).strip()
            
            # Guardar visualizaciones para validar los filtros
            base_name = f"{img_file.stem}_roi{idx}_{cls_name}"
            
            # Guardar el crop original
            cv2.imwrite(str(out_path / f"{base_name}_orig.jpg"), crop)
            # Guardar el crop con el filtro aplicado
            cv2.imwrite(str(out_path / f"{base_name}_filtered.jpg"), filtered_img)
            
            # Guardar el texto extraído
            if text:
                with open(out_path / f"{base_name}_text.txt", "w") as f:
                    f.write(text)
                    
            print(f"  ✔ {img_file.name} -> Detectado {cls_name} ({conf:.2f}). Texto OCR: '{text}'")

    print(f"\n[*] Proceso completado. Revisa la carpeta '{output_dir}' para ver los resultados de los filtros.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aplica inferencia YOLO y filtros OCR adaptativos.")
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo YOLO entrenado (best.pt)")
    parser.add_argument("--source", type=str, default="database", help="Carpeta con imágenes para procesar")
    parser.add_argument("--output", type=str, default="ocr_results", help="Carpeta para guardar resultados")
    
    args = parser.parse_args()
    main(args.model, args.source, args.output)
