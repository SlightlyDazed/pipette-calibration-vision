import cv2
import json
import numpy as np
from ultralytics import YOLO
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio

# Configuración
DATASET_DIR = (ROOT / "data/photos")
YOLO_MODEL = str((ROOT / "models/yolo/roi_detector_v1.pt"))
OUTPUT_MAPPING = str((ROOT / "roi_mapping.json"))

def main():
    print("Cargando modelo YOLO...")
    yolo = YOLO(YOLO_MODEL)
    
    # Buscar una imagen de muestra en el dataset
    images = list(DATASET_DIR.glob("*.jpg"))
    if not images:
        print(f"No se encontraron imágenes en {DATASET_DIR}")
        return
    
    img_path = str(images[0])
    print(f"Usando imagen de muestra: {img_path}")
    img = cv2.imread(img_path)
    
    # Predecir con YOLO
    results = yolo.predict(source=img, conf=0.4, verbose=False)
    boxes = results[0].boxes
    
    if not boxes or len(boxes) == 0:
        print("YOLO no detectó ningún ROI en esta imagen.")
        return
    
    # Ordenar los ROIs de arriba hacia abajo para mostrarlos en orden
    rois = []
    for box in boxes:
        cls_name = yolo.model.names[int(box.cls[0].item())]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        rois.append({
            "cls_name": cls_name,
            "y1": y1,
            "coords": (x1, y1, x2, y2)
        })
    
    rois.sort(key=lambda r: r['y1']) # Ordenar por posición vertical (Y)
    
    mapping_results = []
    
    for i, roi in enumerate(rois):
        display_img = img.copy()
        x1, y1, x2, y2 = roi['coords']
        
        # Dibujar todos los ROIs en azul
        for other_roi in rois:
            ox1, oy1, ox2, oy2 = other_roi['coords']
            cv2.rectangle(display_img, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
            
        # Resaltar el ROI actual en ROJO y más grueso
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 6)
        
        # Añadir un recuadro encima del ROI actual para llamar la atención
        cv2.putText(display_img, f"ROI A NOMBRAR", (x1, max(30, y1-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        # Redimensionar para que quepa en pantalla
        h, w = display_img.shape[:2]
        scale = 800 / max(h, w)
        disp_resized = cv2.resize(display_img, (int(w * scale), int(h * scale)))
        
        cv2.imshow("Nombramiento de ROIs", disp_resized)
        cv2.waitKey(1)  # Muestra la ventana
        
        print(f"\n[{i+1}/{len(rois)}] Clase original de YOLO: {roi['cls_name']}")
        nombre = input("▶ Escribe el nombre o etiqueta para este ROI resaltado en ROJO: ")
        
        mapping_results.append({
            "index": i,
            "yolo_class": roi['cls_name'],
            "coords": [x1, y1, x2, y2],
            "user_label": nombre.strip()
        })
        
    cv2.destroyAllWindows()
    
    # Guardar resultados
    with open(OUTPUT_MAPPING, "w", encoding="utf-8") as f:
        json.dump(mapping_results, f, indent=4, ensure_ascii=False)
        
    print("\n" + "="*50)
    print("Mapeo guardado exitosamente en roi_mapping.json")
    print("Resumen:")
    for r in mapping_results:
        print(f" - {r['user_label']} (Clase original: {r['yolo_class']})")
    print("="*50)

if __name__ == "__main__":
    main()
