"""
verificar_finetune_real.py — Prueba el modelo exportado imagen por imagen
========================================================================
Lee val_list.txt y train_list.txt y ejecuta inferencia directa con PaddleOCR 
en modo rec-only (sin detección de cajas, ya que las imágenes ya son crops).
Compara la predicción con el Ground Truth (GT).
"""

import cv2
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR
ROOT = Path(__file__).resolve().parents[1]  # raíz del repositorio

FINETUNED_DIR = str((ROOT / "models/ocr_white"))
BASE_DIR = (ROOT / "data/finetune_white")

def evaluate_list(paddle_model, list_file, name):
    print(f"\n{'='*70}")
    print(f"  EVALUANDO: {name}")
    print(f"{'='*70}")
    
    with open(list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    total = 0
    correct = 0
    errors = []
    
    print(f"{'Archivo':50s} | {'GT':^10s} | {'Predicción':^10s} | {'Conf':^6s} | {'Estado'}")
    print("-" * 90)
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # El formato es "images/archivo.jpg\tLABEL"
        parts = line.split('\t')
        if len(parts) < 2: continue
        
        img_rel_path, gt = parts[0], parts[1]
        img_path = BASE_DIR / img_rel_path
        
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # Inferir (modo reconocimiento directo)
        res = paddle_model.ocr(img, det=False, cls=False)
        
        pred_text = ""
        conf = 0.0
        
        if res and res[0] and len(res[0]) > 0:
            pred_text = res[0][0][0].strip()
            conf = res[0][0][1]
            
        is_correct = (pred_text == gt)
        total += 1
        
        if is_correct:
            correct += 1
            status = "✓ OK"
        else:
            status = "✗ ERROR"
            errors.append((img_rel_path, gt, pred_text, conf))
            
        # Imprimir todos para el VAL set, o solo los errores para el TRAIN (para no saturar la terminal)
        if name == "VAL SET" or not is_correct:
            print(f"{img_rel_path:50s} | {gt:^10s} | {pred_text:^10s} | {conf*100:>5.1f}% | {status}")
            
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"\nResumen {name}:")
    print(f"Total: {total} | Correctos: {correct} | Errores: {len(errors)} | Accuracy: {acc:.2f}%\n")
    

def main():
    print("Cargando modelo fine-tuned...")
    ocr_ft = PaddleOCR(
        use_angle_cls=False, 
        lang='en', 
        use_gpu=False, 
        show_log=False,
        rec_model_dir=FINETUNED_DIR
    )
    print("[✓] Modelo cargado")
    
    val_file = BASE_DIR / "val_list.txt"
    train_file = BASE_DIR / "train_list.txt"
    
    evaluate_list(ocr_ft, val_file, "VAL SET")
    print("\nEvaluando Train set (mostrando SOLO errores para ser breves)...")
    evaluate_list(ocr_ft, train_file, "TRAIN SET")

if __name__ == "__main__":
    main()
