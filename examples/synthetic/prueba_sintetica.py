"""
prueba_sintetica.py
===================
Genera datos sintéticos realistas de una sesión de calibración ISO 8655-6
y produce el PDF + Excel de ejemplo para verificar el formato de salida.

Simula: Pipeta Eppendorf Research Plus 1000 µL
        10 réplicas válidas con ruido realista
"""

import sys
import json
import random
import csv
from pathlib import Path
from datetime import datetime, timedelta

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
OCR_DIR     = SCRIPT_DIR.parents[1]                 # raíz del repositorio
OUTPUT_DIR  = SCRIPT_DIR                         # ya estamos en prueba_sintetica/

sys.path.insert(0, str(OCR_DIR))

from metrologia import (
    calculate_expanded_uncertainty, evaluar_calibracion,
    volume_from_mass, z_factor
)
from pdf_reporte import generar_certificado_pdf

# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DE SIMULACIÓN
# ══════════════════════════════════════════════════════════════════════════════
random.seed(42)

SESSION = {
    "operador":  "Operador de prueba",
    "pipeta":    "PIPETA-DEMO-1000",
    "v_nominal": 1000.0,
}

# Condiciones ambientales realistas (Lima ~500 m s.n.m.)
T_AGUA_BASE   = 20.45   # °C — promedio del Excursion-Trac
T_AMB_BASE    = 22.3    # °C — Extech SD700
HUMEDAD_BASE  = 58.4    # %RH
PRESION_BASE  = 987.6   # hPa

# Pipeta ligeramente mal calibrada (sesgo de +2 µL → ~0.2%) para que sea interesante
SESGO_UL = 2.1          # µL de error sistemático intencional
CV_REAL   = 0.003       # 0.3% de imprecisión real


# ══════════════════════════════════════════════════════════════════════════════
# GENERAR 10 MEDICIONES SINTÉTICAS
# ══════════════════════════════════════════════════════════════════════════════
def generar_mediciones():
    mediciones = []
    hora_base  = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)

    for i in range(10):
        hora = hora_base + timedelta(minutes=5 * i + random.uniform(-0.5, 0.5))

        # Temperatura del agua: varía ±0.02 °C entre mediciones
        t_agua = round(T_AGUA_BASE + random.gauss(0, 0.02), 2)

        # Factor Z para esta temperatura
        Z = z_factor(t_agua)

        # Volumen real dispensado (con sesgo + ruido)
        v_real_sim = SESSION["v_nominal"] + SESGO_UL + random.gauss(0, SESSION["v_nominal"] * CV_REAL)

        # Masa en mg que correspondería a ese volumen
        masa_mg = round(v_real_sim / Z, 2)

        # Condiciones ambientales con ruido pequeño
        t_amb   = round(T_AMB_BASE  + random.gauss(0, 0.1),  1)
        humedad = round(HUMEDAD_BASE + random.gauss(0, 0.3),  1)
        presion = round(PRESION_BASE + random.gauss(0, 0.2),  1)
        t_min   = round(t_agua - random.uniform(0.05, 0.15),  2)
        t_max   = round(t_agua + random.uniform(0.05, 0.15),  2)

        # Volumen calculado hacia atrás (como haría el sistema real)
        v_calc  = round(volume_from_mass(masa_mg, t_agua), 2)

        mediciones.append({
            "num":      i + 1,
            "hora":     hora.strftime("%H:%M:%S"),
            "peso":     str(masa_mg),
            "humedad":  str(humedad),
            "temp_amb": str(t_amb),
            "presion":  str(presion),
            "t_agua":   str(t_agua),
            "temp_min": str(t_min),
            "temp_max": str(t_max),
            "vol_real": v_calc,
        })

    return mediciones


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 62)
    print("  PRUEBA SINTÉTICA — Sistema de Calibración ISO 8655-6")
    print("  Laboratorio de Calibración (datos sintéticos)")
    print("═" * 62)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Generar mediciones
    print("\n[1/4] Generando 10 mediciones sintéticas...")
    mediciones = generar_mediciones()
    volumenes  = [m["vol_real"] for m in mediciones]
    t_agua_prom = sum(float(m["t_agua"]) for m in mediciones) / len(mediciones)

    for m in mediciones:
        print(f"  #{m['num']:02d}  {m['hora']}  masa={m['peso']} mg  "
              f"T.agua={m['t_agua']}°C  V={m['vol_real']} µL")

    # 2. Calcular incertidumbre
    print("\n[2/4] Calculando presupuesto de incertidumbre (GUM)...")
    unc = calculate_expanded_uncertainty(SESSION["v_nominal"], volumenes, t_agua_prom)
    print(f"  V̄        = {unc['V_media_uL']} µL")
    print(f"  Es        = {unc['error_sistematico_uL']} µL")
    print(f"  CV        = {unc['CV_pct']} %")
    print(f"  Z factor  = {unc['Z_factor']} µL/mg")
    print(f"  uc        = {unc['u_combinada_uL']} µL")
    print(f"  k         = {unc['k_cobertura']} (t-Student, ν={unc['grados_libertad']}, 95%)")
    print(f"  U (k·uc)  = ± {unc['U_expandida_k_uL']} µL  ({unc['U_expandida_k_pct']} %)")

    # 3. Evaluación ISO 8655-2
    print("\n[3/4] Evaluando conformidad ISO 8655-2:2022...")
    ev = evaluar_calibracion(unc, SESSION["v_nominal"])
    emoji = "✓" if ev["aprobado"] else "✗"
    print(f"  {emoji}  RESULTADO: {ev['resultado']}")
    pasa_es = "✓ PASA" if ev["pasa_error_sistematico"]     else "✗ NO PASA"
    pasa_cv = "✓ PASA" if ev["pasa_coeficiente_variacion"] else "✗ NO PASA"
    print(f"  Error sistemático |es|: {ev['es_pct']}% (máx {ev['max_es_pct']}%) → {pasa_es}")
    print(f"  Coef. variación CV:     {ev['cv_pct']}%  (máx {ev['max_cv_pct']}%) → {pasa_cv}")

    # 4a. Exportar Excel
    print("\n[4/4] Exportando archivos...")
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment

        wb = Workbook()
        ws = wb.active
        ws.title = "Calibración ISO 8655-6"

        HEADERS = ["#", "Hora", "Peso (mg)", "Humedad %RH", "T.Amb.Extech °C",
                   "Presión hPa", "T.Agua.ExcursionTrac °C",
                   "T.Agua Mín °C", "T.Agua Máx °C", "Volumen Real (µL)"]

        ws.append(HEADERS)
        for cell in ws[1]:
            cell.font      = Font(bold=True, color="FFFFFF")
            cell.fill      = PatternFill(start_color="1B3A5C", fill_type="solid")
            cell.alignment = Alignment(horizontal="center")

        for m in mediciones:
            ws.append([m["num"], m["hora"], float(m["peso"]),
                       float(m["humedad"]), float(m["temp_amb"]),
                       float(m["presion"]), float(m["t_agua"]),
                       float(m["temp_min"]), float(m["temp_max"]),
                       m["vol_real"]])

        # Bloque de resultados metrológicos
        ws.append([])
        hdr_row = ws.max_row + 1
        ws.append(["RESULTADOS METROLÓGICOS ISO 8655-6:2022 / GUM (JCGM 100:2008)"])
        for cell in ws[hdr_row]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="1B3A5C", fill_type="solid")

        k  = unc["k_cobertura"]
        nu = unc["grados_libertad"]
        ws.append(["N° de réplicas válidas:",          unc["n_mediciones"]])
        ws.append(["Volumen Medio V̄ (µL):",            unc["V_media_uL"]])
        ws.append(["Error Sistemático Es (µL):",        unc["error_sistematico_uL"]])
        ws.append(["CV (%):",                           unc["CV_pct"]])
        ws.append(["Factor Z (µL/mg):",                 unc["Z_factor"]])
        ws.append(["u combinada uc (µL):",              unc["u_combinada_uL"]])
        ws.append([f"  componente balanza (µL):",       unc["u_balanza_uL"]])
        ws.append([f"  componente repetibilidad (µL):", unc["u_repetibilidad_uL"]])
        ws.append([f"  componente temperatura (µL):",   unc["u_temperatura_uL"]])
        ws.append([f"  componente evaporación (µL):",   unc["u_evaporacion_uL"]])
        ws.append([f"Factor de cobertura k (t-Student ν={nu}, 95%):", k])
        ws.append([f"Incertidumbre Expandida U = k·uc (µL):", unc["U_expandida_k_uL"]])
        ws.append(["U (%):", unc["U_expandida_k_pct"]])
        ws.append([])

        verd_row = ws.max_row + 1
        ws.append([f"EVALUACIÓN ISO 8655-2:2022:  {ev['resultado']}"])
        color_v = "22C55E" if ev["aprobado"] else "EF4444"
        for cell in ws[verd_row]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color=color_v, fill_type="solid")
        ws.append(["  Error sistém. |es| (µL):", ev["es_uL"],
                   f"Límite: ≤ {ev['max_es_pct']} %",
                   "PASA" if ev["pasa_error_sistematico"] else "NO PASA"])
        ws.append(["  Coef. variación CV (%):", ev["cv_pct"],
                   f"Límite: ≤ {ev['max_cv_pct']} %",
                   "PASA" if ev["pasa_coeficiente_variacion"] else "NO PASA"])

        # Autoajustar ancho de columnas
        for col in ws.columns:
            max_len = max((len(str(cell.value or "")) for cell in col), default=0)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 50)

        xlsx_path = OUTPUT_DIR / "Sesion_Sintetica_1000uL.xlsx"
        wb.save(xlsx_path)
        print(f"  ✓ Excel guardado: {xlsx_path.name}")
    except Exception as e:
        print(f"  ✗ Error Excel: {e}")

    # 4b. Exportar PDF
    try:
        pdf_path = OUTPUT_DIR / "Certificado_Sintetico_1000uL.pdf"
        ok = generar_certificado_pdf(
            session_data = SESSION,
            measurements = mediciones,
            unc_report   = unc,
            evaluacion   = ev,
            output_path  = pdf_path
        )
        if ok:
            print(f"  ✓ PDF guardado:  {pdf_path.name}")
        else:
            print("  ✗ PDF: reportlab no disponible")
    except Exception as e:
        print(f"  ✗ Error PDF: {e}")

    # 4c. Exportar audit trail de ejemplo
    try:
        import hashlib
        audit_path = OUTPUT_DIR / "audit_trail_sintetico.jsonl"
        with open(audit_path, "w", encoding="utf-8") as f:
            for m in mediciones:
                record = {
                    "timestamp":  (datetime.now()).isoformat(),
                    "action":     "MEASURE_VALID_ISO",
                    "entry_num":  m["num"],
                    "operator":   SESSION["operador"],
                    "data_hash":  hashlib.sha256(
                                      json.dumps(m, sort_keys=True).encode()
                                  ).hexdigest()
                }
                f.write(json.dumps(record) + "\n")
            # Registro de cierre de sesión
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "action":    "SESSION_COMPLETE",
                "entry_num": 10,
                "operator":  SESSION["operador"],
                "data_hash": hashlib.sha256(
                    json.dumps({"veredicto": ev["resultado"],
                                "V_media": unc["V_media_uL"],
                                "U_k": unc["U_expandida_k_uL"],
                                "k": unc["k_cobertura"]},
                               sort_keys=True).encode()
                ).hexdigest()
            }) + "\n")
        print(f"  ✓ Audit trail:   {audit_path.name}")
    except Exception as e:
        print(f"  ✗ Error Audit: {e}")

    # ── Resumen final ─────────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    emoji_r = "✅" if ev["aprobado"] else "❌"
    print(f"  {emoji_r}  RESULTADO FINAL: {ev['resultado']}")
    print(f"  V̄ = {unc['V_media_uL']} µL  |  Es = {unc['error_sistematico_uL']} µL  ({ev['es_pct']}%)")
    print(f"  CV = {unc['CV_pct']}%  |  U = ±{unc['U_expandida_k_uL']} µL  (k={unc['k_cobertura']}, ν={unc['grados_libertad']}, 95%)")
    print(f"\n  Archivos en: {OUTPUT_DIR}")
    print("═" * 62 + "\n")


if __name__ == "__main__":
    main()
