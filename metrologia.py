"""
Módulo de Incertidumbre Metrológica — calibracion_gui.py
Conforme: ISO 8655-6:2022, GUM (JCGM 100:2008), NTP-ISO/IEC 17025:2017 §7.6
Equipos: Balanza Mettler Toledo AX105 DeltaRange, Termómetro Excursion-Trac, Datalogger Extech SD700
"""
import numpy as np
import json
from pathlib import Path
from scipy.stats import t as t_dist  # Para factor de cobertura k según GUM §T.3

CONFIG_EQUIPOS_PATH = Path(__file__).resolve().parent / "config_equipos.json"

DEFAULT_CONFIG_EQUIPOS = {
    "balanza": {
        "modelo": "Mettler Toledo AX105 DeltaRange",
        "rango_operacion": "fino",
        "u_repetibilidad_mg": 0.02,
        "u_linealidad_mg": 0.08,
        "u_certificado_mg": None,
        "fecha_certificado": "",
        "vigencia_certificado": ""
    },
    "termometro_agua": {
        "modelo": "Traceable Excursion-Trac",
        "u_temp_C": 0.25,
        "u_certificado_C": None,
        "fecha_certificado": "",
        "vigencia_certificado": ""
    },
    "termohigrobarometro": {
        "modelo": "Extech SD700",
        "u_temp_C": 0.8,
        "u_humedad_pct_lectura": 0.04,
        "u_humedad_pct_offset": 1.0,
        "u_presion_hPa": 1.0,
        "fecha_certificado": "",
        "vigencia_certificado": ""
    }
}

def load_config_equipos():
    if CONFIG_EQUIPOS_PATH.exists():
        with open(CONFIG_EQUIPOS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(CONFIG_EQUIPOS_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG_EQUIPOS, f, indent=2, ensure_ascii=False)
        return DEFAULT_CONFIG_EQUIPOS

def z_factor(T_water_C: float) -> float:
    """Factor Z (uL/mg) — CIPM-2001 densidad del agua + correccion de flotabilidad."""
    rho_w = (999.842594 + 6.793952e-2*T_water_C - 9.095290e-3*T_water_C**2
             + 1.001685e-4*T_water_C**3 - 1.120083e-6*T_water_C**4
             + 6.536332e-9*T_water_C**5) / 1000.0
    rho_air, rho_weights = 0.0012, 8.0
    return (1.0/rho_w) * (1.0 - rho_air/rho_weights) / (1.0 - rho_air/rho_w)

def volume_from_mass(mass_mg: float, T_water_C: float) -> float:
    """Convierte masa (mg) a volumen (μL) usando factor Z."""
    return mass_mg * z_factor(T_water_C)

def balance_uncertainty_mg(config: dict) -> float:
    """Incertidumbre combinada de balanza (mg), usa certificado si existe."""
    bal = config["balanza"]
    if bal.get("u_certificado_mg"):
        return bal["u_certificado_mg"]
    u_rep = bal["u_repetibilidad_mg"]
    u_lin = bal["u_linealidad_mg"] / np.sqrt(3)
    return float(np.sqrt(u_rep**2 + u_lin**2))

def calculate_expanded_uncertainty(V_nominal_uL: float, volumes_measured_uL: list,
                                    T_water_C: float, config: dict = None) -> dict:
    """
    Presupuesto de incertidumbre completo conforme ISO 8655-6/GUM.
    volumes_measured_uL: lista de >=10 volumenes ya convertidos con factor Z
    """
    if config is None:
        config = load_config_equipos()

    n = len(volumes_measured_uL)
    V_mean = float(np.mean(volumes_measured_uL))
    s = float(np.std(volumes_measured_uL, ddof=1)) if n > 1 else 0.0

    Z = z_factor(T_water_C)
    u_balance_mg = balance_uncertainty_mg(config)
    u_balance_uL = u_balance_mg * Z

    u_rep_uL = s / np.sqrt(n) if n > 0 else 0.0

    u_temp_C = config["termometro_agua"].get("u_certificado_C") or config["termometro_agua"]["u_temp_C"]
    u_temp_rect = u_temp_C / np.sqrt(3)
    dZ_dT = (z_factor(T_water_C + 0.01) - z_factor(T_water_C)) / 0.01
    u_Z_temp_uL = abs(dZ_dT) * u_temp_rect * V_nominal_uL

    u_evap_uL = 0.1 if V_nominal_uL <= 50 else (0.3 if V_nominal_uL <= 200 else 0.0)

    u_combined = float(np.sqrt(u_balance_uL**2 + u_rep_uL**2 + u_Z_temp_uL**2 + u_evap_uL**2))

    # Factor de cobertura k — GUM §T.3 / ISO 8655-6:2022
    # Para n mediciones se usan (n-1) grados de libertad y la distribución t de Student
    # al nivel de confianza del 95 % (dos colas: 0.025 en cada cola).
    # Para n=10 -> nu=9 -> k = t(0.975, 9) ≈ 2.262  (vs k=2.000 fijo = error del 13%)
    nu = max(n - 1, 1)  # grados de libertad; mínimo 1 para evitar división por cero
    k_coverage = float(t_dist.ppf(0.975, nu))  # nivel 95 %, dos colas
    U_expanded = k_coverage * u_combined

    systematic_error = V_mean - V_nominal_uL
    CV_pct = (s / V_mean * 100) if V_mean > 0 else 0.0

    return {
        "n_mediciones": n,
        "V_media_uL": round(V_mean, 4),
        "error_sistematico_uL": round(systematic_error, 4),
        "CV_pct": round(CV_pct, 3),
        "Z_factor": round(Z, 5),
        "u_balanza_uL": round(u_balance_uL, 5),
        "u_repetibilidad_uL": round(u_rep_uL, 5),
        "u_temperatura_uL": round(u_Z_temp_uL, 5),
        "u_evaporacion_uL": round(u_evap_uL, 5),
        "u_combinada_uL": round(u_combined, 5),
        # k se reporta con 3 decimales para trazabilidad (ej. 2.262 para n=10)
        "k_cobertura": round(k_coverage, 3),
        "grados_libertad": nu,
        "nivel_confianza_pct": 95.0,
        "U_expandida_k_uL": round(U_expanded, 4),
        "U_expandida_k_pct": round(100*U_expanded/V_nominal_uL, 3),
        # Alias mantenidos para compatibilidad con el código de la GUI
        "U_expandida_k2_uL": round(U_expanded, 4),
        "U_expandida_k2_pct": round(100*U_expanded/V_nominal_uL, 3),
        "equipos_usados": {
            "balanza": config["balanza"]["modelo"],
            "termometro": config["termometro_agua"]["modelo"],
            "termohigrobarometro": config["termohigrobarometro"]["modelo"]
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUACIÓN DE CONFORMIDAD ISO 8655-2:2022
# ══════════════════════════════════════════════════════════════════════════════

# Tabla 1 — Pipetas de pistón de un canal, desplazamiento de aire (100 % vol. nominal)
# (v_max_uL, max_es_pct, max_cv_pct)
# NOTA: Verificar con la edición vigente del estándar para el tipo exacto de pipeta.
_ISO8655_2_TABLA = [
    (0.1,         12.0,  6.0),
    (0.2,         10.0,  5.0),
    (0.5,          5.0,  3.0),
    (1.0,          3.0,  2.0),
    (2.0,          2.0,  2.0),
    (5.0,          2.0,  1.5),
    (10.0,         1.5,  1.0),
    (25.0,         1.0,  1.0),
    (100.0,        0.8,  0.5),
    (1000.0,       0.8,  0.5),
    (5000.0,       0.8,  0.5),
    (float('inf'), 0.8,  0.5),
]


def iso8655_2_tolerancias(v_nominal_uL: float) -> dict:
    """
    Retorna los límites máximos de error sistemático (es) y coeficiente de
    variación (CV) según ISO 8655-2:2022 para el volumen nominal dado.
    Aplica a pipetas de un canal, desplazamiento de aire, al 100 % del volumen.
    """
    for v_max, es_pct, cv_pct in _ISO8655_2_TABLA:
        if v_nominal_uL <= v_max:
            return {
                "max_es_pct": es_pct,
                "max_cv_pct": cv_pct,
                "max_es_uL":  round(v_nominal_uL * es_pct / 100.0, 4),
                "norma":      "ISO 8655-2:2022 Tabla 1 (pipeta 1 canal, aire)"
            }
    # fallback (no debería alcanzarse)
    return {"max_es_pct": 0.8, "max_cv_pct": 0.5,
            "max_es_uL": round(v_nominal_uL * 0.008, 4),
            "norma": "ISO 8655-2:2022"}


def evaluar_calibracion(unc_report: dict, v_nominal_uL: float) -> dict:
    """
    Compara los resultados de la calibración contra los límites ISO 8655-2:2022.
    Retorna el veredicto APROBADO / RECHAZADO con detalles de cada criterio.

    Args:
        unc_report:    dict de calculate_expanded_uncertainty()
        v_nominal_uL:  volumen nominal declarado de la pipeta (µL)

    Returns:
        dict con resultado, criterios individuales y valores de tolerancia.
    """
    tol    = iso8655_2_tolerancias(v_nominal_uL)
    es_uL  = abs(unc_report["error_sistematico_uL"])
    cv_pct = unc_report["CV_pct"]
    es_pct = (es_uL / v_nominal_uL * 100.0) if v_nominal_uL > 0 else 0.0

    pasa_es  = es_pct <= tol["max_es_pct"]
    pasa_cv  = cv_pct <= tol["max_cv_pct"]
    aprobado = pasa_es and pasa_cv

    return {
        "resultado":                     "APROBADO" if aprobado else "RECHAZADO",
        "aprobado":                      aprobado,
        "es_uL":                         round(es_uL,  4),
        "es_pct":                        round(es_pct, 3),
        "cv_pct":                        cv_pct,
        "max_es_pct":                    tol["max_es_pct"],
        "max_cv_pct":                    tol["max_cv_pct"],
        "max_es_uL":                     tol["max_es_uL"],
        "pasa_error_sistematico":        pasa_es,
        "pasa_coeficiente_variacion":    pasa_cv,
        "norma_referencia":              tol["norma"],
    }

