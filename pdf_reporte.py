"""
pdf_reporte.py — Generador de Certificado de Calibración PDF
============================================================
Conforme: ISO 8655-6:2022, GUM (JCGM 100:2008), NTP-ISO/IEC 17025:2017 §7.8

Dependencia: reportlab (pip install reportlab)
"""

import os
from datetime import datetime
from pathlib import Path

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                    Paragraph, Spacer, HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

LAB_NAME = os.environ.get("CALIBRATION_LAB_NAME", "Laboratorio de Calibración")
# Declaración de trazabilidad (p. ej. "Resultados trazables a patrones nacionales INACAL."): solo si
# los equipos tienen certificados vigentes declarados en config_equipos.json.
TRACEABILITY_NOTE = os.environ.get("CALIBRATION_TRACEABILITY", "")


def generar_certificado_pdf(session_data: dict,
                             measurements: list,
                             unc_report: dict,
                             evaluacion: dict,
                             output_path: Path) -> bool:
    """
    Genera un certificado de calibración no editable en PDF.

    Args:
        session_data:  {"operador", "pipeta", "v_nominal"}
        measurements:  lista de dicts de medición (entry dicts del sistema)
        unc_report:    dict de calculate_expanded_uncertainty()
        evaluacion:    dict de evaluar_calibracion()
        output_path:   Path donde guardar el PDF

    Returns:
        True si se generó correctamente, False si falta reportlab.
    """
    if not REPORTLAB_OK:
        return False

    # ── Paleta corporativa ────────────────────────────────────────────────────
    AZUL_OSCURO  = colors.HexColor("#1B3A5C")
    AZUL_MID     = colors.HexColor("#2563EB")
    VERDE_OK     = colors.HexColor("#22C55E")
    ROJO_ERR     = colors.HexColor("#EF4444")
    GRIS_CLARO   = colors.HexColor("#F0F4FA")
    GRIS_BORDE   = colors.HexColor("#D1D5DB")
    BLANCO       = colors.white

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=1.5*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm,
        title=f"Certificado de Calibración — {LAB_NAME}",
        author=LAB_NAME,
        subject="ISO 8655-6:2022 Calibración de Pipeta",
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Estilos ───────────────────────────────────────────────────────────────
    def _style(name, **kw):
        base = styles["Normal"].clone(name)
        for k, v in kw.items():
            setattr(base, k, v)
        return base

    st_h1  = _style("H1",  fontSize=16, textColor=BLANCO,    alignment=TA_CENTER, fontName="Helvetica-Bold", leading=20)
    st_h2  = _style("H2",  fontSize=11, textColor=AZUL_OSCURO, fontName="Helvetica-Bold", spaceAfter=4)
    st_lab = _style("LAB", fontSize=8,  textColor=colors.HexColor("#6B7280"), leading=10)
    st_val = _style("VAL", fontSize=9,  textColor=AZUL_OSCURO, fontName="Helvetica-Bold", leading=12)
    st_nom = _style("NOM", fontSize=9,  textColor=colors.black, leading=11)
    st_ok  = _style("OK",  fontSize=14, textColor=VERDE_OK,  alignment=TA_CENTER, fontName="Helvetica-Bold")
    st_err = _style("ERR", fontSize=14, textColor=ROJO_ERR,  alignment=TA_CENTER, fontName="Helvetica-Bold")
    st_foot= _style("FT",  fontSize=7.5, textColor=colors.HexColor("#6B7280"), alignment=TA_CENTER)

    # ── Número de certificado ─────────────────────────────────────────────────
    now     = datetime.now()
    cert_no = f"CB-{now.strftime('%Y%m%d')}-{session_data.get('pipeta','P').replace(' ','_')}"
    v_nom   = session_data.get("v_nominal", 0)

    # ════════════════════════════════════════════════════════════════════════════
    # 1. ENCABEZADO
    # ════════════════════════════════════════════════════════════════════════════
    header_data = [[
        Paragraph("CERTIFICADO DE CALIBRACIÓN", st_h1),
        Paragraph(f"No. {cert_no}", _style("CNO", fontSize=8, textColor=BLANCO,
                                            alignment=TA_RIGHT, fontName="Helvetica")),
    ]]
    header_tbl = Table(header_data, colWidths=["70%", "30%"])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), AZUL_OSCURO),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
    ]))
    story.append(header_tbl)

    sub_data = [[
        Paragraph(LAB_NAME, _style("SB1", fontSize=9, textColor=BLANCO,
                                                  fontName="Helvetica-Bold")),
        Paragraph(f"Fecha emisión: {now.strftime('%d/%m/%Y %H:%M')}", 
                  _style("SB2", fontSize=8, textColor=colors.HexColor("#93C5FD"),
                          alignment=TA_RIGHT)),
    ]]
    sub_tbl = Table(sub_data, colWidths=["60%", "40%"])
    sub_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), colors.HexColor("#0F2036")),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING",(0,0), (-1,-1), 12),
    ]))
    story.append(sub_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ════════════════════════════════════════════════════════════════════════════
    # Helper: sección con fondo gris
    # ════════════════════════════════════════════════════════════════════════════
    def _section_title(text):
        tbl = Table([[Paragraph(f"  {text}", st_h2)]], colWidths=["100%"])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), GRIS_CLARO),
            ("LINEBELOW",     (0,0),(-1,-1), 1.5, AZUL_MID),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.15*cm))

    def _kv_table(rows):
        """rows: list of (label, value) tuples"""
        data = [[Paragraph(lbl, st_lab), Paragraph(str(val), st_val)] for lbl, val in rows]
        tbl  = Table(data, colWidths=["45%","55%"])
        tbl.setStyle(TableStyle([
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [BLANCO, GRIS_CLARO]),
            ("LINEBELOW",      (0,0), (-1,-1), 0.3, GRIS_BORDE),
            ("TOPPADDING",     (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
            ("LEFTPADDING",    (0,0), (-1,-1), 8),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.3*cm))

    # ════════════════════════════════════════════════════════════════════════════
    # 2. EQUIPO CALIBRADO
    # ════════════════════════════════════════════════════════════════════════════
    _section_title("1. EQUIPO BAJO CALIBRACIÓN")
    _kv_table([
        ("Descripción:",         "Pipeta de Pistón de un Canal (air-displacement)"),
        ("Identificación / ID:", session_data.get("pipeta", "—")),
        ("Volumen Nominal:",     f"{v_nom} µL"),
        ("Operador Calibrador:", session_data.get("operador", "—")),
        ("Fecha / Hora:",        now.strftime("%d/%m/%Y  %H:%M:%S")),
        ("Norma de calibración:","ISO 8655-6:2022 — Método Gravimétrico"),
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # 3. EQUIPOS PATRÓN USADOS
    # ════════════════════════════════════════════════════════════════════════════
    _section_title("2. EQUIPOS PATRÓN")
    eq = unc_report.get("equipos_usados", {})
    _kv_table([
        ("Balanza analítica:",   eq.get("balanza",        "Mettler Toledo AX105 DeltaRange")),
        ("Termómetro de agua:", eq.get("termometro",      "Traceable Excursion-Trac")),
        ("Termohigrómetro:",    eq.get("termohigrobarometro", "Extech SD700")),
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # 4. CONDICIONES AMBIENTALES
    # ════════════════════════════════════════════════════════════════════════════
    _section_title("3. CONDICIONES AMBIENTALES (ISO 8655-7:2022)")

    # Calcular promedios de condiciones ambientales
    def _safe_avg(field):
        vals = []
        for m in measurements:
            try: vals.append(float(m.get(field, "")))
            except: pass
        return round(sum(vals)/len(vals), 2) if vals else "—"

    _kv_table([
        ("Temperatura ambiente (Extech SD700):",   f"{_safe_avg('temp_amb')} °C   [límite: 15–30 °C]"),
        ("Temperatura del agua (Excursion-Trac):", f"{_safe_avg('t_agua')} °C   [límite: 15–30 °C]"),
        ("Humedad relativa:",                       f"{_safe_avg('humedad')} %   [límite: 45–80 %]"),
        ("Presión atmosférica:",                    f"{_safe_avg('presion')} hPa"),
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # 5. TABLA DE MEDICIONES
    # ════════════════════════════════════════════════════════════════════════════
    _section_title("4. RESULTADOS DE MEDICIÓN (10 RÉPLICAS VÁLIDAS)")

    col_headers = ["No.", "Hora", "Masa (mg)", "T.Agua (°C)", "V.Real (µL)"]
    meas_rows   = [col_headers]
    for i, m in enumerate(measurements, 1):
        meas_rows.append([
            str(i),
            str(m.get("hora", "—")),
            str(m.get("peso", "—")),
            str(m.get("t_agua", "—")),
            str(m.get("vol_real", "—")),
        ])

    meas_tbl = Table(meas_rows, colWidths=["8%","18%","22%","22%","22%"],
                     repeatRows=1)
    meas_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  AZUL_OSCURO),
        ("TEXTCOLOR",     (0,0), (-1,0),  BLANCO),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [BLANCO, GRIS_CLARO]),
        ("LINEBELOW",     (0,0), (-1,-1), 0.3, GRIS_BORDE),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(meas_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ════════════════════════════════════════════════════════════════════════════
    # 6. PRESUPUESTO DE INCERTIDUMBRE
    # ════════════════════════════════════════════════════════════════════════════
    _section_title("5. PRESUPUESTO DE INCERTIDUMBRE — GUM (JCGM 100:2008)")

    k   = unc_report.get("k_cobertura",  2.262)
    nu  = unc_report.get("grados_libertad", 9)
    _kv_table([
        ("Factor Z (µL/mg) — CIPM-2001:",    f"{unc_report.get('Z_factor','—')} µL/mg"),
        ("Volumen medio V̄:",                 f"{unc_report.get('V_media_uL','—')} µL"),
        ("u balanza (Tipo B):",               f"{unc_report.get('u_balanza_uL','—')} µL"),
        ("u repetibilidad (Tipo A):",         f"{unc_report.get('u_repetibilidad_uL','—')} µL"),
        ("u temperatura (Tipo B):",           f"{unc_report.get('u_temperatura_uL','—')} µL"),
        ("u evaporación (Tipo B):",           f"{unc_report.get('u_evaporacion_uL','—')} µL"),
        ("Incertidumbre combinada uc:",       f"{unc_report.get('u_combinada_uL','—')} µL"),
        (f"Factor de cobertura k (t-Student, ν={nu}, 95%):", f"k = {k}"),
        ("INCERTIDUMBRE EXPANDIDA U = k·uc:", f"± {unc_report.get('U_expandida_k_uL','—')} µL  "
                                               f"({unc_report.get('U_expandida_k_pct','—')} %)"),
    ])

    # ════════════════════════════════════════════════════════════════════════════
    # 7. EVALUACIÓN ISO 8655-2
    # ════════════════════════════════════════════════════════════════════════════
    _section_title("6. EVALUACIÓN DE CONFORMIDAD — ISO 8655-2:2022")

    es_pasa = evaluacion.get("pasa_error_sistematico", False)
    cv_pasa = evaluacion.get("pasa_coeficiente_variacion", False)
    aprobado = evaluacion.get("aprobado", False)

    tol_data = [
        ["Criterio", "Resultado", "Límite ISO 8655-2", "¿Pasa?"],
        [
            "Error Sistemático |es|",
            f"{evaluacion.get('es_uL','—')} µL  ({evaluacion.get('es_pct','—')} %)",
            f"≤ {evaluacion.get('max_es_pct','—')} %  (≤ {evaluacion.get('max_es_uL','—')} µL)",
            "✓ PASA" if es_pasa else "✗ NO PASA",
        ],
        [
            "Coef. Variación CV",
            f"{evaluacion.get('cv_pct','—')} %",
            f"≤ {evaluacion.get('max_cv_pct','—')} %",
            "✓ PASA" if cv_pasa else "✗ NO PASA",
        ],
    ]
    tol_tbl = Table(tol_data, colWidths=["30%","25%","25%","20%"])
    tol_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),   AZUL_OSCURO),
        ("TEXTCOLOR",     (0,0), (-1,0),   BLANCO),
        ("FONTNAME",      (0,0), (-1,0),   "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1),  8),
        ("ALIGN",         (0,0), (-1,-1),  "CENTER"),
        ("VALIGN",        (0,0), (-1,-1),  "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1),  [BLANCO, GRIS_CLARO]),
        ("LINEBELOW",     (0,0), (-1,-1),  0.4, GRIS_BORDE),
        ("TOPPADDING",    (0,0), (-1,-1),  5),
        ("BOTTOMPADDING", (0,0), (-1,-1),  5),
        # Colorear la columna "¿Pasa?"
        ("TEXTCOLOR",     (3,1), (3,1),    VERDE_OK if es_pasa else ROJO_ERR),
        ("TEXTCOLOR",     (3,2), (3,2),    VERDE_OK if cv_pasa else ROJO_ERR),
        ("FONTNAME",      (3,1), (3,-1),   "Helvetica-Bold"),
    ]))
    story.append(tol_tbl)
    story.append(Spacer(1, 0.4*cm))

    # Veredicto final
    veredicto_text = ("✔  APROBADO" if aprobado else "✘  RECHAZADO")
    verd_style = st_ok if aprobado else st_err
    verd_data = [[Paragraph(veredicto_text, verd_style)]]
    verd_tbl = Table(verd_data, colWidths=["100%"])
    verd_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#DCFCE7") if aprobado else colors.HexColor("#FEF2F2")),
        ("BOX",           (0,0), (-1,-1), 2, VERDE_OK if aprobado else ROJO_ERR),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(verd_tbl)
    story.append(Spacer(1, 0.5*cm))

    # ════════════════════════════════════════════════════════════════════════════
    # 8. FIRMAS / TRAZABILIDAD
    # ════════════════════════════════════════════════════════════════════════════
    _section_title("7. TRAZABILIDAD Y FIRMAS")

    norma_text = evaluacion.get("norma_referencia", "ISO 8655-2:2022")
    story.append(Paragraph(
        f"{TRACEABILITY_NOTE + ' ' if TRACEABILITY_NOTE else ''}"
        f"Incertidumbre calculada conforme GUM (JCGM 100:2008). "
        f"Evaluación de conformidad conforme {norma_text}.",
        _style("TRZ", fontSize=8, textColor=colors.HexColor("#6B7280"), spaceAfter=12)
    ))

    firma_data = [
        [
            Paragraph("_______________________________\nResponsable Metrológico\n"
                      f"{session_data.get('operador','')}", st_nom),
            Paragraph("_______________________________\nRevisor Técnico / Jefe de Lab.", st_nom),
            Paragraph(f"_______________________________\nFecha y Sello\n{now.strftime('%d/%m/%Y')}", st_nom),
        ]
    ]
    firma_tbl = Table(firma_data, colWidths=["33%","34%","33%"])
    firma_tbl.setStyle(TableStyle([
        ("ALIGN",  (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE",(0,0),(-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(firma_tbl)
    story.append(Spacer(1, 0.4*cm))

    # ── Pie de página ─────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRIS_BORDE))
    story.append(Spacer(1, 0.1*cm))
    story.append(Paragraph(
        f"Documento generado automáticamente por el Sistema de Calibración — {LAB_NAME}  |  "
        f"Certificado No. {cert_no}  |  ISO 8655-6:2022 / GUM (JCGM 100:2008) / NTP-ISO/IEC 17025:2017  |  "
        f"Este certificado no es válido sin firma autógrafa del Responsable Metrológico.",
        st_foot
    ))

    doc.build(story)
    return True
