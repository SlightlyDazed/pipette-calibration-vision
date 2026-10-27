# Pipette Calibration Vision

**Automated gravimetric pipette calibration with computer vision, OCR and ISO 8655 metrology.**

Developed for the calibration laboratory of a biomedical equipment services company. During a
gravimetric calibration the technician dispenses water onto an analytical balance and has to write down
the mass together with the water temperature, ambient temperature, humidity and pressure shown on two
separate instruments. This system captures all of it automatically: the mass from the balance's serial
port, and the environmental readings **from a camera** pointed at the instruments' displays — then it
computes the delivered volume, the measurement uncertainty and the ISO 8655 conformity verdict.

```
Analytical balance ──RS-232──┐
                             ├──► measurement worker ──► ISO 8655 checks ──► Excel · audit trail · PDF
USB camera ──► YOLO11 (6 display ROIs) ──► PaddleOCR (white / green LCD models) ──┘
```

## Features

- **Balance over serial:** Mettler Toledo AX105 DeltaRange (RS-232 → USB, 9600 8N1, 0.01 mg).
  Every reading that arrives triggers a full measurement.
- **Displays read by vision:** a fine-tuned **YOLO11m** detector finds the six readouts of a Traceable
  Excursion-Trac water thermometer (white LCD) and an Extech SD700 thermo-hygro-barometer (green LCD);
  two **PaddleOCR recognition models fine-tuned on real crops** — one per display type — read the
  digits.
- **Validation** of every reading against format rules and the ISO 8655-7 environmental limits.
- **Metrology** (`metrologia.py`): water density and **Z factor** (CIPM-2001), delivered volume,
  systematic and random error over the 10 replicates required by ISO 8655-6, **combined and expanded
  uncertainty** following the GUM with a Student-t coverage factor, and the **ISO 8655-2 acceptance
  verdict** for the pipette's nominal volume.
- **Outputs:** daily and per-session Excel reports with the metrological summary, a **SHA-256 audit
  trail** (append-only JSONL) and a **calibration certificate in PDF**.
- **Desktop GUI** (CustomTkinter) with live camera preview, detected ROIs, sound alarms and settings.
- **Robust threading:** camera, serial and GUI threads plus a single persistent OCR worker — PaddleOCR's
  oneDNN backend caches primitives per OS thread and fails when called from new threads.

## Try it without hardware

`examples/synthetic/prueba_sintetica.py` simulates a 10-replicate session of a 1000 µL pipette and runs
the full metrology chain. Its outputs are included:

```bash
python examples/synthetic/prueba_sintetica.py
```

| File | Content |
|---|---|
| `examples/synthetic/Sesion_Sintetica_1000uL.xlsx` | Session report with the metrological summary |
| `examples/synthetic/Certificado_Sintetico_1000uL.pdf` | Calibration certificate generated from it |
| `examples/synthetic/audit_trail_sintetico.jsonl` | Audit trail of the 10 replicates |

Result of the synthetic session: mean volume 1002.003 µL, systematic error 0.2 %, CV 0.166 %,
U = ±1.20 µL (k = 2.262, ν = 9, 95 %) → **pass**.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Linux. Inference runs on the CPU; a GPU is only useful for training.

## Usage

```bash
python gui_v2.py
```

- The balance port is auto-detected or set in the settings dialog (stored in `config.json`).
- Equipment uncertainties (balance repeatability and linearity, thermometer, hygro-barometer, certificate
  values and dates) are in `config_equipos.json`.
- `CALIBRATION_LAB_NAME` sets the laboratory name shown in the GUI and the certificate.
  `CALIBRATION_TRACEABILITY` sets the traceability statement printed on the certificate; leave it empty
  unless the instruments have valid calibration certificates declared in `config_equipos.json`.

Reports are written to `resultados/` (daily Excel, `sesiones/`, `audit_trail.jsonl`).

## Models

| Path | Model |
|---|---|
| `models/yolo/roi_detector_v1.pt` | YOLO11m fine-tuned to detect the white and green display ROIs |
| `models/ocr_white/` | PaddleOCR recognition model for the Excursion-Trac display (2 decimals) |
| `models/ocr_green/` | PaddleOCR recognition model for the Extech SD700 display (1 decimal) |

## Training your own models

The laboratory photos used for training are not included. The tools in `tools/` reproduce the process
with your own images (in `data/photos/` and `dataset/`):

| Step | Script |
|---|---|
| Label display ROIs | `tools/roi_labeler.py`, `tools/roi_selector.py`, `tools/nombrar_rois.py` |
| Train the YOLO detector | `tools/train_roi_yolo.py` (YOLO11m, 640 px, no flips, mild augmentation) |
| Extract crops and label ground truth | `tools/extract_crops.py`, `tools/label_gt.py`, `tools/verificar_white.py`, `tools/verificar_green.py` |
| Fine-tune PaddleOCR | `tools/preparar_finetune.py`, `tools/preparar_finetune_green.py` |
| Evaluate | `tools/evaluar_finetune.py`, `tools/verificar_finetune_real.py`, `tools/verificar_finetune_green.py`, `tools/compare_roi*.py` |

`legacy/` keeps earlier experiments: the first GUI, image-filter tuning and an NSGA-II filter optimizer.

## Limitations

- The detector and OCR models were trained on one laboratory setup (camera position, lighting,
  instrument models); another setup needs new training data.
- Traceability of the uncertainty budget depends on filling in the instruments' certificate values in
  `config_equipos.json`.
- Excel reports can be edited afterwards: the audit trail detects changes but does not prevent them.
- The software produces the data and the certificate draft; the validity of a calibration certificate
  depends on the laboratory's own accreditation (ISO/IEC 17025).

## License

Code released under the [MIT License](LICENSE).
Detection uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), licensed under
AGPL-3.0; the trained detector in `models/yolo/` derives from YOLO11m and is subject to Ultralytics'
terms. OCR uses [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache 2.0).
