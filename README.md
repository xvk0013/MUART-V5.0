# MUART V5.0 — portable D1 research package

This directory contains only the current V6-D1 model, its transitive V6-A/A2
dependencies, the D1 data generator, required tests, and reproducibility docs.
Historical D0/P0-P9 diagnostics and generated reports are intentionally kept
outside this runtime package.

## Runtime requirements

- Python with `numpy`, `scipy`, `torch`, and `torchaudio` (see
  `requirements-d1.txt`; use the PyTorch build matching the target GPU/CUDA).
- MATLAB for Check/Generate; Audit/Smoke/Train use Python.
- Windows PowerShell 5.1 or PowerShell 7.

## Portable layout

```text
MUART V5.0/
  run_v6d1.ps1
  train_v6d1.py
  ...model dependencies...
  data_gen_v2/
    main_cargo_sl_d1.m
    ...generator dependencies...
```

No source file requires a machine-specific drive letter. `run_v6d1.ps1`
locates `ModelDir` from its own directory and `data_gen_v2` beneath it.
External data and outputs are passed at runtime.

The external data root must contain:

```text
data_cargo_sl_sir15/SS2_bp12/dataset
data_cargo_sl_sir15_d1_recording_uniform/SS2_bp12/dataset
data_sl_templates_bp12
data_sl_split_bp12
```

Example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\run_v6d1.ps1" `
  -Mode Check `
  -DataRoot "<DATA_ROOT>" `
  -OutputRoot "<OUTPUT_ROOT>"
```

Use the same `-DataRoot` and `-OutputRoot` for Generate, Audit, Smoke, and
Train. For Train, the A2 reference defaults to
`<OutputRoot>/checkpoints_v6a2_shared_evidence_s42`; use `-ReferenceRun` when
it is stored elsewhere. `-OutputDir` may also explicitly select a result path.

Alternatively set `MUART_DATA_ROOT` and `MUART_OUTPUT_ROOT` once per shell.
