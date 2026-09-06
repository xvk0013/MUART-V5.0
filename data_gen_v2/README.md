# V6-D1 data generator

This directory is the minimal D1 generation and audit package.

- `main_cargo_sl_d1.m`: generate recording-uniform Train and copy frozen Val.
- `validate_d1_recording_uniform.m`: read-only protocol and input validation.
- `test_d1_recording_uniform.m`: synthetic two-stage sampling test.
- `audit_d1_dataset.py`: frozen-Val, split-isolation, exposure, count and SIR audit.
- `.arr_gen/`: required precomputed Bellhop arrival assets.

The data root is external. Set `MUART_DATA_ROOT` before direct MATLAB use; the
top-level `run_v6d1.ps1 -DataRoot <path>` does this automatically. The root
must directly contain `data_sl_templates_bp12`, `data_sl_split_bp12`, and the
frozen `data_cargo_sl_sir15` dataset.

This package does not construct, copy, or evaluate Test.
