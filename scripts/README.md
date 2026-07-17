# Script entrypoints

The top level contains only current CGR/data entrypoints plus four compatibility
modules imported by archived regression tests.

## Current entrypoints

- `cgr_v3_1_vs_liu_humans.py`
- `cgr_v3_2_vs_liu_humans.py`
- `cgr_v3_heldout_blind.py`
- `cgr_v3_vs_liu_humans.py`
- `audit_liu2026_human_data.py`
- `fetch_liu2026_human_data.py`

## Compatibility support

`audit_ciranka2022_exp4.py`, `generate_ciranka2022_frozen_predictions.py`,
`diagnose_phase5j_order_conflict.py` and `model_recovery_phase5j_path.py` remain
here because archived regression tests import them directly. They are not active
candidate entrypoints.

Concluded standalone scripts are immutable snapshots under
`archive/legacy_candidates/`. They are retained for source and hash audit, not
as top-level commands.
