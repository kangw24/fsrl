# Test suite map

Run the complete suite from the repository root:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Current CGR audit surface

- `test_constructive_global_rank.py`: CGR model invariants, online state and
  batch/online endpoint parity.
- `test_cgr_evaluation.py`: evaluator metrics, presentation complementarity,
  RNG isolation and report provenance.
- `test_liu2026_human_data.py`: public Liu data parsing and integrity checks.
- `test_liu2026_symbolic_tasks.py`: symbolic support/query observation contract.

## Shared cognitive and analysis checks

- `test_analysis_alignment.py`
- `test_cognitive_validation.py`

## Archived candidates and controls

`archive/legacy_candidates/` retains 21 regression modules for earlier candidate
families (AADM, DCR, bounded rank, Miconi-Kay, online ordinal, Phase 5j and
WBDM). The archive remains part of full test discovery, but is not evidence that
those models remain active candidates. Candidate status is documented in
`docs/current_status.md` and `docs/research_ledger.md`.

Keep tests deterministic and self-contained.  Tests may read committed fixtures
or explicit external-data paths, but must not depend on disposable `_tmp*`,
`_smoke*` or log files under `outputs/`.
