# Outputs directory contract

`outputs/` contains local experiment products.  Generated files are ignored by
default; this index is the only file in the directory intended for routine
version control.

## Active root

Only these entries belong at the top level:

- `cgr_v3_1_vs_liu_humans/` and `cgr_v3_2_vs_liu_humans/`: current matched
  endpoint and online-process reports;
- `cgr_v3_heldout_blind/` and `cgr_v3_vs_liu_humans/`: canonical v3 audit
  reports retained by `docs/reports.md`;
- `external_data/` and `liu2026_human_audit/`: shared source data and its audit;
- `archive/`: immutable historical batches.

Everything else must be temporary (`_tmp_` or `_smoke_`) or placed in a dated
archive batch after its conclusion is recorded.

## Retention classes

| Class | Examples | Policy |
|---|---|---|
| Current canonical reports | `cgr_v3_1_vs_liu_humans/`, `cgr_v3_2_vs_liu_humans/` | Keep locally; regenerate with the corresponding script |
| Frozen or historical evidence | `archive/<date>_<label>/` | Preserve in a dated, classified batch |
| External source data | `external_data/`, `liu2026_human_audit/raw/` | Preserve; do not treat as disposable output |
| Exploratory runs | phase, sweep, diagnostic and lesion directories | Keep only while needed to audit a recorded conclusion |
| Disposable artifacts | `_tmp*`, `_smoke*`, `_repair*`, `*.log`, caches | Delete after the run |

The canonical report inventory and contamination labels are maintained in
`docs/reports.md`.  A report is not confirmatory merely because it is retained
here; use the status recorded in that inventory and in the report provenance.

## Naming

New durable output directories should use a descriptive experiment name and
contain a machine-readable report with seeds, evaluator protocol, parameters,
and data provenance. Temporary runs must begin with `_tmp_` or `_smoke_` so
they can be cleaned safely. Do not overwrite or append to an existing archive
batch.
