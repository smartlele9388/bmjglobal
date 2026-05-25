# BMJ Global Health Analysis Code

This repository contains the analysis code used to build the public-source BMJ Global Health analysis panel and recovery analyses.

## Included

- `scripts/download_bmjgh_sources.ps1`: downloads public source files and records a download manifest.
- `scripts/export_bmjgh_to_csv.ps1`: converts downloaded Excel workbooks to CSV extracts.
- `scripts/prepare_bmjgh_aux_sources.ps1`: prepares auxiliary country, WHO region, and climate files.
- `scripts/build_bmjgh_main_panel.ps1`: builds the 2016-2024 country-year analytic panel and QC summaries.
- `scripts/run_bmjgh_recovery_analysis.ps1`: runs descriptive, recovery, modified Poisson, and event-study analyses.
- `scripts/export_bmjgh_png_figures.ps1`: exports manuscript-style PNG figures and main-text table CSVs.
- `docs/analysis_code_manifest.md`: code inventory, run order, inputs, and outputs.

## Not Included

- Raw downloaded data files.
- Generated analysis outputs, figures, and tables.
- Local logs, caches, credentials, or temporary files.
- Restricted clinical or patient-level data.

## Reproducibility Notes

The scripts were written for a local Windows PowerShell workflow. Paths in the original scripts point to the local project directory used during analysis. To rerun elsewhere, update the `$root` or `$analysisDir` variables near the top of each script before execution.

Recommended run order:

1. `scripts/download_bmjgh_sources.ps1`
2. `scripts/export_bmjgh_to_csv.ps1`
3. `scripts/prepare_bmjgh_aux_sources.ps1`
4. `scripts/build_bmjgh_main_panel.ps1`
5. `scripts/run_bmjgh_recovery_analysis.ps1`
6. `scripts/export_bmjgh_png_figures.ps1`

The repository intentionally tracks code only. Recreated data and outputs should remain local unless separately reviewed for sharing.
