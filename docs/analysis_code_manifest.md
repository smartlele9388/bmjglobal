# Analysis Code Manifest

## Scope

This upload contains only the analysis scripts used for the BMJ Global Health public-source panel and recovery analyses. It excludes raw data, generated data extracts, figures, tables, and local logs.

## Script Inventory

| Script | Purpose | Main outputs when run locally |
| --- | --- | --- |
| `scripts/download_bmjgh_sources.ps1` | Downloads public source files including WUENIC, WHO immunization portal exports, WPP documentation/indicators, World Bank indicators, UCDP conflict data, and climate/source metadata where available. | `downloads/bmjgh_sources/raw/`, `download_manifest.json` |
| `scripts/export_bmjgh_to_csv.ps1` | Converts downloaded Excel workbooks into CSV folders for reproducible downstream parsing. | `downloads/bmjgh_sources/csv/`, `csv_manifest.json` |
| `scripts/prepare_bmjgh_aux_sources.ps1` | Builds auxiliary mappings and climate anomaly files from public APIs. | `downloads/bmjgh_sources/aux/` |
| `scripts/build_bmjgh_main_panel.ps1` | Harmonizes public source extracts into a country-year analytic panel for 2016-2024 and creates QC summaries. | `downloads/bmjgh_sources/analysis/bmjgh_main_panel_2016_2024.csv`, QC JSON/Markdown |
| `scripts/run_bmjgh_recovery_analysis.ps1` | Runs descriptive summaries, recovery classification, modified Poisson models, event-study models, and supporting analysis tables. | `downloads/bmjgh_sources/analysis/recovery_analysis/` |
| `scripts/export_bmjgh_png_figures.ps1` | Draws PNG figures and exports main-text table CSVs from the recovery-analysis outputs. | `figures_png/`, `tables_main_text/` |

## Run Order

Run the scripts from repository root in this order:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_bmjgh_sources.ps1
powershell -ExecutionPolicy Bypass -File scripts/export_bmjgh_to_csv.ps1
powershell -ExecutionPolicy Bypass -File scripts/prepare_bmjgh_aux_sources.ps1
powershell -ExecutionPolicy Bypass -File scripts/build_bmjgh_main_panel.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_bmjgh_recovery_analysis.ps1
powershell -ExecutionPolicy Bypass -File scripts/export_bmjgh_png_figures.ps1
```

## Data Governance

No patient-level or restricted clinical data are required for these scripts. The code uses public aggregate sources only. Generated data and outputs are reproducible from the scripts and are intentionally excluded from this repository upload.

## Local Path Note

The original scripts use absolute local Windows paths. Before rerunning on another machine, edit the `$root` or `$analysisDir` variables near the top of each script to match the local checkout location.
