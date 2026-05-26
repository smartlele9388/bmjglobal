# The Hidden Gender Tax of Stroke

Analysis code for the manuscript:

**"The Hidden Gender Tax of Stroke: Disability Burden, Unmet Care Needs, and Informal Care Across Seven Aging Societies"**

## Cohorts

The analysis uses harmonized aging cohort data from:

- CHARLS
- HRS
- ELSA
- SHARE
- KLoSA
- LASI
- MHAS

## Data Availability

Raw cohort data are **not included** in this repository and are **not publicly redistributed here**.

Access to the underlying individual-level data requires approval or registration through the relevant cohort data providers or harmonized data portals. Users who wish to reproduce the analyses must obtain the source files directly from CHARLS, HRS, ELSA, SHARE, KLoSA, LASI, and MHAS according to each cohort's data-use requirements.

This repository tracks analysis code and documentation only. It intentionally excludes raw data, cleaned individual-level analytic data, generated result tables, figures, logs, and local archives.

## Analysis Workflow

The workflow is organized around patient-side disability and care burden analyses, helper-level caregiver eligibility screening, an HRS-only provider-side gender care matrix, robustness diagnostics, and manuscript integration.

1. Initial harmonized cohort feasibility and latest post-stroke sample construction.
2. Step 2b patient-side disability, unmet need, informal care gap, YLD-like disability burden, care-hour missingness, and age-sex standardization.
3. Step 3A helper-level caregiver variable inventory and eligibility screening.
4. Step 3B HRS-only provider-sex by patient-sex helper-level care-hour matrix.
5. Step 3C HRS-only helper-level missingness and robustness diagnostics.
6. Step 4 integrated results and claim-support audit.
7. Step 5A recipient-level 2050 care-demand projection framework and external-data ingestion layer.
8. Step 6 submission-ready statistics and figures.

Provider-side gender care decomposition is currently HRS-only. The scripts do **not** infer provider sex from patient sex or relationship and do **not** calculate a seven-country hidden gender care tax.

## Scripts Run Order

Run from the repository root after placing approved cohort data in the expected local folders.

```powershell
python scripts/pcgi_initial_analysis.py
python scripts/gender_tax_step1_protocol_analysis.py
python scripts/extract_step2b_cohort_wave_years.py
python scripts/step2b_poststroke_disability_care_revision.py
python scripts/step3a_helper_level_inventory.py
python scripts/run_step3b.py
python scripts/step3c_hrs_helper_missingness_diagnostics.py
python scripts/step4_integrated_results_and_claim_audit.py
python scripts/step5a_2050_recipient_level_care_projection.py
python scripts/step5a_fix_external_data_ingestion.py
python scripts/step5a_fix_rerun_projection.py
python scripts/run_step6.py
```

Older exploratory scripts are retained only where they document earlier reproducible checks. The main manuscript workflow starts from the Step 2b revision and proceeds through Step 6.

## Output Map

See [docs/analysis_workflow_and_output_manifest.md](docs/analysis_workflow_and_output_manifest.md) for the detailed mapping between scripts and expected main tables, figures, and supplementary tables.

## Important Interpretation Limits

- Seven-cohort patient-side disability and survey-based post-stroke YLD-like burden are supported.
- Recipient-level care-hour analysis is supported only for cohorts with harmonized recipient-level care hours; SHARE is not forced into care-hour analysis.
- Strict unmet need is estimated only where required care-source variables are constructible.
- Provider-side gender care decomposition is HRS-only.
- HRS helper-level care hours are not labeled as pure informal care unless paid/formal status is validated.
- Hidden gender care tax is not calculated without adult sex-specific population denominators.

