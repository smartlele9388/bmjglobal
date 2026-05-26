# Analysis Workflow and Output Manifest

## Manuscript Title

**The Hidden Gender Tax of Stroke: Disability Burden, Unmet Care Needs, and Informal Care Across Seven Aging Societies**

## Cohort Names

CHARLS, HRS, ELSA, SHARE, KLoSA, LASI, and MHAS.

## Data Governance Statement

The original cohort data are not public in this repository. They must be obtained through the relevant cohort data application or registration processes. This code repository does not include raw cohort files, restricted individual-level data, intermediate person-level analytic datasets, generated result files, or local logs.

## Workflow Summary

| Step | Purpose | Primary script(s) | Main output folder |
| --- | --- | --- | --- |
| Initial scan | Harmonized variable scan and latest post-stroke analytic sample construction | `pcgi_initial_analysis.py`; `gender_tax_step1_protocol_analysis.py`; `extract_step2b_cohort_wave_years.py` | `output/tables/`; `output/inventory/` |
| Step 2b | Patient-side disability, care hours, strict unmet need, informal care gap, YLD-like burden, missingness, age-sex standardization | `step2b_poststroke_disability_care_revision.py` | `results/step2b/` |
| Step 3A | Helper-level caregiver variable inventory and eligibility screening | `step3a_helper_level_inventory.py` | `results/step3a/` |
| Step 3B | HRS-only provider-sex by patient-sex helper-level care-hour matrix | `step3b_gender_care_matrix.py`; `run_step3b.py` | `results/step3b/` |
| Step 3C | HRS-only helper-level missingness and robustness diagnostics | `step3c_hrs_helper_missingness_diagnostics.py` | `results/step3c/` |
| Step 4 | Integrated results, claim-support audit, manuscript-ready snippets | `step4_integrated_results_and_claim_audit.py` | `results/step4/` |
| Step 5A | Recipient-level 2050 care-demand projection framework | `step5a_2050_recipient_level_care_projection.py` | `results/step5a/` |
| Step 5A-fix | External population/prevalence ingestion and projection rerun when inputs exist | `step5a_fix_external_data_ingestion.py`; `step5a_fix_rerun_projection.py` | `results/step5a_fix/` |
| Step 6 | Submission-ready statistics and figures | `step6_submission_ready_statistics.py`; `step6_submission_ready_figures.py`; `run_step6.py` | `results/step6/` |

## Recommended Scripts Run Order

```text
1. python scripts/pcgi_initial_analysis.py
2. python scripts/gender_tax_step1_protocol_analysis.py
3. python scripts/extract_step2b_cohort_wave_years.py
4. python scripts/step2b_poststroke_disability_care_revision.py
5. python scripts/step3a_helper_level_inventory.py
6. python scripts/run_step3b.py
7. python scripts/step3c_hrs_helper_missingness_diagnostics.py
8. python scripts/step4_integrated_results_and_claim_audit.py
9. python scripts/step5a_2050_recipient_level_care_projection.py
10. python scripts/step5a_fix_external_data_ingestion.py
11. python scripts/step5a_fix_rerun_projection.py
12. python scripts/run_step6.py
```

## Main Tables and Figures

| Manuscript item | Source output |
| --- | --- |
| Main analysis scope by cohort | `results/step4/table20_analysis_scope_by_cohort.csv` |
| Patient-side main disability, YLD-like burden, strict unmet need, informal care gap, and recipient-level care-hour results | `results/step4/table21_patient_side_main_results.csv` |
| Patient-side sex inequality table | `results/step4/table22_patient_side_sex_inequality.csv` |
| HRS-only provider-side helper-level summary | `results/step4/table23_hrs_provider_side_summary.csv` |
| Claim-support audit | `results/step4/table24_claim_support_audit.csv` |
| Abstract-ready key numbers | `results/step4/table25_abstract_ready_key_numbers.csv` |
| YLD-like burden figure | `results/step4/figures/fig1_step4_yld_like_burden_by_cohort.png` |
| Care gap figure | `results/step4/figures/fig2_step4_care_gap_by_cohort.png` |
| Recipient-level care-hour figure | `results/step4/figures/fig3_step4_care_hours_by_cohort.png` |
| HRS-only gender matrix and sensitivity figure | `results/step4/figures/fig4_step4_hrs_gender_matrix_and_sensitivity.png` |
| Submission-ready figures | `results/step6/figures/` |

## Supplementary Tables and Diagnostics

| Supplementary domain | Source output(s) |
| --- | --- |
| Sample flow | `results/step2b/table_s0_sample_flow.csv` |
| Strict unmet need variable availability | `results/step2b/table_s1_unmet_variable_availability.csv` |
| Care-hour missingness | `results/step2b/table_s2_care_hour_missingness.csv` |
| Care-hour p95/p99 cutoffs | `results/step2b/table_s3_care_hour_p99_cutoffs.csv` |
| Care-hour sensitivity | `results/step2b/table_s4_care_hour_sensitivity.csv` |
| Covariate availability | `results/step2b/table_s5_covariate_availability.csv` |
| Helper variable availability | `results/step2b/table_s6_helper_variable_availability.csv`; `results/step3a/step3a_helper_variable_inventory.csv` |
| YLD-like sensitivity | `results/step2b/table5_sensitivity_yld_no_disability_dw0019.csv` |
| HRS helper-level main matrix | `results/step3b/table7_step3b_gender_care_matrix.csv` |
| HRS provider care shares | `results/step3b/table8_step3b_gender_care_shares.csv` |
| Step 3B exclusion diagnostics | `results/step3b/table9_step3b_exclusion_diagnostics.csv` |
| HRS patient-level provider-hour aggregation | `results/step3b/table10_step3b_patient_level_provider_hours.csv` |
| Step 3B paid/cap sensitivity | `results/step3b/supp_table_step3b_sensitivity_paid_and_cap.csv` |
| Step 3B bootstrap uncertainty | `results/step3b/supp_table_step3b_bootstrap_uncertainty.csv` |
| Step 3C helper-row missingness | `results/step3c/table12_step3c_helper_row_missingness.csv` |
| Step 3C patient-level inclusion diagnostics | `results/step3c/table13_step3c_patient_level_inclusion_diagnostics.csv` |
| Step 3C included-vs-excluded balance | `results/step3c/table14_step3c_included_vs_excluded_balance.csv` |
| Step 3C missing provider-sex bounds | `results/step3c/table15_step3c_missing_provider_sex_bounds.csv` |
| Step 3C missing-hours sensitivity | `results/step3c/table16_step3c_missing_hours_sensitivity.csv` |
| Step 3C combined missingness sensitivity | `results/step3c/table17_step3c_combined_missingness_sensitivity.csv` |
| Step 3C paid-care missingness | `results/step3c/table18_step3c_paid_care_missingness.csv` |
| Step 3C robustness summary | `results/step3c/table19_step3c_robustness_summary.csv` |
| Step 5A external data inventory and projection readiness | `results/step5a/`; `results/step5a_fix/` |
| Quality logs | `results/step2b/*quality_check_log.txt`; `results/step3b/*quality_check_log.txt`; `results/step3c/*quality_check_log.txt`; `results/step4/*quality_check_log.txt`; `results/step5a*/*quality_check_log.txt` |

## Interpretation Guardrails

- Do not describe provider-side decomposition as seven-country evidence.
- Do not calculate or report hidden gender care tax without adult sex-specific denominator inputs.
- Do not infer provider sex from patient sex, relationship, marital status, or household roster unless explicitly validated.
- Do not treat missing care hours as zero in main analyses.
- Do not call survey-based post-stroke YLD-like estimates official GBD YLDs.
- Do not call HRS helper-level care hours pure informal care unless formal/paid status is validated.

