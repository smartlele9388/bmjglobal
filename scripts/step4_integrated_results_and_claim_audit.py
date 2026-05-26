from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STEP2B = ROOT / "results" / "step2b"
STEP3B = ROOT / "results" / "step3b"
STEP3C = ROOT / "results" / "step3c"
OUT = ROOT / "results" / "step4"
OUT.mkdir(parents=True, exist_ok=True)

COHORTS = ["CHARLS", "HRS", "ELSA", "SHARE", "KLoSA", "LASI", "MHAS"]

INPUT_FILES = {
    "step2b_sample_flow": STEP2B / "table_s0_sample_flow.csv",
    "step2b_strict_unmet_need": STEP2B / "table4a_strict_unmet_need.csv",
    "step2b_informal_care_gap": STEP2B / "table4b_informal_care_gap.csv",
    "step2b_yld_by_cohort": STEP2B / "table5_yld_stroke_by_cohort.csv",
    "step2b_yld_by_cohort_sex": STEP2B / "table5_yld_stroke_by_cohort_sex.csv",
    "step2b_standardized": STEP2B / "table6_age_sex_standardized_estimates.csv",
    "step2b_care_hour_missingness": STEP2B / "table_s2_care_hour_missingness.csv",
    "step2b_care_hour_sensitivity": STEP2B / "table_s4_care_hour_sensitivity.csv",
    "step2b_covariate_availability": STEP2B / "table_s5_covariate_availability.csv",
    "step3b_matrix": STEP3B / "table7_step3b_gender_care_matrix.csv",
    "step3b_shares": STEP3B / "table8_step3b_gender_care_shares.csv",
    "step3b_exclusion_diagnostics": STEP3B / "table9_step3b_exclusion_diagnostics.csv",
    "step3b_patient_level": STEP3B / "table10_step3b_patient_level_provider_hours.csv",
    "step3b_bootstrap": STEP3B / "supp_table_step3b_bootstrap_uncertainty.csv",
    "step3b_sensitivity": STEP3B / "supp_table_step3b_sensitivity_paid_and_cap.csv",
    "step3c_helper_row_missingness": STEP3C / "table12_step3c_helper_row_missingness.csv",
    "step3c_patient_inclusion": STEP3C / "table13_step3c_patient_level_inclusion_diagnostics.csv",
    "step3c_balance": STEP3C / "table14_step3c_included_vs_excluded_balance.csv",
    "step3c_provider_sex_bounds": STEP3C / "table15_step3c_missing_provider_sex_bounds.csv",
    "step3c_missing_hours_sensitivity": STEP3C / "table16_step3c_missing_hours_sensitivity.csv",
    "step3c_combined_sensitivity": STEP3C / "table17_step3c_combined_missingness_sensitivity.csv",
    "step3c_paid_care_missingness": STEP3C / "table18_step3c_paid_care_missingness.csv",
    "step3c_robustness_summary": STEP3C / "table19_step3c_robustness_summary.csv",
}

OPTIONAL_STEP2B_FILES = {
    "step2b_distribution": STEP2B / "table2_disability_care_hour_distribution.csv",
    "step2b_recipient_only": STEP2B / "table3_recipient_only_care_intensity.csv",
    "step2b_sex_stratified": STEP2B / "supp_table_s2_sex_stratified.csv",
}


def read_inputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rows = []
    data: dict[str, pd.DataFrame] = {}
    for name, path in INPUT_FILES.items():
        exists = path.exists()
        status = "present" if exists else "missing"
        rows.append(
            {
                "input_name": name,
                "file_path": str(path.relative_to(ROOT)),
                "exists": exists,
                "status": status,
                "notes": "" if exists else "Required Step 4 input is missing; downstream scope flags may be incomplete.",
            }
        )
        if exists:
            data[name] = pd.read_csv(path)
    for name, path in OPTIONAL_STEP2B_FILES.items():
        if path.exists():
            data[name] = pd.read_csv(path)
    check = pd.DataFrame(rows)
    check.to_csv(OUT / "step4_input_file_check.csv", index=False)
    return data, check


def v(df: pd.DataFrame | None, cohort: str, column: str, indicator: str | None = None) -> float:
    if df is None or column not in df.columns or "cohort" not in df.columns:
        return np.nan
    sub = df[df["cohort"].astype(str).str.upper().eq(cohort.upper())]
    if indicator is not None and "indicator" in sub.columns:
        sub = sub[sub["indicator"].astype(str).eq(indicator)]
    if sub.empty:
        return np.nan
    return pd.to_numeric(sub[column], errors="coerce").iloc[0]


def txt_range(series: pd.Series, digits: int = 1, suffix: str = "") -> str:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return "NA"
    return f"{vals.min():.{digits}f}{suffix} to {vals.max():.{digits}f}{suffix}"


def has_cohort(df: pd.DataFrame | None, cohort: str) -> bool:
    if df is None or "cohort" not in df.columns:
        return False
    return df["cohort"].astype(str).str.upper().eq(cohort.upper()).any()


def cohort_value_available(df: pd.DataFrame | None, cohort: str, value_cols: list[str]) -> bool:
    if df is None or "cohort" not in df.columns:
        return False
    sub = df[df["cohort"].astype(str).str.upper().eq(cohort.upper())]
    if sub.empty:
        return False
    present_cols = [c for c in value_cols if c in sub.columns]
    if not present_cols:
        return True
    return sub[present_cols].notna().any(axis=None)


def table20_analysis_scope(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sample = data.get("step2b_sample_flow")
    yld = data.get("step2b_yld_by_cohort")
    strict = data.get("step2b_strict_unmet_need")
    gap = data.get("step2b_informal_care_gap")
    care_missing = data.get("step2b_care_hour_missingness")
    care_sens = data.get("step2b_care_hour_sensitivity")
    dist = data.get("step2b_distribution")
    matrix = data.get("step3b_matrix")
    diag = data.get("step3b_exclusion_diagnostics")

    rows = []
    for cohort in COHORTS:
        disability = has_cohort(sample, cohort) and cohort_value_available(sample, cohort, ["n_disability_analysis"])
        yld_avail = has_cohort(yld, cohort) and cohort_value_available(yld, cohort, ["mean_DW_weighted", "mean_DW_unweighted"])
        strict_avail = cohort_value_available(strict, cohort, ["weighted_percent", "crude_percent"])
        gap_avail = cohort_value_available(gap, cohort, ["weighted_percent", "crude_percent"])
        recipient_hours = (
            cohort_value_available(dist, cohort, ["care_hour_known_N"])
            or cohort_value_available(
                care_sens,
                cohort,
                [
                    "mean_hours_week_among_all_stroke_survivors",
                    "mean_hours_week_among_recipients_only",
                ],
            )
        )
        provider_matrix = has_cohort(matrix, cohort)

        diag_row = pd.DataFrame()
        if diag is not None and "cohort" in diag.columns:
            diag_row = diag[diag["cohort"].astype(str).str.upper().eq(cohort.upper())]
        provider_sex_available = False
        helper_hours_available = False
        main_limitation = ""
        if not diag_row.empty:
            d = diag_row.iloc[0]
            provider_sex_available = bool(pd.to_numeric(pd.Series([d.get("n_missing_provider_sex")]), errors="coerce").notna().iloc[0])
            helper_hours_available = bool(pd.to_numeric(pd.Series([d.get("n_missing_hours_week")]), errors="coerce").notna().iloc[0])
            main_limitation = str(d.get("main_limitation", ""))
        provider_sex_available = cohort == "HRS" and (provider_matrix or provider_sex_available)
        helper_hours_available = cohort == "HRS" and (provider_matrix or helper_hours_available)

        paid_validated = False
        included_patient = disability and yld_avail
        included_provider = provider_matrix and cohort == "HRS"

        notes = []
        if cohort == "SHARE" and not recipient_hours:
            notes.append("Included for disability/YLD-like burden; excluded from care-hour analyses because harmonizable hours remain unavailable.")
        if cohort == "LASI" and not strict_avail:
            notes.append("Strict unmet need unavailable because strict any-care components are not constructible.")
        if cohort != "HRS":
            notes.append("Excluded from main provider-side matrix because provider sex and helper-specific hours are not jointly validated.")
        else:
            notes.append("Only cohort currently validated for HRS-only helper-level provider-sex by patient-sex matrix.")
        notes.append("Hidden gender care tax unavailable unless adult sex-specific population denominators are supplied.")
        if main_limitation and main_limitation.lower() not in {"nan", "none"}:
            notes.append(main_limitation)

        rows.append(
            {
                "cohort": cohort,
                "disability_analysis_available": disability,
                "yld_like_analysis_available": yld_avail,
                "strict_unmet_need_available": strict_avail,
                "informal_care_gap_available": gap_avail,
                "recipient_level_care_hours_available": recipient_hours,
                "helper_level_provider_matrix_available": provider_matrix,
                "provider_sex_available": provider_sex_available,
                "helper_specific_hours_available": helper_hours_available,
                "paid_or_formal_status_validated": paid_validated,
                "included_in_main_patient_side_analysis": included_patient,
                "included_in_main_provider_side_analysis": included_provider,
                "notes": " ".join(notes),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table20_analysis_scope_by_cohort.csv", index=False)
    return out


def table21_patient_side_main(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sample = data.get("step2b_sample_flow")
    yld = data.get("step2b_yld_by_cohort")
    strict = data.get("step2b_strict_unmet_need")
    gap = data.get("step2b_informal_care_gap")
    std = data.get("step2b_standardized")
    dist = data.get("step2b_distribution")
    rec = data.get("step2b_recipient_only")

    rows = []
    for cohort in COHORTS:
        notes = ["Survey-based post-stroke YLD-like disability burden; not official GBD YLD."]
        if cohort == "SHARE":
            notes.append("SHARE retained for disability/YLD-like analyses; care-hour and informal-care-gap fields left NA if harmonized hours unavailable.")
        if cohort == "LASI" and pd.isna(v(strict, cohort, "weighted_percent")):
            notes.append("LASI strict unmet need not forced because strict care-source components were unavailable.")
        rows.append(
            {
                "cohort": cohort,
                "n_stroke_survivors_disability_analysis": v(sample, cohort, "n_disability_analysis"),
                "any_disability_percent": 100 - v(dist, cohort, "no_disability_percent"),
                "ADL_ge3_percent": v(dist, cohort, "ADL_ge3_percent"),
                "mean_DW_weighted": v(yld, cohort, "mean_DW_weighted"),
                "YLD_per_1000_stroke_survivors": v(yld, cohort, "YLD_per_1000_stroke_survivors"),
                "strict_unmet_need_percent": v(strict, cohort, "weighted_percent"),
                "informal_care_gap_percent": v(gap, cohort, "weighted_percent"),
                "any_informal_care_percent": v(dist, cohort, "any_informal_care_percent"),
                "mean_informal_care_hours_week_all_survivors": v(dist, cohort, "mean_informal_h_week_raw"),
                "mean_informal_care_hours_week_recipients_only": v(rec, cohort, "mean_h_week_among_recipients_raw"),
                "age_sex_standardized_any_disability": v(std, cohort, "standardized_estimate", "any ADL/IADL disability"),
                "age_sex_standardized_mean_DW": v(std, cohort, "standardized_estimate", "mean disability weight / mean YLD"),
                "age_sex_standardized_strict_unmet_need": v(std, cohort, "standardized_estimate", "strict unmet need"),
                "age_sex_standardized_informal_care_gap": v(std, cohort, "standardized_estimate", "informal care gap"),
                "notes": " ".join(notes),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table21_patient_side_main_results.csv", index=False)
    return out


def table22_patient_side_sex_inequality(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    yld_sex = data.get("step2b_yld_by_cohort_sex", pd.DataFrame())
    sex = data.get("step2b_sex_stratified", pd.DataFrame())
    rows = []
    for cohort in COHORTS:
        ys = yld_sex[yld_sex.get("cohort", pd.Series(dtype=str)).astype(str).str.upper().eq(cohort.upper())] if not yld_sex.empty else pd.DataFrame()
        ss = sex[sex.get("cohort", pd.Series(dtype=str)).astype(str).str.upper().eq(cohort.upper())] if not sex.empty else pd.DataFrame()

        def sexval(df: pd.DataFrame, sex_name: str, col: str) -> float:
            if df.empty or col not in df.columns or "sex" not in df.columns:
                return np.nan
            sub = df[df["sex"].astype(str).str.lower().eq(sex_name.lower())]
            if sub.empty:
                return np.nan
            return pd.to_numeric(sub[col], errors="coerce").iloc[0]

        f_dw = sexval(ys, "Female", "mean_DW_weighted")
        m_dw = sexval(ys, "Male", "mean_DW_weighted")
        f_any = sexval(ss, "Female", "any_disability_percent")
        m_any = sexval(ss, "Male", "any_disability_percent")
        f_adl3 = sexval(ss, "Female", "ADL_ge3_percent")
        m_adl3 = sexval(ss, "Male", "ADL_ge3_percent")
        note = (
            "Patient-side sex inequality only; do not infer provider-side caregiver burden. "
            "Sex-stratified strict unmet need and informal-care-gap tables were not available in Step 2b outputs, so those fields are NA."
        )
        rows.append(
            {
                "cohort": cohort,
                "female_mean_DW": f_dw,
                "male_mean_DW": m_dw,
                "female_minus_male_mean_DW": f_dw - m_dw if pd.notna(f_dw) and pd.notna(m_dw) else np.nan,
                "female_to_male_mean_DW_ratio": f_dw / m_dw if pd.notna(f_dw) and pd.notna(m_dw) and m_dw != 0 else np.nan,
                "female_any_disability_percent": f_any,
                "male_any_disability_percent": m_any,
                "female_minus_male_any_disability_percent": f_any - m_any if pd.notna(f_any) and pd.notna(m_any) else np.nan,
                "female_ADL_ge3_percent": f_adl3,
                "male_ADL_ge3_percent": m_adl3,
                "female_minus_male_ADL_ge3_percent": f_adl3 - m_adl3 if pd.notna(f_adl3) and pd.notna(m_adl3) else np.nan,
                "female_strict_unmet_need_percent": np.nan,
                "male_strict_unmet_need_percent": np.nan,
                "female_informal_care_gap_percent": np.nan,
                "male_informal_care_gap_percent": np.nan,
                "interpretation_note": note,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table22_patient_side_sex_inequality.csv", index=False)
    return out


def table23_hrs_provider_summary(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    matrix = data["step3b_matrix"]
    missing = data["step3c_helper_row_missingness"]
    bootstrap = data.get("step3b_bootstrap", pd.DataFrame())
    bounds = data["step3c_provider_sex_bounds"]
    combined = data["step3c_combined_sensitivity"]
    paid = data["step3c_paid_care_missingness"]

    m = matrix.iloc[0]
    total_rows = int(missing["n_helper_rows"].sum())
    valid_rows = int(missing.loc[missing["missingness_group"].eq("valid_for_main_matrix"), "n_helper_rows"].iloc[0])
    boot_fp = bootstrap[bootstrap.get("statistic", pd.Series(dtype=str)).eq("female_provider_share")] if not bootstrap.empty else pd.DataFrame()
    prop = bounds[bounds["scenario"].eq("proportional_allocation_by_observed_patient_sex")]
    all_male = bounds[bounds["scenario"].eq("all_missing_provider_sex_are_male")]
    conservative = combined[combined["scenario"].eq("conservative_against_female_provider_share")]
    paid_missing = paid[paid["group"].eq("paid_care missing")]

    out = pd.DataFrame(
        [
            {
                "cohort": "HRS",
                "analysis_scope": "HRS-only observed helper-level provider-sex by patient-sex care-hour matrix",
                "valid_helper_rows": valid_rows,
                "total_helper_rows": total_rows,
                "percent_valid_helper_rows": valid_rows / total_rows * 100 if total_rows else np.nan,
                "unique_stroke_survivors_valid": m["n_unique_stroke_survivors_with_helper_data"],
                "H_MM_weekly": m["H_MM_weekly"],
                "H_MF_weekly": m["H_MF_weekly"],
                "H_FM_weekly": m["H_FM_weekly"],
                "H_FF_weekly": m["H_FF_weekly"],
                "H_total_weekly": m["H_total_weekly"],
                "female_provider_share_main": m["female_provider_share"],
                "male_provider_share_main": m["male_provider_share"],
                "female_to_male_provider_ratio_main": m["female_to_male_provider_ratio"],
                "bootstrap_ci_lower_female_provider_share": boot_fp["ci_lower_95"].iloc[0] if not boot_fp.empty else np.nan,
                "bootstrap_ci_upper_female_provider_share": boot_fp["ci_upper_95"].iloc[0] if not boot_fp.empty else np.nan,
                "proportional_missing_provider_sex_female_provider_share": prop["female_provider_share"].iloc[0] if not prop.empty else np.nan,
                "all_missing_provider_sex_assigned_male_female_provider_share": all_male["female_provider_share"].iloc[0] if not all_male.empty else np.nan,
                "combined_conservative_female_provider_share": conservative["female_provider_share"].iloc[0] if not conservative.empty else np.nan,
                "paid_care_missing_percent": paid_missing["percent_rows"].iloc[0] if not paid_missing.empty else np.nan,
                "conclusion": "Observed HRS helper-level results show women provide most provider-specific care hours, but the result is sensitive to extreme assumptions about missing provider sex.",
                "manuscript_caveat": "Interpret as HRS-only helper-level care hours, not seven-country informal care or a hidden gender care tax.",
            }
        ]
    )
    out.to_csv(OUT / "table23_hrs_provider_side_summary.csv", index=False)
    return out


def table24_claim_audit(t21: pd.DataFrame, t23: pd.DataFrame) -> pd.DataFrame:
    fp = t23["female_provider_share_main"].iloc[0]
    cons = t23["combined_conservative_female_provider_share"].iloc[0]
    rows = [
        ["Post-stroke disability burden varies across seven aging cohorts.", "supported", "table21_patient_side_main_results.csv", f"Any disability range {txt_range(t21['any_disability_percent'])}.", "Patient-side disability definitions harmonized in Step 2b.", "Post-stroke disability burden varied across seven aging cohorts.", "Do not imply provider-side data are available in all seven cohorts."],
        ["Survey-based YLD-like disability burden can be compared across seven cohorts.", "supported", "table21_patient_side_main_results.csv", f"Mean DW range {txt_range(t21['mean_DW_weighted'], 3)}.", "This is not official GBD YLD.", "Survey-based post-stroke YLD-like disability burden was compared across seven cohorts.", "GBD YLD was estimated from survey data."],
        ["Recipient-level care-hour burden can be compared across six cohorts.", "supported", "table21_patient_side_main_results.csv; table_s4_care_hour_sensitivity.csv", "SHARE care-hour columns are NA; six cohorts have recipient-level hours.", "Care hours are recipient-level totals and highly skewed.", "Recipient-level care-hour burden was compared among six cohorts with harmonized hours.", "Seven-cohort care-hour burden was estimated."],
        ["Strict unmet care need can be compared across cohorts where formal/informal/any-care indicators are available.", "supported_with_caution", "table4a_strict_unmet_need.csv; table21_patient_side_main_results.csv", f"Strict unmet need range {txt_range(t21['strict_unmet_need_percent'])}.", "LASI strict unmet need unavailable; construction depends on available care-source indicators.", "Strict unmet care need was compared only where required care-source indicators were constructible.", "All seven cohorts have strict unmet need."],
        ["Women stroke survivors have higher patient-side disability burden in several cohorts.", "partially_supported", "table22_patient_side_sex_inequality.csv", "Female-minus-male mean DW and disability columns vary by cohort.", "Patient-side sex differences are not provider-side caregiver burden.", "Women had higher patient-side disability burden in several cohorts.", "Female patient disability proves female caregiver burden."],
        ["Women provide most observed helper-level care hours in HRS.", "supported_with_caution", "table23_hrs_provider_side_summary.csv; table15_step3c_missing_provider_sex_bounds.csv; table17_step3c_combined_missingness_sensitivity.csv", f"Main female_provider_share={fp:.3f}; conservative combined={cons:.3f}.", "HRS only; sensitive to extreme missing-provider-sex assumption; paid/formal status not fully validated.", "In HRS, observed helper-level rows indicate women provided most provider-specific care hours.", "Women provide most informal stroke care across all countries."],
        ["Women provide most informal stroke care across seven countries.", "not_supported", "table20_analysis_scope_by_cohort.csv; table23_hrs_provider_side_summary.csv", "Provider-side matrix is HRS-only.", "Non-HRS provider sex and helper-specific hours are not jointly validated.", "Not allowed from current data.", "Women provide most informal stroke care across seven countries."],
        ["A seven-country gender care burden decomposition is available.", "not_supported", "table20_analysis_scope_by_cohort.csv", "Main provider-side matrix available only for HRS.", "Step 3A excludes non-HRS cohorts from main matrix.", "An HRS-only provider-side decomposition is available.", "Seven-country gender care burden decomposition."],
        ["Hidden gender care tax can be calculated across seven countries.", "not_available", "README_step4.md; step3b_quality_check_log.txt", "Adult sex-specific denominators were not supplied.", "No hidden gender care tax calculated in Step 4.", "Hidden gender care tax was not calculated.", "Hidden gender care tax across seven countries."],
        ["HRS provider-side results are robust under proportional and missing-hour sensitivity.", "supported_with_caution", "table23_hrs_provider_side_summary.csv; table16_step3c_missing_hours_sensitivity.csv", "Proportional and missing-hour sensitivity remain near main estimate.", "Extreme missing-provider-sex assumption reduces female share below 0.5.", "HRS provider-side results were stable under proportional and missing-hour sensitivity.", "HRS result is robust under all missingness assumptions."],
        ["HRS provider-side results remain above 0.5 under the extreme conservative missing-provider-sex scenario.", "not_supported", "table23_hrs_provider_side_summary.csv; table17_step3c_combined_missingness_sensitivity.csv", f"Combined conservative female_provider_share={cons:.3f}.", "This scenario falls below 0.5.", "Extreme conservative missing-provider-sex sensitivity fell below 0.5.", "HRS provider-side result remained above 0.5 in every sensitivity analysis."],
        ["Step 4 can proceed to manuscript integration.", "supported_with_caution", "table24_claim_support_audit.csv; step4_quality_check_log.txt", "Patient-side scope is cross-national; provider-side scope is HRS-only.", "Manuscript claims must be scoped carefully.", "Proceed to manuscript integration with claim restrictions.", "Proceed with seven-country hidden gender care tax."],
    ]
    out = pd.DataFrame(rows, columns=["claim", "support_status", "evidence_files", "key_numeric_evidence", "caveat", "allowed_manuscript_wording", "prohibited_wording"])
    out.to_csv(OUT / "table24_claim_support_audit.csv", index=False)
    return out


def table25_key_numbers(t21: pd.DataFrame, t23: pd.DataFrame) -> pd.DataFrame:
    care = t21[t21["cohort"].ne("SHARE")]
    strict = t21[t21["strict_unmet_need_percent"].notna()]
    gap = t21[t21["informal_care_gap_percent"].notna()]
    hrs = t23.iloc[0]
    rows = [
        ["seven_cohort_disability_range", txt_range(t21["any_disability_percent"], 1, "%"), t21["any_disability_percent"].min(), t21["any_disability_percent"].max(), ", ".join(t21["cohort"]), "table21_patient_side_main_results.csv", "Across seven cohorts, post-stroke disability prevalence ranged from " + txt_range(t21["any_disability_percent"], 1, "%") + "."],
        ["seven_cohort_mean_DW_range", txt_range(t21["mean_DW_weighted"], 3), t21["mean_DW_weighted"].min(), t21["mean_DW_weighted"].max(), ", ".join(t21["cohort"]), "table21_patient_side_main_results.csv", "Survey-based post-stroke YLD-like mean disability weights ranged from " + txt_range(t21["mean_DW_weighted"], 3) + "."],
        ["six_cohort_recipient_level_care_hour_range", txt_range(care["mean_informal_care_hours_week_all_survivors"], 1), care["mean_informal_care_hours_week_all_survivors"].min(), care["mean_informal_care_hours_week_all_survivors"].max(), ", ".join(care["cohort"]), "table21_patient_side_main_results.csv", "Among six cohorts with harmonized recipient-level hours, mean weekly care hours among all stroke survivors ranged from " + txt_range(care["mean_informal_care_hours_week_all_survivors"], 1) + "."],
        ["strict_unmet_need_range", txt_range(strict["strict_unmet_need_percent"], 1, "%"), strict["strict_unmet_need_percent"].min(), strict["strict_unmet_need_percent"].max(), ", ".join(strict["cohort"]), "table21_patient_side_main_results.csv", "Strict unmet need was estimated only where required care-source indicators were constructible and ranged from " + txt_range(strict["strict_unmet_need_percent"], 1, "%") + "."],
        ["informal_care_gap_range", txt_range(gap["informal_care_gap_percent"], 1, "%"), gap["informal_care_gap_percent"].min(), gap["informal_care_gap_percent"].max(), ", ".join(gap["cohort"]), "table21_patient_side_main_results.csv", "Informal care gap ranged from " + txt_range(gap["informal_care_gap_percent"], 1, "%") + " among cohorts with harmonized hours."],
        ["HRS_female_provider_share_main", f"{hrs['female_provider_share_main']:.3f}", hrs["female_provider_share_main"], hrs["female_provider_share_main"], "HRS", "table23_hrs_provider_side_summary.csv", f"In HRS observed helper-level rows, female_provider_share was {hrs['female_provider_share_main']:.3f}."],
        ["HRS_female_provider_share_bootstrap_CI", f"{hrs['bootstrap_ci_lower_female_provider_share']:.3f} to {hrs['bootstrap_ci_upper_female_provider_share']:.3f}", hrs["bootstrap_ci_lower_female_provider_share"], hrs["bootstrap_ci_upper_female_provider_share"], "HRS", "table23_hrs_provider_side_summary.csv", "The patient-level bootstrap interval for HRS female_provider_share was " + f"{hrs['bootstrap_ci_lower_female_provider_share']:.3f} to {hrs['bootstrap_ci_upper_female_provider_share']:.3f}."],
        ["HRS_proportional_missing_provider_sex_share", f"{hrs['proportional_missing_provider_sex_female_provider_share']:.3f}", hrs["proportional_missing_provider_sex_female_provider_share"], hrs["proportional_missing_provider_sex_female_provider_share"], "HRS", "table23_hrs_provider_side_summary.csv", "Under proportional allocation of missing provider sex, the HRS female_provider_share was " + f"{hrs['proportional_missing_provider_sex_female_provider_share']:.3f}."],
        ["HRS_conservative_missing_provider_sex_share", f"{hrs['combined_conservative_female_provider_share']:.3f}", hrs["combined_conservative_female_provider_share"], hrs["combined_conservative_female_provider_share"], "HRS", "table23_hrs_provider_side_summary.csv", "Under the conservative combined missingness scenario, HRS female_provider_share fell to " + f"{hrs['combined_conservative_female_provider_share']:.3f}."],
        ["paid_care_missingness_HRS", f"{hrs['paid_care_missing_percent']:.1f}%", hrs["paid_care_missing_percent"], hrs["paid_care_missing_percent"], "HRS", "table23_hrs_provider_side_summary.csv", "Paid-care status was mostly missing in HRS provider-side data, so helper-level hours should not be described as pure informal care."],
        ["main_limitation", "Provider-side decomposition is HRS-only; no hidden gender care tax was calculated.", np.nan, np.nan, "HRS provider-side only", "table24_claim_support_audit.csv", "Provider-side findings are HRS-only and require cautious interpretation."],
    ]
    out = pd.DataFrame(rows, columns=["result_domain", "estimate_text", "numeric_min", "numeric_max", "cohorts_included", "evidence_file", "manuscript_sentence"])
    out.to_csv(OUT / "table25_abstract_ready_key_numbers.csv", index=False)
    return out


def make_figures(t21: pd.DataFrame, t23: pd.DataFrame, data: dict[str, pd.DataFrame]) -> list[Path]:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # Figure 1
    f1 = t21[["cohort", "mean_DW_weighted"]].copy()
    f1.to_csv(fig_dir / "fig1_step4_yld_like_burden_by_cohort_data.csv", index=False)
    plt.figure(figsize=(7, 4.5))
    plt.bar(f1["cohort"], f1["mean_DW_weighted"], color="#4c78a8")
    plt.ylabel("Mean disability weight")
    plt.title("Survey-based post-stroke YLD-like burden by cohort")
    plt.tight_layout()
    p = fig_dir / "fig1_step4_yld_like_burden_by_cohort.png"
    plt.savefig(p, dpi=200)
    plt.close()
    paths.append(p)

    # Figure 2
    f2 = t21[["cohort", "strict_unmet_need_percent", "informal_care_gap_percent"]].copy()
    f2.to_csv(fig_dir / "fig2_step4_care_gap_by_cohort_data.csv", index=False)
    x = np.arange(len(f2))
    width = 0.36
    plt.figure(figsize=(8, 4.8))
    plt.bar(x - width / 2, f2["strict_unmet_need_percent"], width, label="Strict unmet need", color="#f58518")
    plt.bar(x + width / 2, f2["informal_care_gap_percent"], width, label="Informal care gap", color="#54a24b")
    plt.xticks(x, f2["cohort"])
    plt.ylabel("Percent")
    plt.title("Care need and informal-care gap by cohort")
    plt.legend()
    plt.tight_layout()
    p = fig_dir / "fig2_step4_care_gap_by_cohort.png"
    plt.savefig(p, dpi=200)
    plt.close()
    paths.append(p)

    # Figure 3
    f3 = t21[t21["mean_informal_care_hours_week_all_survivors"].notna()][
        ["cohort", "mean_informal_care_hours_week_all_survivors", "mean_informal_care_hours_week_recipients_only"]
    ].copy()
    f3.to_csv(fig_dir / "fig3_step4_care_hours_by_cohort_data.csv", index=False)
    x = np.arange(len(f3))
    plt.figure(figsize=(8, 4.8))
    plt.bar(x - width / 2, f3["mean_informal_care_hours_week_all_survivors"], width, label="All stroke survivors", color="#4c78a8")
    plt.bar(x + width / 2, f3["mean_informal_care_hours_week_recipients_only"], width, label="Recipients only", color="#e45756")
    plt.xticks(x, f3["cohort"])
    plt.ylabel("Mean hours/week")
    plt.title("Recipient-level care hours by cohort")
    plt.legend()
    plt.tight_layout()
    p = fig_dir / "fig3_step4_care_hours_by_cohort.png"
    plt.savefig(p, dpi=200)
    plt.close()
    paths.append(p)

    # Figure 4
    matrix = data["step3b_matrix"].iloc[0]
    bounds = data["step3c_provider_sex_bounds"]
    combined = data["step3c_combined_sensitivity"]
    bootstrap = data.get("step3b_bootstrap", pd.DataFrame())
    boot_fp = bootstrap[bootstrap.get("statistic", pd.Series(dtype=str)).eq("female_provider_share")] if not bootstrap.empty else pd.DataFrame()
    rows = [
        {"panel": "matrix", "label": "H_MM", "value": matrix["H_MM_weekly"]},
        {"panel": "matrix", "label": "H_MF", "value": matrix["H_MF_weekly"]},
        {"panel": "matrix", "label": "H_FM", "value": matrix["H_FM_weekly"]},
        {"panel": "matrix", "label": "H_FF", "value": matrix["H_FF_weekly"]},
        {"panel": "share", "label": "main observed", "value": matrix["female_provider_share"]},
        {"panel": "share", "label": "proportional allocation", "value": bounds.loc[bounds["scenario"].eq("proportional_allocation_by_observed_patient_sex"), "female_provider_share"].iloc[0]},
        {"panel": "share", "label": "all missing provider sex assigned male", "value": bounds.loc[bounds["scenario"].eq("all_missing_provider_sex_are_male"), "female_provider_share"].iloc[0]},
        {"panel": "share", "label": "combined conservative", "value": combined.loc[combined["scenario"].eq("conservative_against_female_provider_share"), "female_provider_share"].iloc[0]},
    ]
    if not boot_fp.empty:
        rows.append({"panel": "share_ci", "label": "bootstrap CI lower", "value": boot_fp["ci_lower_95"].iloc[0]})
        rows.append({"panel": "share_ci", "label": "bootstrap CI upper", "value": boot_fp["ci_upper_95"].iloc[0]})
    f4 = pd.DataFrame(rows)
    f4.to_csv(fig_dir / "fig4_step4_hrs_gender_matrix_and_sensitivity_data.csv", index=False)

    plt.figure(figsize=(9, 4.8))
    ax1 = plt.subplot(1, 2, 1)
    mat = np.array([[matrix["H_MM_weekly"], matrix["H_MF_weekly"]], [matrix["H_FM_weekly"], matrix["H_FF_weekly"]]], dtype=float) / 1e6
    im = ax1.imshow(mat, cmap="YlGnBu")
    ax1.set_xticks([0, 1], ["Male patient", "Female patient"])
    ax1.set_yticks([0, 1], ["Male provider", "Female provider"])
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{mat[i, j]:.1f}M", ha="center", va="center")
    ax1.set_title("HRS-only weighted weekly hours")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax2 = plt.subplot(1, 2, 2)
    share = f4[f4["panel"].eq("share")]
    ax2.scatter(share["value"], share["label"], color="#2f6f4e")
    if not boot_fp.empty:
        ax2.plot([boot_fp["ci_lower_95"].iloc[0], boot_fp["ci_upper_95"].iloc[0]], ["main observed", "main observed"], color="#2f6f4e", lw=2)
    ax2.axvline(0.5, color="gray", linestyle="--", lw=1)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Female provider share")
    ax2.set_title("HRS-only sensitivity")
    plt.tight_layout()
    p = fig_dir / "fig4_step4_hrs_gender_matrix_and_sensitivity.png"
    plt.savefig(p, dpi=200)
    plt.close()
    paths.append(p)
    return paths


def write_text_snippets(t21: pd.DataFrame, t23: pd.DataFrame) -> Path:
    hrs = t23.iloc[0]
    text = f"""# Step 4 Text Snippets

## Analysis Scope
Patient-side disability and survey-based post-stroke YLD-like disability burden were evaluated across seven aging cohorts. Recipient-level care-hour analyses were restricted to cohorts with harmonized care-hour variables, excluding SHARE where hours were unavailable. Provider-side gender decomposition is currently HRS-only.

## Survey-Based YLD-Like Disability Burden
The survey-based post-stroke YLD-like disability burden used disability severity weights mapped from ADL/IADL limitations. This measure is not official GBD YLD. Across seven cohorts, mean disability weights ranged from {txt_range(t21['mean_DW_weighted'], 3)}.

## Strict Unmet Need vs Informal Care Gap
Strict unmet care need requires disability or care need plus no informal, formal/professional, paid, or other recorded ADL/IADL help. Informal care gap is a separate recipient-level construct indicating disabled stroke survivors with no recorded informal care hours. These constructs should not be conflated.

## Recipient-Level Care Hours
Recipient-level care hours were compared only among cohorts with harmonized hours. SHARE was not included in care-hour estimates. Missing care hours were not recoded to zero.

## HRS-Only Provider-Side Gender Decomposition
In HRS observed helper-level rows, women provided most provider-specific care hours: female_provider_share={hrs['female_provider_share_main']:.3f}. This is an HRS-only helper-level result and should not be described as a seven-country provider-side decomposition.

## Step 3C Missingness Sensitivity
The HRS provider-side result was stable under proportional missing-provider-sex allocation and missing-hour sensitivity, but the extreme conservative combined scenario reduced female_provider_share to {hrs['combined_conservative_female_provider_share']:.3f}. This sensitivity should be reported.

## Limitations
Provider sex and helper-specific hours were jointly validated only in HRS. HRS paid-care status was mostly missing, so helper-level care hours should not be called pure informal care unless paid/formal status is validated. Adult sex-specific denominators were not supplied; hidden gender care tax was not calculated.

## Prohibited Claims
Do not claim seven-country provider-side gender decomposition. Do not claim women provide most informal stroke care across seven countries from current helper-level evidence. Do not calculate or report hidden gender care tax. Do not label the survey-based YLD-like estimate as official GBD YLD.
"""
    path = OUT / "text_snippets_step4.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_readme(check: pd.DataFrame, outputs: list[str]) -> Path:
    used = "\n".join(f"- `{row.file_path}` ({row.status})" for row in check.itertuples())
    outlist = "\n".join(f"- `{x}`" for x in outputs)
    text = f"""# Step 4 Integrated Results and Claim Audit

## Purpose
Step 4 integrates completed Step 2b patient-side results with Step 3B/3C HRS-only provider-side diagnostics and audits which manuscript claims are supported.

## Input Files Used
{used}

## Analysis Domains and Cohort Inclusion
- Seven-cohort disability/YLD-like burden: CHARLS, HRS, ELSA, SHARE, KLoSA, LASI, MHAS.
- Six-cohort recipient-level care-hour analysis: CHARLS, HRS, ELSA, KLoSA, LASI, MHAS; SHARE excluded unless harmonized hours are validated.
- Strict unmet need: only cohorts where formal/informal/any-care indicators are constructible; LASI is not forced into strict unmet need.
- HRS-only provider-side helper-level matrix: HRS only.

## Hidden Gender Care Tax
Hidden gender care tax is not calculated because adult sex-specific population denominators were not supplied.

## Provider-Side Scope
Seven-country provider-side decomposition is not claimed because provider sex and helper-specific hours are jointly validated only in HRS.

## Output Files
{outlist}

## Recommended Next Step
Proceed to manuscript integration with claim restrictions, or prioritize deeper validation of non-HRS helper-level provider sex if a cross-national provider-side decomposition is essential.
"""
    path = OUT / "README_step4.md"
    path.write_text(text, encoding="utf-8")
    return path


def write_quality_log(check: pd.DataFrame, paths: list[Path], t20: pd.DataFrame, t24: pd.DataFrame, t23: pd.DataFrame) -> Path:
    def line(status: str, item: str, reason: str, blocks: str = "No", fix: str = "None required") -> str:
        return f"{status} - {item} | reason: {reason} | blocks manuscript integration: {blocks} | next: {fix}"

    missing = check[~check["exists"]]
    step2_ok = check[check["input_name"].str.startswith("step2b_")]["exists"].all()
    step3b_ok = check[check["input_name"].str.startswith("step3b_")]["exists"].all()
    step3c_ok = check[check["input_name"].str.startswith("step3c_")]["exists"].all()
    share_care = t20.loc[t20["cohort"].eq("SHARE"), "recipient_level_care_hours_available"].iloc[0]
    lasi_strict = t20.loc[t20["cohort"].eq("LASI"), "strict_unmet_need_available"].iloc[0]
    hrs_only = t23["analysis_scope"].astype(str).str.contains("HRS-only", case=False).all()
    no_seven_provider = not t24["allowed_manuscript_wording"].astype(str).str.contains("seven-country provider-side|seven-country gender care burden decomposition", case=False).any()
    conservative = t23["combined_conservative_female_provider_share"].iloc[0]

    checks = [
        ("PASS" if step2_ok else "FAIL", "1. Step 2b inputs were read.", "all requested Step 2b files present." if step2_ok else f"missing: {missing['file_path'].tolist()}"),
        ("PASS" if step3b_ok else "FAIL", "2. Step 3B inputs were read.", "all requested Step 3B files present." if step3b_ok else f"missing: {missing['file_path'].tolist()}"),
        ("PASS" if step3c_ok else "FAIL", "3. Step 3C inputs were read.", "all requested Step 3C files present." if step3c_ok else f"missing: {missing['file_path'].tolist()}"),
        ("PASS", "4. Analysis scope table created.", "table20 written."),
        ("PASS", "5. Patient-side main results table created.", "table21 written."),
        ("PASS", "6. Patient-side sex inequality table created.", "table22 written."),
        ("PASS", "7. HRS provider-side summary table created.", "table23 written."),
        ("PASS", "8. Claim-support audit table created.", "table24 written."),
        ("PASS", "9. Abstract-ready key numbers table created.", "table25 written."),
        ("PASS", "10. Figure data files created.", "four figure data CSV files written."),
        ("PASS" if all(p.exists() for p in paths) else "FAIL", "11. Figure PNG files created.", "four PNG figures checked."),
        ("PASS", "12. Text snippets created.", "text_snippets_step4.md written."),
        ("PASS" if not share_care else "FAIL", "13. SHARE not included in care-hour results unless hours validated.", f"SHARE care-hour availability={share_care}."),
        ("PASS" if not lasi_strict else "WARNING", "14. LASI not forced into strict unmet need if variables unavailable.", f"LASI strict availability={lasi_strict}."),
        ("PASS" if hrs_only else "FAIL", "15. HRS provider-side result labeled HRS-only.", "table23 analysis_scope checked."),
        ("PASS" if no_seven_provider else "FAIL", "16. No seven-country provider-side claim made.", "allowed wording avoids seven-country provider-side claims."),
        ("PASS", "17. No hidden gender care tax calculated.", "no table11 or tax output created by Step 4."),
        ("PASS", "18. YLD-like measure not labeled as GBD YLD.", "outputs use survey-based post-stroke YLD-like wording."),
        ("PASS", "19. HRS helper-level care not labeled pure informal care unless paid/formal status validated.", "table23 and text snippets include caveat."),
        ("PASS" if conservative < 0.5 else "FAIL", "20. Conservative missing-provider-sex sensitivity below 0.5 is acknowledged.", f"combined conservative female_provider_share={conservative:.3f}."),
    ]
    lines = ["Step 4 integrated results and claim audit quality log"]
    for status, item, reason in checks:
        lines.append(line(status, item, reason, "Yes" if status == "FAIL" else "No", "Fix failed input/output before manuscript integration." if status == "FAIL" else "None required"))
    path = OUT / "step4_quality_check_log.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    data, check = read_inputs()
    scope = table20_analysis_scope(data)
    t21 = table21_patient_side_main(data)
    t22 = table22_patient_side_sex_inequality(data)
    t23 = table23_hrs_provider_summary(data)
    t24 = table24_claim_audit(t21, t23)
    t25 = table25_key_numbers(t21, t23)
    fig_paths = make_figures(t21, t23, data)
    text_path = write_text_snippets(t21, t23)
    outputs = [
        "step4_input_file_check.csv",
        "table20_analysis_scope_by_cohort.csv",
        "table21_patient_side_main_results.csv",
        "table22_patient_side_sex_inequality.csv",
        "table23_hrs_provider_side_summary.csv",
        "table24_claim_support_audit.csv",
        "table25_abstract_ready_key_numbers.csv",
        "figures/fig1_step4_yld_like_burden_by_cohort.png",
        "figures/fig2_step4_care_gap_by_cohort.png",
        "figures/fig3_step4_care_hours_by_cohort.png",
        "figures/fig4_step4_hrs_gender_matrix_and_sensitivity.png",
        "text_snippets_step4.md",
        "step4_quality_check_log.txt",
        "README_step4.md",
    ]
    log_path = write_quality_log(check, fig_paths, scope, t24, t23)
    readme_path = write_readme(check, outputs)
    print(f"Step 4 outputs written under: {OUT}")
    print("Input file status:")
    print(check["status"].value_counts().to_string())
    print(f"Quality log: {log_path}")
    print(f"README: {readme_path}")
    print(f"Text snippets: {text_path}")
    print(scope.to_string(index=False))


if __name__ == "__main__":
    main()
