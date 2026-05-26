from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
STEP3B = ROOT / "results" / "step3b"
OUT = ROOT / "results" / "step3c"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

REQUIRED_INPUTS = {
    "helper_flags": STEP3B / "step3b_helper_long_with_flags.csv",
    "matrix": STEP3B / "table7_step3b_gender_care_matrix.csv",
    "shares": STEP3B / "table8_step3b_gender_care_shares.csv",
    "patient_level": STEP3B / "table10_step3b_patient_level_provider_hours.csv",
    "sensitivity": STEP3B / "supp_table_step3b_sensitivity_paid_and_cap.csv",
    "bootstrap": STEP3B / "supp_table_step3b_bootstrap_uncertainty.csv",
}


def yn_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def load_inputs() -> dict[str, pd.DataFrame]:
    missing = [str(path) for path in REQUIRED_INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Step 3B input(s): " + "; ".join(missing))
    return {name: pd.read_csv(path) for name, path in REQUIRED_INPUTS.items()}


def prep_helper(df: pd.DataFrame) -> pd.DataFrame:
    hrs = df[df["cohort"].astype(str).eq("HRS")].copy()
    hrs["eligible_main_matrix"] = yn_series(hrs["eligible_main_matrix"])
    for col in [
        "flag_provider_sex_missing",
        "flag_hours_week_missing",
        "flag_hours_week_above_168",
        "flag_hours_week_negative",
        "flag_patient_weight_nonpositive",
        "flag_patient_weight_missing",
        "flag_patient_sex_missing",
    ]:
        hrs[col] = yn_series(hrs[col])
    hrs["hours_week_num"] = pd.to_numeric(hrs["hours_week_num"], errors="coerce")
    hrs["patient_weight_num"] = pd.to_numeric(hrs["patient_weight_num"], errors="coerce")
    hrs["patient_age"] = pd.to_numeric(hrs["patient_age"], errors="coerce")
    hrs["patient_yld_like_disability_weight"] = pd.to_numeric(hrs["patient_yld_like_disability_weight"], errors="coerce")

    valid = hrs["eligible_main_matrix"]
    miss_provider = hrs["flag_provider_sex_missing"]
    miss_hours = hrs["flag_hours_week_missing"]
    conditions = [
        valid,
        ~valid & miss_provider & ~miss_hours,
        ~valid & ~miss_provider & miss_hours,
        ~valid & miss_provider & miss_hours,
    ]
    choices = [
        "valid_for_main_matrix",
        "missing_provider_sex_only",
        "missing_hours_only",
        "missing_both_provider_sex_and_hours",
    ]
    hrs["missingness_group"] = np.select(conditions, choices, default="invalid_other_reason")
    return hrs


def weighted_sum_weight(rows: pd.DataFrame) -> float:
    w = rows["patient_weight_num"]
    return float(w.where(w.gt(0)).sum())


def percent(series: pd.Series) -> float:
    denom = series.notna().sum()
    if denom == 0:
        return np.nan
    return float(series.sum() / denom * 100)


def table12_helper_row_missingness(hrs: pd.DataFrame) -> pd.DataFrame:
    total = len(hrs)
    groups = [
        "valid_for_main_matrix",
        "missing_provider_sex_only",
        "missing_hours_only",
        "missing_both_provider_sex_and_hours",
        "invalid_other_reason",
    ]
    rows = []
    for group in groups:
        sub = hrs[hrs["missingness_group"].eq(group)].copy()
        observed_hours = sub["hours_week_num"].dropna()
        sev = sub["patient_disability_severity"].astype(str)
        rows.append(
            {
                "cohort": "HRS",
                "missingness_group": group,
                "n_helper_rows": len(sub),
                "percent_helper_rows": len(sub) / total * 100 if total else np.nan,
                "n_unique_patients": sub["patient_id"].nunique(),
                "weighted_n_helper_rows": weighted_sum_weight(sub),
                "mean_patient_age": sub["patient_age"].mean(),
                "female_patient_percent": percent(sub["patient_sex"].astype(str).str.lower().eq("female").where(sub["patient_sex"].notna())),
                "male_patient_percent": percent(sub["patient_sex"].astype(str).str.lower().eq("male").where(sub["patient_sex"].notna())),
                "ADL_ge3_percent": percent(sev.str.upper().eq("ADL >=3").where(sub["patient_disability_severity"].notna())),
                "IADL_only_percent": percent(sev.str.upper().eq("IADL-ONLY").where(sub["patient_disability_severity"].notna())),
                "mean_yld_like_disability_weight": sub["patient_yld_like_disability_weight"].mean(),
                "mean_hours_week_observed": observed_hours.mean() if len(observed_hours) else np.nan,
                "median_hours_week_observed": observed_hours.median() if len(observed_hours) else np.nan,
                "p90_hours_week_observed": observed_hours.quantile(0.90) if len(observed_hours) else np.nan,
                "notes": "Provider sex and hours are not recoded; hours summaries use observed non-missing hours only.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table12_step3c_helper_row_missingness.csv", index=False)
    return out


def w_hours(rows: pd.DataFrame, hours_col: str = "hours_week_num") -> pd.Series:
    return pd.to_numeric(rows[hours_col], errors="coerce") * rows["patient_weight_num"]


def matrix_from_rows(rows: pd.DataFrame, hours_col: str = "hours_week_num") -> dict:
    d = rows.copy()
    d["_wh"] = w_hours(d, hours_col)
    def s(provider, patient):
        return float(d.loc[d["provider_sex_std"].eq(provider) & d["patient_sex_std"].eq(patient), "_wh"].sum())
    h_mm = s("Male", "Male")
    h_mf = s("Male", "Female")
    h_fm = s("Female", "Male")
    h_ff = s("Female", "Female")
    total = h_mm + h_mf + h_fm + h_ff
    female = h_fm + h_ff
    male = h_mm + h_mf
    return {
        "H_MM_weekly": h_mm,
        "H_MF_weekly": h_mf,
        "H_FM_weekly": h_fm,
        "H_FF_weekly": h_ff,
        "H_total_weekly": total,
        "female_provider_share": female / total if total else np.nan,
        "male_provider_share": male / total if total else np.nan,
        "cross_sex_care_share": (h_mf + h_fm) / total if total else np.nan,
        "same_sex_care_share": (h_mm + h_ff) / total if total else np.nan,
        "female_to_male_provider_ratio": female / male if male else np.nan,
    }


def main_valid(hrs: pd.DataFrame) -> pd.DataFrame:
    return hrs[hrs["eligible_main_matrix"]].copy()


def patient_summary(rows: pd.DataFrame, patient_ids: pd.Index | np.ndarray, group: str) -> dict:
    sub = rows[rows["patient_id"].isin(patient_ids)].copy()
    patient = sub.sort_values("patient_id").drop_duplicates("patient_id")
    obs_hours = sub.dropna(subset=["hours_week_num"]).groupby("patient_id")["hours_week_num"].sum()
    sev = patient["patient_disability_severity"].astype(str)
    w = patient["patient_weight_num"].where(patient["patient_weight_num"].gt(0))
    return {
        "cohort": "HRS",
        "patient_inclusion_group": group,
        "n_patients": patient["patient_id"].nunique(),
        "weighted_n_patients": float(w.sum()),
        "mean_age": patient["patient_age"].mean(),
        "female_patient_percent": percent(patient["patient_sex"].astype(str).str.lower().eq("female").where(patient["patient_sex"].notna())),
        "male_patient_percent": percent(patient["patient_sex"].astype(str).str.lower().eq("male").where(patient["patient_sex"].notna())),
        "ADL_ge3_percent": percent(sev.str.upper().eq("ADL >=3").where(patient["patient_disability_severity"].notna())),
        "IADL_only_percent": percent(sev.str.upper().eq("IADL-ONLY").where(patient["patient_disability_severity"].notna())),
        "no_disability_percent": percent(sev.str.upper().eq("NO ADL/IADL").where(patient["patient_disability_severity"].notna())),
        "mean_yld_like_disability_weight": patient["patient_yld_like_disability_weight"].mean(),
        "mean_total_observed_helper_hours_week": obs_hours.mean() if len(obs_hours) else np.nan,
        "median_total_observed_helper_hours_week": obs_hours.median() if len(obs_hours) else np.nan,
        "notes": "Patients may appear in multiple diagnostic groups; table assesses missingness and selection only.",
    }


def table13_patient_level_inclusion(hrs: pd.DataFrame) -> pd.DataFrame:
    valid_patients = hrs.loc[hrs["eligible_main_matrix"], "patient_id"].dropna().unique()
    all_patients = hrs["patient_id"].dropna().unique()
    none_valid = np.setdiff1d(all_patients, valid_patients)
    groups = [
        ("patient_has_at_least_one_valid_helper_row", valid_patients),
        ("patient_has_helper_rows_but_none_valid", none_valid),
        ("patient_has_missing_provider_sex_rows", hrs.loc[hrs["flag_provider_sex_missing"], "patient_id"].dropna().unique()),
        ("patient_has_missing_hours_rows", hrs.loc[hrs["flag_hours_week_missing"], "patient_id"].dropna().unique()),
        ("patient_has_hours_above_168_rows", hrs.loc[hrs["flag_hours_week_above_168"], "patient_id"].dropna().unique()),
    ]
    out = pd.DataFrame([patient_summary(hrs, ids, name) for name, ids in groups])
    out.to_csv(OUT / "table13_step3c_patient_level_inclusion_diagnostics.csv", index=False)
    return out


def row_group(hrs: pd.DataFrame) -> pd.Series:
    return np.select(
        [
            hrs["eligible_main_matrix"],
            ~hrs["eligible_main_matrix"] & hrs["flag_provider_sex_missing"] & ~hrs["flag_hours_week_missing"],
            ~hrs["eligible_main_matrix"] & ~hrs["flag_provider_sex_missing"] & hrs["flag_hours_week_missing"],
            ~hrs["eligible_main_matrix"] & hrs["flag_provider_sex_missing"] & hrs["flag_hours_week_missing"],
        ],
        ["included_main_matrix", "excluded_missing_provider_sex", "excluded_missing_hours", "excluded_missing_both"],
        default="excluded_other",
    )


def smd_cont(x: pd.Series, ref: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    ref = pd.to_numeric(ref, errors="coerce").dropna()
    if len(x) < 2 or len(ref) < 2:
        return np.nan
    pooled = np.sqrt((x.var(ddof=1) + ref.var(ddof=1)) / 2)
    return float((x.mean() - ref.mean()) / pooled) if pooled else np.nan


def std_diff_prop(p: float, pref: float) -> float:
    denom = np.sqrt((p * (1 - p) + pref * (1 - pref)) / 2)
    return float((p - pref) / denom) if denom else np.nan


def table14_balance(hrs: pd.DataFrame) -> pd.DataFrame:
    d = hrs.copy()
    d["balance_group"] = row_group(d)
    ref = d[d["balance_group"].eq("included_main_matrix")]
    rows = []
    cont_vars = ["patient_age", "patient_yld_like_disability_weight", "hours_week_num"]
    for var in cont_vars:
        for group, sub in d.groupby("balance_group"):
            rows.append({
                "cohort": "HRS", "variable": var, "group": group, "n": sub[var].notna().sum(),
                "mean_or_percent": pd.to_numeric(sub[var], errors="coerce").mean(),
                "standardized_difference_vs_included": 0 if group == "included_main_matrix" else smd_cont(sub[var], ref[var]),
                "notes": "Continuous variable; standardized mean difference. Hours use observed non-missing hours only." if var == "hours_week_num" else "Continuous variable; standardized mean difference.",
            })
    cat_specs = {
        "patient_sex": ["Female", "Male"],
        "patient_age_group": sorted(d["patient_age_group"].dropna().astype(str).unique()),
        "patient_disability_severity": sorted(d["patient_disability_severity"].dropna().astype(str).unique()),
        "provider_relationship_to_patient": sorted(d["provider_relationship_to_patient"].dropna().astype(str).unique())[:30],
        "paid_care": sorted(d["paid_care"].dropna().astype(str).unique()),
    }
    for var, levels in cat_specs.items():
        if d[var].notna().mean() < 0.05:
            rows.append({"cohort": "HRS", "variable": var, "group": "all", "n": int(d[var].notna().sum()), "mean_or_percent": np.nan, "standardized_difference_vs_included": np.nan, "notes": "Variable unavailable or mostly missing; comparison not forced."})
            continue
        for level in levels:
            ref_p = (ref[var].astype(str).eq(str(level))).mean() if len(ref) else np.nan
            for group, sub in d.groupby("balance_group"):
                p = (sub[var].astype(str).eq(str(level))).mean() if len(sub) else np.nan
                rows.append({
                    "cohort": "HRS", "variable": f"{var}={level}", "group": group, "n": len(sub),
                    "mean_or_percent": p * 100 if pd.notna(p) else np.nan,
                    "standardized_difference_vs_included": 0 if group == "included_main_matrix" else std_diff_prop(p, ref_p),
                    "notes": "Categorical variable; standardized difference in proportions.",
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table14_step3c_included_vs_excluded_balance.csv", index=False)
    return out


def add_row_to_cell(base: dict, provider: str, patient: str, weighted_hours: float) -> None:
    key = f"H_{provider[0]}{patient[0]}_weekly"
    base[key] += weighted_hours


def finalize_matrix_row(row: dict) -> dict:
    total = row["H_MM_weekly"] + row["H_MF_weekly"] + row["H_FM_weekly"] + row["H_FF_weekly"]
    female = row["H_FM_weekly"] + row["H_FF_weekly"]
    male = row["H_MM_weekly"] + row["H_MF_weekly"]
    row["H_total_weekly"] = total
    row["female_provider_share"] = female / total if total else np.nan
    row["male_provider_share"] = male / total if total else np.nan
    row["cross_sex_care_share"] = (row["H_MF_weekly"] + row["H_FM_weekly"]) / total if total else np.nan
    row["same_sex_care_share"] = (row["H_MM_weekly"] + row["H_FF_weekly"]) / total if total else np.nan
    row["female_to_male_provider_ratio"] = female / male if male else np.nan
    return row


def table15_provider_sex_bounds(hrs: pd.DataFrame) -> pd.DataFrame:
    valid = main_valid(hrs)
    base = matrix_from_rows(valid)
    candidate = hrs[
        hrs["flag_provider_sex_missing"] & hrs["hours_week_num"].notna() & hrs["patient_weight_num"].gt(0) & hrs["patient_sex_std"].isin(["Male", "Female"])
    ].copy()
    candidate["_wh"] = w_hours(candidate)
    valid["_wh"] = w_hours(valid)
    rows = []
    def start(name, notes):
        r = {"cohort": "HRS", "scenario": name, **base, "n_missing_provider_sex_rows_added": 0, "weighted_hours_added": 0.0, "notes": notes}
        return r
    rows.append(start("main_observed", "Observed Step 3B main matrix; no missing provider sex rows added."))
    for scenario, provider in [("all_missing_provider_sex_are_female", "Female"), ("all_missing_provider_sex_are_male", "Male")]:
        r = start(scenario, "Sensitivity bounds only; actual provider sex not imputed in dataset.")
        for _, x in candidate.iterrows():
            add_row_to_cell(r, provider, x["patient_sex_std"], x["_wh"])
        r["n_missing_provider_sex_rows_added"] = len(candidate)
        r["weighted_hours_added"] = float(candidate["_wh"].sum())
        rows.append(finalize_matrix_row(r))
    # proportional by patient sex
    r = start("proportional_allocation_by_observed_patient_sex", "Allocated missing-provider-sex observed-hour rows by observed provider distribution within patient sex.")
    for patient_sex, sub in candidate.groupby("patient_sex_std"):
        obs = valid[valid["patient_sex_std"].eq(patient_sex)]
        denom = obs["_wh"].sum()
        fshare = obs.loc[obs["provider_sex_std"].eq("Female"), "_wh"].sum() / denom if denom else valid.loc[valid["provider_sex_std"].eq("Female"), "_wh"].sum() / valid["_wh"].sum()
        for _, x in sub.iterrows():
            add_row_to_cell(r, "Female", patient_sex, x["_wh"] * fshare)
            add_row_to_cell(r, "Male", patient_sex, x["_wh"] * (1 - fshare))
    r["n_missing_provider_sex_rows_added"] = len(candidate)
    r["weighted_hours_added"] = float(candidate["_wh"].sum())
    rows.append(finalize_matrix_row(r))
    # overall distribution
    r = start("overall_observed_provider_distribution", "Allocated missing-provider-sex observed-hour rows by overall observed provider distribution.")
    overall_f = valid.loc[valid["provider_sex_std"].eq("Female"), "_wh"].sum() / valid["_wh"].sum()
    for _, x in candidate.iterrows():
        add_row_to_cell(r, "Female", x["patient_sex_std"], x["_wh"] * overall_f)
        add_row_to_cell(r, "Male", x["patient_sex_std"], x["_wh"] * (1 - overall_f))
    r["n_missing_provider_sex_rows_added"] = len(candidate)
    r["weighted_hours_added"] = float(candidate["_wh"].sum())
    rows.append(finalize_matrix_row(r))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table15_step3c_missing_provider_sex_bounds.csv", index=False)
    return out


def impute_hours_rows(hrs: pd.DataFrame, stat: str) -> tuple[pd.DataFrame, str]:
    valid = main_valid(hrs)
    miss = hrs[
        hrs["hours_week_num"].isna() & hrs["provider_sex_std"].isin(["Male", "Female"]) & hrs["patient_sex_std"].isin(["Male", "Female"]) & hrs["patient_weight_num"].gt(0)
    ].copy()
    obs = valid.copy()
    overall = getattr(obs["hours_week_num"], stat)() if stat in ["median", "mean"] else obs["hours_week_num"].quantile(0.75)
    notes = []
    vals = []
    for _, x in miss.iterrows():
        cell = obs[obs["provider_sex_std"].eq(x["provider_sex_std"]) & obs["patient_sex_std"].eq(x["patient_sex_std"])]["hours_week_num"].dropna()
        if len(cell):
            val = getattr(cell, stat)() if stat in ["median", "mean"] else cell.quantile(0.75)
        else:
            val = overall
            notes.append(f"Fallback used for {x['provider_sex_std']}->{x['patient_sex_std']}")
        vals.append(val)
    miss["hours_week_imputed"] = vals
    combined = pd.concat([valid.assign(hours_week_imputed=valid["hours_week_num"]), miss], ignore_index=True)
    return combined, "; ".join(sorted(set(notes))) if notes else "Cell-specific observed hours used."


def table16_missing_hours(hrs: pd.DataFrame) -> pd.DataFrame:
    valid = main_valid(hrs)
    miss_n = len(hrs[hrs["hours_week_num"].isna() & hrs["provider_sex_std"].isin(["Male", "Female"]) & hrs["patient_sex_std"].isin(["Male", "Female"]) & hrs["patient_weight_num"].gt(0)])
    rows = []
    base = {"cohort": "HRS", "scenario": "main_observed", **matrix_from_rows(valid), "n_missing_hours_rows_added": 0, "notes": "Observed Step 3B main matrix."}
    rows.append(base)
    zero = hrs[hrs["hours_week_num"].isna() & hrs["provider_sex_std"].isin(["Male", "Female"]) & hrs["patient_sex_std"].isin(["Male", "Female"]) & hrs["patient_weight_num"].gt(0)].copy()
    zero["hours_week_zero"] = 0
    zcomb = pd.concat([valid.assign(hours_week_zero=valid["hours_week_num"]), zero], ignore_index=True)
    rows.append({"cohort": "HRS", "scenario": "missing_hours_set_to_zero", **matrix_from_rows(zcomb, "hours_week_zero"), "n_missing_hours_rows_added": len(zero), "notes": "Lower-bound sensitivity only; missing hours not used in main analysis."})
    for scenario, stat in [
        ("missing_hours_assigned_median_by_provider_patient_cell", "median"),
        ("missing_hours_assigned_mean_by_provider_patient_cell", "mean"),
        ("missing_hours_assigned_p75_by_provider_patient_cell", "p75"),
    ]:
        comb, note = impute_hours_rows(hrs, stat)
        rows.append({"cohort": "HRS", "scenario": scenario, **matrix_from_rows(comb, "hours_week_imputed"), "n_missing_hours_rows_added": miss_n, "notes": f"Sensitivity only; imputed values are not true observed hours. {note}"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table16_step3c_missing_hours_sensitivity.csv", index=False)
    return out


def combined_scenario(hrs: pd.DataFrame, provider_mode: str, hour_stat: str) -> pd.DataFrame:
    valid = main_valid(hrs).copy()
    valid["hours_combined"] = valid["hours_week_num"]
    parts = [valid]
    # missing hours with observed provider sex
    hmiss, _ = impute_hours_rows(hrs, hour_stat)
    hmiss = hmiss[hmiss["eligible_main_matrix"].ne(True) & hmiss["hours_week_num"].isna()].copy()
    if not hmiss.empty:
        hmiss["hours_combined"] = hmiss["hours_week_imputed"]
        parts.append(hmiss)
    # missing provider sex observed hours
    pmiss = hrs[
        hrs["flag_provider_sex_missing"] & hrs["hours_week_num"].notna() & hrs["patient_weight_num"].gt(0) & hrs["patient_sex_std"].isin(["Male", "Female"])
    ].copy()
    if provider_mode in ["Female", "Male"]:
        pmiss["provider_sex_std"] = provider_mode
        pmiss["hours_combined"] = pmiss["hours_week_num"]
        parts.append(pmiss)
    else:
        obs = valid.copy()
        obs["_wh"] = w_hours(obs)
        overall_f = obs.loc[obs["provider_sex_std"].eq("Female"), "_wh"].sum() / obs["_wh"].sum()
        f = pmiss.copy(); f["provider_sex_std"] = "Female"; f["hours_combined"] = f["hours_week_num"] * overall_f
        m = pmiss.copy(); m["provider_sex_std"] = "Male"; m["hours_combined"] = m["hours_week_num"] * (1 - overall_f)
        parts.extend([f, m])
    return pd.concat(parts, ignore_index=True)


def table17_combined(hrs: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("main_observed", None, None, "Original Step 3B main matrix."),
        ("conservative_against_female_provider_share", "Male", "median", "Missing provider sex assigned male; missing hours assigned median. female_provider_share_gt_0_5={flag}"),
        ("favorable_to_female_provider_share", "Female", "median", "Missing provider sex assigned female; missing hours assigned median. female_provider_share_gt_0_5={flag}"),
        ("proportional_provider_sex_plus_median_hours", "proportional", "median", "Missing provider sex allocated by observed distribution; missing hours median. female_provider_share_gt_0_5={flag}"),
        ("proportional_provider_sex_plus_mean_hours", "proportional", "mean", "Missing provider sex allocated by observed distribution; missing hours mean. female_provider_share_gt_0_5={flag}"),
    ]
    rows = []
    for name, pmode, hstat, interp in specs:
        data = main_valid(hrs) if name == "main_observed" else combined_scenario(hrs, pmode, hstat)
        row = {"cohort": "HRS", "scenario": name, **matrix_from_rows(data, "hours_combined" if "hours_combined" in data else "hours_week_num")}
        row["scenario_interpretation"] = interp.format(flag=row["female_provider_share"] > 0.5)
        row["notes"] = "Sensitivity scenario only; does not alter main matrix."
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table17_step3c_combined_missingness_sensitivity.csv", index=False)
    return out


def table18_paid(hrs: pd.DataFrame) -> pd.DataFrame:
    valid = main_valid(hrs).copy()
    total_rows = len(valid)
    total_wh = w_hours(valid).sum()
    groups = [
        ("paid_care == 1", pd.to_numeric(valid["paid_care"], errors="coerce").eq(1)),
        ("paid_care == 0", pd.to_numeric(valid["paid_care"], errors="coerce").eq(0)),
        ("paid_care missing", valid["paid_care"].isna()),
    ]
    rows = []
    for name, mask in groups:
        sub = valid[mask].copy()
        mat = matrix_from_rows(sub) if len(sub) else {"female_provider_share": np.nan, "male_provider_share": np.nan}
        wh = w_hours(sub).sum()
        rows.append({
            "cohort": "HRS", "group": name, "n_rows": len(sub), "percent_rows": len(sub)/total_rows*100 if total_rows else np.nan,
            "weighted_hours_week": wh, "percent_weighted_hours": wh/total_wh*100 if total_wh else np.nan,
            "female_provider_share": mat["female_provider_share"], "male_provider_share": mat["male_provider_share"],
            "notes": "Missing paid_care is not treated as unpaid. Main Step 3B results are helper-level care hours, not pure informal care unless formal/paid status is fully validated.",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table18_step3c_paid_care_missingness.csv", index=False)
    return out


def table19_robustness_summary(
    table15: pd.DataFrame,
    table16: pd.DataFrame,
    table17: pd.DataFrame,
    table18: pd.DataFrame,
    sensitivity: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def classify(vals: pd.Series) -> str:
        vals = pd.to_numeric(vals, errors="coerce").dropna()
        if vals.empty:
            return "cannot_assess"
        if vals.min() > 0.5:
            return "robust"
        if vals.max() > 0.5:
            return "somewhat_sensitive"
        return "highly_sensitive"

    def add(domain, analysis, estimates, ratio_vals=None, recommendation=None):
        vals = pd.to_numeric(estimates, errors="coerce").dropna()
        main = float(vals.iloc[0]) if len(vals) else np.nan
        conclusion = classify(vals)
        if recommendation is None:
            recommendation = (
                "Use as main HRS provider-side result with missingness sensitivity in supplement."
                if conclusion == "robust"
                else "Use cautiously; emphasize sensitivity to missing helper-level data."
            )
        rows.append(
            {
                "domain": domain,
                "analysis": analysis,
                "main_estimate": main,
                "sensitivity_min": vals.min() if len(vals) else np.nan,
                "sensitivity_max": vals.max() if len(vals) else np.nan,
                "conclusion": conclusion,
                "manuscript_use_recommendation": recommendation,
            }
        )
        if ratio_vals is not None:
            r = pd.to_numeric(ratio_vals, errors="coerce").dropna()
            rows.append(
                {
                    "domain": domain,
                    "analysis": analysis + "_female_to_male_provider_ratio",
                    "main_estimate": r.iloc[0] if len(r) else np.nan,
                    "sensitivity_min": r.min() if len(r) else np.nan,
                    "sensitivity_max": r.max() if len(r) else np.nan,
                    "conclusion": "cannot_assess" if r.empty else "robust" if r.min() > 1 else "somewhat_sensitive",
                    "manuscript_use_recommendation": recommendation,
                }
            )

    add("provider_sex_missingness", "female_provider_share", table15["female_provider_share"], table15["female_to_male_provider_ratio"])
    add("hours_week_missingness", "female_provider_share", table16["female_provider_share"], table16["female_to_male_provider_ratio"])
    add("combined_missingness", "female_provider_share", table17["female_provider_share"], table17["female_to_male_provider_ratio"])
    add(
        "paid_care_missingness",
        "female_provider_share",
        table18["female_provider_share"],
        None,
        "Use cautiously; emphasize helper-level care hours rather than informal care because paid_care is incompletely observed.",
    )
    add("hours_above_168", "female_provider_share", sensitivity["female_provider_share"], sensitivity["female_to_male_provider_ratio"])
    if not bootstrap.empty:
        fp = bootstrap[bootstrap["statistic"].eq("female_provider_share")]
        vals = pd.Series([fp["ci_lower_95"].iloc[0], fp["estimate"].iloc[0], fp["ci_upper_95"].iloc[0]]) if not fp.empty else pd.Series(dtype=float)
        add("bootstrap_uncertainty", "female_provider_share_ci", vals)
    else:
        rows.append(
            {
                "domain": "bootstrap_uncertainty",
                "analysis": "female_provider_share_ci",
                "main_estimate": np.nan,
                "sensitivity_min": np.nan,
                "sensitivity_max": np.nan,
                "conclusion": "cannot_assess",
                "manuscript_use_recommendation": "Bootstrap file unavailable.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table19_step3c_robustness_summary.csv", index=False)
    return out


def make_figures(
    table12: pd.DataFrame,
    table15: pd.DataFrame,
    table16: pd.DataFrame,
    table17: pd.DataFrame,
    table18: pd.DataFrame,
    matrix: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    # Figure-ready provider share data
    rows = []
    for source, df, label_col in [
        ("provider_sex_bounds", table15, "scenario"),
        ("missing_hours", table16, "scenario"),
        ("combined", table17, "scenario"),
        ("paid_care", table18, "group"),
    ]:
        for _, r in df.iterrows():
            rows.append(
                {
                    "source": source,
                    "scenario": r[label_col],
                    "female_provider_share": r.get("female_provider_share", np.nan),
                    "ci_lower_95": np.nan,
                    "ci_upper_95": np.nan,
                }
            )
    if not bootstrap.empty:
        b = bootstrap[bootstrap["statistic"].eq("female_provider_share")]
        if not b.empty:
            rows.append(
                {
                    "source": "bootstrap",
                    "scenario": "bootstrap_95_CI",
                    "female_provider_share": b["estimate"].iloc[0],
                    "ci_lower_95": b["ci_lower_95"].iloc[0],
                    "ci_upper_95": b["ci_upper_95"].iloc[0],
                }
            )
    figdata = pd.DataFrame(rows)
    figdata.to_csv(OUT / "figures" / "fig_step3c_provider_share_sensitivity_data.csv", index=False)
    table12.to_csv(OUT / "figures" / "fig_step3c_missingness_flow_data.csv", index=False)

    m = matrix.iloc[0]
    heat = pd.DataFrame(
        [
            {"provider_sex": "Male", "patient_sex": "Male", "weighted_hours_week": m["H_MM_weekly"]},
            {"provider_sex": "Male", "patient_sex": "Female", "weighted_hours_week": m["H_MF_weekly"]},
            {"provider_sex": "Female", "patient_sex": "Male", "weighted_hours_week": m["H_FM_weekly"]},
            {"provider_sex": "Female", "patient_sex": "Female", "weighted_hours_week": m["H_FF_weekly"]},
        ]
    )
    heat.to_csv(OUT / "figures" / "fig_step3c_gender_matrix_heatmap_HRS_data.csv", index=False)

    # Figure 1
    plt.figure(figsize=(8, 4.8))
    labels = table12["missingness_group"].str.replace("_", " ")
    plt.bar(labels, table12["n_helper_rows"], color="#4c78a8")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Helper rows")
    plt.title("HRS-only helper row missingness flow")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(OUT / "figures" / f"fig_step3c_missingness_flow.{ext}", dpi=200)
    plt.close()

    # Figure 2
    plotdf = figdata.dropna(subset=["female_provider_share"]).copy()
    plotdf["label"] = plotdf["source"] + ": " + plotdf["scenario"].astype(str)
    plt.figure(figsize=(9, max(5, 0.28 * len(plotdf))))
    y = np.arange(len(plotdf))
    x = plotdf["female_provider_share"].astype(float)
    plt.scatter(x, y, color="#2f6f4e", zorder=3)
    has_ci = plotdf["ci_lower_95"].notna() & plotdf["ci_upper_95"].notna()
    for i, row in plotdf[has_ci].iterrows():
        yi = plotdf.index.get_loc(i)
        plt.plot([row["ci_lower_95"], row["ci_upper_95"]], [yi, yi], color="#2f6f4e", lw=2)
    plt.axvline(0.5, color="gray", linestyle="--", lw=1)
    plt.yticks(y, plotdf["label"], fontsize=8)
    plt.xlabel("Female provider share")
    plt.title("HRS-only provider share sensitivity")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(OUT / "figures" / f"fig_step3c_provider_share_sensitivity.{ext}", dpi=200)
    plt.close()

    # Figure 3
    mat = np.array([[m["H_MM_weekly"], m["H_MF_weekly"]], [m["H_FM_weekly"], m["H_FF_weekly"]]], dtype=float) / 1e6
    plt.figure(figsize=(5.6, 4.6))
    im = plt.imshow(mat, cmap="YlGnBu")
    plt.xticks([0, 1], ["Male patient", "Female patient"])
    plt.yticks([0, 1], ["Male provider", "Female provider"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{mat[i, j]:.1f}M", ha="center", va="center", color="black")
    plt.colorbar(im, label="Weighted weekly hours, millions")
    plt.title("HRS-only provider by patient sex care-hour matrix")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(OUT / "figures" / f"fig_step3c_gender_matrix_heatmap_HRS.{ext}", dpi=200)
    plt.close()
    return figdata


def write_log(
    hrs: pd.DataFrame,
    table12: pd.DataFrame,
    table13: pd.DataFrame,
    table14: pd.DataFrame,
    table15: pd.DataFrame,
    table16: pd.DataFrame,
    table17: pd.DataFrame,
    table18: pd.DataFrame,
    table19: pd.DataFrame,
    figdata: pd.DataFrame,
) -> None:
    def status_line(status: str, check: str, reason: str, blocks: str = "No", fix: str = "None required") -> str:
        return f"{status} - {check} | reason: {reason} | blocks Step 4: {blocks} | next: {fix}"

    conservative = table17[table17["scenario"].eq("conservative_against_female_provider_share")]
    conservative_share = (
        float(conservative["female_provider_share"].iloc[0])
        if not conservative.empty and pd.notna(conservative["female_provider_share"].iloc[0])
        else np.nan
    )
    robust_status = "PASS" if pd.notna(conservative_share) and conservative_share > 0.5 else "WARNING"
    robust_reason = (
        f"conservative combined scenario female_provider_share={conservative_share:.3f}."
        if pd.notna(conservative_share)
        else "conservative combined scenario could not be evaluated."
    )

    figure_paths = [
        FIG / "fig_step3c_missingness_flow.png",
        FIG / "fig_step3c_provider_share_sensitivity.png",
        FIG / "fig_step3c_gender_matrix_heatmap_HRS.png",
    ]
    lines = [
        "Step 3C HRS helper-level missingness diagnostics quality log",
        status_line("PASS", "1. Step 3B helper flag file was read.", "required Step 3B inputs were found and loaded."),
        status_line("PASS", "2. HRS-only restriction applied.", f"analysis dataset contains cohort values: {sorted(hrs['cohort'].dropna().unique().tolist())}."),
        status_line("PASS", "3. Missing provider sex was not recoded in the main analysis.", "main matrix uses observed valid provider_sex_std only."),
        status_line("PASS", "4. Missing hours were not recoded in the main analysis.", "main matrix uses observed non-missing hours_week_num only."),
        status_line("PASS", "5. Missingness groups were created.", f"{len(table12)} helper-row groups written to table12."),
        status_line("PASS", "6. Patient-level inclusion diagnostics were created.", f"{len(table13)} patient diagnostic groups written to table13."),
        status_line("PASS", "7. Included vs excluded balance table was created.", f"{len(table14)} balance rows written to table14."),
        status_line("PASS", "8. Missing provider sex bound sensitivity was created.", f"{len(table15)} scenarios written to table15."),
        status_line("PASS", "9. Missing hours sensitivity was created.", f"{len(table16)} scenarios written to table16."),
        status_line("PASS", "10. Combined missingness sensitivity was created.", f"{len(table17)} scenarios written to table17."),
        status_line("PASS", "11. Paid-care missingness table was created.", f"{len(table18)} paid-care groups written to table18."),
        status_line("PASS", "12. Robustness summary was created.", f"{len(table19)} rows written to table19."),
        status_line("PASS" if all(p.exists() for p in figure_paths) else "FAIL", "13. Figures were created.", "PNG figure files were checked.", "Yes" if not all(p.exists() for p in figure_paths) else "No", "Re-run figure generation if any PNG is missing."),
        status_line(robust_status, "14. Main conclusion remains or does not remain robust under conservative scenarios.", robust_reason, "No", "Use cautious supplement language if sensitivity is below 0.5."),
        status_line("PASS", "15. No seven-country provider-side claim was made.", "all Step 3C outputs are labelled HRS-only."),
        status_line("PASS", "16. Hidden gender care tax was not calculated.", "no adult sex-specific denominator was supplied or used."),
        "",
        f"HRS total helper rows: {len(hrs)}",
        f"HRS valid main-matrix rows: {int(hrs['eligible_main_matrix'].sum())}",
        f"Provider share sensitivity figure rows: {len(figdata)}",
    ]
    for _, row in table12.iterrows():
        lines.append(f"{row['missingness_group']}: n={row['n_helper_rows']}, percent={row['percent_helper_rows']:.2f}")
    (OUT / "step3c_quality_check_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(hrs: pd.DataFrame, table12: pd.DataFrame, table17: pd.DataFrame) -> None:
    valid_n = int(hrs["eligible_main_matrix"].sum())
    total_n = len(hrs)
    missing_provider = int(hrs["provider_sex_std"].isna().sum())
    missing_hours = int(hrs["hours_week_num"].isna().sum())
    conservative = table17[table17["scenario"].eq("conservative_against_female_provider_share")]
    conservative_share = conservative["female_provider_share"].iloc[0] if not conservative.empty else np.nan
    recommendation = (
        "The HRS Step 3B provider-side result can be used as an HRS-only manuscript result with missingness sensitivity in the supplement."
        if pd.notna(conservative_share) and conservative_share > 0.5
        else "Use the HRS Step 3B provider-side result cautiously; conservative missingness sensitivity weakens the directional conclusion."
    )
    outputs = [
        "table12_step3c_helper_row_missingness.csv",
        "table13_step3c_patient_level_inclusion_diagnostics.csv",
        "table14_step3c_included_vs_excluded_balance.csv",
        "table15_step3c_missing_provider_sex_bounds.csv",
        "table16_step3c_missing_hours_sensitivity.csv",
        "table17_step3c_combined_missingness_sensitivity.csv",
        "table18_step3c_paid_care_missingness.csv",
        "table19_step3c_robustness_summary.csv",
        "figures/fig_step3c_missingness_flow.png",
        "figures/fig_step3c_provider_share_sensitivity.png",
        "figures/fig_step3c_gender_matrix_heatmap_HRS.png",
        "step3c_quality_check_log.txt",
        "README_step3c.md",
    ]
    text = f"""# Step 3C HRS Helper Missingness Diagnostics

## Purpose
Step 3C evaluates whether the HRS-only provider-side gender care matrix from Step 3B is robust to helper-level missingness and whether excluded helper rows or patients differ from included rows.

## Scope
All analyses are restricted to HRS. Non-HRS cohorts are not included because Step 3A/3B did not jointly validate provider sex and helper-specific hours for those cohorts.

## Major Missingness
- Total HRS helper rows: {total_n}
- Valid helper rows for the Step 3B main matrix: {valid_n} ({valid_n / total_n * 100:.2f}%)
- Rows missing provider sex: {missing_provider} ({missing_provider / total_n * 100:.2f}%)
- Rows missing hours_week: {missing_hours} ({missing_hours / total_n * 100:.2f}%)

## Sensitivity Scenarios
The sensitivity tables examine missing provider sex bounds, missing hours assignments, combined missingness assumptions, paid-care missingness, hours above 168 per week, and bootstrap uncertainty from Step 3B.

These sensitivity analyses do not replace the main observed Step 3B matrix. Provider sex and hours are not imputed in the main analysis.

## Recommendation
{recommendation}

Step 3C still does not calculate a hidden gender care tax. Adult sex-specific denominator files would be required for that later step.

## Output Files
""" + "\n".join(f"- `{name}`" for name in outputs) + "\n"
    (OUT / "README_step3c.md").write_text(text, encoding="utf-8")


def create_zip() -> Path:
    zip_path = OUT / "step3c_outputs.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in OUT.rglob("*"):
            if path.is_file() and path != zip_path:
                zf.write(path, path.relative_to(OUT))
        script_path = ROOT / "scripts" / "step3c_hrs_helper_missingness_diagnostics.py"
        if script_path.exists():
            zf.write(script_path, Path("scripts") / script_path.name)
    return zip_path


def main() -> None:
    inputs = load_inputs()
    hrs = prep_helper(inputs["helper_flags"])
    table12 = table12_helper_row_missingness(hrs)
    table13 = table13_patient_level_inclusion(hrs)
    table14 = table14_balance(hrs)
    table15 = table15_provider_sex_bounds(hrs)
    table16 = table16_missing_hours(hrs)
    table17 = table17_combined(hrs)
    table18 = table18_paid(hrs)
    table19 = table19_robustness_summary(table15, table16, table17, table18, inputs["sensitivity"], inputs["bootstrap"])
    figdata = make_figures(table12, table15, table16, table17, table18, inputs["matrix"], inputs["bootstrap"])
    write_log(hrs, table12, table13, table14, table15, table16, table17, table18, table19, figdata)
    write_readme(hrs, table12, table17)
    zip_path = create_zip()
    print(f"Step 3C outputs written under: {OUT}")
    print(f"Step 3C zip written to: {zip_path}")
    print(table12.to_string(index=False))


if __name__ == "__main__":
    main()
