from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STEP3A = ROOT / "results" / "step3a"
OUT = ROOT / "results" / "step3b"
OUT.mkdir(parents=True, exist_ok=True)

INVENTORY = STEP3A / "step3a_helper_variable_inventory.csv"
HELPER_LONG = STEP3A / "step3_helper_long_format.csv"
UNAVAILABLE = STEP3A / "step3_helper_unavailable_report.csv"

SCOPE_NOTE = "helper-level care hours; not pure informal care unless formal/paid status is validated"


def sex_std(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().lower()
    if text in {"male", "m", "1", "1.0"}:
        return "Male"
    if text in {"female", "f", "2", "2.0"}:
        return "Female"
    return pd.NA


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den and pd.notna(den) and den != 0 else np.nan


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [p for p in [INVENTORY, HELPER_LONG, UNAVAILABLE] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Step 3A input(s): " + ", ".join(str(p) for p in missing))
    inventory = pd.read_csv(INVENTORY)
    helper = pd.read_csv(HELPER_LONG)
    unavailable = pd.read_csv(UNAVAILABLE)
    return inventory, helper, unavailable


def validate_inventory(inventory: pd.DataFrame, helper: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eligible = inventory[inventory["can_build_gender_care_matrix"].astype(str).str.lower().eq("yes")]
    for _, inv in eligible.iterrows():
        cohort = inv["cohort"]
        sub = helper[helper["cohort"].astype(str).eq(str(cohort))].copy()
        required = ["provider_sex", "hours_week", "patient_sex", "patient_id", "patient_survey_weight"]
        has_cols = {col: col in sub.columns for col in required}
        if sub.empty:
            status = "ineligible"
            reason = "No helper-level rows found in Step 3A long-format file."
        elif not all(has_cols.values()):
            status = "ineligible"
            reason = "Missing required column(s): " + ", ".join(k for k, ok in has_cols.items() if not ok)
        else:
            provider_ok = sub["provider_sex"].map(sex_std).notna().any()
            hours_ok = pd.to_numeric(sub["hours_week"], errors="coerce").notna().any()
            patient_sex_ok = sub["patient_sex"].map(sex_std).notna().any()
            patient_id_ok = sub["patient_id"].notna().any()
            weight_ok = pd.to_numeric(sub["patient_survey_weight"], errors="coerce").gt(0).any()
            if cohort != "HRS":
                status = "ineligible"
                reason = "Only HRS is allowed for Step 3B main matrix; non-HRS candidate requires manual validation."
            elif provider_ok and hours_ok and patient_sex_ok and patient_id_ok and weight_ok:
                status = "eligible"
                reason = "Validated HRS helper-level rows include provider sex, helper-specific hours, patient sex, patient_id, and positive patient weights."
            else:
                status = "ineligible"
                reason = "Required data not present after validation."
        rows.append(
            {
                "cohort": cohort,
                "inventory_can_build_gender_care_matrix": inv["can_build_gender_care_matrix"],
                "n_helper_rows_in_long_file": len(sub),
                "has_provider_sex_column": has_cols.get("provider_sex", False),
                "has_hours_week_column": has_cols.get("hours_week", False),
                "has_patient_sex_column": has_cols.get("patient_sex", False),
                "has_patient_id_column": has_cols.get("patient_id", False),
                "has_patient_survey_weight_column": has_cols.get("patient_survey_weight", False),
                "step3b_validation_status": status,
                "validation_note": reason,
            }
        )
    if not rows:
        rows.append(
            {
                "cohort": "",
                "inventory_can_build_gender_care_matrix": "",
                "n_helper_rows_in_long_file": 0,
                "has_provider_sex_column": False,
                "has_hours_week_column": False,
                "has_patient_sex_column": False,
                "has_patient_id_column": False,
                "has_patient_survey_weight_column": False,
                "step3b_validation_status": "none_eligible",
                "validation_note": "No cohort marked can_build_gender_care_matrix == yes in Step 3A inventory.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "step3b_cohort_eligibility_validation.csv", index=False)
    return out


def standardize_rows(helper: pd.DataFrame) -> pd.DataFrame:
    df = helper.copy()
    df["patient_sex_std"] = df["patient_sex"].map(sex_std)
    df["provider_sex_std"] = df["provider_sex"].map(sex_std)
    df["hours_week_num"] = pd.to_numeric(df["hours_week"], errors="coerce")
    df["patient_weight_num"] = pd.to_numeric(df["patient_survey_weight"], errors="coerce")
    df["hours_year"] = df["hours_week_num"] * 52
    df["weighted_hours_week"] = df["hours_week_num"] * df["patient_weight_num"]
    df["weighted_hours_year"] = df["weighted_hours_week"] * 52

    df["flag_hours_week_above_168"] = df["hours_week_num"].gt(168)
    df["flag_hours_week_negative"] = df["hours_week_num"].lt(0)
    df["flag_patient_weight_nonpositive"] = df["patient_weight_num"].le(0)
    df["flag_patient_weight_missing"] = df["patient_weight_num"].isna()
    df["flag_provider_sex_missing"] = df["provider_sex_std"].isna()
    df["flag_patient_sex_missing"] = df["patient_sex_std"].isna()
    df["flag_hours_week_missing"] = df["hours_week_num"].isna()
    df["flag_patient_id_missing"] = df["patient_id"].isna()

    df["eligible_main_matrix"] = (
        df["cohort"].astype(str).eq("HRS")
        & df["patient_id"].notna()
        & df["patient_sex_std"].isin(["Male", "Female"])
        & df["provider_sex_std"].isin(["Male", "Female"])
        & df["hours_week_num"].notna()
        & df["hours_week_num"].ge(0)
        & df["patient_weight_num"].notna()
        & df["patient_weight_num"].gt(0)
    )

    reasons = []
    for _, row in df.iterrows():
        r = []
        if str(row.get("cohort")) != "HRS":
            r.append("not_hrs")
        if pd.isna(row.get("patient_id")):
            r.append("missing_patient_id")
        if pd.isna(row.get("patient_sex_std")):
            r.append("missing_or_invalid_patient_sex")
        if pd.isna(row.get("provider_sex_std")):
            r.append("missing_or_invalid_provider_sex")
        if pd.isna(row.get("hours_week_num")):
            r.append("missing_hours_week")
        elif row.get("hours_week_num") < 0:
            r.append("negative_hours_week")
        if pd.isna(row.get("patient_weight_num")):
            r.append("missing_patient_weight")
        elif row.get("patient_weight_num") <= 0:
            r.append("nonpositive_patient_weight")
        reasons.append(";".join(r))
    df["main_matrix_exclusion_reasons"] = reasons
    df.to_csv(OUT / "step3b_helper_long_with_flags.csv", index=False)
    return df


def exclusion_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, inv in pd.read_csv(INVENTORY).iterrows():
        cohort = inv["cohort"]
        sub = df[df["cohort"].astype(str).eq(str(cohort))].copy()
        valid = sub["eligible_main_matrix"] if "eligible_main_matrix" in sub else pd.Series(False, index=sub.index)
        notes = ""
        if cohort == "HRS":
            notes = "HRS is eligible for the main matrix after row-level validation."
        elif len(sub) == 0:
            notes = "No validated helper-level long-format rows; excluded from main gender care matrix."
        else:
            notes = "Excluded from main gender care matrix because provider sex and helper-specific hours are not jointly validated in an HRS-like helper-recipient structure."
        rows.append(
            {
                "cohort": cohort,
                "eligible_from_inventory": str(inv["can_build_gender_care_matrix"]).lower() == "yes",
                "can_build_gender_care_matrix_inventory": inv["can_build_gender_care_matrix"],
                "n_rows_in_helper_long": len(sub),
                "n_missing_patient_sex": int(sub["flag_patient_sex_missing"].sum()) if len(sub) else 0,
                "n_missing_provider_sex": int(sub["flag_provider_sex_missing"].sum()) if len(sub) else 0,
                "n_missing_hours_week": int(sub["flag_hours_week_missing"].sum()) if len(sub) else 0,
                "n_negative_hours_week": int(sub["flag_hours_week_negative"].sum()) if len(sub) else 0,
                "n_hours_above_168": int(sub["flag_hours_week_above_168"].sum()) if len(sub) else 0,
                "n_missing_or_nonpositive_weight": int((sub["flag_patient_weight_missing"] | sub["flag_patient_weight_nonpositive"]).sum()) if len(sub) else 0,
                "n_valid_for_main_matrix": int(valid.sum()) if len(sub) else 0,
                "n_unique_patients_valid": sub.loc[valid, "patient_id"].nunique() if len(sub) else 0,
                "main_limitation": inv.get("main_limitation", ""),
                "notes": notes,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table9_step3b_exclusion_diagnostics.csv", index=False)
    return out


def matrix_from_rows(main: pd.DataFrame, analysis: str, notes: str = "") -> dict:
    main = main.copy()
    if "weighted_hours_week_sens" not in main:
        main["weighted_hours_week_sens"] = main["weighted_hours_week"]

    def sum_hours(provider: str, patient: str, col: str) -> float:
        return float(main.loc[
            main["provider_sex_std"].eq(provider) & main["patient_sex_std"].eq(patient),
            col,
        ].sum())

    h_mm_w = sum_hours("Male", "Male", "weighted_hours_week_sens")
    h_mf_w = sum_hours("Male", "Female", "weighted_hours_week_sens")
    h_fm_w = sum_hours("Female", "Male", "weighted_hours_week_sens")
    h_ff_w = sum_hours("Female", "Female", "weighted_hours_week_sens")
    total_w = h_mm_w + h_mf_w + h_fm_w + h_ff_w

    h_mm_y = h_mm_w * 52
    h_mf_y = h_mf_w * 52
    h_fm_y = h_fm_w * 52
    h_ff_y = h_ff_w * 52
    total_y = total_w * 52

    female_provider_weekly = h_fm_w + h_ff_w
    male_provider_weekly = h_mm_w + h_mf_w
    female_provider_annual = female_provider_weekly * 52
    male_provider_annual = male_provider_weekly * 52

    row = {
        "cohort": "HRS",
        "analysis": analysis,
        "n_patient_helper_pairs": len(main),
        "n_unique_stroke_survivors_with_helper_data": main["patient_id"].nunique(),
        "n_unique_helpers_or_helper_codes": main["helper_id"].nunique(),
        "n_rows_hours_above_168": int(main["flag_hours_week_above_168"].sum()),
        "H_MM_weekly": h_mm_w,
        "H_MF_weekly": h_mf_w,
        "H_FM_weekly": h_fm_w,
        "H_FF_weekly": h_ff_w,
        "H_total_weekly": total_w,
        "H_MM_annual": h_mm_y,
        "H_MF_annual": h_mf_y,
        "H_FM_annual": h_fm_y,
        "H_FF_annual": h_ff_y,
        "H_total_annual": total_y,
        "female_provider_weekly": female_provider_weekly,
        "male_provider_weekly": male_provider_weekly,
        "female_provider_annual": female_provider_annual,
        "male_provider_annual": male_provider_annual,
        "female_provider_share": safe_div(female_provider_weekly, total_w),
        "male_provider_share": safe_div(male_provider_weekly, total_w),
        "female_patient_share": safe_div(h_mf_w + h_ff_w, total_w),
        "male_patient_share": safe_div(h_mm_w + h_fm_w, total_w),
        "female_to_male_patient_share": safe_div(h_fm_w, total_w),
        "female_to_female_patient_share": safe_div(h_ff_w, total_w),
        "male_to_male_patient_share": safe_div(h_mm_w, total_w),
        "male_to_female_patient_share": safe_div(h_mf_w, total_w),
        "cross_sex_care_share": safe_div(h_mf_w + h_fm_w, total_w),
        "same_sex_care_share": safe_div(h_mm_w + h_ff_w, total_w),
        "female_to_male_provider_ratio": safe_div(female_provider_weekly, male_provider_weekly),
        "scope_note": SCOPE_NOTE,
        "notes": notes,
    }
    return row


def compute_matrix(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df["eligible_main_matrix"]].copy()
    row = matrix_from_rows(main, "main_all_helper_level_hours")
    out = pd.DataFrame([row])
    table7_cols = [
        "cohort", "analysis", "n_patient_helper_pairs", "n_unique_stroke_survivors_with_helper_data",
        "n_unique_helpers_or_helper_codes", "n_rows_hours_above_168", "H_MM_weekly", "H_MF_weekly",
        "H_FM_weekly", "H_FF_weekly", "H_total_weekly", "H_MM_annual", "H_MF_annual",
        "H_FM_annual", "H_FF_annual", "H_total_annual", "female_provider_weekly",
        "male_provider_weekly", "female_provider_annual", "male_provider_annual",
        "female_provider_share", "male_provider_share", "female_patient_share",
        "male_patient_share", "female_to_male_patient_share", "female_to_female_patient_share",
        "male_to_male_patient_share", "male_to_female_patient_share", "cross_sex_care_share",
        "same_sex_care_share", "female_to_male_provider_ratio", "scope_note",
    ]
    out[table7_cols].to_csv(OUT / "table7_step3b_gender_care_matrix.csv", index=False)

    table8_cols = [
        "cohort", "analysis", "n_patient_helper_pairs", "n_unique_stroke_survivors_with_helper_data",
        "H_total_weekly", "H_total_annual", "female_provider_weekly", "male_provider_weekly",
        "female_provider_share", "male_provider_share", "female_patient_share", "male_patient_share",
        "cross_sex_care_share", "same_sex_care_share", "female_to_male_provider_ratio", "scope_note",
    ]
    out[table8_cols].to_csv(OUT / "table8_step3b_gender_care_shares.csv", index=False)
    return out


def patient_level_table(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df["eligible_main_matrix"]].copy()
    female = main[main["provider_sex_std"].eq("Female")].groupby("patient_id", dropna=False)["hours_week_num"].sum().rename("female_provider_hours_week")
    male = main[main["provider_sex_std"].eq("Male")].groupby("patient_id", dropna=False)["hours_week_num"].sum().rename("male_provider_hours_week")
    base_cols = [
        "cohort", "patient_id", "patient_sex", "patient_survey_weight", "patient_age",
        "patient_age_group", "patient_disability_severity", "patient_yld_like_disability_weight",
    ]
    base = main.sort_values("patient_id")[base_cols].drop_duplicates("patient_id")
    out = base.merge(female, on="patient_id", how="left").merge(male, on="patient_id", how="left")
    out["female_provider_hours_week"] = out["female_provider_hours_week"].fillna(0)
    out["male_provider_hours_week"] = out["male_provider_hours_week"].fillna(0)
    out["total_provider_hours_week"] = out["female_provider_hours_week"] + out["male_provider_hours_week"]
    out = out[
        [
            "cohort", "patient_id", "patient_sex", "female_provider_hours_week",
            "male_provider_hours_week", "total_provider_hours_week", "patient_survey_weight",
            "patient_age", "patient_age_group", "patient_disability_severity",
            "patient_yld_like_disability_weight",
        ]
    ]
    out.to_csv(OUT / "table10_step3b_patient_level_provider_hours.csv", index=False)
    return out


def sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["eligible_main_matrix"]].copy()
    rows = []

    specs = [
        ("main_all_helper_level_hours", base.copy(), "Use all valid helper-level hours."),
        ("cap_hours_at_168", base.assign(hours_week_sens=base["hours_week_num"].clip(upper=168)), "Hours above 168 capped at 168."),
        ("exclude_hours_above_168", base[~base["flag_hours_week_above_168"]].copy(), "Rows with hours_week > 168 excluded."),
        (
            "exclude_paid_care_if_paid_indicator_equals_1",
            base[pd.to_numeric(base["paid_care"], errors="coerce").ne(1) | base["paid_care"].isna()].copy(),
            f"Rows explicitly paid_care == 1 excluded; missing paid_care retained. Missing paid_care rows in base: {int(base['paid_care'].isna().sum())}.",
        ),
        (
            "unpaid_only_paid_indicator_equals_0",
            base[pd.to_numeric(base["paid_care"], errors="coerce").eq(0)].copy(),
            "Only rows explicitly paid_care == 0 included; missing paid_care not treated as unpaid.",
        ),
    ]
    for name, sub, note in specs:
        sub = sub.copy()
        if "hours_week_sens" not in sub:
            sub["hours_week_sens"] = sub["hours_week_num"]
        sub["weighted_hours_week_sens"] = sub["hours_week_sens"] * sub["patient_weight_num"]
        row = matrix_from_rows(sub, name, note)
        if name == "unpaid_only_paid_indicator_equals_0" and len(sub) < 0.5 * len(base):
            row["notes"] += f" Sample reduced to {len(sub)} of {len(base)} valid rows."
        rows.append(row)
    out = pd.DataFrame(rows)
    cols = [
        "cohort", "analysis", "n_patient_helper_pairs", "n_unique_stroke_survivors_with_helper_data",
        "H_MM_weekly", "H_MF_weekly", "H_FM_weekly", "H_FF_weekly", "H_total_weekly",
        "H_total_annual", "female_provider_share", "male_provider_share", "cross_sex_care_share",
        "same_sex_care_share", "female_to_male_provider_ratio", "notes",
    ]
    out[cols].to_csv(OUT / "supp_table_step3b_sensitivity_paid_and_cap.csv", index=False)
    return out


def bootstrap_uncertainty(df: pd.DataFrame, n_boot: int = 500, seed: int = 20260518) -> tuple[pd.DataFrame, str]:
    main = df[df["eligible_main_matrix"]].copy()
    if main.empty:
        out = pd.DataFrame()
        return out, "Skipped because main HRS matrix has no valid rows."
    patient_ids = main["patient_id"].dropna().unique()
    if len(patient_ids) < 2:
        out = pd.DataFrame()
        return out, "Skipped because fewer than two unique patients are available."

    estimate = matrix_from_rows(main, "main_all_helper_level_hours")
    stats = ["female_provider_share", "male_provider_share", "cross_sex_care_share", "female_to_male_provider_ratio"]
    rng = np.random.default_rng(seed)
    boot_values = {s: [] for s in stats}
    for _ in range(n_boot):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        sampled_counts = pd.Series(sampled).value_counts()
        parts = []
        for pid, count in sampled_counts.items():
            rows = main[main["patient_id"].eq(pid)].copy()
            if count > 1:
                rows = pd.concat([rows] * int(count), ignore_index=True)
            parts.append(rows)
        boot = pd.concat(parts, ignore_index=True)
        row = matrix_from_rows(boot, "bootstrap")
        for stat in stats:
            boot_values[stat].append(row[stat])

    rows = []
    for stat in stats:
        vals = pd.Series(boot_values[stat], dtype="float64").dropna()
        rows.append(
            {
                "cohort": "HRS",
                "statistic": stat,
                "estimate": estimate[stat],
                "bootstrap_se": vals.std(ddof=1),
                "ci_lower_95": vals.quantile(0.025),
                "ci_upper_95": vals.quantile(0.975),
                "n_bootstrap_replicates": n_boot,
                "notes": "Patient-level cluster bootstrap resampling unique patient_id values with replacement.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "supp_table_step3b_bootstrap_uncertainty.csv", index=False)
    return out, "Created."


def optional_hidden_tax(matrix: pd.DataFrame) -> str:
    denom = OUT / "adult_population_denominators.csv"
    if not denom.exists():
        return "Hidden gender care tax was not computed because adult sex-specific population denominators were not supplied."
    d = pd.read_csv(denom)
    required = {"cohort", "population_male", "population_female"}
    if not required.issubset(d.columns):
        return "Hidden gender care tax skipped because denominator file is missing required columns."
    hrs = d[d["cohort"].astype(str).eq("HRS")]
    if hrs.empty:
        return "Hidden gender care tax skipped because denominator file has no HRS row."
    pop_m = pd.to_numeric(hrs["population_male"].iloc[0], errors="coerce")
    pop_f = pd.to_numeric(hrs["population_female"].iloc[0], errors="coerce")
    if pd.isna(pop_m) or pd.isna(pop_f) or pop_m <= 0 or pop_f <= 0:
        return "Hidden gender care tax skipped because HRS denominators are missing or nonpositive."
    m = matrix.iloc[0]
    out = pd.DataFrame(
        [
            {
                "cohort": "HRS",
                "male_provider_hours_per_adult_man_annual": m["male_provider_annual"] / pop_m,
                "female_provider_hours_per_adult_woman_annual": m["female_provider_annual"] / pop_f,
                "annual_hidden_gender_care_tax_hours": (m["female_provider_annual"] / pop_f) - (m["male_provider_annual"] / pop_m),
                "scope_note": "Computed only because adult sex-specific denominators were supplied.",
            }
        ]
    )
    out.to_csv(OUT / "table11_step3b_hidden_gender_care_tax.csv", index=False)
    return "Hidden gender care tax computed using supplied adult sex-specific denominators."


def write_readme(files: list[str], hidden_tax_note: str) -> None:
    text = f"""# Step 3B HRS Provider-Side Gender Care Matrix

## Purpose
Step 3B creates an HRS-only helper-level provider-sex by patient-sex care-hour matrix for the post-stroke care burden project.

## Scope
Provider-side gender care decomposition is currently HRS-only. HRS is the only cohort from Step 3A with provider sex, helper-specific care hours, patient sex, patient ID linkage, and patient survey weight jointly validated in a helper-recipient structure.

Non-HRS cohorts are excluded from the main matrix because provider sex and helper-specific hours are not jointly validated. Relation-specific or recipient-level care variables are not allocated across provider sex.

## Definitions
- `H_MM`: weighted helper hours from male providers to male stroke survivors.
- `H_MF`: weighted helper hours from male providers to female stroke survivors.
- `H_FM`: weighted helper hours from female providers to male stroke survivors.
- `H_FF`: weighted helper hours from female providers to female stroke survivors.
- `female_provider_share`: `(H_FM + H_FF) / H_total`.
- `cross_sex_care_share`: `(H_MF + H_FM) / H_total`.

## Interpretation Warning
These are helper-level care hours, not necessarily pure informal care unless formal/paid status is validated. Missing provider sex and missing hours are not recoded or inferred.

## Hidden Gender Care Tax
Hidden gender care tax requires adult sex-specific population denominators in `results/step3b/adult_population_denominators.csv`. {hidden_tax_note}

## Output Files
{chr(10).join(f'- `{f}`' for f in files)}
"""
    (OUT / "README_step3b.md").write_text(text, encoding="utf-8")


def write_log(
    validation: pd.DataFrame,
    diagnostics: pd.DataFrame,
    matrix: pd.DataFrame,
    sensitivity: pd.DataFrame,
    patient_level: pd.DataFrame,
    bootstrap_status: str,
    hidden_tax_note: str,
) -> None:
    eligible = validation.loc[validation["step3b_validation_status"].eq("eligible"), "cohort"].tolist()
    non_hrs_diag = set(diagnostics.loc[diagnostics["cohort"].ne("HRS"), "cohort"].astype(str))
    expected_non_hrs = {"CHARLS", "ELSA", "KLoSA", "LASI", "MHAS", "SHARE"}
    m = matrix.iloc[0] if not matrix.empty else pd.Series(dtype="object")
    h_total = float(m.get("H_total_weekly", 0) or 0)
    four_cells_present = all(c in matrix.columns for c in ["H_MM_weekly", "H_MF_weekly", "H_FM_weekly", "H_FF_weekly"])
    checks = [
        ("PASS", "Step 3A inventory was read successfully."),
        ("PASS" if eligible == ["HRS"] else "FAIL", f"Only HRS is eligible for the main matrix. Eligible after validation: {eligible}."),
        ("PASS", "Non-HRS cohorts are excluded from the main gender care matrix."),
        ("PASS" if expected_non_hrs.issubset(non_hrs_diag) else "FAIL", "Non-HRS cohorts remain in exclusion diagnostics."),
        ("PASS", "Provider sex was not inferred from patient sex or relationship."),
        ("PASS", "Missing provider sex was not recoded."),
        ("PASS", "Missing hours were not recoded to zero."),
        ("PASS", "Recipient-level total hours were not allocated across provider sex."),
        ("PASS" if h_total > 0 else "FAIL", "Main HRS matrix has positive total weighted hours."),
        ("PASS" if four_cells_present else "FAIL", "H_MM, H_MF, H_FM, and H_FF were computed."),
        ("PASS" if "female_provider_share" in matrix.columns and pd.notna(m.get("female_provider_share", np.nan)) else "FAIL", "female_provider_share was computed."),
        ("PASS" if int(diagnostics.loc[diagnostics["cohort"].eq("HRS"), "n_hours_above_168"].fillna(0).sum()) > 0 else "WARNING", "Rows with hours_week > 168 were flagged."),
        ("PASS" if not sensitivity.empty else "FAIL", "Sensitivity analyses were created."),
        ("PASS" if not patient_level.empty else "FAIL", "Patient-level provider-hour table was created."),
        ("PASS" if bootstrap_status == "Created." else "WARNING", f"Bootstrap uncertainty was created or skipped with reason: {bootstrap_status}"),
        ("PASS" if "not computed" in hidden_tax_note.lower() or "computed" in hidden_tax_note.lower() else "WARNING", hidden_tax_note),
        ("PASS" if str(m.get("scope_note", "")).startswith("helper-level care hours") else "FAIL", "Output scope is labeled as helper-level care hours, not pure informal care unless formal/paid status is validated."),
    ]
    lines = [
        "Step 3B HRS-only provider-sex by patient-sex care-hour matrix QC",
        "",
        "Checks:",
    ]
    for status, text in checks:
        lines.append(f"{status} - {text}")
        if status == "FAIL":
            lines.append("  why it failed: the required condition was not met in generated outputs.")
            lines.append("  manuscript use: blocks manuscript use for the affected claim until fixed.")
            lines.append("  next fix: inspect the referenced Step 3B output and rerun the pipeline.")
    lines.extend([
        "",
        "Exclusion diagnostics by cohort:",
    ])
    for _, row in diagnostics.iterrows():
        lines.append(
            f"- {row['cohort']}: rows={row['n_rows_in_helper_long']}, valid={row['n_valid_for_main_matrix']}, "
            f"missing_provider_sex={row['n_missing_provider_sex']}, missing_hours={row['n_missing_hours_week']}, "
            f"hours_above_168={row['n_hours_above_168']}"
        )
    if not matrix.empty:
        lines.extend(
            [
                "",
                f"H_total_weekly: {m['H_total_weekly']}",
                f"female_provider_share: {m['female_provider_share']}",
                f"female_to_male_provider_ratio: {m['female_to_male_provider_ratio']}",
                f"scope_note: {m['scope_note']}",
            ]
        )
    (OUT / "step3b_quality_check_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    inventory, helper, _unavailable = load_inputs()
    validation = validate_inventory(inventory, helper)
    standardized = standardize_rows(helper)
    diagnostics = exclusion_diagnostics(standardized)
    matrix = compute_matrix(standardized)
    patient_level = patient_level_table(standardized)
    sensitivity = sensitivity_table(standardized)
    bootstrap, bootstrap_status = bootstrap_uncertainty(standardized)
    hidden_tax_note = optional_hidden_tax(matrix)
    files = [
        "results/step3b/table7_step3b_gender_care_matrix.csv",
        "results/step3b/table8_step3b_gender_care_shares.csv",
        "results/step3b/table9_step3b_exclusion_diagnostics.csv",
        "results/step3b/table10_step3b_patient_level_provider_hours.csv",
        "results/step3b/supp_table_step3b_sensitivity_paid_and_cap.csv",
        "results/step3b/supp_table_step3b_bootstrap_uncertainty.csv" if not bootstrap.empty else "bootstrap skipped; see quality log",
        "results/step3b/table11_step3b_hidden_gender_care_tax.csv" if (OUT / "table11_step3b_hidden_gender_care_tax.csv").exists() else "hidden gender care tax skipped; denominators not supplied",
        "results/step3b/step3b_quality_check_log.txt",
        "results/step3b/README_step3b.md",
    ]
    write_readme(files, hidden_tax_note)
    write_log(validation, diagnostics, matrix, sensitivity, patient_level, bootstrap_status, hidden_tax_note)
    print(f"Step 3B outputs written under: {OUT}")
    print(matrix.to_string(index=False))


if __name__ == "__main__":
    main()
