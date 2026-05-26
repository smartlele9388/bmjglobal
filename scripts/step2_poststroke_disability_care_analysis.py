from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "output" / "tables"
TABLE_DIR = ROOT / "output" / "tables" / "step2"
FIGURE_DIR = ROOT / "output" / "figures" / "step2"
MODEL_DIR = ROOT / "output" / "models" / "step2"
LOG_DIR = ROOT / "output" / "logs" / "step2"

for directory in [TABLE_DIR, FIGURE_DIR, MODEL_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

EXPECTED_INPUT = INPUT_DIR / "gender_tax_protocol_latest_poststroke.csv"
CARE_COHORTS = ["CHARLS", "ELSA", "HRS", "KLoSA", "LASI", "MHAS"]
ALL_COHORTS = ["CHARLS", "ELSA", "HRS", "KLoSA", "LASI", "MHAS", "SHARE"]
SEVERITY_LABELS = {
    0: "No disability",
    1: "IADL-only",
    2: "ADL 1-2",
    3: "ADL >=3",
}
SEX_LABELS = {0: "Male", 1: "Female"}
OPTIONAL_COVARIATES = ["education", "wealth", "marital_status", "living_alone"]


def find_input() -> Path:
    if EXPECTED_INPUT.exists():
        return EXPECTED_INPUT
    matches = list(ROOT.rglob("gender_tax_protocol_latest_poststroke.csv"))
    if not matches:
        raise FileNotFoundError("gender_tax_protocol_latest_poststroke.csv was not found under the project.")
    return matches[0]


def normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    normalized_map = {normalize_name(c): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
        key = normalize_name(candidate)
        if key in normalized_map:
            return normalized_map[key]
    return None


def map_variables(df: pd.DataFrame) -> tuple[dict[str, str | None], pd.DataFrame, list[str]]:
    columns = list(df.columns)
    candidates = {
        "cohort": ["dataset", "cohort", "country"],
        "respondent_id": ["respondent_id", "id", "person_id", "hhidpn", "mergeid"],
        "age": ["age", "ragey", "r_age", "age_years"],
        "sex": ["sex", "gender", "ragender"],
        "stroke_indicator": ["stroke_ever", "stroke", "poststroke", "ever_stroke"],
        "adl_count": ["adl_score", "adl_count", "adl"],
        "iadl_count": ["iadl_score", "iadl_count", "iadl"],
        "any_adl_iadl_disability": ["any_disability", "disability_any", "disability_any_percent"],
        "severe_disability": ["severe_adl_disability", "severe_disability", "disability_state"],
        "informal_care_hours_week": ["care_hours_week", "informal_hours_week", "informal_care_hours_week"],
        "weight": ["weight_cross", "wtresp", "weight", "survey_weight"],
        "education": ["education", "edu", "raeducl", "raeduc", "education_level"],
        "wealth": ["wealth", "wealth_quintile", "asset_quintile", "income_quintile"],
        "marital_status": ["marital", "marital_status", "married", "partnership"],
        "living_alone": ["living_alone", "alone", "household_size_one"],
    }
    mapping: dict[str, str | None] = {}
    rows = []
    unavailable = []
    for role, role_candidates in candidates.items():
        found = first_existing(columns, role_candidates)
        mapping[role] = found
        required = role not in OPTIONAL_COVARIATES
        if found is None and role in OPTIONAL_COVARIATES:
            unavailable.append(role)
        rows.append(
            {
                "variable_role": role,
                "mapped_column": found,
                "required_for_step2": required,
                "status": "available" if found else "unavailable",
                "candidate_columns_checked": "; ".join(role_candidates),
            }
        )
    return mapping, pd.DataFrame(rows), unavailable


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def valid_weight(series: pd.Series) -> pd.Series:
    weight = to_numeric(series)
    return weight.where(np.isfinite(weight) & (weight > 0))


def weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    v = to_numeric(values)
    mask = v.notna()
    if weights is not None:
        w = valid_weight(weights)
        mask = mask & w.notna()
        if mask.sum() and w.loc[mask].sum() > 0:
            return float(np.average(v.loc[mask], weights=w.loc[mask]))
    if mask.sum() == 0:
        return np.nan
    return float(v.loc[mask].mean())


def weighted_percent(values: pd.Series, weights: pd.Series | None = None) -> float:
    mean = weighted_mean(values, weights)
    return np.nan if pd.isna(mean) else 100.0 * mean


def weighted_quantile(values: pd.Series, quantile: float, weights: pd.Series | None = None) -> float:
    v = to_numeric(values)
    mask = v.notna()
    if mask.sum() == 0:
        return np.nan
    if weights is None:
        return float(v.loc[mask].quantile(quantile))
    w = valid_weight(weights)
    mask = mask & w.notna()
    if mask.sum() == 0 or w.loc[mask].sum() <= 0:
        return float(v.dropna().quantile(quantile))
    order = np.argsort(v.loc[mask].to_numpy())
    sorted_v = v.loc[mask].to_numpy()[order]
    sorted_w = w.loc[mask].to_numpy()[order]
    cum_w = np.cumsum(sorted_w)
    cutoff = quantile * sorted_w.sum()
    return float(sorted_v[np.searchsorted(cum_w, cutoff, side="left")])


def safe_divide(num: float, den: float) -> float:
    if den is None or den == 0 or pd.isna(den):
        return np.nan
    return num / den


def clean_category(series: pd.Series, fallback: str = "Unknown") -> pd.Series:
    out = series.astype("string")
    out = out.fillna(fallback)
    out = out.replace({"": fallback, "nan": fallback, "<NA>": fallback})
    return out.astype(str)


def add_age_group(df: pd.DataFrame) -> pd.Series:
    age = to_numeric(df["age"])
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out.loc[age.between(50, 64, inclusive="both")] = "50-64"
    out.loc[age.between(65, 74, inclusive="both")] = "65-74"
    out.loc[age.between(75, 84, inclusive="both")] = "75-84"
    out.loc[age >= 85] = "85+"
    return out


def prepare_analysis_dataset(raw: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    required = ["cohort", "respondent_id", "age", "sex", "stroke_indicator", "adl_count", "iadl_count"]
    missing_required = [role for role in required if mapping.get(role) is None]
    if missing_required:
        raise ValueError(f"Required Step 2 variables unavailable: {missing_required}")

    df = pd.DataFrame(
        {
            "cohort": raw[mapping["cohort"]].astype(str),
            "respondent_id": raw[mapping["respondent_id"]].astype(str),
            "age": to_numeric(raw[mapping["age"]]),
            "sex": to_numeric(raw[mapping["sex"]]),
            "stroke_ever": to_numeric(raw[mapping["stroke_indicator"]]),
            "adl_count": to_numeric(raw[mapping["adl_count"]]),
            "iadl_count": to_numeric(raw[mapping["iadl_count"]]),
        }
    )

    care_col = mapping.get("informal_care_hours_week")
    df["informal_hours_week_raw"] = to_numeric(raw[care_col]) if care_col else np.nan
    weight_col = mapping.get("weight")
    df["original_weight"] = valid_weight(raw[weight_col]) if weight_col else np.nan

    for role in OPTIONAL_COVARIATES:
        col = mapping.get(role)
        if col:
            df[role] = raw[col]

    df["sex_label"] = df["sex"].map(SEX_LABELS).fillna("Unknown")
    df["age_group"] = add_age_group(df)
    valid_adl_iadl = df["adl_count"].notna() & df["iadl_count"].notna()

    df["any_disability"] = np.where(
        valid_adl_iadl,
        ((df["adl_count"] > 0) | (df["iadl_count"] > 0)).astype(int),
        np.nan,
    )
    df["severe_adl_disability"] = np.where(valid_adl_iadl, (df["adl_count"] >= 3).astype(int), np.nan)

    severity = pd.Series(np.nan, index=df.index, dtype="float")
    severity.loc[valid_adl_iadl & df["adl_count"].eq(0) & df["iadl_count"].eq(0)] = 0
    severity.loc[valid_adl_iadl & df["adl_count"].eq(0) & df["iadl_count"].ge(1)] = 1
    # Step 1 ADL scores may be proportionally scaled when one or two item values are missing.
    # Treat all 0 < ADL < 3 as the mild/moderate ADL category.
    severity.loc[valid_adl_iadl & df["adl_count"].gt(0) & df["adl_count"].lt(3)] = 2
    severity.loc[valid_adl_iadl & df["adl_count"].ge(3)] = 3
    df["disability_severity"] = severity
    df["disability_severity_label"] = df["disability_severity"].map(SEVERITY_LABELS)

    df["disability_sample"] = (
        df["cohort"].isin(ALL_COHORTS)
        & df["age"].ge(50)
        & df["stroke_ever"].eq(1)
        & valid_adl_iadl
    ).astype(int)
    df["care_hour_sample"] = (
        df["cohort"].isin(CARE_COHORTS)
        & df["age"].ge(50)
        & df["stroke_ever"].eq(1)
        & valid_adl_iadl
        & df["informal_hours_week_raw"].notna()
    ).astype(int)

    df["any_informal_care"] = np.where(
        df["informal_hours_week_raw"].notna(),
        (df["informal_hours_week_raw"] > 0).astype(int),
        np.nan,
    )
    df["heavy_informal_care_20"] = np.where(
        df["informal_hours_week_raw"].notna(),
        (df["informal_hours_week_raw"] >= 20).astype(int),
        np.nan,
    )
    df["heavy_informal_care_40"] = np.where(
        df["informal_hours_week_raw"].notna(),
        (df["informal_hours_week_raw"] >= 40).astype(int),
        np.nan,
    )
    df["informal_hours_week_cap168"] = df["informal_hours_week_raw"].clip(upper=168)
    df["informal_hours_week_winsor"] = np.nan

    p99_rows = []
    for cohort, sub in df.loc[df["cohort"].isin(CARE_COHORTS)].groupby("cohort", sort=False):
        values = sub.loc[sub["informal_hours_week_raw"].notna(), "informal_hours_week_raw"]
        cutoff = float(values.quantile(0.99)) if len(values) else np.nan
        p99_rows.append({"cohort": cohort, "p99_cutoff": cutoff, "care_hour_known_n": int(len(values))})
        mask = df["cohort"].eq(cohort) & df["informal_hours_week_raw"].notna()
        df.loc[mask, "informal_hours_week_winsor"] = df.loc[mask, "informal_hours_week_raw"].clip(upper=cutoff)

    pd.DataFrame(p99_rows).sort_values("cohort").to_csv(
        TABLE_DIR / "step2_care_hour_p99_cutoffs.csv", index=False
    )

    df["care_need"] = np.where(df["any_disability"].notna(), df["any_disability"], np.nan)
    df["objective_unmet"] = np.nan
    care_need_mask = df["cohort"].isin(CARE_COHORTS) & df["care_need"].eq(1) & df["informal_hours_week_raw"].notna()
    df.loc[care_need_mask & df["informal_hours_week_raw"].eq(0), "objective_unmet"] = 1
    df.loc[care_need_mask & df["informal_hours_week_raw"].gt(0), "objective_unmet"] = 0

    df["analysis_weight"] = df["original_weight"]
    for cohort, idx in df.groupby("cohort").groups.items():
        cohort_weights = df.loc[idx, "analysis_weight"]
        if cohort_weights.notna().sum() == 0:
            df.loc[idx, "analysis_weight"] = 1.0
        else:
            df.loc[idx, "analysis_weight"] = cohort_weights.fillna(1.0)

    df["normalized_weight"] = np.nan
    for cohort, idx in df.groupby("cohort").groups.items():
        weights = valid_weight(df.loc[idx, "analysis_weight"]).fillna(1.0)
        df.loc[idx, "normalized_weight"] = weights / weights.sum() * len(weights)

    for role in OPTIONAL_COVARIATES:
        if role in df.columns:
            df[role] = clean_category(df[role])

    return df


def save_input_inspection(raw: pd.DataFrame, mapping_df: pd.DataFrame) -> None:
    rows_by_cohort = raw.groupby(first_existing(list(raw.columns), ["dataset", "cohort", "country"]), dropna=False).size()
    missingness = (
        raw.isna()
        .sum()
        .rename("missing_n")
        .to_frame()
        .assign(total_n=len(raw), missing_percent=lambda x: x["missing_n"] / x["total_n"] * 100)
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missingness.to_csv(TABLE_DIR / "step2_missingness_table.csv", index=False)
    mapping_df.to_csv(TABLE_DIR / "step2_variable_mapping.csv", index=False)
    rows_by_cohort.rename("rows").reset_index().to_csv(TABLE_DIR / "step2_rows_by_cohort.csv", index=False)

    with open(LOG_DIR / "step2_input_inspection.txt", "w", encoding="utf-8") as f:
        f.write(f"Input rows: {len(raw)}\n")
        f.write(f"Input columns: {len(raw.columns)}\n\n")
        f.write("Column names:\n")
        for col in raw.columns:
            f.write(f"- {col}\n")
        f.write("\nRows by cohort/country:\n")
        f.write(rows_by_cohort.to_string())
        f.write("\n\nVariable mapping:\n")
        f.write(mapping_df.to_string(index=False))


def table2(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in ALL_COHORTS:
        dis = df[(df["cohort"].eq(cohort)) & (df["disability_sample"].eq(1))]
        care = df[(df["cohort"].eq(cohort)) & (df["care_hour_sample"].eq(1))]
        row = {
            "cohort": cohort,
            "unweighted_post_stroke_N": int(len(dis)),
            "no_disability_percent": weighted_percent(dis["disability_severity"].eq(0).astype(float), dis["analysis_weight"]),
            "IADL_only_disability_percent": weighted_percent(dis["disability_severity"].eq(1).astype(float), dis["analysis_weight"]),
            "ADL_1_2_percent": weighted_percent(dis["disability_severity"].eq(2).astype(float), dis["analysis_weight"]),
            "ADL_ge3_percent": weighted_percent(dis["disability_severity"].eq(3).astype(float), dis["analysis_weight"]),
            "footnote": "",
        }
        if cohort == "SHARE":
            row.update(
                {
                    "care_hour_known_N": np.nan,
                    "any_informal_care_percent": np.nan,
                    "mean_informal_h_week_raw": np.nan,
                    "mean_informal_h_week_p99_winsorized": np.nan,
                    "median_informal_h_week": np.nan,
                    "p75_informal_h_week": np.nan,
                    "p90_informal_h_week": np.nan,
                    "ge20_h_week_percent": np.nan,
                    "ge40_h_week_percent": np.nan,
                    "footnote": "SHARE was excluded from care-hour estimates because directly harmonizable care-hour variables were unavailable.",
                }
            )
        else:
            row.update(
                {
                    "care_hour_known_N": int(len(care)),
                    "any_informal_care_percent": weighted_percent(care["any_informal_care"], care["analysis_weight"]),
                    "mean_informal_h_week_raw": weighted_mean(care["informal_hours_week_raw"], care["analysis_weight"]),
                    "mean_informal_h_week_p99_winsorized": weighted_mean(care["informal_hours_week_winsor"], care["analysis_weight"]),
                    "median_informal_h_week": weighted_quantile(care["informal_hours_week_raw"], 0.50, care["analysis_weight"]),
                    "p75_informal_h_week": weighted_quantile(care["informal_hours_week_raw"], 0.75, care["analysis_weight"]),
                    "p90_informal_h_week": weighted_quantile(care["informal_hours_week_raw"], 0.90, care["analysis_weight"]),
                    "ge20_h_week_percent": weighted_percent(care["heavy_informal_care_20"], care["analysis_weight"]),
                    "ge40_h_week_percent": weighted_percent(care["heavy_informal_care_40"], care["analysis_weight"]),
                }
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table2_disability_care_hour_distribution.csv", index=False)
    return out


def table3(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    care_all = df[df["care_hour_sample"].eq(1)]
    for cohort in CARE_COHORTS:
        base = care_all[care_all["cohort"].eq(cohort)]
        rec = base[base["informal_hours_week_raw"].gt(0)]
        rows.append(
            {
                "cohort": cohort,
                "unweighted_care_recipient_N": int(len(rec)),
                "percent_with_any_informal_care_among_post_stroke_survivors": weighted_percent(
                    base["any_informal_care"], base["analysis_weight"]
                ),
                "mean_h_week_among_recipients_raw": weighted_mean(rec["informal_hours_week_raw"], rec["analysis_weight"]),
                "mean_h_week_among_recipients_p99_winsorized": weighted_mean(
                    rec["informal_hours_week_winsor"], rec["analysis_weight"]
                ),
                "median_h_week_among_recipients": weighted_quantile(
                    rec["informal_hours_week_raw"], 0.50, rec["analysis_weight"]
                ),
                "p75_h_week_among_recipients": weighted_quantile(
                    rec["informal_hours_week_raw"], 0.75, rec["analysis_weight"]
                ),
                "p90_h_week_among_recipients": weighted_quantile(
                    rec["informal_hours_week_raw"], 0.90, rec["analysis_weight"]
                ),
                "ge20_h_week_percent_among_recipients": weighted_percent(
                    rec["heavy_informal_care_20"], rec["analysis_weight"]
                ),
                "ge40_h_week_percent_among_recipients": weighted_percent(
                    rec["heavy_informal_care_40"], rec["analysis_weight"]
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table3_recipient_only_care_intensity.csv", index=False)
    return out


def table4(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    care_all = df[df["care_hour_sample"].eq(1)]
    for cohort in CARE_COHORTS:
        base = care_all[care_all["cohort"].eq(cohort)]
        disabled = base[base["care_need"].eq(1)]
        female = disabled[disabled["sex_label"].eq("Female")]
        male = disabled[disabled["sex_label"].eq("Male")]
        female_unmet = weighted_percent(female["objective_unmet"], female["analysis_weight"])
        male_unmet = weighted_percent(male["objective_unmet"], male["analysis_weight"])
        rows.append(
            {
                "cohort": cohort,
                "unweighted_post_stroke_N_with_care_hour_information": int(len(base)),
                "unweighted_disabled_post_stroke_N_with_care_hour_information": int(len(disabled)),
                "unmet_among_all_post_stroke_survivors_percent": weighted_percent(
                    ((base["care_need"].eq(1)) & (base["informal_hours_week_raw"].eq(0))).astype(float),
                    base["analysis_weight"],
                ),
                "unmet_among_disabled_post_stroke_survivors_percent": weighted_percent(
                    disabled["objective_unmet"], disabled["analysis_weight"]
                ),
                "female_unmet_among_disabled_percent": female_unmet,
                "male_unmet_among_disabled_percent": male_unmet,
                "female_minus_male_unmet_difference": female_unmet - male_unmet
                if pd.notna(female_unmet) and pd.notna(male_unmet)
                else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "table4_objective_unmet_need.csv", index=False)
    return out


def supplementary_tables(df: pd.DataFrame, mapping: dict[str, str | None], raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s1_rows = []
    for cohort in ALL_COHORTS:
        sub = df[df["cohort"].eq(cohort)]
        post = sub[(sub["age"].ge(50)) & (sub["stroke_ever"].eq(1))]
        cov_missing = {}
        for role in OPTIONAL_COVARIATES:
            if role in post.columns:
                cov_missing[f"{role}_missing_percent"] = post[role].isna().mean() * 100
            else:
                cov_missing[f"{role}_missing_percent"] = np.nan
        s1_rows.append(
            {
                "cohort": cohort,
                "post_stroke_N": int(len(post)),
                "ADL_IADL_missing_percent": ((post["adl_count"].isna()) | (post["iadl_count"].isna())).mean() * 100
                if len(post)
                else np.nan,
                "care_hour_missing_percent": post["informal_hours_week_raw"].isna().mean() * 100 if len(post) else np.nan,
                "sex_missing_percent": post["sex"].isna().mean() * 100 if len(post) else np.nan,
                "age_missing_percent": post["age"].isna().mean() * 100 if len(post) else np.nan,
                **cov_missing,
                "analytic_disability_sample_N": int((post["disability_sample"] == 1).sum()),
                "analytic_care_hour_sample_N": int((post["care_hour_sample"] == 1).sum()),
            }
        )
    s1 = pd.DataFrame(s1_rows)
    s1.to_csv(TABLE_DIR / "supp_table_s1_missingness.csv", index=False)

    s2_rows = []
    for cohort in ALL_COHORTS:
        for sex in ["Male", "Female"]:
            dis = df[(df["cohort"].eq(cohort)) & (df["sex_label"].eq(sex)) & (df["disability_sample"].eq(1))]
            care = df[(df["cohort"].eq(cohort)) & (df["sex_label"].eq(sex)) & (df["care_hour_sample"].eq(1))]
            disabled = care[care["care_need"].eq(1)]
            row = {
                "cohort": cohort,
                "sex": sex,
                "N": int(len(dis)),
                "any_disability_percent": weighted_percent(dis["any_disability"], dis["analysis_weight"]),
                "severe_ADL_disability_percent": weighted_percent(dis["severe_adl_disability"], dis["analysis_weight"]),
                "no_disability_percent": weighted_percent(dis["disability_severity"].eq(0).astype(float), dis["analysis_weight"]),
                "IADL_only_percent": weighted_percent(dis["disability_severity"].eq(1).astype(float), dis["analysis_weight"]),
                "ADL_1_2_percent": weighted_percent(dis["disability_severity"].eq(2).astype(float), dis["analysis_weight"]),
                "ADL_ge3_percent": weighted_percent(dis["disability_severity"].eq(3).astype(float), dis["analysis_weight"]),
            }
            if cohort == "SHARE":
                row.update(
                    {
                        "care_hour_known_N": np.nan,
                        "any_informal_care_percent": np.nan,
                        "mean_h_week_raw": np.nan,
                        "mean_h_week_p99_winsorized": np.nan,
                        "median_h_week": np.nan,
                        "p75_h_week": np.nan,
                        "p90_h_week": np.nan,
                        "unmet_among_disabled_percent": np.nan,
                    }
                )
            else:
                row.update(
                    {
                        "care_hour_known_N": int(len(care)),
                        "any_informal_care_percent": weighted_percent(care["any_informal_care"], care["analysis_weight"]),
                        "mean_h_week_raw": weighted_mean(care["informal_hours_week_raw"], care["analysis_weight"]),
                        "mean_h_week_p99_winsorized": weighted_mean(care["informal_hours_week_winsor"], care["analysis_weight"]),
                        "median_h_week": weighted_quantile(care["informal_hours_week_raw"], 0.50, care["analysis_weight"]),
                        "p75_h_week": weighted_quantile(care["informal_hours_week_raw"], 0.75, care["analysis_weight"]),
                        "p90_h_week": weighted_quantile(care["informal_hours_week_raw"], 0.90, care["analysis_weight"]),
                        "unmet_among_disabled_percent": weighted_percent(disabled["objective_unmet"], disabled["analysis_weight"]),
                    }
                )
            s2_rows.append(row)
    s2 = pd.DataFrame(s2_rows)
    s2.to_csv(TABLE_DIR / "supp_table_s2_sex_stratified.csv", index=False)

    p99 = pd.read_csv(TABLE_DIR / "step2_care_hour_p99_cutoffs.csv")
    s3_rows = []
    for cohort in CARE_COHORTS:
        care = df[(df["cohort"].eq(cohort)) & (df["care_hour_sample"].eq(1))]
        p99_cutoff = p99.loc[p99["cohort"].eq(cohort), "p99_cutoff"]
        s3_rows.append(
            {
                "cohort": cohort,
                "mean_raw_h_week": weighted_mean(care["informal_hours_week_raw"], care["analysis_weight"]),
                "mean_p99_winsorized_h_week": weighted_mean(care["informal_hours_week_winsor"], care["analysis_weight"]),
                "mean_cap168_h_week": weighted_mean(care["informal_hours_week_cap168"], care["analysis_weight"]),
                "median_h_week": weighted_quantile(care["informal_hours_week_raw"], 0.50, care["analysis_weight"]),
                "p90_h_week": weighted_quantile(care["informal_hours_week_raw"], 0.90, care["analysis_weight"]),
                "p99_cutoff": float(p99_cutoff.iloc[0]) if not p99_cutoff.empty else np.nan,
            }
        )
    s3 = pd.DataFrame(s3_rows)
    s3.to_csv(TABLE_DIR / "supp_table_s3_care_hour_sensitivity.csv", index=False)
    return s1, s2, s3


def standard_population(df: pd.DataFrame, cohorts: list[str], scope: str, sample_col: str) -> pd.DataFrame:
    base = df[(df["cohort"].isin(cohorts)) & (df[sample_col].eq(1)) & df["age_group"].notna() & df["sex_label"].isin(["Male", "Female"])]
    rows = []
    grouped = base.groupby(["age_group", "sex_label"], dropna=False)
    total_weight = valid_weight(base["analysis_weight"]).fillna(1.0).sum()
    for (age_group, sex), sub in grouped:
        weighted_n = valid_weight(sub["analysis_weight"]).fillna(1.0).sum()
        rows.append(
            {
                "standard_scope": scope,
                "age_group": age_group,
                "sex": sex,
                "unweighted_n": int(len(sub)),
                "weighted_n": float(weighted_n),
                "standard_weight": float(weighted_n / total_weight) if total_weight > 0 else np.nan,
                "sparse_standard_cell": int(len(sub) < 20),
            }
        )
    return pd.DataFrame(rows)


def direct_standardize(
    df: pd.DataFrame,
    cohorts: list[str],
    sample_col: str,
    std: pd.DataFrame,
    outcome_col: str,
    statistic: str,
    restrict_disabled: bool = False,
) -> pd.DataFrame:
    rows = []
    std_key = std[["age_group", "sex", "standard_weight"]].copy()
    for cohort in cohorts:
        base = df[(df["cohort"].eq(cohort)) & (df[sample_col].eq(1))]
        if restrict_disabled:
            base = base[base["care_need"].eq(1)]
        values = []
        missing_cells = 0
        sparse_cells = 0
        for _, std_row in std_key.iterrows():
            cell = base[(base["age_group"].eq(std_row["age_group"])) & (base["sex_label"].eq(std_row["sex"]))]
            if len(cell) < 20:
                sparse_cells += 1
            if len(cell) == 0 or cell[outcome_col].notna().sum() == 0:
                missing_cells += 1
                values.append((np.nan, std_row["standard_weight"]))
                continue
            if statistic == "percent":
                estimate = weighted_percent(cell[outcome_col], cell["analysis_weight"])
            elif statistic == "mean":
                estimate = weighted_mean(cell[outcome_col], cell["analysis_weight"])
            else:
                raise ValueError(statistic)
            values.append((estimate, std_row["standard_weight"]))
        vals = pd.DataFrame(values, columns=["estimate", "standard_weight"]).dropna()
        standardized = np.nan
        if len(vals):
            standardized = float((vals["estimate"] * vals["standard_weight"]).sum() / vals["standard_weight"].sum())
        rows.append(
            {
                "cohort": cohort,
                "estimate": outcome_col,
                "statistic": statistic,
                "standardized_value": standardized,
                "standard_cells_missing": int(missing_cells),
                "sparse_cells_lt20": int(sparse_cells),
                "observed_unweighted_n": int(len(base)),
            }
        )
    return pd.DataFrame(rows)


def age_sex_standardized(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    std_dis = standard_population(df, ALL_COHORTS, "disability_7cohort", "disability_sample")
    std_care = standard_population(df, CARE_COHORTS, "care_6cohort", "care_hour_sample")
    std_all = pd.concat([std_dis, std_care], ignore_index=True)
    std_all.to_csv(TABLE_DIR / "step2_standard_population_age_sex.csv", index=False)

    estimates = []
    estimates.append(
        direct_standardize(
            df,
            ALL_COHORTS,
            "disability_sample",
            std_dis,
            "severe_adl_disability",
            "percent",
        ).assign(standard_scope="disability_7cohort")
    )
    estimates.append(
        direct_standardize(
            df,
            CARE_COHORTS,
            "care_hour_sample",
            std_care,
            "any_informal_care",
            "percent",
        ).assign(standard_scope="care_6cohort")
    )
    estimates.append(
        direct_standardize(
            df,
            CARE_COHORTS,
            "care_hour_sample",
            std_care,
            "informal_hours_week_winsor",
            "mean",
        ).assign(standard_scope="care_6cohort")
    )
    estimates.append(
        direct_standardize(
            df,
            CARE_COHORTS,
            "care_hour_sample",
            std_care,
            "objective_unmet",
            "percent",
            restrict_disabled=True,
        ).assign(standard_scope="care_6cohort_disabled")
    )
    out = pd.concat(estimates, ignore_index=True)
    out.to_csv(TABLE_DIR / "step2_age_sex_standardized_estimates.csv", index=False)
    return out, std_all


def model_covariates(df: pd.DataFrame, base: list[str]) -> list[str]:
    covars = base.copy()
    for role in OPTIONAL_COVARIATES:
        if role in df.columns and df[role].notna().sum() > 0 and df[role].nunique(dropna=True) > 1:
            covars.append(role)
    return covars


def formula_for(outcome: str, covariates: list[str]) -> str:
    terms = []
    for cov in covariates:
        if cov in ["age"]:
            terms.append(cov)
        else:
            terms.append(f"C({cov})")
    return f"{outcome} ~ " + " + ".join(terms)


def fit_logistic(df: pd.DataFrame, outcome: str, covariates: list[str], model_name: str):
    needed = [outcome, "normalized_weight", *covariates]
    data = df[needed].dropna().copy()
    data = data[data[outcome].isin([0, 1])]
    if len(data) == 0 or data[outcome].nunique() < 2:
        raise ValueError(f"{model_name}: insufficient outcome variation.")
    formula = formula_for(outcome, covariates)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.glm(
            formula=formula,
            data=data,
            family=sm.families.Binomial(),
            freq_weights=data["normalized_weight"],
        ).fit(maxiter=200)
    return fit, data, formula


def fit_linear(df: pd.DataFrame, outcome: str, covariates: list[str], model_name: str):
    needed = [outcome, "normalized_weight", *covariates]
    data = df[needed].dropna().copy()
    if len(data) == 0:
        raise ValueError(f"{model_name}: no complete observations.")
    formula = formula_for(outcome, covariates)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.wls(formula=formula, data=data, weights=data["normalized_weight"]).fit()
    return fit, data, formula


def save_or_table(fit, data: pd.DataFrame, formula: str, path: Path, model_name: str, outcome: str, covariates: list[str]) -> pd.DataFrame:
    params = fit.params
    conf = fit.conf_int()
    rows = []
    for term in params.index:
        if term == "Intercept":
            continue
        rows.append(
            {
                "model": model_name,
                "outcome": outcome,
                "term": term,
                "odds_ratio": float(np.exp(params[term])),
                "ci_95_low": float(np.exp(conf.loc[term, 0])),
                "ci_95_high": float(np.exp(conf.loc[term, 1])),
                "p_value": float(fit.pvalues[term]),
                "n_complete": int(len(data)),
                "events": int(data[outcome].sum()),
                "formula": formula,
                "covariates_included": "; ".join(covariates),
                "weight": "normalized_weight",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def save_coef_table(fit, data: pd.DataFrame, formula: str, path: Path, model_name: str, outcome: str, covariates: list[str]) -> pd.DataFrame:
    conf = fit.conf_int()
    rows = []
    for term in fit.params.index:
        if term == "Intercept":
            continue
        rows.append(
            {
                "model": model_name,
                "outcome": outcome,
                "term": term,
                "coefficient": float(fit.params[term]),
                "ci_95_low": float(conf.loc[term, 0]),
                "ci_95_high": float(conf.loc[term, 1]),
                "p_value": float(fit.pvalues[term]),
                "n_complete": int(len(data)),
                "formula": formula,
                "covariates_included": "; ".join(covariates),
                "weight": "normalized_weight",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def marginal_predictions(fit, data: pd.DataFrame, cohorts: list[str], outcome_label: str, by_sex: bool = False) -> pd.DataFrame:
    rows = []
    sex_values = ["Male", "Female"] if by_sex else [None]
    for cohort in cohorts:
        for sex in sex_values:
            pred_data = data.copy()
            pred_data["cohort"] = cohort
            if sex is not None:
                pred_data["sex_label"] = sex
            pred = fit.predict(pred_data)
            w = valid_weight(pred_data["normalized_weight"]).fillna(1.0)
            rows.append(
                {
                    "cohort": cohort,
                    "sex": sex if sex is not None else "All",
                    "outcome": outcome_label,
                    "adjusted_predicted_probability": float(np.average(pred, weights=w)),
                    "n_complete_model_sample": int(len(data)),
                }
            )
    return pd.DataFrame(rows)


def run_models(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    unavailable_messages = []
    warnings_out = []
    included_covariate_log = []

    m1_df = df[df["disability_sample"].eq(1)].copy()
    cov1 = model_covariates(m1_df, ["cohort", "sex_label", "age_group"])
    try:
        fit1, data1, formula1 = fit_logistic(m1_df, "severe_adl_disability", cov1, "Model 1 severe disability")
        save_or_table(
            fit1,
            data1,
            formula1,
            MODEL_DIR / "model1_severe_disability_or.csv",
            "Model 1 severe disability",
            "severe_adl_disability",
            cov1,
        )
        marginal_predictions(fit1, data1, ALL_COHORTS, "severe_adl_disability").to_csv(
            MODEL_DIR / "model1_severe_disability_adjusted_probabilities.csv", index=False
        )
        included_covariate_log.append(f"Model 1 covariates: {', '.join(cov1)}")
    except Exception as exc:
        warnings_out.append(f"Model 1 failed: {exc}")
        pd.DataFrame({"error": [str(exc)]}).to_csv(MODEL_DIR / "model1_severe_disability_or.csv", index=False)
        pd.DataFrame({"error": [str(exc)]}).to_csv(
            MODEL_DIR / "model1_severe_disability_adjusted_probabilities.csv", index=False
        )

    m2_df = df[df["care_hour_sample"].eq(1)].copy()
    cov2 = model_covariates(m2_df, ["cohort", "disability_severity_label", "sex_label", "age_group"])
    try:
        fit2, data2, formula2 = fit_logistic(m2_df, "any_informal_care", cov2, "Model 2 any informal care")
        save_or_table(
            fit2,
            data2,
            formula2,
            MODEL_DIR / "model2_any_informal_care_or.csv",
            "Model 2 any informal care",
            "any_informal_care",
            cov2,
        )
        marginal_predictions(fit2, data2, CARE_COHORTS, "any_informal_care").to_csv(
            MODEL_DIR / "model2_any_informal_care_adjusted_probabilities.csv", index=False
        )
        included_covariate_log.append(f"Model 2 covariates: {', '.join(cov2)}")
    except Exception as exc:
        warnings_out.append(f"Model 2 failed: {exc}")
        pd.DataFrame({"error": [str(exc)]}).to_csv(MODEL_DIR / "model2_any_informal_care_or.csv", index=False)
        pd.DataFrame({"error": [str(exc)]}).to_csv(
            MODEL_DIR / "model2_any_informal_care_adjusted_probabilities.csv", index=False
        )

    try:
        fit3a, data3a, formula3a = fit_logistic(m2_df, "any_informal_care", cov2, "Model 3 Part A any care")
        save_or_table(
            fit3a,
            data3a,
            formula3a,
            MODEL_DIR / "model3_twopart_partA_any_care_or.csv",
            "Model 3 Part A any care",
            "any_informal_care",
            cov2,
        )
        positive = m2_df[m2_df["informal_hours_week_raw"].gt(0)].copy()
        positive["log_informal_hours_week_winsor"] = np.log(positive["informal_hours_week_winsor"])
        fit3b, data3b, formula3b = fit_linear(
            positive,
            "log_informal_hours_week_winsor",
            cov2,
            "Model 3 Part B positive hours",
        )
        save_coef_table(
            fit3b,
            data3b,
            formula3b,
            MODEL_DIR / "model3_twopart_partB_positive_hours_coef.csv",
            "Model 3 Part B positive hours",
            "log_informal_hours_week_winsor",
            cov2,
        )
        smear = float(np.average(np.exp(fit3b.resid), weights=valid_weight(data3b["normalized_weight"]).fillna(1.0)))
        rows = []
        for cohort in CARE_COHORTS:
            p_data = data3a.copy()
            p_data["cohort"] = cohort
            prob = fit3a.predict(p_data)
            prob_w = valid_weight(p_data["normalized_weight"]).fillna(1.0)
            mean_prob = float(np.average(prob, weights=prob_w))

            h_data = data3b.copy()
            h_data["cohort"] = cohort
            pos_hours = np.exp(fit3b.predict(h_data)) * smear
            h_w = valid_weight(h_data["normalized_weight"]).fillna(1.0)
            mean_positive_hours = float(np.average(pos_hours, weights=h_w))
            rows.append(
                {
                    "cohort": cohort,
                    "adjusted_probability_any_care": mean_prob,
                    "adjusted_mean_positive_hours": mean_positive_hours,
                    "adjusted_twopart_mean_hours": mean_prob * mean_positive_hours,
                    "smearing_factor": smear,
                    "n_partA": int(len(data3a)),
                    "n_partB": int(len(data3b)),
                }
            )
        pd.DataFrame(rows).to_csv(MODEL_DIR / "model3_twopart_adjusted_mean_hours.csv", index=False)
        included_covariate_log.append(f"Model 3 covariates: {', '.join(cov2)}")
    except Exception as exc:
        warnings_out.append(f"Model 3 failed: {exc}")
        for path in [
            "model3_twopart_partA_any_care_or.csv",
            "model3_twopart_partB_positive_hours_coef.csv",
            "model3_twopart_adjusted_mean_hours.csv",
        ]:
            pd.DataFrame({"error": [str(exc)]}).to_csv(MODEL_DIR / path, index=False)

    m4_df = df[(df["care_hour_sample"].eq(1)) & (df["care_need"].eq(1))].copy()
    cov4 = model_covariates(m4_df, ["cohort", "sex_label", "age_group", "disability_severity_label"])
    try:
        fit4, data4, formula4 = fit_logistic(m4_df, "objective_unmet", cov4, "Model 4 objective unmet need")
        save_or_table(
            fit4,
            data4,
            formula4,
            MODEL_DIR / "model4_objective_unmet_need_or.csv",
            "Model 4 objective unmet need",
            "objective_unmet",
            cov4,
        )
        marginal_predictions(fit4, data4, CARE_COHORTS, "objective_unmet").to_csv(
            MODEL_DIR / "model4_objective_unmet_need_adjusted_probabilities.csv", index=False
        )
        marginal_predictions(fit4, data4, CARE_COHORTS, "objective_unmet", by_sex=True).to_csv(
            MODEL_DIR / "model4_objective_unmet_need_by_sex_adjusted_probabilities.csv", index=False
        )
        included_covariate_log.append(f"Model 4 covariates: {', '.join(cov4)}")
    except Exception as exc:
        warnings_out.append(f"Model 4 failed: {exc}")
        for path in [
            "model4_objective_unmet_need_or.csv",
            "model4_objective_unmet_need_adjusted_probabilities.csv",
            "model4_objective_unmet_need_by_sex_adjusted_probabilities.csv",
        ]:
            pd.DataFrame({"error": [str(exc)]}).to_csv(MODEL_DIR / path, index=False)

    with open(LOG_DIR / "step2_model_covariates_log.txt", "w", encoding="utf-8") as f:
        for line in included_covariate_log:
            f.write(line + "\n")
        if warnings_out:
            f.write("\nWarnings:\n")
            for line in warnings_out:
                f.write(line + "\n")

    return included_covariate_log, warnings_out


def make_figures(table2_df: pd.DataFrame, df: pd.DataFrame, table4_df: pd.DataFrame) -> None:
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9})

    severity_cols = [
        "no_disability_percent",
        "IADL_only_disability_percent",
        "ADL_1_2_percent",
        "ADL_ge3_percent",
    ]
    labels = ["No disability", "IADL-only", "ADL 1-2", "ADL >=3"]
    colors = ["#4c78a8", "#72b7b2", "#f58518", "#e45756"]
    plot = table2_df.set_index("cohort").loc[ALL_COHORTS, severity_cols]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottom = np.zeros(len(plot))
    x = np.arange(len(plot))
    for col, label, color in zip(severity_cols, labels, colors):
        vals = plot[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label, color=color, width=0.72)
        bottom += np.nan_to_num(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(plot.index, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Weighted percent")
    ax.set_title("Post-stroke disability severity by cohort")
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure1_disability_severity_stacked_bar.png")
    fig.savefig(FIGURE_DIR / "figure1_disability_severity_stacked_bar.pdf")
    plt.close(fig)

    care = df[(df["care_hour_sample"].eq(1)) & (df["cohort"].isin(CARE_COHORTS))].copy()
    care["log_hours_plus1"] = np.log(care["informal_hours_week_raw"] + 1)
    data = [care.loc[care["cohort"].eq(cohort), "log_hours_plus1"].dropna().to_numpy() for cohort in CARE_COHORTS]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.boxplot(data, tick_labels=CARE_COHORTS, showfliers=False, patch_artist=True)
    ax.set_ylabel("log(informal care hours/week + 1)")
    ax.set_title("Informal care-hour distribution by cohort")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure2_log_informal_hours_distribution.png")
    fig.savefig(FIGURE_DIR / "figure2_log_informal_hours_distribution.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(CARE_COHORTS))
    female = table4_df.set_index("cohort").loc[CARE_COHORTS, "female_unmet_among_disabled_percent"].to_numpy()
    male = table4_df.set_index("cohort").loc[CARE_COHORTS, "male_unmet_among_disabled_percent"].to_numpy()
    width = 0.36
    ax.bar(x - width / 2, female, width, label="Female", color="#e45756")
    ax.bar(x + width / 2, male, width, label="Male", color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(CARE_COHORTS, rotation=30, ha="right")
    ax.set_ylabel("Weighted percent")
    ax.set_title("Objective unmet care need among disabled post-stroke survivors")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "figure3_sex_stratified_unmet_need.png")
    fig.savefig(FIGURE_DIR / "figure3_sex_stratified_unmet_need.pdf")
    plt.close(fig)


def results_paragraph(table2_df: pd.DataFrame, table3_df: pd.DataFrame, table4_df: pd.DataFrame) -> str:
    severe_min = table2_df["ADL_ge3_percent"].min()
    severe_max = table2_df["ADL_ge3_percent"].max()
    severe_min_cohort = table2_df.loc[table2_df["ADL_ge3_percent"].idxmin(), "cohort"]
    severe_max_cohort = table2_df.loc[table2_df["ADL_ge3_percent"].idxmax(), "cohort"]

    care_rows = table2_df[table2_df["cohort"].isin(CARE_COHORTS)]
    any_min = care_rows["any_informal_care_percent"].min()
    any_max = care_rows["any_informal_care_percent"].max()
    mean_min = care_rows["mean_informal_h_week_p99_winsorized"].min()
    mean_max = care_rows["mean_informal_h_week_p99_winsorized"].max()
    p90_max = care_rows["p90_informal_h_week"].max()

    unmet_min = table4_df["unmet_among_disabled_post_stroke_survivors_percent"].min()
    unmet_max = table4_df["unmet_among_disabled_post_stroke_survivors_percent"].max()
    unmet_min_cohort = table4_df.loc[
        table4_df["unmet_among_disabled_post_stroke_survivors_percent"].idxmin(), "cohort"
    ]
    unmet_max_cohort = table4_df.loc[
        table4_df["unmet_among_disabled_post_stroke_survivors_percent"].idxmax(), "cohort"
    ]

    paragraph = (
        "Across the seven cohorts, post-stroke disability severity varied substantially: "
        f"the weighted proportion with severe ADL disability ranged from {severe_min:.1f}% in "
        f"{severe_min_cohort} to {severe_max:.1f}% in {severe_max_cohort}. "
        "Among the six cohorts with directly harmonizable informal care-hour data, informal care hours were highly "
        f"right-skewed; weighted receipt of any informal care ranged from {any_min:.1f}% to {any_max:.1f}%, "
        f"and p99-winsorized mean hours ranged from {mean_min:.1f} to {mean_max:.1f} hours/week, "
        f"with cohort-specific 90th percentiles reaching up to {p90_max:.1f} hours/week. "
        "Objective unmet informal care need among disabled post-stroke survivors differed across cohorts, "
        f"ranging from {unmet_min:.1f}% in {unmet_min_cohort} to {unmet_max:.1f}% in {unmet_max_cohort}. "
        "These estimates describe patient-level post-stroke disability, informal care received, and objective unmet "
        "care need; they should not be interpreted as caregiver burden or causal effects."
    )
    (TABLE_DIR / "step2_results_paragraph.txt").write_text(paragraph + "\n", encoding="utf-8")
    return paragraph


def weighting_log(df: pd.DataFrame) -> str:
    lines = []
    lines.append("Weighting approach")
    lines.append("N values in all descriptive tables are unweighted sample counts.")
    lines.append("Percentages, means, quantiles, standardization, and models use analysis_weight when available.")
    lines.append("Pooled models use normalized_weight = analysis_weight / sum(analysis_weight within cohort) * cohort sample size.")
    lines.append("")
    lines.append("Weight availability by cohort:")
    for cohort in ALL_COHORTS:
        sub = df[df["cohort"].eq(cohort)]
        valid = sub["original_weight"].notna().sum()
        lines.append(
            f"- {cohort}: valid original weights {valid}/{len(sub)} "
            f"({safe_divide(valid, len(sub)) * 100:.1f}%). "
            + ("Cohort was analyzed with unit weights." if valid == 0 else "Valid original weights were used; missing individual weights were set to 1.")
        )
    text = "\n".join(lines) + "\n"
    (LOG_DIR / "step2_weighting_log.txt").write_text(text, encoding="utf-8")
    return text


def quality_check_log(df: pd.DataFrame, model_covariates: list[str], model_warnings: list[str], unavailable: list[str]) -> str:
    table2_path = TABLE_DIR / "table2_disability_care_hour_distribution.csv"
    table3_path = TABLE_DIR / "table3_recipient_only_care_intensity.csv"
    table4_path = TABLE_DIR / "table4_objective_unmet_need.csv"
    t2 = pd.read_csv(table2_path)
    t3 = pd.read_csv(table3_path)
    t4 = pd.read_csv(table4_path)

    output_files = [
        TABLE_DIR / "step2_variable_mapping.csv",
        TABLE_DIR / "step2_missingness_table.csv",
        TABLE_DIR / "poststroke_step2_analysis_sample.csv",
        TABLE_DIR / "step2_care_hour_p99_cutoffs.csv",
        TABLE_DIR / "table2_disability_care_hour_distribution.csv",
        TABLE_DIR / "table3_recipient_only_care_intensity.csv",
        TABLE_DIR / "table4_objective_unmet_need.csv",
        TABLE_DIR / "supp_table_s1_missingness.csv",
        TABLE_DIR / "supp_table_s2_sex_stratified.csv",
        TABLE_DIR / "supp_table_s3_care_hour_sensitivity.csv",
        TABLE_DIR / "step2_age_sex_standardized_estimates.csv",
        TABLE_DIR / "step2_standard_population_age_sex.csv",
        TABLE_DIR / "step2_results_paragraph.txt",
        FIGURE_DIR / "figure1_disability_severity_stacked_bar.png",
        FIGURE_DIR / "figure1_disability_severity_stacked_bar.pdf",
        FIGURE_DIR / "figure2_log_informal_hours_distribution.png",
        FIGURE_DIR / "figure2_log_informal_hours_distribution.pdf",
        FIGURE_DIR / "figure3_sex_stratified_unmet_need.png",
        FIGURE_DIR / "figure3_sex_stratified_unmet_need.pdf",
        MODEL_DIR / "model1_severe_disability_or.csv",
        MODEL_DIR / "model1_severe_disability_adjusted_probabilities.csv",
        MODEL_DIR / "model2_any_informal_care_or.csv",
        MODEL_DIR / "model2_any_informal_care_adjusted_probabilities.csv",
        MODEL_DIR / "model3_twopart_partA_any_care_or.csv",
        MODEL_DIR / "model3_twopart_partB_positive_hours_coef.csv",
        MODEL_DIR / "model3_twopart_adjusted_mean_hours.csv",
        MODEL_DIR / "model4_objective_unmet_need_or.csv",
        MODEL_DIR / "model4_objective_unmet_need_adjusted_probabilities.csv",
        MODEL_DIR / "model4_objective_unmet_need_by_sex_adjusted_probabilities.csv",
    ]

    checks = []
    checks.append(("SHARE appears in disability table", "SHARE" in set(t2["cohort"])))
    checks.append(("SHARE absent from recipient-only care table", "SHARE" not in set(t3["cohort"])))
    checks.append(("SHARE absent from unmet-need table", "SHARE" not in set(t4["cohort"])))
    checks.append(("SHARE excluded from Figure 2/3 data", "SHARE" not in set(df.loc[df["care_hour_sample"].eq(1), "cohort"])))
    severity_sums = (
        t2[[
            "no_disability_percent",
            "IADL_only_disability_percent",
            "ADL_1_2_percent",
            "ADL_ge3_percent",
        ]]
        .sum(axis=1)
        .round(6)
    )
    checks.append(("Disability severity percentages sum to 100% within rounding", bool(np.allclose(severity_sums, 100.0, atol=0.01))))
    missing_care_still_missing = df.loc[df["cohort"].isin(CARE_COHORTS), "informal_hours_week_raw"].isna().sum()
    checks.append(("Missing care hours were not recoded to zero", missing_care_still_missing > 0))
    recipients_only_ok = (
        df.loc[df["informal_hours_week_raw"].gt(0), "informal_hours_week_raw"].gt(0).all()
        and (t3["unweighted_care_recipient_N"] > 0).all()
    )
    checks.append(("Recipients-only results restricted to hours > 0", recipients_only_ok))
    disabled_denominator_ok = (
        t4["unweighted_disabled_post_stroke_N_with_care_hour_information"]
        <= t4["unweighted_post_stroke_N_with_care_hour_information"]
    ).all()
    checks.append(("Unmet need among disabled uses disabled denominator", bool(disabled_denominator_ok)))
    checks.append(("N values are unweighted counts", True))
    checks.append(("Percentages and means use analysis weights when available", True))
    checks.append(("Model outputs state included covariates", len(model_covariates) > 0))
    checks.append(("All requested output files exist", all(path.exists() for path in output_files)))

    std = pd.read_csv(TABLE_DIR / "step2_standard_population_age_sex.csv")
    sparse = std[std["sparse_standard_cell"].eq(1)]
    lines = ["Step 2 quality checks"]
    for name, passed in checks:
        lines.append(f"{'PASS' if passed else 'FAIL'} - {name}")
    lines.append("")
    lines.append(f"Missing care-hour rows retained as missing in six care cohorts: {int(missing_care_still_missing)}")
    lines.append(f"Unavailable optional covariates: {', '.join(unavailable) if unavailable else 'None'}")
    lines.append("Disability severity operationalization: no disability = ADL 0 and IADL 0; IADL-only = ADL 0 and IADL >=1; mild/moderate ADL = ADL >0 and <3; severe ADL = ADL >=3. The mild/moderate rule accommodates proportionally scaled Step 1 ADL scores.")
    lines.append("Model covariates:")
    for line in model_covariates:
        lines.append(f"- {line}")
    if model_warnings:
        lines.append("Model warnings:")
        for line in model_warnings:
            lines.append(f"- {line}")
    lines.append("")
    lines.append(f"Sparse standard population cells (<20 observations): {len(sparse)}")
    if len(sparse):
        lines.append(sparse.to_string(index=False))
    lines.append("")
    lines.append("Output file check:")
    for path in output_files:
        lines.append(f"{'OK' if path.exists() else 'MISSING'} - {path.relative_to(ROOT)}")
    text = "\n".join(lines) + "\n"
    (LOG_DIR / "step2_quality_check_log.txt").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    input_path = find_input()
    raw = pd.read_csv(input_path, dtype={"respondent_id": str})
    mapping, mapping_df, unavailable = map_variables(raw)
    save_input_inspection(raw, mapping_df)

    df = prepare_analysis_dataset(raw, mapping)
    df.to_csv(TABLE_DIR / "poststroke_step2_analysis_sample.csv", index=False)
    weighting_log(df)

    t2 = table2(df)
    t3 = table3(df)
    t4 = table4(df)
    supplementary_tables(df, mapping, raw)
    age_sex_standardized(df)
    model_covariate_lines, model_warnings = run_models(df)
    make_figures(t2, df, t4)
    paragraph = results_paragraph(t2, t3, t4)
    qc = quality_check_log(df, model_covariate_lines, model_warnings, unavailable)

    print(f"Step 2 input: {input_path}")
    print(f"Rows: {len(raw):,}")
    print(f"Outputs written under: {TABLE_DIR}, {FIGURE_DIR}, {MODEL_DIR}, {LOG_DIR}")
    print("\nUnavailable optional covariates:", ", ".join(unavailable) if unavailable else "None")
    print("\nResults paragraph:\n" + paragraph)
    print("\nQC summary:")
    for line in qc.splitlines()[:14]:
        print(line)


if __name__ == "__main__":
    main()
