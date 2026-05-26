from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "output" / "tables"
OUTPUT_DIR = ROOT / "results" / "step2b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_INPUT = INPUT_DIR / "gender_tax_protocol_latest_poststroke.csv"
PROTOCOL_MANIFEST = INPUT_DIR / "gender_tax_protocol_variable_manifest.csv"
GBD_COUNTRY_TABLE = INPUT_DIR / "gbd_2023_stroke_60plus_country_table.csv"
PCGI_LATEST = INPUT_DIR / "pcgi_latest_poststroke_dataset.csv"
STEP1_LATEST_WITH_HOURS = INPUT_DIR / "gender_tax_step1_latest_with_hours.csv"

CARE_COHORTS = ["CHARLS", "ELSA", "HRS", "KLoSA", "LASI", "MHAS"]
ALL_COHORTS = ["CHARLS", "ELSA", "HRS", "KLoSA", "LASI", "MHAS", "SHARE"]
SEX_LABELS = {0: "Male", 1: "Female"}
SEVERITY_LABELS = {
    0: "No disability",
    1: "IADL-only",
    2: "ADL 1-2",
    3: "ADL >=3",
}
OPTIONAL_COVARIATES = ["education", "wealth", "marital_status", "living_alone"]
GBD_LOCATION = {
    "CHARLS": "China",
    "ELSA": "United Kingdom",
    "HRS": "United States of America",
    "KLoSA": "Republic of Korea",
    "LASI": "India",
    "MHAS": "Mexico",
    "SHARE": None,
}
DISABILITY_WEIGHTS = {
    0: 0.0,
    1: 0.019,
    2: 0.070,
    3: 0.552,
}
DISABILITY_WEIGHTS_SENS_NO_DISABILITY_0019 = {
    0: 0.019,
    1: 0.019,
    2: 0.070,
    3: 0.552,
}


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


def weighted_sum(values: pd.Series, weights: pd.Series | None = None) -> float:
    v = to_numeric(values)
    mask = v.notna()
    if weights is not None:
        w = valid_weight(weights)
        mask = mask & w.notna()
        if mask.sum():
            return float((v.loc[mask] * w.loc[mask]).sum())
    if mask.sum() == 0:
        return np.nan
    return float(v.loc[mask].sum())


def weighted_binary_percent_se(values: pd.Series, weights: pd.Series | None = None) -> float:
    v = to_numeric(values)
    mask = v.isin([0, 1])
    if weights is None:
        n = int(mask.sum())
        if n <= 1:
            return np.nan
        p = float(v.loc[mask].mean())
        return float(np.sqrt(p * (1 - p) / n) * 100)
    w = valid_weight(weights)
    mask = mask & w.notna()
    if mask.sum() <= 1:
        return np.nan
    wv = w.loc[mask]
    p = float(np.average(v.loc[mask], weights=wv))
    n_eff = float((wv.sum() ** 2) / (wv.pow(2).sum())) if wv.pow(2).sum() > 0 else np.nan
    if pd.isna(n_eff) or n_eff <= 1:
        return np.nan
    return float(np.sqrt(p * (1 - p) / n_eff) * 100)


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


def map_variables(raw: pd.DataFrame) -> tuple[dict[str, str | None], pd.DataFrame]:
    columns = list(raw.columns)
    candidates = {
        "cohort": ["dataset", "cohort", "country"],
        "respondent_id": ["respondent_id", "id", "person_id", "hhidpn", "mergeid"],
        "wave": ["wave"],
        "age": ["age", "ragey", "r_age", "age_years"],
        "sex": ["sex", "gender", "ragender"],
        "stroke_indicator": ["stroke_ever", "stroke", "poststroke", "ever_stroke"],
        "adl_count": ["adl_score", "adl_count", "adl"],
        "iadl_count": ["iadl_score", "iadl_count", "iadl"],
        "disability_state": ["disability_state", "severe_adl_disability", "severe_disability"],
        "informal_care_hours_week": ["care_hours_week", "informal_hours_week", "informal_care_hours_week"],
        "care_intensity": ["care_intensity"],
        "unmet_need_objective": ["unmet_need_objective", "objective_unmet", "unmet_care_objective"],
        "weight": ["weight_cross", "wtresp", "weight", "survey_weight"],
        "adl_bath": ["adl_bath"],
        "adl_dress": ["adl_dress"],
        "adl_eat": ["adl_eat"],
        "adl_bed": ["adl_bed"],
        "adl_toilet": ["adl_toilet"],
        "adl_continence": ["adl_continence"],
        "iadl_phone": ["iadl_phone"],
        "iadl_money": ["iadl_money"],
        "iadl_meds": ["iadl_meds"],
        "iadl_shop": ["iadl_shop"],
        "iadl_meal": ["iadl_meal"],
        "education": ["education", "edu", "raeducl", "raeduc", "education_level"],
        "wealth": ["wealth", "wealth_quintile", "asset_quintile", "income_quintile"],
        "marital_status": ["marital", "marital_status", "married", "partnership"],
        "living_alone": ["living_alone", "alone", "household_size_one"],
        "helper_level_id": ["helper_id", "caregiver_id", "provider_id"],
        "helper_sex": ["helper_sex", "caregiver_sex", "provider_sex"],
        "helper_relationship": ["helper_relationship", "caregiver_relationship", "provider_relationship"],
        "helper_hours": ["helper_hours", "caregiver_hours", "provider_hours"],
        "helper_paid_status": ["helper_paid", "paid_helper", "provider_paid"],
        "formal_care_indicator": ["formal_care", "paid_care", "professional_care"],
        "informal_care_indicator": ["informal_care", "family_care", "unpaid_care"],
    }
    required = {
        "cohort",
        "respondent_id",
        "age",
        "sex",
        "stroke_indicator",
        "adl_count",
        "iadl_count",
        "informal_care_hours_week",
        "unmet_need_objective",
        "weight",
    }
    helper_roles = {"helper_level_id", "helper_sex", "helper_relationship", "helper_hours", "helper_paid_status"}
    mapping: dict[str, str | None] = {}
    rows = []
    for role, role_candidates in candidates.items():
        found = first_existing(columns, role_candidates)
        mapping[role] = found
        rows.append(
            {
                "variable_role": role,
                "mapped_column": found,
                "required_for_step2b": role in required,
                "helper_level_preflight_role": role in helper_roles,
                "status": "available" if found else "unavailable",
                "candidate_columns_checked": "; ".join(role_candidates),
                "note": "",
            }
        )
    availability = pd.DataFrame(rows)
    availability.loc[
        availability["variable_role"].isin(helper_roles) & availability["mapped_column"].isna(),
        "note",
    ] = "Unavailable in the Step 1 latest-poststroke file; Step 2b does not impute helper-level provider data."
    availability.to_csv(OUTPUT_DIR / "step2b_variable_availability.csv", index=False)
    return mapping, availability


def add_age_group(age: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=age.index, dtype="object")
    out.loc[age.between(50, 64, inclusive="both")] = "50-64"
    out.loc[age.between(65, 74, inclusive="both")] = "65-74"
    out.loc[age.between(75, 84, inclusive="both")] = "75-84"
    out.loc[age >= 85] = "85+"
    return out


def prepare_dataset(raw: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    required = ["cohort", "respondent_id", "age", "sex", "stroke_indicator", "adl_count", "iadl_count"]
    missing_required = [role for role in required if mapping.get(role) is None]
    if missing_required:
        raise ValueError(f"Required Step 2b variables unavailable: {missing_required}")

    df = pd.DataFrame(
        {
            "cohort": raw[mapping["cohort"]].astype(str),
            "respondent_id": raw[mapping["respondent_id"]].astype(str),
            "wave": to_numeric(raw[mapping["wave"]]) if mapping.get("wave") else np.nan,
            "age": to_numeric(raw[mapping["age"]]),
            "sex": to_numeric(raw[mapping["sex"]]),
            "stroke_ever": to_numeric(raw[mapping["stroke_indicator"]]),
            "adl_count": to_numeric(raw[mapping["adl_count"]]),
            "iadl_count": to_numeric(raw[mapping["iadl_count"]]),
            "informal_hours_week_raw": to_numeric(raw[mapping["informal_care_hours_week"]])
            if mapping.get("informal_care_hours_week")
            else np.nan,
            "unmet_need_objective_source": to_numeric(raw[mapping["unmet_need_objective"]])
            if mapping.get("unmet_need_objective")
            else np.nan,
            "original_weight": valid_weight(raw[mapping["weight"]]) if mapping.get("weight") else np.nan,
        }
    )

    for role in [
        "adl_bath",
        "adl_dress",
        "adl_eat",
        "adl_bed",
        "adl_toilet",
        "adl_continence",
        "iadl_phone",
        "iadl_money",
        "iadl_meds",
        "iadl_shop",
        "iadl_meal",
    ]:
        if mapping.get(role):
            df[role] = to_numeric(raw[mapping[role]])

    df["sex_label"] = df["sex"].map(SEX_LABELS).fillna("Unknown")
    df["age_group"] = add_age_group(df["age"])
    df["poststroke_eligible"] = (
        df["cohort"].isin(ALL_COHORTS) & df["age"].ge(50) & df["stroke_ever"].eq(1)
    ).astype(int)

    valid_adl_iadl = df["adl_count"].notna() & df["iadl_count"].notna()
    df["any_disability"] = np.where(valid_adl_iadl, ((df["adl_count"] > 0) | (df["iadl_count"] > 0)).astype(int), np.nan)
    df["severe_adl_disability"] = np.where(valid_adl_iadl, (df["adl_count"] >= 3).astype(int), np.nan)
    df["adl_care_need"] = np.where(df["adl_count"].notna(), (df["adl_count"] >= 1).astype(int), np.nan)

    severity = pd.Series(np.nan, index=df.index, dtype="float")
    severity.loc[valid_adl_iadl & df["adl_count"].eq(0) & df["iadl_count"].eq(0)] = 0
    severity.loc[valid_adl_iadl & df["adl_count"].eq(0) & df["iadl_count"].ge(1)] = 1
    severity.loc[valid_adl_iadl & df["adl_count"].gt(0) & df["adl_count"].lt(3)] = 2
    severity.loc[valid_adl_iadl & df["adl_count"].ge(3)] = 3
    df["disability_severity"] = severity
    df["disability_severity_label"] = df["disability_severity"].map(SEVERITY_LABELS)
    df["disability_weight_i"] = df["disability_severity"].map(DISABILITY_WEIGHTS)
    df["yld_stroke_i"] = df["disability_weight_i"]
    df["disability_weight_sens_no_disability_0019"] = df["disability_severity"].map(
        DISABILITY_WEIGHTS_SENS_NO_DISABILITY_0019
    )
    df["yld_stroke_i_sens_no_disability_0019"] = df["disability_weight_sens_no_disability_0019"]

    df["disability_known_sample"] = (
        df["poststroke_eligible"].eq(1) & valid_adl_iadl
    ).astype(int)
    # Step 2b denominator fix: care-hour estimates no longer require complete ADL/IADL.
    df["care_hour_known_sample"] = (
        df["cohort"].isin(CARE_COHORTS)
        & df["poststroke_eligible"].eq(1)
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
    for cohort in CARE_COHORTS:
        mask = df["cohort"].eq(cohort) & df["care_hour_known_sample"].eq(1)
        values = df.loc[mask, "informal_hours_week_raw"].dropna()
        cutoff = float(values.quantile(0.99)) if len(values) else np.nan
        p99_rows.append({"cohort": cohort, "p99_cutoff": cutoff, "care_hour_known_n": int(len(values))})
        if pd.notna(cutoff):
            df.loc[mask, "informal_hours_week_winsor"] = df.loc[mask, "informal_hours_week_raw"].clip(upper=cutoff)
    pd.DataFrame(p99_rows).to_csv(OUTPUT_DIR / "step2b_care_hour_p99_cutoffs.csv", index=False)

    df["objective_unmet"] = np.where(
        df["unmet_need_objective_source"].isin([0, 1]),
        df["unmet_need_objective_source"],
        np.nan,
    )
    df["unmet_known_sample"] = (
        df["cohort"].isin(CARE_COHORTS)
        & df["poststroke_eligible"].eq(1)
        & df["objective_unmet"].notna()
    ).astype(int)
    df["unmet_need_denominator_sample"] = (
        df["unmet_known_sample"].eq(1) & df["adl_care_need"].eq(1)
    ).astype(int)

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

    return df


def save_missingness(raw: pd.DataFrame, df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    role_rows = []
    roles = [
        "age",
        "sex",
        "weight",
        "adl_count",
        "iadl_count",
        "informal_care_hours_week",
        "unmet_need_objective",
    ]
    post = df[df["poststroke_eligible"].eq(1)].copy()
    for cohort in ALL_COHORTS:
        sub = post[post["cohort"].eq(cohort)]
        for role in roles:
            if role == "weight":
                col = "original_weight"
            elif role == "informal_care_hours_week":
                col = "informal_hours_week_raw"
            elif role == "unmet_need_objective":
                col = "objective_unmet"
            elif role == "adl_count":
                col = "adl_count"
            elif role == "iadl_count":
                col = "iadl_count"
            else:
                col = role
            missing_n = int(sub[col].isna().sum()) if col in sub else int(len(sub))
            role_rows.append(
                {
                    "cohort": cohort,
                    "variable_role": role,
                    "mapped_column": mapping.get(role),
                    "poststroke_eligible_n": int(len(sub)),
                    "missing_n": missing_n,
                    "missing_percent": safe_divide(missing_n, len(sub)) * 100,
                }
            )
    out = pd.DataFrame(role_rows)
    out.to_csv(OUTPUT_DIR / "step2b_missingness_by_cohort.csv", index=False)

    raw_missing = (
        raw.isna()
        .sum()
        .rename("missing_n")
        .to_frame()
        .assign(total_n=len(raw), missing_percent=lambda x: x["missing_n"] / x["total_n"] * 100)
        .reset_index()
        .rename(columns={"index": "column"})
    )
    raw_missing.to_csv(OUTPUT_DIR / "step2b_raw_column_missingness.csv", index=False)
    return out


def denominator_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in ALL_COHORTS:
        sub = df[df["cohort"].eq(cohort)]
        post = sub[sub["poststroke_eligible"].eq(1)]
        rows.append(
            {
                "cohort": cohort,
                "poststroke_eligible_n": int(len(post)),
                "disability_known_denominator_n": int(post["disability_known_sample"].sum()),
                "excluded_from_disability_missing_adl_or_iadl_n": int(
                    len(post) - post["disability_known_sample"].sum()
                ),
                "care_hour_known_denominator_n": int(post["care_hour_known_sample"].sum())
                if cohort in CARE_COHORTS
                else np.nan,
                "care_hour_denominator_requires_adl_iadl_complete": False if cohort in CARE_COHORTS else np.nan,
                "care_recipient_positive_hours_n": int(
                    ((post["care_hour_known_sample"].eq(1)) & post["informal_hours_week_raw"].gt(0)).sum()
                )
                if cohort in CARE_COHORTS
                else np.nan,
                "unmet_status_known_denominator_n": int(post["unmet_known_sample"].sum())
                if cohort in CARE_COHORTS
                else np.nan,
                "unmet_adl_need_denominator_n": int(post["unmet_need_denominator_sample"].sum())
                if cohort in CARE_COHORTS
                else np.nan,
                "excluded_from_care_unmet_because_no_direct_harmonized_hours": cohort not in CARE_COHORTS,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "step2b_denominator_audit.csv", index=False)
    return out


def sample_flow_table(raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in ALL_COHORTS:
        raw_sub = raw[raw["dataset"].astype(str).eq(cohort)] if "dataset" in raw.columns else pd.DataFrame()
        sub = df[df["cohort"].eq(cohort)]
        age = sub[sub["age"].ge(50)]
        stroke = age[age["stroke_ever"].eq(1)]
        adl_known = stroke["adl_count"].notna()
        iadl_known = stroke["iadl_count"].notna()
        weight_known = stroke["original_weight"].notna()
        missing_weight_n = int((~weight_known).sum())
        missing_weight_excluded_n = 0
        care_known_n = int(stroke["care_hour_known_sample"].sum()) if cohort in CARE_COHORTS else 0
        disability_n = int(stroke["disability_known_sample"].sum())
        unmet_gap_n = int(
            (
                stroke["disability_known_sample"].eq(1)
                & stroke["any_disability"].eq(1)
                & stroke["care_hour_known_sample"].eq(1)
            ).sum()
        ) if cohort in CARE_COHORTS else 0
        yld_n = disability_n

        notes = []
        if len(raw_sub) != len(stroke):
            notes.append(
                f"Closest reproducible Step 1 latest-poststroke denominator is raw latest-poststroke N={len(raw_sub)}; "
                f"age>=50 and stroke_ever==1 gives N={len(stroke)}."
            )
        if missing_weight_n:
            notes.append(
                f"Original survey weight missing for {missing_weight_n}; excluded due missing weights={missing_weight_excluded_n} "
                "because Step 2b retains them with unit analysis weights."
            )
        else:
            notes.append("Original survey weight available for all age-eligible stroke survivors; excluded due missing weights=0.")
        if cohort == "MHAS":
            neither = int((~adl_known & ~iadl_known).sum())
            adl_only = int((adl_known & ~iadl_known).sum())
            iadl_only = int((~adl_known & iadl_known).sum())
            notes.append(
                "MHAS drop from Step 1 raw latest-poststroke N=1248 to Step 2 disability N=812 is driven by the "
                f"ADL/IADL complete-case denominator after age eligibility: 3 were not age>=50; among 1245 age-eligible "
                f"stroke survivors, {adl_only} had ADL but missing IADL, {iadl_only} had IADL but missing ADL, and "
                f"{neither} had both ADL and IADL missing."
            )
        if cohort == "SHARE":
            source_hours = int(stroke["informal_hours_week_raw"].notna().sum())
            notes.append(
                f"SHARE has {source_hours} nonmissing care_hours_week values in the latest file, but the Step 1 manifest "
                "reports no directly harmonized care-hour variables; Step 2b therefore excludes SHARE from care-hour and informal-care-gap analyses. "
                "Survey-based post-stroke YLD remains included when disability severity is known."
            )
        if cohort in CARE_COHORTS:
            notes.append(
                "n_unmet_or_gap_analysis reports the informal-care-gap denominator: disabled post-stroke survivors "
                "with known directly harmonized informal care hours. Strict-unmet denominators are reported separately "
                "in table4a_strict_unmet_need.csv because they require PCGI care-source indicators."
            )

        rows.append(
            {
                "cohort": cohort,
                "n_raw_latest_wave": int(len(raw_sub)),
                "n_age_eligible": int(len(age)),
                "n_stroke_survivors": int(len(stroke)),
                "n_with_adl_iadl": int((adl_known & iadl_known).sum()),
                "n_with_analysis_weight": int(weight_known.sum()),
                "n_with_care_hour_variables": care_known_n,
                "n_disability_analysis": disability_n,
                "n_care_hour_analysis": care_known_n,
                "n_unmet_or_gap_analysis": unmet_gap_n,
                "n_yld_analysis": yld_n,
                "notes": " ".join(notes),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "table_s0_sample_flow.csv", index=False)
    return out


def table2(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in ALL_COHORTS:
        post = df[(df["cohort"].eq(cohort)) & df["poststroke_eligible"].eq(1)]
        dis = post[post["disability_known_sample"].eq(1)]
        care = post[post["care_hour_known_sample"].eq(1)]
        row = {
            "cohort": cohort,
            "unweighted_post_stroke_eligible_N": int(len(post)),
            "unweighted_disability_known_N": int(len(dis)),
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
                    "footnote": "SHARE excluded from care-hour estimates because directly harmonizable care-hour variables were unavailable in the Step 1 manifest.",
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
    out.to_csv(OUTPUT_DIR / "table2_disability_care_hour_distribution.csv", index=False)
    return out


def table3(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in CARE_COHORTS:
        base = df[(df["cohort"].eq(cohort)) & (df["care_hour_known_sample"].eq(1))]
        rec = base[base["informal_hours_week_raw"].gt(0)]
        rows.append(
            {
                "cohort": cohort,
                "unweighted_care_hour_known_N": int(len(base)),
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
    out.to_csv(OUTPUT_DIR / "table3_recipient_only_care_intensity.csv", index=False)
    return out


def table4(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in CARE_COHORTS:
        known = df[(df["cohort"].eq(cohort)) & (df["unmet_known_sample"].eq(1))]
        need = known[known["adl_care_need"].eq(1)]
        female = need[need["sex_label"].eq("Female")]
        male = need[need["sex_label"].eq("Male")]
        female_unmet = weighted_percent(female["objective_unmet"], female["analysis_weight"])
        male_unmet = weighted_percent(male["objective_unmet"], male["analysis_weight"])
        rows.append(
            {
                "cohort": cohort,
                "unweighted_post_stroke_N_with_unmet_status": int(len(known)),
                "unweighted_ADL_care_need_N_with_unmet_status": int(len(need)),
                "unmet_among_all_post_stroke_survivors_percent": weighted_percent(
                    known["objective_unmet"], known["analysis_weight"]
                ),
                "unmet_among_ADL_care_need_post_stroke_survivors_percent": weighted_percent(
                    need["objective_unmet"], need["analysis_weight"]
                ),
                "female_unmet_among_ADL_care_need_percent": female_unmet,
                "male_unmet_among_ADL_care_need_percent": male_unmet,
                "female_minus_male_unmet_difference": female_unmet - male_unmet
                if pd.notna(female_unmet) and pd.notna(male_unmet)
                else np.nan,
                "definition": "Uses source unmet_need_objective; denominator is ADL care need (ADL score >=1) with unmet status known.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "table4_objective_unmet_need.csv", index=False)
    return out


def load_pcgi_latest(df: pd.DataFrame) -> pd.DataFrame:
    if not PCGI_LATEST.exists():
        return pd.DataFrame()
    pcgi = pd.read_csv(PCGI_LATEST, dtype={"respondent_id": str})
    pcgi = pcgi.rename(columns={"dataset": "cohort"})
    keep = [
        "cohort",
        "respondent_id",
        "post_stroke",
        "any_care",
        "informal_care",
        "formal_care",
        "care_need",
        "pcgi_code",
        "pcgi_label",
        "education_level",
        "wealth",
        "wealth_tertile",
        "n_children",
        "coresident_child",
        "coupled",
        "rural",
        "share_country_code",
    ]
    keep = [col for col in keep if col in pcgi.columns]
    pcgi = pcgi[keep].copy()
    for col in ["post_stroke", "any_care", "informal_care", "formal_care", "care_need", "pcgi_code"]:
        if col in pcgi.columns:
            pcgi[col] = to_numeric(pcgi[col])
    weights = df[
        [
            "cohort",
            "respondent_id",
            "age_group",
            "sex_label",
            "poststroke_eligible",
            "any_disability",
            "analysis_weight",
        ]
    ].copy()
    merged = pcgi.merge(weights, on=["cohort", "respondent_id"], how="left")
    return merged


def unmet_variable_availability(df: pd.DataFrame, pcgi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in ALL_COHORTS:
        step2 = df[df["cohort"].eq(cohort) & df["poststroke_eligible"].eq(1)]
        p = pcgi[pcgi["cohort"].eq(cohort)] if not pcgi.empty else pd.DataFrame()
        need = p[
            p.get("post_stroke", pd.Series(dtype=float)).eq(1)
            & (
                p.get("care_need", pd.Series(dtype=float)).eq(1)
                | p.get("any_disability", pd.Series(dtype=float)).eq(1)
            )
        ] if not p.empty else pd.DataFrame()
        role_specs = [
            ("adl_iadl_disability", "Step 2b latest-poststroke", "any_disability"),
            ("recipient_informal_care_hours", "Step 2b latest-poststroke", "informal_hours_week_raw"),
            ("any_recorded_adl_iadl_help", "PCGI latest-poststroke", "any_care"),
            ("informal_care_indicator", "PCGI latest-poststroke", "informal_care"),
            ("formal_or_professional_care_indicator", "PCGI latest-poststroke", "formal_care"),
            ("paid_or_professional_care_indicator", "PCGI latest-poststroke", "formal_care"),
            ("care_need_indicator", "PCGI latest-poststroke", "care_need"),
            ("pcgi_strict_unmet_proxy", "PCGI latest-poststroke", "pcgi_code"),
        ]
        for role, source, col in role_specs:
            if source.startswith("Step 2b"):
                available = col in step2.columns and step2[col].notna().sum() > 0
                nonmissing = int(step2[col].notna().sum()) if col in step2.columns else 0
            else:
                available = col in p.columns and need[col].notna().sum() > 0 if not need.empty else False
                nonmissing = int(need[col].notna().sum()) if col in need.columns else 0
            decision = "available"
            note = ""
            if not available:
                decision = "unavailable"
                note = "Not constructible for strict unmet need in this cohort."
            if role == "paid_or_professional_care_indicator" and available:
                decision = "proxy_available"
                note = "No separate paid-care variable is present; formal_care is used as the paid/professional-care proxy."
            if role == "recipient_informal_care_hours" and cohort == "SHARE":
                available = False
                decision = "excluded_from_gap"
                note = "SHARE is excluded from care-hour and informal-care-gap analyses because the Step 1 manifest reports no directly harmonized care-hour variables."
            rows.append(
                {
                    "cohort": cohort,
                    "concept": "strict_unmet_need" if role != "recipient_informal_care_hours" else "informal_care_gap",
                    "variable_role": role,
                    "source_file": source,
                    "mapped_column": col if available or col in ["formal_care", "informal_hours_week_raw"] else "",
                    "available": bool(available),
                    "nonmissing_in_relevant_denominator_n": nonmissing,
                    "decision": decision,
                    "notes": note,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "table_s1_unmet_variable_availability.csv", index=False)
    return out


def table4a_strict_unmet_need(pcgi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in ALL_COHORTS:
        p = pcgi[pcgi["cohort"].eq(cohort)].copy() if not pcgi.empty else pd.DataFrame()
        notes = []
        if p.empty:
            rows.append(
                {
                    "cohort": cohort,
                    "n": np.nan,
                    "weighted_n": np.nan,
                    "denominator_definition": "Not constructible",
                    "numerator_definition": "Not constructible",
                    "crude_percent": np.nan,
                    "weighted_percent": np.nan,
                    "weighted_percent_se": np.nan,
                    "notes": "PCGI latest-poststroke file unavailable or cohort absent.",
                }
            )
            continue
        required = ["any_care", "informal_care", "formal_care", "care_need", "analysis_weight"]
        need_or_disability = p["post_stroke"].eq(1) & (p["care_need"].eq(1) | p["any_disability"].eq(1))
        need = p[need_or_disability].copy()
        unavailable = [col for col in required if col not in need.columns or need[col].notna().sum() == 0]
        if unavailable:
            rows.append(
                {
                    "cohort": cohort,
                    "n": np.nan,
                    "weighted_n": np.nan,
                    "denominator_definition": "Stroke survivors with ADL/IADL disability or reported care need and complete care-source indicators",
                    "numerator_definition": "No informal care, no formal/professional care, and no other recorded ADL/IADL help",
                    "crude_percent": np.nan,
                    "weighted_percent": np.nan,
                    "weighted_percent_se": np.nan,
                    "notes": f"Strict unmet need cannot be constructed because required variables are unavailable or all missing: {', '.join(unavailable)}.",
                }
            )
            continue
        complete = need.dropna(subset=["any_care", "informal_care", "formal_care", "analysis_weight"]).copy()
        missing_complete = int(len(need) - len(complete))
        strict = (
            complete["informal_care"].eq(0)
            & complete["formal_care"].eq(0)
            & complete["any_care"].eq(0)
        ).astype(float)
        if missing_complete:
            notes.append(f"Excluded {missing_complete} care-need/disability rows with incomplete care-source indicators or weights.")
        notes.append("formal_care is used as the available formal/professional and paid/professional proxy.")
        rows.append(
            {
                "cohort": cohort,
                "n": int(len(complete)),
                "weighted_n": float(valid_weight(complete["analysis_weight"]).sum()),
                "denominator_definition": "Stroke survivors with ADL/IADL disability or reported care need and complete care-source indicators",
                "numerator_definition": "No informal care, no formal/professional care, and no other recorded ADL/IADL help",
                "crude_percent": float(strict.mean() * 100) if len(strict) else np.nan,
                "weighted_percent": weighted_percent(strict, complete["analysis_weight"]),
                "weighted_percent_se": weighted_binary_percent_se(strict, complete["analysis_weight"]),
                "notes": " ".join(notes),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "table4a_strict_unmet_need.csv", index=False)
    return out


def table4b_informal_care_gap(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in ALL_COHORTS:
        notes = []
        if cohort == "SHARE":
            rows.append(
                {
                    "cohort": cohort,
                    "n": np.nan,
                    "weighted_n": np.nan,
                    "denominator_definition": "Not applicable: directly harmonizable care-hour variables unavailable",
                    "numerator_definition": "Not applicable",
                    "crude_percent": np.nan,
                    "weighted_percent": np.nan,
                    "weighted_percent_se": np.nan,
                    "notes": "SHARE retained in disability analyses but excluded from care-hour and informal-care-gap analyses.",
                }
            )
            continue
        base = df[
            df["cohort"].eq(cohort)
            & df["disability_known_sample"].eq(1)
            & df["any_disability"].eq(1)
            & df["care_hour_known_sample"].eq(1)
        ].copy()
        missing_hours = int(
            (
                df["cohort"].eq(cohort)
                & df["disability_known_sample"].eq(1)
                & df["any_disability"].eq(1)
                & df["informal_hours_week_raw"].isna()
            ).sum()
        )
        gap = base["informal_hours_week_raw"].eq(0).astype(float)
        if missing_hours:
            notes.append(f"{missing_hours} disabled stroke survivors had missing care hours and were not classified as gap.")
        rows.append(
            {
                "cohort": cohort,
                "n": int(len(base)),
                "weighted_n": float(valid_weight(base["analysis_weight"]).sum()),
                "denominator_definition": "Post-stroke survivors with ADL/IADL disability and known directly harmonized informal care hours",
                "numerator_definition": "No recorded informal care hours (care_hours_week == 0)",
                "crude_percent": float(gap.mean() * 100) if len(gap) else np.nan,
                "weighted_percent": weighted_percent(gap, base["analysis_weight"]),
                "weighted_percent_se": weighted_binary_percent_se(gap, base["analysis_weight"]),
                "notes": " ".join(notes),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "table4b_informal_care_gap.csv", index=False)
    return out


def add_covariates_to_analysis(df: pd.DataFrame, pcgi: pd.DataFrame) -> pd.DataFrame:
    if pcgi.empty:
        return df
    covars = [
        "education_level",
        "wealth",
        "wealth_tertile",
        "n_children",
        "coresident_child",
        "coupled",
        "rural",
        "share_country_code",
    ]
    keep = ["cohort", "respondent_id", *[col for col in covars if col in pcgi.columns]]
    cov = pcgi[keep].drop_duplicates(["cohort", "respondent_id"])
    out = df.merge(cov, on=["cohort", "respondent_id"], how="left")
    if "coupled" in out.columns:
        out["marital_status_proxy"] = out["coupled"]
    return out


def care_hour_missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    post = df[df["poststroke_eligible"].eq(1)].copy()
    post["severity_for_missingness"] = post["disability_severity_label"].fillna("ADL/IADL missing")
    for (cohort, sex, severity), sub in post.groupby(["cohort", "sex_label", "severity_for_missingness"], dropna=False):
        care_available = cohort in CARE_COHORTS
        missing = int(sub["informal_hours_week_raw"].isna().sum()) if care_available else int(len(sub))
        zero = int(sub["informal_hours_week_raw"].eq(0).sum()) if care_available else np.nan
        positive = int(sub["informal_hours_week_raw"].gt(0).sum()) if care_available else np.nan
        notes = []
        if not care_available:
            notes.append("No directly harmonizable care-hour variables; SHARE excluded from care-hour analyses.")
        else:
            notes.append("Missing care hours retained as missing; zero hours counted separately.")
        rows.append(
            {
                "cohort": cohort,
                "sex": sex,
                "disability_severity": severity,
                "n_stroke_survivors": int(len(sub)),
                "n_missing_care_hours": missing,
                "percent_missing_care_hours": safe_divide(missing, len(sub)) * 100,
                "n_zero_care_hours": zero,
                "n_positive_care_hours": positive,
                "notes": " ".join(notes),
            }
        )
    out = pd.DataFrame(rows).sort_values(["cohort", "sex", "disability_severity"])
    out.to_csv(OUTPUT_DIR / "table_s2_care_hour_missingness.csv", index=False)
    return out


def care_hour_conversion_note(cohort: str) -> str:
    if not PROTOCOL_MANIFEST.exists():
        return "Conversion metadata unavailable; Step 2b uses weekly care_hours_week from the harmonized Step 1 protocol file."
    manifest = pd.read_csv(PROTOCOL_MANIFEST)
    sub = manifest[manifest["dataset"].eq(cohort)]
    variables = "; ".join(sub["care_hour_variables"].dropna().astype(str).tolist())
    if not variables:
        return "No directly harmonizable care-hour variables in Step 1 manifest."
    if "hpw" in variables:
        return "Already reported as hours per week in the harmonized source variables."
    if "hr" in variables and "dpm" in variables:
        return "Converted from hours/day times days/month to weekly hours using 52.1775/12 weeks per month in Step 1."
    return "Step 2b uses weekly care_hours_week from the harmonized Step 1 protocol file."


def care_hour_p99_cutoff_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in ALL_COHORTS:
        if cohort not in CARE_COHORTS:
            rows.append(
                {
                    "cohort": cohort,
                    "p95_hours_week": np.nan,
                    "p99_hours_week": np.nan,
                    "max_hours_week": np.nan,
                    "n_above_168_hours_week": np.nan,
                    "action_taken": "Excluded from care-hour analyses because harmonizable weekly hours are unavailable.",
                }
            )
            continue
        vals = df.loc[df["cohort"].eq(cohort) & df["care_hour_known_sample"].eq(1), "informal_hours_week_raw"].dropna()
        rows.append(
            {
                "cohort": cohort,
                "p95_hours_week": float(vals.quantile(0.95)) if len(vals) else np.nan,
                "p99_hours_week": float(vals.quantile(0.99)) if len(vals) else np.nan,
                "max_hours_week": float(vals.max()) if len(vals) else np.nan,
                "n_above_168_hours_week": int(vals.gt(168).sum()) if len(vals) else 0,
                "action_taken": f"Raw weekly hours retained; sensitivity tables compare no winsorization, p99, p95, excluding >168, and capping >168. {care_hour_conversion_note(cohort)}",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "table_s3_care_hour_p99_cutoffs.csv", index=False)
    return out


def care_hour_sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scenarios = [
        ("no winsorization", lambda s: s),
        ("p99 winsorization", lambda s: s.clip(upper=s.quantile(0.99)) if s.notna().sum() else s),
        ("p95 winsorization", lambda s: s.clip(upper=s.quantile(0.95)) if s.notna().sum() else s),
        ("excluding >168 hours/week", lambda s: s.where(s <= 168)),
        ("capping >168 hours/week at 168", lambda s: s.clip(upper=168)),
    ]
    for cohort in ALL_COHORTS:
        if cohort not in CARE_COHORTS:
            rows.append(
                {
                    "cohort": cohort,
                    "sensitivity_scenario": "not applicable",
                    "mean_hours_week_among_all_stroke_survivors": np.nan,
                    "mean_hours_week_among_recipients_only": np.nan,
                    "median_hours_week_among_recipients_only": np.nan,
                    "p75_hours_week_among_recipients_only": np.nan,
                    "p90_hours_week_among_recipients_only": np.nan,
                    "notes": "SHARE excluded from care-hour sensitivity analyses because harmonizable weekly hours are unavailable.",
                }
            )
            continue
        base = df[df["cohort"].eq(cohort) & df["care_hour_known_sample"].eq(1)].copy()
        for scenario, transform in scenarios:
            hours = transform(base["informal_hours_week_raw"])
            rec_mask = hours.gt(0)
            rec = base.loc[rec_mask].copy()
            rec_hours = hours.loc[rec_mask]
            rows.append(
                {
                    "cohort": cohort,
                    "sensitivity_scenario": scenario,
                    "mean_hours_week_among_all_stroke_survivors": weighted_mean(hours, base["analysis_weight"]),
                    "mean_hours_week_among_recipients_only": weighted_mean(rec_hours, rec["analysis_weight"]),
                    "median_hours_week_among_recipients_only": weighted_quantile(rec_hours, 0.50, rec["analysis_weight"]),
                    "p75_hours_week_among_recipients_only": weighted_quantile(rec_hours, 0.75, rec["analysis_weight"]),
                    "p90_hours_week_among_recipients_only": weighted_quantile(rec_hours, 0.90, rec["analysis_weight"]),
                    "notes": f"Missing care hours not recoded to zero. Recipient-only estimates restrict to transformed hours > 0. {care_hour_conversion_note(cohort)} Hours are recipient-level total hours, not helper-level provider hours.",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "table_s4_care_hour_sensitivity.csv", index=False)
    return out


def covariate_availability_table(df: pd.DataFrame, pcgi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    post = df[df["poststroke_eligible"].eq(1)].copy()
    domains = [
        ("education", "education_level", "education_level", "PCGI processing reads raeducl/raeduc variants where available."),
        ("marital status", "coupled", "marital_status_proxy", "Available as coupled/partnered proxy, not full marital-status categories."),
        ("living alone", "", "", "No harmonized living-alone variable found in current Step 1/PCGI outputs."),
        ("household size", "", "", "No harmonized household-size variable found in current Step 1/PCGI outputs."),
        ("wealth or assets", "wealth; wealth_tertile", "wealth; wealth_tertile", "PCGI processing reads h* asset/income totals and derives cohort wealth tertiles."),
        ("number of children", "n_children", "n_children", "PCGI processing reads h*child/r*child where available."),
        ("health insurance or long-term care insurance", "", "", "Insurance variables appear in some raw/code paths but are not harmonized into Step 2b analytic data."),
        ("comorbidities", "", "", "Comorbidity variables appear in some CHARLS-specific stroke scripts but are not harmonized across Step 2b cohorts."),
        ("self-rated health", "", "", "No harmonized self-rated health variable found in current Step 1/PCGI outputs."),
        ("age", "age", "age", "Harmonized age in years."),
        ("sex", "sex", "sex", "Harmonized as 0 male / 1 female, with sex_label added."),
        ("survey weight", "weight_cross", "analysis_weight", "Original survey weights used when available; missing weights retained with unit analysis weights."),
    ]
    for cohort in ALL_COHORTS:
        sub = post[post["cohort"].eq(cohort)]
        for domain, found, harmonized, note in domains:
            cols = [c.strip() for c in harmonized.split(";") if c.strip()]
            available = bool(cols) and any(c in sub.columns and sub[c].notna().sum() > 0 for c in cols)
            if domain == "survey weight":
                miss = sub["original_weight"].isna().mean() * 100 if len(sub) else np.nan
                available = sub["original_weight"].notna().sum() > 0
            elif available:
                primary = cols[0]
                miss = sub[primary].isna().mean() * 100 if primary in sub.columns and len(sub) else np.nan
            else:
                miss = np.nan
            rows.append(
                {
                    "cohort": cohort,
                    "covariate_domain": domain,
                    "variable_name_found": found if available or found else "",
                    "harmonized_variable_name": harmonized if available or harmonized else "",
                    "available_yes_no": "yes" if available else "no",
                    "coding_notes": note,
                    "missing_percent": miss,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "table_s5_covariate_availability.csv", index=False)
    return out


def supp_sex_stratified(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in ALL_COHORTS:
        for sex in ["Male", "Female"]:
            post = df[(df["cohort"].eq(cohort)) & (df["sex_label"].eq(sex)) & (df["poststroke_eligible"].eq(1))]
            dis = post[post["disability_known_sample"].eq(1)]
            care = post[post["care_hour_known_sample"].eq(1)]
            need = post[post["unmet_need_denominator_sample"].eq(1)]
            row = {
                "cohort": cohort,
                "sex": sex,
                "poststroke_eligible_N": int(len(post)),
                "disability_known_N": int(len(dis)),
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
                        "unmet_among_ADL_care_need_percent": np.nan,
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
                        "unmet_among_ADL_care_need_percent": weighted_percent(need["objective_unmet"], need["analysis_weight"]),
                    }
                )
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "supp_table_s2_sex_stratified.csv", index=False)
    return out


def standard_population(df: pd.DataFrame, cohorts: list[str], sample_col: str, scope: str) -> pd.DataFrame:
    base = df[
        df["cohort"].isin(cohorts)
        & df[sample_col].eq(1)
        & df["age_group"].notna()
        & df["sex_label"].isin(["Male", "Female"])
    ].copy()
    rows = []
    total_weight = valid_weight(base["analysis_weight"]).fillna(1.0).sum()
    for age_group in ["50-64", "65-74", "75-84", "85+"]:
        for sex in ["Male", "Female"]:
            sub = base[(base["age_group"].eq(age_group)) & (base["sex_label"].eq(sex))]
            weighted_n = valid_weight(sub["analysis_weight"]).fillna(1.0).sum()
            rows.append(
                {
                    "standard_scope": scope,
                    "source_sample": sample_col,
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
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cell_rows = []
    std_key = std[["age_group", "sex", "standard_weight"]].copy()
    for cohort in cohorts:
        base = df[(df["cohort"].eq(cohort)) & df[sample_col].eq(1)]
        values = []
        missing_cells = 0
        sparse_cells = 0
        for _, std_row in std_key.iterrows():
            cell = base[(base["age_group"].eq(std_row["age_group"])) & (base["sex_label"].eq(std_row["sex"]))]
            if len(cell) < 20:
                sparse_cells += 1
            if len(cell) == 0 or cell[outcome_col].notna().sum() == 0:
                missing_cells += 1
                estimate = np.nan
            elif statistic == "percent":
                estimate = weighted_percent(cell[outcome_col], cell["analysis_weight"])
            elif statistic == "mean":
                estimate = weighted_mean(cell[outcome_col], cell["analysis_weight"])
            else:
                raise ValueError(statistic)
            values.append((estimate, std_row["standard_weight"]))
            cell_rows.append(
                {
                    "cohort": cohort,
                    "estimate": label,
                    "sample_col": sample_col,
                    "age_group": std_row["age_group"],
                    "sex": std_row["sex"],
                    "cell_unweighted_n": int(len(cell)),
                    "cell_nonmissing_outcome_n": int(cell[outcome_col].notna().sum()) if len(cell) else 0,
                    "cell_estimate": estimate,
                    "standard_weight": std_row["standard_weight"],
                }
            )
        vals = pd.DataFrame(values, columns=["estimate", "standard_weight"]).dropna()
        standardized = np.nan
        if len(vals) and vals["standard_weight"].sum() > 0:
            standardized = float((vals["estimate"] * vals["standard_weight"]).sum() / vals["standard_weight"].sum())
        rows.append(
            {
                "cohort": cohort,
                "estimate": label,
                "statistic": statistic,
                "standardized_value": standardized,
                "standard_cells_missing": int(missing_cells),
                "sparse_cells_lt20": int(sparse_cells),
                "observed_unweighted_n": int(len(base)),
                "standard_scope": std["standard_scope"].iloc[0],
                "denominator_sample": sample_col,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(cell_rows)


def strict_unmet_standardization_dataset(pcgi: pd.DataFrame) -> pd.DataFrame:
    if pcgi.empty:
        return pd.DataFrame()
    required = ["any_care", "informal_care", "formal_care", "care_need", "analysis_weight", "age_group", "sex_label"]
    if any(col not in pcgi.columns for col in required):
        return pd.DataFrame()
    data = pcgi[
        pcgi["post_stroke"].eq(1)
        & (pcgi["care_need"].eq(1) | pcgi["any_disability"].eq(1))
    ].copy()
    data = data.dropna(subset=required)
    if data.empty:
        return data
    data["strict_unmet_need"] = (
        data["informal_care"].eq(0)
        & data["formal_care"].eq(0)
        & data["any_care"].eq(0)
    ).astype(float)
    return data


def estimate_with_weights(data: pd.DataFrame, value_col: str, statistic: str) -> float:
    if data.empty or value_col not in data.columns:
        return np.nan
    if statistic == "percent":
        return weighted_percent(data[value_col], data["analysis_weight"])
    if statistic == "mean":
        return weighted_mean(data[value_col], data["analysis_weight"])
    raise ValueError(statistic)


def standardize_indicator(
    data: pd.DataFrame,
    cohorts: list[str],
    indicator: str,
    value_col: str,
    statistic: str,
    std: pd.DataFrame,
    standard_population_name: str,
    base_note: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cell_rows = []
    std_key = std[["age_group", "sex", "standard_weight"]].copy()
    for cohort in cohorts:
        base = data[data["cohort"].eq(cohort)].copy()
        crude = estimate_with_weights(base, value_col, statistic)
        cell_estimates = []
        included_cells = 0
        excluded_cells = 0
        sparse_cells = 0
        for _, std_row in std_key.iterrows():
            cell = base[
                base["age_group"].eq(std_row["age_group"])
                & base["sex_label"].eq(std_row["sex"])
            ]
            nonmissing = int(cell[value_col].notna().sum()) if value_col in cell.columns else 0
            if nonmissing == 0:
                excluded_cells += 1
                estimate = np.nan
            else:
                included_cells += 1
                if len(cell) < 20:
                    sparse_cells += 1
                estimate = estimate_with_weights(cell, value_col, statistic)
            cell_estimates.append((estimate, std_row["standard_weight"]))
            cell_rows.append(
                {
                    "cohort": cohort,
                    "indicator": indicator,
                    "age_group": std_row["age_group"],
                    "sex": std_row["sex"],
                    "cell_unweighted_n": int(len(cell)),
                    "cell_nonmissing_outcome_n": nonmissing,
                    "cell_estimate": estimate,
                    "standard_weight": std_row["standard_weight"],
                    "sparse_cell_lt20": int(nonmissing > 0 and len(cell) < 20),
                }
            )
        vals = pd.DataFrame(cell_estimates, columns=["estimate", "standard_weight"]).dropna()
        standardized = np.nan
        if len(vals) and vals["standard_weight"].sum() > 0:
            standardized = float((vals["estimate"] * vals["standard_weight"]).sum() / vals["standard_weight"].sum())
        notes = [base_note]
        if excluded_cells:
            notes.append(f"{excluded_cells} age-sex cells had no estimable observations and were excluded from direct standardization.")
        if sparse_cells:
            notes.append(f"{sparse_cells} included age-sex cells had fewer than 20 observations; estimates are flagged as sparse.")
        rows.append(
            {
                "cohort": cohort,
                "indicator": indicator,
                "crude_estimate": crude,
                "standardized_estimate": standardized,
                "standard_population": standard_population_name,
                "included_age_sex_cells": included_cells,
                "excluded_age_sex_cells": excluded_cells,
                "notes": " ".join(notes),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(cell_rows)


def age_sex_standardized(df: pd.DataFrame, pcgi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    standard_name = "Pooled weighted age-sex distribution of seven-cohort post-stroke disability-analysis sample"
    std = standard_population(df, ALL_COHORTS, "disability_known_sample", "disability_known_7cohort")
    std["standard_population"] = standard_name
    std.to_csv(OUTPUT_DIR / "step2b_standard_population_age_sex.csv", index=False)

    disability = df[df["disability_known_sample"].eq(1)].copy()
    care_all = df[df["care_hour_known_sample"].eq(1)].copy()
    care_recipients = care_all[care_all["informal_hours_week_raw"].gt(0)].copy()
    gap = df[
        df["cohort"].isin(CARE_COHORTS)
        & df["disability_known_sample"].eq(1)
        & df["any_disability"].eq(1)
        & df["care_hour_known_sample"].eq(1)
    ].copy()
    gap["informal_care_gap"] = gap["informal_hours_week_raw"].eq(0).astype(float)
    strict = strict_unmet_standardization_dataset(pcgi)

    specs = [
        (disability, ALL_COHORTS, "any ADL/IADL disability", "any_disability", "percent", "Disability indicator; SHARE included."),
        (disability, ALL_COHORTS, "ADL >=3", "severe_adl_disability", "percent", "Severe ADL disability indicator; SHARE included."),
        (disability, ALL_COHORTS, "mean disability weight / mean YLD", "yld_stroke_i", "mean", "Survey-based post-stroke YLD-like mean disability weight; not GBD YLD."),
        (care_all, CARE_COHORTS, "any informal care", "any_informal_care", "percent", "Care-hour indicator; SHARE excluded because directly harmonizable hours are unavailable."),
        (care_all, CARE_COHORTS, "mean informal care hours per week among all stroke survivors", "informal_hours_week_raw", "mean", "Mean recipient-level informal care hours among post-stroke survivors with known harmonized hours."),
        (care_recipients, CARE_COHORTS, "mean informal care hours per week among recipients only", "informal_hours_week_raw", "mean", "Recipient-only mean restricted to informal_hours_week_raw > 0."),
        (gap, CARE_COHORTS, "informal care gap", "informal_care_gap", "percent", "Informal care gap; denominator is disabled post-stroke survivors with known harmonized informal care hours."),
    ]
    if strict.empty:
        strict_rows = pd.DataFrame(
            [
                {
                    "cohort": cohort,
                    "indicator": "strict unmet need",
                    "crude_estimate": np.nan,
                    "standardized_estimate": np.nan,
                    "standard_population": standard_name,
                    "included_age_sex_cells": 0,
                    "excluded_age_sex_cells": 8,
                    "notes": "Strict unmet need is not constructible because care-source variables are unavailable.",
                }
                for cohort in ALL_COHORTS
            ]
        )
        strict_cells = pd.DataFrame()
    else:
        strict_rows, strict_cells = standardize_indicator(
            strict,
            ALL_COHORTS,
            "strict unmet need",
            "strict_unmet_need",
            "percent",
            std,
            standard_name,
            "Strict unmet need uses PCGI care-source indicators; formal_care is the paid/professional-care proxy where no separate paid-care variable exists.",
        )

    estimate_frames = []
    cell_frames = []
    for data, cohorts, indicator, value_col, statistic, note in specs:
        estimates, cells = standardize_indicator(data, cohorts, indicator, value_col, statistic, std, standard_name, note)
        estimate_frames.append(estimates)
        cell_frames.append(cells)
    estimate_frames.append(strict_rows)
    if not strict_cells.empty:
        cell_frames.append(strict_cells)

    out = pd.concat(estimate_frames, ignore_index=True)
    cells = pd.concat(cell_frames, ignore_index=True) if cell_frames else pd.DataFrame()
    out.to_csv(OUTPUT_DIR / "table6_age_sex_standardized_estimates.csv", index=False)
    out.to_csv(OUTPUT_DIR / "step2b_age_sex_standardized_estimates.csv", index=False)
    cells.to_csv(OUTPUT_DIR / "step2b_standardization_cell_estimates.csv", index=False)
    return out, std


def survey_yld_summary(df: pd.DataFrame, by_sex: bool = False, sensitivity: bool = False) -> pd.DataFrame:
    value_col = "yld_stroke_i_sens_no_disability_0019" if sensitivity else "yld_stroke_i"
    group_cols = ["cohort", "sex_label"] if by_sex else ["cohort"]
    rows = []
    data = df[df["disability_known_sample"].eq(1)].copy()
    for keys, sub in data.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {"cohort": keys[0]}
        if by_sex:
            row["sex"] = keys[1]
        values = sub[value_col]
        weights = sub["analysis_weight"]
        weighted_n = float(valid_weight(weights).sum())
        mean_weighted = weighted_mean(values, weights)
        notes = [
            "Survey-based post-stroke YLD-like disability burden estimate; not GBD YLD.",
            "Individual yld_stroke_i equals the disability-severity weight.",
        ]
        if sensitivity:
            notes.append("Sensitivity assigns no-disability stroke survivors DW=0.019 instead of 0.")
        notes.append("YLD_per_100000_older_adults is NA because the Step 2b input contains post-stroke survivors only, not a full older-adult population denominator.")
        row.update(
            {
                "n": int(len(sub)),
                "weighted_n": weighted_n,
                "mean_DW_unweighted": float(values.mean()) if values.notna().sum() else np.nan,
                "mean_DW_weighted": mean_weighted,
                "total_weighted_YLD_among_stroke_survivors": weighted_sum(values, weights),
                "YLD_per_1000_stroke_survivors": mean_weighted * 1000 if pd.notna(mean_weighted) else np.nan,
                "YLD_per_100000_older_adults": np.nan,
                "notes": " ".join(notes),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if by_sex:
        out = out[[
            "cohort",
            "sex",
            "n",
            "weighted_n",
            "mean_DW_unweighted",
            "mean_DW_weighted",
            "total_weighted_YLD_among_stroke_survivors",
            "YLD_per_1000_stroke_survivors",
            "YLD_per_100000_older_adults",
            "notes",
        ]]
    else:
        out = out[[
            "cohort",
            "n",
            "weighted_n",
            "mean_DW_unweighted",
            "mean_DW_weighted",
            "total_weighted_YLD_among_stroke_survivors",
            "YLD_per_1000_stroke_survivors",
            "YLD_per_100000_older_adults",
            "notes",
        ]]
    return out.sort_values(group_cols if not by_sex else ["cohort", "sex"]).reset_index(drop=True)


def helper_level_preflight(df: pd.DataFrame) -> pd.DataFrame:
    manifest = pd.DataFrame()
    if PROTOCOL_MANIFEST.exists():
        manifest = pd.read_csv(PROTOCOL_MANIFEST)
    rows = []
    for cohort in ALL_COHORTS:
        sub = df[df["cohort"].eq(cohort) & df["poststroke_eligible"].eq(1)]
        m = manifest[manifest["dataset"].eq(cohort)] if not manifest.empty else pd.DataFrame()
        care_hour_waves = int((m["care_hour_variables_found"] > 0).sum()) if not m.empty else np.nan
        care_vars = "; ".join(
            sorted(
                set(
                    "; ".join(m["care_hour_variables"].dropna().astype(str).tolist())
                    .replace(", ", "; ")
                    .split("; ")
                )
                - {""}
            )
        ) if not m.empty and "care_hour_variables" in m else ""
        rows.append(
            {
                "cohort": cohort,
                "poststroke_eligible_n": int(len(sub)),
                "step1_waves_with_care_hour_variables": care_hour_waves,
                "care_hour_variables_seen_upstream": care_vars,
                "recipient_level_care_hours_available_for_step2b": cohort in CARE_COHORTS,
                "helper_level_identifier_available": False,
                "helper_sex_available": False,
                "helper_relationship_available": False,
                "helper_specific_hours_available": False,
                "helper_level_ready_for_provider_burden": False,
                "step2b_decision": "Use recipient-level care hours only"
                if cohort in CARE_COHORTS
                else "Exclude from care-hour and informal-care-gap estimates",
                "blocking_issue": "No helper-level/provider-level table or helper identifiers in the Step 1 latest-poststroke analytic file.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "step2b_helper_level_preflight.csv", index=False)
    return out


def helper_variable_availability_table(df: pd.DataFrame, pcgi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in ALL_COHORTS:
        p = pcgi[pcgi["cohort"].eq(cohort)] if not pcgi.empty else pd.DataFrame()
        has_formal_informal = (
            not p.empty
            and "informal_care" in p.columns
            and "formal_care" in p.columns
            and (p["informal_care"].notna().sum() > 0 or p["formal_care"].notna().sum() > 0)
        )
        has_paid = "formal_care" in p.columns and p["formal_care"].notna().sum() > 0 if not p.empty else False
        main_limitation = "provider_sex unavailable; helper-level hours unavailable"
        notes = [
            "Current Step 2b inputs contain recipient-level care indicators/hours, not one row per helper-care-recipient pair.",
            "Provider sex is the key Step 3 variable and is not present.",
            "Do not infer provider sex from patient sex or relationship.",
        ]
        rows.append(
            {
                "cohort": cohort,
                "has_helper_level_data": "no",
                "has_helper_id": "no",
                "has_helper_sex": "no",
                "has_helper_relation": "no",
                "has_helper_hours": "no",
                "has_formal_informal_indicator": "yes" if has_formal_informal else "no",
                "has_paid_indicator": "proxy" if has_paid else "no",
                "can_build_gender_care_matrix": "no",
                "main_limitation": main_limitation,
                "notes": " ".join(notes),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "table_s6_helper_variable_availability.csv", index=False)
    return out


def step3_helper_long_format_preflight(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in ALL_COHORTS:
        sub = df[df["cohort"].eq(cohort) & df["poststroke_eligible"].eq(1)]
        rows.append(
            {
                "cohort": cohort,
                "patient_id": pd.NA,
                "helper_id": pd.NA,
                "patient_sex": pd.NA,
                "patient_age": np.nan,
                "patient_age_group": pd.NA,
                "provider_sex": pd.NA,
                "provider_age": np.nan,
                "relation_to_patient": pd.NA,
                "informal_care": np.nan,
                "formal_care": np.nan,
                "paid_care": np.nan,
                "care_type": pd.NA,
                "hours_week": np.nan,
                "hours_year": np.nan,
                "disability_severity": pd.NA,
                "yld_stroke": np.nan,
                "patient_weight": np.nan,
                "source_variable_notes": (
                    f"{cohort}: no helper-level provider-sex and helper-level hour data in current harmonized Step 2b inputs. "
                    f"Age-eligible stroke survivors available at patient level: {len(sub)}. Recipient-level total hours are not treated as helper-level hours."
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "step3_helper_long_format_preflight.csv", index=False)
    return out


def preliminary_gender_care_matrix(helper_availability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in ALL_COHORTS:
        avail = helper_availability[helper_availability["cohort"].eq(cohort)]
        can_build = (
            not avail.empty
            and avail["has_helper_sex"].iloc[0] == "yes"
            and avail["has_helper_hours"].iloc[0] == "yes"
            and avail["can_build_gender_care_matrix"].iloc[0] in ["yes", "partial"]
        )
        if not can_build:
            rows.append(
                {
                    "cohort": cohort,
                    "n_patient_helper_pairs": np.nan,
                    "n_unique_stroke_survivors_with_helper_data": np.nan,
                    "H_MM_weekly": np.nan,
                    "H_MF_weekly": np.nan,
                    "H_FM_weekly": np.nan,
                    "H_FF_weekly": np.nan,
                    "H_total_weekly": np.nan,
                    "H_MM_annual": np.nan,
                    "H_MF_annual": np.nan,
                    "H_FM_annual": np.nan,
                    "H_FF_annual": np.nan,
                    "H_total_annual": np.nan,
                    "female_provider_share": np.nan,
                    "female_to_male_patient_share": np.nan,
                    "female_to_female_patient_share": np.nan,
                    "notes": "Not generated for this cohort because provider sex and helper-level hours are not both available. No hidden gender care tax is calculated.",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "table7_preliminary_gender_care_matrix.csv", index=False)
    return out


def results_paragraph(table2_df: pd.DataFrame, gap_df: pd.DataFrame, survey_yld_df: pd.DataFrame) -> str:
    severe_min = table2_df["ADL_ge3_percent"].min()
    severe_max = table2_df["ADL_ge3_percent"].max()
    severe_min_cohort = table2_df.loc[table2_df["ADL_ge3_percent"].idxmin(), "cohort"]
    severe_max_cohort = table2_df.loc[table2_df["ADL_ge3_percent"].idxmax(), "cohort"]
    care_rows = table2_df[table2_df["cohort"].isin(CARE_COHORTS)]
    any_min = care_rows["any_informal_care_percent"].min()
    any_max = care_rows["any_informal_care_percent"].max()
    gap_valid = gap_df[gap_df["weighted_percent"].notna()]
    gap_min = gap_valid["weighted_percent"].min()
    gap_max = gap_valid["weighted_percent"].max()
    gap_min_cohort = gap_valid.loc[gap_valid["weighted_percent"].idxmin(), "cohort"]
    gap_max_cohort = gap_valid.loc[gap_valid["weighted_percent"].idxmax(), "cohort"]
    yld_min = survey_yld_df["mean_DW_weighted"].min()
    yld_max = survey_yld_df["mean_DW_weighted"].max()
    yld_min_cohort = survey_yld_df.loc[survey_yld_df["mean_DW_weighted"].idxmin(), "cohort"]
    yld_max_cohort = survey_yld_df.loc[survey_yld_df["mean_DW_weighted"].idxmax(), "cohort"]
    paragraph = (
        "Step 2b revises Step 2 without overwriting the original outputs. Disability estimates use the ADL/IADL-known "
        f"post-stroke denominator; severe ADL disability ranges from {severe_min:.1f}% in {severe_min_cohort} to "
        f"{severe_max:.1f}% in {severe_max_cohort}. Care-hour receipt uses all post-stroke survivors with known "
        f"recipient-level care hours, no longer requiring ADL/IADL completeness; any informal care ranges from "
        f"{any_min:.1f}% to {any_max:.1f}% across the six care-hour cohorts. The informal care gap, reported separately "
        f"from strict unmet need, ranges from {gap_min:.1f}% in {gap_min_cohort} to {gap_max:.1f}% in {gap_max_cohort} "
        "among disabled post-stroke survivors with known harmonized care hours. Survey-based post-stroke YLD-like "
        f"mean disability weight ranges from {yld_min:.3f} in {yld_min_cohort} to {yld_max:.3f} in {yld_max_cohort}; "
        "this estimate is not GBD YLD and no projection is performed."
    )
    (OUTPUT_DIR / "step2b_results_paragraph.txt").write_text(paragraph + "\n", encoding="utf-8")
    return paragraph


def write_readme(input_path: Path, outputs: list[Path]) -> None:
    lines = [
        "# Step 2b Revision",
        "",
        "This folder contains a revision of Step 2 for the cross-national post-stroke disability and care-burden project.",
        "The original Step 2 outputs under `output/` were not overwritten.",
        "",
        "## Key Changes",
        "",
        "- Denominators are explicit: disability uses ADL/IADL-known post-stroke survivors; care-hour estimates use care-hour-known post-stroke survivors and do not require complete ADL/IADL; unmet need uses ADL care-need survivors with known source unmet status.",
        "- Objective unmet need uses the `unmet_need_objective` variable from the Step 1 latest-poststroke file rather than silently reconstructing unmet need from any ADL/IADL disability plus zero hours.",
        "- Strict unmet need and informal care gap are reported as separate concepts. The informal-care-gap outputs are not labeled as unmet care need.",
        "- Survey-based post-stroke YLD-like disability burden estimates use disability-severity weights and are kept distinct from GBD YLD.",
        "- Missing values are retained as missing. Required unavailable variables are recorded in `step2b_variable_availability.csv` and the quality log.",
        "- Age-sex standardization uses the pooled weighted age-sex distribution of the seven-cohort disability-analysis sample as the default standard population.",
        "- Survey-based post-stroke YLD is calculated from disability severity weights only; it is not GBD YLD and no 2050 projection is run.",
        "- Helper/provider-level data are not inferred. `step2b_helper_level_preflight.csv` documents that only recipient-level care hours are ready for Step 2b.",
        "",
        "## Input",
        "",
        f"- `{input_path}`",
        "",
        "## Output Files",
        "",
    ]
    for path in outputs:
        lines.append(f"- `{path.relative_to(ROOT)}`")
    lines.append("")
    (OUTPUT_DIR / "README_step2b.md").write_text("\n".join(lines), encoding="utf-8")


def quality_check_log(
    df: pd.DataFrame,
    availability: pd.DataFrame,
    denominators: pd.DataFrame,
    sample_flow: pd.DataFrame,
    unmet_availability: pd.DataFrame,
    strict_unmet: pd.DataFrame,
    informal_gap: pd.DataFrame,
    survey_yld: pd.DataFrame,
    survey_yld_sens: pd.DataFrame,
    std: pd.DataFrame,
    care_missingness: pd.DataFrame,
    care_sensitivity: pd.DataFrame,
    covariates: pd.DataFrame,
    helper: pd.DataFrame,
    helper_availability: pd.DataFrame,
    gender_matrix: pd.DataFrame,
    outputs: list[Path],
) -> str:
    checks = []

    def add_check(name: str, passed: bool, fail_why: str, blocks_step3: bool, fix_next: str, warning: bool = False) -> None:
        status = "PASS" if passed and not warning else "WARNING" if passed and warning else "FAIL"
        checks.append(
            {
                "status": status,
                "name": name,
                "why": "" if status == "PASS" else fail_why,
                "blocks_step3": "yes" if blocks_step3 and status != "PASS" else "no",
                "fix_next": "" if status == "PASS" else fix_next,
            }
        )

    share_dis_yld = (
        "SHARE" in set(df.loc[df["disability_known_sample"].eq(1), "cohort"])
        and "SHARE" in set(survey_yld["cohort"])
    )
    add_check(
        "1. SHARE included in disability and YLD analysis",
        share_dis_yld,
        "SHARE is absent from disability-known rows or survey-based YLD table.",
        True,
        "Restore SHARE disability rows and rerun survey-based YLD.",
    )
    share_care_excluded = "SHARE" not in set(df.loc[df["care_hour_known_sample"].eq(1), "cohort"])
    add_check(
        "2. SHARE excluded from care-hour analysis if harmonizable hours are unavailable",
        share_care_excluded,
        "SHARE appears in the care-hour analysis sample despite no harmonizable weekly hours in the Step 1 manifest.",
        False,
        "Remove SHARE from care-hour and informal-care-gap denominators unless harmonized hours are built.",
    )
    denom_ok = (denominators.loc[denominators["cohort"].isin(CARE_COHORTS), "disability_known_denominator_n"] >= 0).all() and (
        df.loc[df["cohort"].isin(CARE_COHORTS) & df["informal_hours_week_raw"].isna(), "disability_known_sample"].sum() > 0
    )
    add_check(
        "3. Disability denominator is not reduced due to care-hour missingness",
        bool(denom_ok),
        "Disability rows with missing care hours were not retained, or denominator audit is inconsistent.",
        True,
        "Use disability_known_sample independent of care_hour_known_sample.",
    )
    missing_not_zero = int(df.loc[df["cohort"].isin(CARE_COHORTS), "informal_hours_week_raw"].isna().sum()) > 0
    add_check(
        "4. Missing care hours are not recoded to zero",
        missing_not_zero and len(care_missingness) > 0,
        "Care-hour missingness is absent or not documented separately from zero hours.",
        True,
        "Retain missing hours as NA and regenerate table_s2.",
    )
    separated = len(strict_unmet) == len(ALL_COHORTS) and len(informal_gap) == len(ALL_COHORTS)
    add_check(
        "5. Strict unmet need and informal care gap are separated",
        separated,
        "Strict unmet and informal care gap tables are missing or merged.",
        True,
        "Regenerate table4a and table4b as separate concepts.",
    )
    relabel_ok = (OUTPUT_DIR / "table4a_strict_unmet_need.csv").exists() and (OUTPUT_DIR / "table4b_informal_care_gap.csv").exists()
    add_check(
        "6. table4_objective_unmet_need.csv is not reused without relabeling",
        relabel_ok,
        "Only the legacy table4_objective_unmet_need.csv is present.",
        True,
        "Use table4a/table4b for Step 2b concepts and keep legacy table4 only as a backward-compatibility artifact.",
    )
    mhas_flow = sample_flow[sample_flow["cohort"].eq("MHAS")].iloc[0]
    add_check(
        "7. MHAS sample attrition is explained",
        "missing IADL" in mhas_flow["notes"],
        "MHAS attrition note does not explain ADL/IADL complete-case loss.",
        False,
        "Update table_s0 and QC log with MHAS ADL/IADL missingness counts.",
    )
    weights_used = "analysis_weight" in df.columns and df["analysis_weight"].notna().all()
    add_check(
        "8. Survey weights are used where available",
        bool(weights_used),
        "analysis_weight is missing for some rows.",
        False,
        "Fill missing original weights with unit analysis weights and document this.",
    )
    yld_ok = df.loc[df["disability_known_sample"].eq(1), "disability_severity"].map(DISABILITY_WEIGHTS).equals(
        df.loc[df["disability_known_sample"].eq(1), "yld_stroke_i"]
    )
    add_check(
        "9. YLD mapping is implemented correctly",
        bool(yld_ok),
        "yld_stroke_i does not match the requested disability-weight mapping.",
        True,
        "Recompute yld_stroke_i from disability_severity.",
    )
    add_check(
        "10. YLD sensitivity analysis is created",
        (OUTPUT_DIR / "table5_sensitivity_yld_no_disability_dw0019.csv").exists() and len(survey_yld_sens) > 0,
        "Sensitivity table is missing.",
        False,
        "Regenerate table5_sensitivity_yld_no_disability_dw0019.csv.",
    )
    add_check(
        "11. Age-sex standardization output is created",
        (OUTPUT_DIR / "table6_age_sex_standardized_estimates.csv").exists(),
        "table6_age_sex_standardized_estimates.csv is missing.",
        False,
        "Run direct standardization and write table6.",
    )
    add_check(
        "12. Care-hour p95/p99 sensitivity output is created",
        (OUTPUT_DIR / "table_s4_care_hour_sensitivity.csv").exists() and len(care_sensitivity) > 0,
        "Care-hour sensitivity output is missing.",
        False,
        "Regenerate table_s3 and table_s4.",
    )
    add_check(
        "13. Covariate availability table is created",
        (OUTPUT_DIR / "table_s5_covariate_availability.csv").exists() and len(covariates) > 0,
        "Covariate availability table is missing.",
        False,
        "Search harmonized outputs/scripts and regenerate table_s5.",
    )
    add_check(
        "14. Helper-level preflight table is created",
        (OUTPUT_DIR / "step3_helper_long_format_preflight.csv").exists() and len(helper) > 0,
        "Step 3 helper long preflight is missing.",
        True,
        "Create step3_helper_long_format_preflight.csv.",
    )
    provider_checked = (OUTPUT_DIR / "table_s6_helper_variable_availability.csv").exists() and "has_helper_sex" in helper_availability.columns
    add_check(
        "15. Provider sex availability is checked",
        provider_checked,
        "Provider sex availability was not recorded.",
        True,
        "Regenerate table_s6_helper_variable_availability.csv.",
    )
    any_eligible = (helper_availability["can_build_gender_care_matrix"].isin(["yes", "partial"])).any() if len(helper_availability) else False
    matrix_rule_ok = (not any_eligible and gender_matrix[["H_MM_weekly", "H_MF_weekly", "H_FM_weekly", "H_FF_weekly"]].isna().all().all()) or any_eligible
    add_check(
        "16. Gender care matrix is generated only where provider sex and helper-level hours are available",
        bool(matrix_rule_ok),
        "Gender matrix contains hour estimates without confirmed provider sex and helper-level hours.",
        True,
        "Suppress matrix estimates until provider sex and helper-level hours are available.",
    )

    sparse = std[std["sparse_standard_cell"].eq(1)]
    unavailable_required = availability[
        availability["required_for_step2b"].eq(True) & availability["status"].eq("unavailable")
    ]
    unavailable_helper = availability[
        availability["helper_level_preflight_role"].eq(True) & availability["status"].eq("unavailable")
    ]

    lines = ["Step 2b quality checks"]
    for check in checks:
        lines.append(f"{check['status']} - {check['name']}")
        if check["status"] != "PASS":
            lines.append(f"  why_failed_or_warning: {check['why']}")
            lines.append(f"  blocks_step3: {check['blocks_step3']}")
            lines.append(f"  fix_next: {check['fix_next']}")
    lines.append("")
    lines.append(f"Unavailable required Step 2b variables: {', '.join(unavailable_required['variable_role']) if len(unavailable_required) else 'None'}")
    lines.append(f"Unavailable helper-level variables: {', '.join(unavailable_helper['variable_role']) if len(unavailable_helper) else 'None'}")
    lines.append("No missing values were imputed inside Step 2b. The script uses complete-case denominators that are recorded in step2b_denominator_audit.csv.")
    lines.append("Unmet need definition: source unmet_need_objective; conditional unmet-need denominator is ADL care need (ADL score >=1) with known unmet status.")
    lines.append("Strict unmet need and informal care gap are now separated in table4a_strict_unmet_need.csv and table4b_informal_care_gap.csv. The informal-care-gap table does not label the concept as unmet care need.")
    lines.append("Strict unmet need uses PCGI care-source indicators where constructible; cohorts with unavailable required care-source indicators are set to NA and explained in table_s1_unmet_variable_availability.csv.")
    lines.append("Survey-based post-stroke YLD uses disability severity weights only and is not labeled as GBD YLD.")
    lines.append("YLD calculation: individual yld_stroke_i equals the disability-severity weight; no GBD YLD linkage and no 2050 projection are run in Step 2b.")
    lines.append(
        "MHAS sample-flow check: Step 1 raw latest-poststroke N=1248; age>=50/stroke N=1245; "
        "ADL/IADL complete disability-analysis N=812. The loss is mainly IADL missingness: 416 had ADL known but IADL missing, "
        "2 had IADL known but ADL missing, and 15 had both ADL and IADL missing."
    )
    lines.append("")
    lines.append(f"Sparse standard population cells (<20 observations): {len(sparse)}")
    if len(sparse):
        lines.append(sparse.to_string(index=False))
    lines.append("")
    lines.append("Output file check:")
    for path in outputs:
        lines.append(f"{'OK' if path.exists() else 'MISSING'} - {path.relative_to(ROOT)}")
    text = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "step2b_quality_check_log.txt").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    input_path = find_input()
    obsolete_yld_context = OUTPUT_DIR / "step2b_yld_burden_table.csv"
    if obsolete_yld_context.exists():
        obsolete_yld_context.unlink()
    raw = pd.read_csv(input_path, dtype={"respondent_id": str})
    mapping, availability = map_variables(raw)
    df = prepare_dataset(raw, mapping)
    pcgi = load_pcgi_latest(df)
    df = add_covariates_to_analysis(df, pcgi)
    df.to_csv(OUTPUT_DIR / "poststroke_step2b_analysis_sample.csv", index=False)
    pcgi = load_pcgi_latest(df)
    missingness = save_missingness(raw, df, mapping)
    denominators = denominator_audit(df)
    sample_flow = sample_flow_table(raw, df)
    care_missingness = care_hour_missingness_table(df)
    care_cutoffs = care_hour_p99_cutoff_table(df)
    care_sensitivity = care_hour_sensitivity_table(df)
    t2 = table2(df)
    t3 = table3(df)
    t4 = table4(df)
    unmet_availability = unmet_variable_availability(df, pcgi)
    t4a = table4a_strict_unmet_need(pcgi)
    t4b = table4b_informal_care_gap(df)
    covariates = covariate_availability_table(df, pcgi)
    s2 = supp_sex_stratified(df)
    survey_yld = survey_yld_summary(df)
    survey_yld.to_csv(OUTPUT_DIR / "table5_yld_stroke_by_cohort.csv", index=False)
    survey_yld_sex = survey_yld_summary(df, by_sex=True)
    survey_yld_sex.to_csv(OUTPUT_DIR / "table5_yld_stroke_by_cohort_sex.csv", index=False)
    survey_yld_sens = survey_yld_summary(df, sensitivity=True)
    survey_yld_sens.to_csv(OUTPUT_DIR / "table5_sensitivity_yld_no_disability_dw0019.csv", index=False)
    std_estimates, std = age_sex_standardized(df, pcgi)
    helper_legacy = helper_level_preflight(df)
    helper_availability = helper_variable_availability_table(df, pcgi)
    helper_long = step3_helper_long_format_preflight(df)
    gender_matrix = preliminary_gender_care_matrix(helper_availability)
    paragraph = results_paragraph(t2, t4b, survey_yld)

    outputs = [
        OUTPUT_DIR / "step2b_variable_availability.csv",
        OUTPUT_DIR / "step2b_missingness_by_cohort.csv",
        OUTPUT_DIR / "step2b_raw_column_missingness.csv",
        OUTPUT_DIR / "step2b_denominator_audit.csv",
        OUTPUT_DIR / "table_s0_sample_flow.csv",
        OUTPUT_DIR / "poststroke_step2b_analysis_sample.csv",
        OUTPUT_DIR / "step2b_care_hour_p99_cutoffs.csv",
        OUTPUT_DIR / "table2_disability_care_hour_distribution.csv",
        OUTPUT_DIR / "table3_recipient_only_care_intensity.csv",
        OUTPUT_DIR / "table4_objective_unmet_need.csv",
        OUTPUT_DIR / "table4a_strict_unmet_need.csv",
        OUTPUT_DIR / "table4b_informal_care_gap.csv",
        OUTPUT_DIR / "table_s1_unmet_variable_availability.csv",
        OUTPUT_DIR / "table_s2_care_hour_missingness.csv",
        OUTPUT_DIR / "table_s3_care_hour_p99_cutoffs.csv",
        OUTPUT_DIR / "table_s4_care_hour_sensitivity.csv",
        OUTPUT_DIR / "table_s5_covariate_availability.csv",
        OUTPUT_DIR / "table_s6_helper_variable_availability.csv",
        OUTPUT_DIR / "supp_table_s2_sex_stratified.csv",
        OUTPUT_DIR / "step2b_standard_population_age_sex.csv",
        OUTPUT_DIR / "table6_age_sex_standardized_estimates.csv",
        OUTPUT_DIR / "step2b_age_sex_standardized_estimates.csv",
        OUTPUT_DIR / "step2b_standardization_cell_estimates.csv",
        OUTPUT_DIR / "table5_yld_stroke_by_cohort.csv",
        OUTPUT_DIR / "table5_yld_stroke_by_cohort_sex.csv",
        OUTPUT_DIR / "table5_sensitivity_yld_no_disability_dw0019.csv",
        OUTPUT_DIR / "step2b_helper_level_preflight.csv",
        OUTPUT_DIR / "step3_helper_long_format_preflight.csv",
        OUTPUT_DIR / "table7_preliminary_gender_care_matrix.csv",
        OUTPUT_DIR / "step2b_results_paragraph.txt",
        OUTPUT_DIR / "step2b_quality_check_log.txt",
        OUTPUT_DIR / "README_step2b.md",
    ]
    write_readme(input_path, outputs)
    qc = quality_check_log(
        df,
        availability,
        denominators,
        sample_flow,
        unmet_availability,
        t4a,
        t4b,
        survey_yld,
        survey_yld_sens,
        std,
        care_missingness,
        care_sensitivity,
        covariates,
        helper_long,
        helper_availability,
        gender_matrix,
        outputs,
    )

    print(f"Step 2b input: {input_path}")
    print(f"Rows: {len(raw):,}")
    print(f"Outputs written under: {OUTPUT_DIR}")
    print("\nResults paragraph:\n" + paragraph)
    print("\nQC summary:")
    for line in qc.splitlines()[:14]:
        print(line)

    # Keep references alive for lint-minded readers and accidental future edits.
    _ = (t3, t4, s2, std_estimates, missingness, survey_yld_sex, care_cutoffs, helper_legacy)


if __name__ == "__main__":
    main()
