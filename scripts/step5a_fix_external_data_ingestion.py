from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / "results" / "step5a"
OUT = ROOT / "results" / "step5a_fix"
RAW = OUT / "external_data_raw"
HARM = OUT / "external_data_harmonized"
TEMPLATES = OUT / "templates"
LOGS = OUT / "logs"
for path in [OUT, RAW, HARM, TEMPLATES, LOGS]:
    path.mkdir(parents=True, exist_ok=True)

WPP_CSV = RAW / "wpp_2024_population_age_sex.csv"
WPP_XLSX = RAW / "wpp_2024_population_age_sex.xlsx"
PREV_CSV = RAW / "stroke_prevalence_age_sex.csv"
MAPPING_CSV = RAW / "cohort_country_mapping.csv"
SHARE_WEIGHTS = RAW / "share_country_weights.csv"
PATIENT_CANDIDATES = [
    ROOT / "results" / "step2b" / "step2b_patient_level_analytic.csv",
    ROOT / "results" / "step2b" / "patient_level_analytic.csv",
    ROOT / "results" / "step2b" / "cleaned_patient_level_data.csv",
    ROOT / "results" / "step2b" / "poststroke_step2b_analysis_sample.csv",
]

AGE_GROUPS = ["50-64", "65-74", "75-84", "85+"]
SEXES = ["Male", "Female"]


def write_templates() -> None:
    if not WPP_CSV.exists() and not WPP_XLSX.exists():
        pd.DataFrame(
            columns=["location", "year", "sex", "age", "age_group_original", "population", "source", "notes"]
        ).to_csv(TEMPLATES / "template_wpp_2024_population_age_sex.csv", index=False)
    if not PREV_CSV.exists():
        pd.DataFrame(
            columns=["location", "year", "sex", "age_group", "stroke_prevalence_rate", "stroke_prevalent_cases", "source", "source_detail", "notes"]
        ).to_csv(TEMPLATES / "template_stroke_prevalence_age_sex.csv", index=False)
    if not MAPPING_CSV.exists():
        pd.DataFrame(
            [
                ["CHARLS", "China", "national", "baseline_year_to_fill", "yes", "yes", "yes", "China national denominator"],
                ["HRS", "United States of America", "national", "baseline_year_to_fill", "yes", "yes", "yes", "US national denominator"],
                ["ELSA", "England or United Kingdom", "to_confirm", "baseline_year_to_fill", "yes", "yes", "yes", "ELSA denominator must be confirmed"],
                ["SHARE", "SHARE participating countries or Europe approximation", "to_confirm", "baseline_year_to_fill", "no", "yes", "yes", "no harmonized care hours"],
                ["KLoSA", "Republic of Korea", "national", "baseline_year_to_fill", "yes", "yes", "yes", "South Korea national denominator"],
                ["LASI", "India", "national", "baseline_year_to_fill", "yes", "no", "yes", "strict unmet unavailable"],
                ["MHAS", "Mexico", "national", "baseline_year_to_fill", "yes", "yes", "yes", "Mexico national denominator"],
            ],
            columns=[
                "cohort",
                "projection_location",
                "denominator_scope",
                "baseline_year",
                "care_hour_projection_eligible",
                "strict_unmet_projection_eligible",
                "disability_projection_eligible",
                "notes",
            ],
        ).to_csv(TEMPLATES / "template_cohort_country_mapping.csv", index=False)


def read_previous_diagnostics() -> pd.DataFrame:
    prev_inv = pd.read_csv(PREV / "table26_step5a_external_data_inventory.csv") if (PREV / "table26_step5a_external_data_inventory.csv").exists() else pd.DataFrame()
    rows = [
        ["WPP population data", "missing" if not WPP_CSV.exists() and not WPP_XLSX.exists() else "available_raw", "WPP population data unavailable" if not WPP_CSV.exists() and not WPP_XLSX.exists() else "", "Provide raw WPP age-sex population file and rerun ingestion.", "yes", "Previous Step 5A could not run population-level projection without external population."],
        ["Stroke prevalence data", "missing" if not PREV_CSV.exists() else "available_raw", "GBD/external stroke prevalence unavailable" if not PREV_CSV.exists() else "", "Provide age-sex stroke prevalence or full-sample baseline prevalence.", "yes", "Scenario B requires 2050 prevalence; Scenario A can hold baseline prevalence if baseline prevalence is available."],
        ["Cohort-country mapping", "missing" if not MAPPING_CSV.exists() else "available_raw", "cohort-country mapping unavailable" if not MAPPING_CSV.exists() else "", "Complete cohort-country mapping template.", "yes", "ELSA and SHARE require explicit denominator decisions."],
        ["SHARE care hours", "unavailable", "SHARE care hours unavailable", "Provide harmonized SHARE recipient-level care hours or exclude SHARE from care-hour projection.", "yes_for_care_hours", "SHARE can remain in disability/YLD-like summaries but not care-hour projections."],
        ["LASI strict unmet", "unavailable", "LASI strict unmet unavailable", "Do not force LASI into strict-unmet required-care projection unless variables are validated.", "no_for_demographic_only", "LASI can be used for care-hour projection but not strict-unmet required-care scenario."],
    ]
    out = pd.DataFrame(rows, columns=["domain", "previous_status", "blocker", "required_fix", "blocks_projection_yes_no", "notes"])
    if not prev_inv.empty:
        out["notes"] = out["notes"] + " Previous inventory reviewed."
    out.to_csv(OUT / "table33_step5a_fix_previous_run_diagnostics.csv", index=False)
    return out


def standard_sex(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return np.select(
        [s.isin(["male", "m", "1", "1.0"]), s.isin(["female", "f", "2", "2.0"]), s.isin(["both", "both sexes", "3", "3.0"])],
        ["Male", "Female", "Both"],
        default=pd.NA,
    )


def age_to_group(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().lower().replace(" years", "").replace("year", "").replace("age", "").strip()
    if text in {"85+", "85 plus", "85 and over", "85_over"}:
        return "85+"
    nums = [int(x) for x in re.findall(r"\d+", text)]
    if not nums:
        return pd.NA
    lo = nums[0]
    hi = nums[-1] if len(nums) > 1 else nums[0]
    if lo >= 85:
        return "85+"
    if hi < 50:
        return pd.NA
    if lo <= 64 and hi >= 50:
        return "50-64"
    if lo <= 74 and hi >= 65:
        return "65-74"
    if lo <= 84 and hi >= 75:
        return "75-84"
    return pd.NA


def parse_wpp() -> tuple[pd.DataFrame, pd.DataFrame]:
    if WPP_CSV.exists():
        raw = pd.read_csv(WPP_CSV)
        source_file = WPP_CSV.name
        source_format = "csv"
    elif WPP_XLSX.exists():
        raw = pd.read_excel(WPP_XLSX)
        source_file = WPP_XLSX.name
        source_format = "xlsx"
    else:
        empty = pd.DataFrame(columns=["location", "year", "sex", "age_group", "population", "source_file", "source_format", "notes"])
        empty.to_csv(HARM / "wpp_population_harmonized.csv", index=False)
        return empty, pd.DataFrame([{"domain": "WPP", "status": "missing", "notes": "Raw WPP file absent; template created."}])

    cols = {c.lower().strip(): c for c in raw.columns}
    required = ["location", "year", "sex", "population"]
    missing = [c for c in required if c not in cols]
    if missing:
        empty = pd.DataFrame(columns=["location", "year", "sex", "age_group", "population", "source_file", "source_format", "notes"])
        empty.to_csv(HARM / "wpp_population_harmonized.csv", index=False)
        return empty, pd.DataFrame([{"domain": "WPP", "status": "invalid", "notes": "Missing columns: " + ", ".join(missing)}])

    df = pd.DataFrame(
        {
            "location": raw[cols["location"]],
            "year": pd.to_numeric(raw[cols["year"]], errors="coerce"),
            "sex": standard_sex(raw[cols["sex"]]),
            "population": pd.to_numeric(raw[cols["population"]], errors="coerce"),
        }
    )
    if "age_group_original" in cols:
        age_source = raw[cols["age_group_original"]]
    elif "age_group" in cols:
        age_source = raw[cols["age_group"]]
    elif "age" in cols:
        age_source = raw[cols["age"]]
    else:
        age_source = pd.Series([pd.NA] * len(raw))
    df["age_group"] = age_source.map(age_to_group)
    unit_note = "Population assumed persons because no explicit thousands unit column was found."
    for key in ["unit", "units"]:
        if key in cols:
            units = raw[cols[key]].astype(str).str.lower()
            if units.str.contains("thousand").any():
                df["population"] = df["population"] * 1000
                unit_note = "Population converted from thousands to persons based on unit column."
    df = df[df["sex"].isin(SEXES) & df["age_group"].isin(AGE_GROUPS) & df["population"].gt(0)].copy()
    grouped = (
        df.groupby(["location", "year", "sex", "age_group"], as_index=False)["population"]
        .sum()
        .assign(source_file=source_file, source_format=source_format, notes=unit_note)
    )
    grouped.to_csv(HARM / "wpp_population_harmonized.csv", index=False)
    val = pd.DataFrame(
        [{
            "domain": "WPP",
            "status": "usable" if not grouped.empty and 2050 in grouped["year"].dropna().astype(int).unique() else "incomplete",
            "notes": f"Harmonized rows={len(grouped)}. Required 2050 present={2050 in grouped['year'].dropna().astype(int).unique() if not grouped.empty else False}.",
        }]
    )
    return grouped, val


def parse_prevalence() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PREV_CSV.exists():
        empty = pd.DataFrame(columns=["location", "year", "sex", "age_group", "stroke_prevalence_rate", "stroke_prevalence_source", "notes"])
        empty.to_csv(HARM / "stroke_prevalence_harmonized.csv", index=False)
        return empty, pd.DataFrame([{"domain": "stroke_prevalence", "status": "missing", "notes": "Raw stroke prevalence file absent; template created."}])
    raw = pd.read_csv(PREV_CSV)
    cols = {c.lower().strip(): c for c in raw.columns}
    required = ["location", "year", "sex", "age_group", "stroke_prevalence_rate"]
    missing = [c for c in required if c not in cols]
    if missing:
        empty = pd.DataFrame(columns=["location", "year", "sex", "age_group", "stroke_prevalence_rate", "stroke_prevalence_source", "notes"])
        empty.to_csv(HARM / "stroke_prevalence_harmonized.csv", index=False)
        return empty, pd.DataFrame([{"domain": "stroke_prevalence", "status": "invalid", "notes": "Missing columns: " + ", ".join(missing)}])
    df = pd.DataFrame(
        {
            "location": raw[cols["location"]],
            "year": pd.to_numeric(raw[cols["year"]], errors="coerce"),
            "sex": standard_sex(raw[cols["sex"]]),
            "age_group": raw[cols["age_group"]].map(age_to_group),
            "stroke_prevalence_rate": pd.to_numeric(raw[cols["stroke_prevalence_rate"]], errors="coerce"),
            "stroke_prevalence_source": raw[cols.get("source", cols["stroke_prevalence_rate"])].astype(str) if "source" in cols else "external",
        }
    )
    population_col = next((cols[c] for c in ["population", "denominator", "population_denominator"] if c in cols), None)
    cases_col = next((cols[c] for c in ["stroke_prevalent_cases", "prevalent_cases", "cases"] if c in cols), None)
    if population_col:
        df["population_denominator"] = pd.to_numeric(raw[population_col], errors="coerce")
    else:
        df["population_denominator"] = np.nan
    if cases_col:
        df["stroke_prevalent_cases"] = pd.to_numeric(raw[cases_col], errors="coerce")
    else:
        df["stroke_prevalent_cases"] = np.nan
    notes = []
    unit_text = " ".join(
        raw[c].astype(str).str.lower().str.cat(sep=" ")
        for key, c in cols.items()
        if key in {"unit", "units", "source_detail", "notes", "measure"}
    )
    explicit_percent = ("percent" in unit_text) or ("%" in unit_text)
    probable_percent = df["stroke_prevalence_rate"].gt(1).any() and df["stroke_prevalence_rate"].le(100).all()
    if probable_percent and explicit_percent:
        df["stroke_prevalence_rate"] = df["stroke_prevalence_rate"] / 100
        notes.append("Values explicitly documented as percent and converted to proportion.")
    elif probable_percent:
        df["stroke_prevalence_rate"] = df["stroke_prevalence_rate"] / 100
        notes.append("Values >1 and <=100 flagged as probable percent and converted to proportion; verify source before manuscript use.")
    else:
        notes.append("Prevalence interpreted as proportion.")
    df = df[df["sex"].isin(SEXES) & df["age_group"].isin(AGE_GROUPS) & df["stroke_prevalence_rate"].between(0, 1)].copy()

    def agg_prev(g: pd.DataFrame) -> pd.Series:
        pop = pd.to_numeric(g["population_denominator"], errors="coerce")
        cases = pd.to_numeric(g["stroke_prevalent_cases"], errors="coerce")
        rate = pd.to_numeric(g["stroke_prevalence_rate"], errors="coerce")
        if cases.notna().any() and pop.notna().any() and pop.sum(skipna=True) > 0:
            out_rate = cases.sum(skipna=True) / pop.sum(skipna=True)
            method = "Aggregated as sum(stroke_prevalent_cases) / sum(population_denominator)."
        elif pop.notna().any() and pop.sum(skipna=True) > 0:
            mask = rate.notna() & pop.notna() & pop.gt(0)
            out_rate = (rate[mask] * pop[mask]).sum() / pop[mask].sum() if mask.any() else np.nan
            method = "Aggregated using population-weighted mean prevalence."
        else:
            out_rate = rate.mean(skipna=True)
            method = "Population denominators unavailable; aggregated using unweighted mean fallback."
        return pd.Series(
            {
                "stroke_prevalence_rate": out_rate,
                "stroke_prevalence_source": g["stroke_prevalence_source"].dropna().iloc[0] if g["stroke_prevalence_source"].notna().any() else "external",
                "aggregation_note": method,
            }
        )

    grouped = (
        df.groupby(["location", "year", "sex", "age_group"], as_index=False)
        .apply(agg_prev, include_groups=False)
        .reset_index(drop=True)
    )
    grouped["notes"] = " ".join(notes) + " " + grouped["aggregation_note"].astype(str)
    grouped = grouped.drop(columns=["aggregation_note"])
    grouped.to_csv(HARM / "stroke_prevalence_harmonized.csv", index=False)
    val = pd.DataFrame(
        [{
            "domain": "stroke_prevalence",
            "status": "usable" if not grouped.empty else "incomplete",
            "notes": f"Harmonized rows={len(grouped)}. 2050 present={2050 in grouped['year'].dropna().astype(int).unique() if not grouped.empty else False}.",
        }]
    )
    return grouped, val


def parse_mapping() -> pd.DataFrame:
    if MAPPING_CSV.exists():
        mapping = pd.read_csv(MAPPING_CSV)
    else:
        mapping = pd.read_csv(TEMPLATES / "template_cohort_country_mapping.csv")
    mapping.to_csv(HARM / "cohort_country_mapping_harmonized.csv", index=False)
    return mapping


def data_validation_table(wpp: pd.DataFrame, prev: pd.DataFrame, mapping: pd.DataFrame, val_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cohort in mapping.get("cohort", pd.Series(dtype=str)).astype(str):
        m = mapping[mapping["cohort"].astype(str).eq(cohort)].iloc[0]
        loc = str(m.get("projection_location", ""))
        wloc = wpp[wpp["location"].astype(str).eq(loc)] if not wpp.empty else pd.DataFrame()
        ploc = prev[prev["location"].astype(str).eq(loc)] if not prev.empty else pd.DataFrame()
        rows.append(
            {
                "cohort": cohort,
                "projection_location": loc,
                "mapping_status": "usable" if MAPPING_CSV.exists() else "template_only",
                "wpp_has_baseline_or_any_year": not wloc.empty,
                "wpp_has_2050": (not wloc.empty) and (2050 in wloc["year"].dropna().astype(int).unique()),
                "stroke_prev_has_any_year": not ploc.empty,
                "stroke_prev_has_2050": (not ploc.empty) and (2050 in ploc["year"].dropna().astype(int).unique()),
                "can_run_demographic_only": bool((not wloc.empty) and (2050 in wloc["year"].dropna().astype(int).unique()) and not ploc.empty),
                "can_run_prevalence_2050": bool((not wloc.empty) and (2050 in wloc["year"].dropna().astype(int).unique()) and (not ploc.empty) and (2050 in ploc["year"].dropna().astype(int).unique())),
                "notes": str(m.get("notes", "")),
            }
        )
    out = pd.DataFrame(rows)
    status = pd.concat(val_frames, ignore_index=True) if val_frames else pd.DataFrame()
    status.to_csv(LOGS / "step5a_fix_ingestion_status.csv", index=False)
    out.to_csv(OUT / "table34_step5a_fix_external_data_validation.csv", index=False)
    return out


def detailed_external_validation(wpp: pd.DataFrame, prev: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(domain: str, file_used: str, location: str, year: object, sex: object, age_group: object, value_type: str, value: object, status: str, issue: str, action: str) -> None:
        rows.append(
            {
                "input_domain": domain,
                "file_used": file_used,
                "location": location,
                "year": year,
                "sex": sex,
                "age_group": age_group,
                "value_type": value_type,
                "value": value,
                "validation_status": status,
                "issue": issue,
                "action_taken": action,
            }
        )

    if wpp.empty:
        add("WPP population", str(WPP_CSV if WPP_CSV.exists() else WPP_XLSX), "", "", "", "", "population", np.nan, "fail", "WPP raw data missing or harmonized output empty.", "Template created; projection disabled.")
    else:
        for _, r in wpp.iterrows():
            add("WPP population", str(HARM / "wpp_population_harmonized.csv"), r["location"], r["year"], r["sex"], r["age_group"], "population", r["population"], "pass" if r["population"] > 0 else "fail", "" if r["population"] > 0 else "Population <=0.", "Kept only positive numeric populations.")
        for loc in mapping.get("projection_location", pd.Series(dtype=str)).astype(str):
            sub = wpp[wpp["location"].astype(str).eq(loc)]
            add("WPP location coverage", str(HARM / "wpp_population_harmonized.csv"), loc, "", "", "", "coverage", len(sub), "pass" if not sub.empty else "fail", "" if not sub.empty else "Projection location not found in WPP.", "Location matching checked against cohort-country mapping.")
            for sex in SEXES:
                add("WPP sex coverage", str(HARM / "wpp_population_harmonized.csv"), loc, "", sex, "", "coverage", int(sub["sex"].eq(sex).sum()) if not sub.empty else 0, "pass" if (not sub.empty and sub["sex"].eq(sex).any()) else "fail", "" if (not sub.empty and sub["sex"].eq(sex).any()) else f"{sex} missing.", "Male/Female required for main age-sex projection.")
            for age in AGE_GROUPS:
                add("WPP age coverage", str(HARM / "wpp_population_harmonized.csv"), loc, "", "", age, "coverage", int(sub["age_group"].eq(age).sum()) if not sub.empty else 0, "pass" if (not sub.empty and sub["age_group"].eq(age).any()) else "fail", "" if (not sub.empty and sub["age_group"].eq(age).any()) else f"{age} missing.", "Ages harmonized to project groups.")
            for year in ["baseline_or_any", 2050]:
                ok = not sub.empty and ((sub["year"].notna().any()) if year == "baseline_or_any" else sub["year"].astype(int).eq(2050).any())
                add("WPP year coverage", str(HARM / "wpp_population_harmonized.csv"), loc, year, "", "", "coverage", int(ok), "pass" if ok else "fail", "" if ok else f"Required year {year} missing.", "Baseline/2050 year coverage checked.")
        add("WPP units", str(HARM / "wpp_population_harmonized.csv"), "", "", "", "", "units", "", "warning", "Units must be documented in source notes.", "Conversion note stored in harmonized notes.")

    if prev.empty:
        add("Stroke prevalence", str(PREV_CSV), "", "", "", "", "stroke_prevalence_rate", np.nan, "fail", "Stroke prevalence raw data missing or harmonized output empty.", "Template created; projection disabled unless survey prevalence becomes available.")
    else:
        for _, r in prev.iterrows():
            ok = 0 <= r["stroke_prevalence_rate"] <= 1
            add("Stroke prevalence", str(HARM / "stroke_prevalence_harmonized.csv"), r["location"], r["year"], r["sex"], r["age_group"], "stroke_prevalence_rate", r["stroke_prevalence_rate"], "pass" if ok else "fail", "" if ok else "Prevalence outside [0,1].", "Values validated as proportions.")

    if mapping.empty:
        add("cohort-country mapping", str(MAPPING_CSV), "", "", "", "", "mapping", "", "fail", "Mapping missing.", "Template created.")
    else:
        for _, r in mapping.iterrows():
            cohort = str(r.get("cohort", ""))
            loc = str(r.get("projection_location", ""))
            notes = str(r.get("notes", ""))
            scope = str(r.get("denominator_scope", ""))
            status = "pass"
            issue = ""
            if cohort == "ELSA" and ("confirm" in scope.lower() or "or" in loc.lower()):
                status = "warning"
                issue = "ELSA denominator scope still requires confirmation."
            if cohort == "SHARE" and ("europe" in loc.lower() or "participating" in loc.lower()):
                status = "warning"
                issue = "SHARE mapping requires explicit country composition or documented approximation."
            add("cohort-country mapping", str(HARM / "cohort_country_mapping_harmonized.csv"), loc, "", "", "", "mapping", cohort, status, issue, notes)

    add("fabrication check", "", "", "", "", "", "no fabricated values", "", "pass", "", "No missing population or prevalence values were filled.")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table36_step5a_fix_external_data_validation.csv", index=False)
    return out


def projection_readiness(wpp: pd.DataFrame, prev: pd.DataFrame, mapping: pd.DataFrame, feasibility: pd.DataFrame) -> pd.DataFrame:
    scope_path = ROOT / "results" / "step4" / "table20_analysis_scope_by_cohort.csv"
    scope = pd.read_csv(scope_path) if scope_path.exists() else pd.DataFrame()
    rows = []
    for _, m in mapping.iterrows():
        cohort = str(m.get("cohort", ""))
        loc = str(m.get("projection_location", ""))
        baseline_year = pd.to_numeric(pd.Series([m.get("baseline_year")]), errors="coerce").iloc[0]
        wloc = wpp[wpp["location"].astype(str).eq(loc)] if not wpp.empty else pd.DataFrame()
        ploc = prev[prev["location"].astype(str).eq(loc)] if not prev.empty else pd.DataFrame()
        frow = feasibility[feasibility["cohort"].astype(str).eq(cohort)] if not feasibility.empty else pd.DataFrame()
        survey_prev = not frow.empty and frow["can_calculate_baseline_stroke_prevalence"].iloc[0] == "yes"
        has_care = False
        has_strict = False
        if not scope.empty:
            srow = scope[scope["cohort"].astype(str).eq(cohort)]
            if not srow.empty:
                has_care = bool(srow["recipient_level_care_hours_available"].iloc[0])
                has_strict = bool(srow["strict_unmet_need_available"].iloc[0])
        has_wpp_base = (not wloc.empty) and (wloc["year"].notna().any())
        has_wpp_2050 = (not wloc.empty) and (2050 in wloc["year"].dropna().astype(int).unique())
        has_prev_base = survey_prev or (not ploc.empty)
        has_prev_2050 = (not ploc.empty) and (2050 in ploc["year"].dropna().astype(int).unique())
        can_demo = has_care and has_wpp_2050 and has_prev_base and cohort != "SHARE"
        can_prev = can_demo and has_prev_2050
        can_req = can_demo and (has_strict or cohort != "LASI")
        reasons = []
        if cohort == "SHARE" and not has_care:
            reasons.append("SHARE lacks harmonized recipient-level care hours.")
        if not has_wpp_2050:
            reasons.append("WPP 2050 population unavailable.")
        if not has_prev_base:
            reasons.append("Baseline stroke prevalence unavailable.")
        if cohort == "LASI" and not has_strict:
            reasons.append("LASI strict unmet need unavailable; strict-unmet required-care scenario disabled.")
        rows.append(
            {
                "cohort": cohort,
                "projection_location": loc,
                "baseline_year": baseline_year,
                "has_wpp_population_baseline": has_wpp_base,
                "has_wpp_population_2050": has_wpp_2050,
                "has_baseline_stroke_prevalence": has_prev_base,
                "has_2050_stroke_prevalence": has_prev_2050,
                "has_recipient_level_care_hours": has_care,
                "has_strict_unmet_need": has_strict,
                "can_run_demographic_only_projection": can_demo,
                "can_run_prevalence_adjusted_projection": can_prev,
                "can_run_required_care_projection": can_req,
                "reason_if_not_ready": " ".join(reasons),
                "notes": str(m.get("notes", "")),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table37_step5a_fix_projection_readiness.csv", index=False)
    return out


def baseline_survey_prevalence_feasibility() -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_path = next((p for p in PATIENT_CANDIDATES if p.exists()), None)
    rows = []
    prevalence_rows = []
    if patient_path is None:
        for cohort in ["CHARLS", "HRS", "ELSA", "SHARE", "KLoSA", "LASI", "MHAS"]:
            rows.append(
                {
                    "cohort": cohort,
                    "patient_level_file": "",
                    "includes_non_stroke_older_adults_yes_no": "no",
                    "stroke_indicator_found": "no",
                    "age_found": "no",
                    "sex_found": "no",
                    "weight_found": "no",
                    "can_calculate_baseline_stroke_prevalence": "no",
                    "notes": "No patient-level Step 2b analytic file found; external stroke prevalence is required.",
                }
            )
    else:
        df = pd.read_csv(patient_path)
        cols_lower = {c.lower(): c for c in df.columns}
        stroke_col = next((cols_lower[c] for c in ["stroke_ever", "stroke", "stroke_indicator", "has_stroke"] if c in cols_lower), None)
        age_col = next((cols_lower[c] for c in ["age", "age_years"] if c in cols_lower), None)
        sex_col = next((cols_lower[c] for c in ["sex_label", "sex", "gender"] if c in cols_lower), None)
        weight_col = next((cols_lower[c] for c in ["analysis_weight", "survey_weight", "weight", "original_weight"] if c in cols_lower), None)
        cohort_col = cols_lower.get("cohort")
        denom_flag = next((cols_lower[c] for c in ["full_older_adult_sample", "older_adult_denominator", "denominator_sample"] if c in cols_lower), None)
        if cohort_col is None:
            cohorts = ["UNKNOWN"]
        else:
            cohorts = sorted(df[cohort_col].dropna().astype(str).unique().tolist())
        for cohort in cohorts:
            sub = df[df[cohort_col].astype(str).eq(cohort)].copy() if cohort_col else df.copy()
            stroke_vals = pd.to_numeric(sub[stroke_col], errors="coerce") if stroke_col else pd.Series(dtype=float)
            has_non_stroke = stroke_col is not None and stroke_vals.eq(0).any()
            has_stroke = stroke_col is not None and stroke_vals.eq(1).any()
            denom_ok = denom_flag is not None and pd.to_numeric(sub[denom_flag], errors="coerce").eq(1).any()
            can_calc = has_non_stroke and has_stroke and age_col is not None and sex_col is not None and weight_col is not None
            rows.append(
                {
                    "cohort": cohort,
                    "patient_level_file": str(patient_path.relative_to(ROOT)),
                    "includes_non_stroke_older_adults_yes_no": "yes" if has_non_stroke else "no",
                    "stroke_indicator_found": "yes" if stroke_col else "no",
                    "age_found": "yes" if age_col else "no",
                    "sex_found": "yes" if sex_col else "no",
                    "weight_found": "yes" if weight_col else "no",
                    "can_calculate_baseline_stroke_prevalence": "yes" if can_calc else "no",
                    "notes": (
                        "Full older-adult denominator appears available; baseline age-sex stroke prevalence can be estimated."
                        if can_calc
                        else "Patient-level Step 2b file appears to contain only stroke survivors or lacks denominator variables; external stroke prevalence is required for population-level projection."
                    ),
                }
            )
            if can_calc:
                sub["age_group_project"] = sub[age_col].map(age_to_group)
                if sex_col == "sex_label":
                    sub["sex_project"] = sub[sex_col].astype(str).where(sub[sex_col].astype(str).isin(["Male", "Female"]))
                else:
                    sub["sex_project"] = pd.Series(standard_sex(sub[sex_col]), index=sub.index)
                sub["weight_project"] = pd.to_numeric(sub[weight_col], errors="coerce")
                sub["stroke_project"] = pd.to_numeric(sub[stroke_col], errors="coerce").eq(1)
                for sex in ["Male", "Female"]:
                    for age in ["50-64", "65-74", "75-84", "85+"]:
                        cell = sub[sub["sex_project"].eq(sex) & sub["age_group_project"].eq(age) & sub["weight_project"].gt(0)]
                        n = len(cell)
                        wden = cell["weight_project"].sum()
                        nstroke = int(cell["stroke_project"].sum()) if n else 0
                        wnum = cell.loc[cell["stroke_project"], "weight_project"].sum() if n else np.nan
                        prevalence_rows.append(
                            {
                                "cohort": cohort,
                                "location": "",
                                "baseline_year": np.nan,
                                "sex": sex,
                                "age_group": age,
                                "n_older_adults": n,
                                "weighted_n_older_adults": wden,
                                "n_stroke_survivors": nstroke,
                                "weighted_n_stroke_survivors": wnum,
                                "stroke_prevalence_rate": wnum / wden if wden and pd.notna(wden) else np.nan,
                                "notes": "Small cell flagged." if n < 30 else "Survey-weighted baseline stroke prevalence.",
                            }
                        )
            else:
                for sex in ["Male", "Female"]:
                    for age in ["50-64", "65-74", "75-84", "85+"]:
                        prevalence_rows.append(
                            {
                                "cohort": cohort,
                                "location": "",
                                "baseline_year": np.nan,
                                "sex": sex,
                                "age_group": age,
                                "n_older_adults": np.nan,
                                "weighted_n_older_adults": np.nan,
                                "n_stroke_survivors": np.nan,
                                "weighted_n_stroke_survivors": np.nan,
                                "stroke_prevalence_rate": np.nan,
                                "notes": "Not calculated because patient-level file contains only stroke survivors or lacks full older-adult denominator.",
                            }
                        )
    feas = pd.DataFrame(rows)
    prev = pd.DataFrame(
        prevalence_rows,
        columns=[
            "cohort",
            "location",
            "baseline_year",
            "sex",
            "age_group",
            "n_older_adults",
            "weighted_n_older_adults",
            "n_stroke_survivors",
            "weighted_n_stroke_survivors",
            "stroke_prevalence_rate",
            "notes",
        ],
    )
    feas.to_csv(OUT / "table34_step5a_fix_baseline_survey_prevalence_feasibility.csv", index=False)
    prev.to_csv(OUT / "table35_step5a_fix_baseline_survey_stroke_prevalence.csv", index=False)
    return feas, prev


def main() -> None:
    write_templates()
    prev_diag = read_previous_diagnostics()
    wpp, wpp_val = parse_wpp()
    prev, prev_val = parse_prevalence()
    mapping = parse_mapping()
    validation = data_validation_table(wpp, prev, mapping, [wpp_val, prev_val])
    feasibility, survey_prev = baseline_survey_prevalence_feasibility()
    detailed = detailed_external_validation(wpp, prev, mapping)
    readiness = projection_readiness(wpp, prev, mapping, feasibility)
    print(f"Step 5A-fix ingestion outputs written under: {OUT}")
    print(prev_diag.to_string(index=False))
    print(validation.to_string(index=False))
    print(feasibility.to_string(index=False))
    print(readiness.to_string(index=False))


if __name__ == "__main__":
    main()
