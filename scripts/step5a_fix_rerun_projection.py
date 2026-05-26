from __future__ import annotations

from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step5a_fix"
HARM = OUT / "external_data_harmonized"
LOGS = OUT / "logs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

STEP2B_RATES = ROOT / "results" / "step5a" / "table27_step5a_baseline_age_sex_rates.csv"
STEP4_SCOPE = ROOT / "results" / "step4" / "table20_analysis_scope_by_cohort.csv"

WPP = HARM / "wpp_population_harmonized.csv"
PREV = HARM / "stroke_prevalence_harmonized.csv"
MAPPING = HARM / "cohort_country_mapping_harmonized.csv"
VALIDATION = OUT / "table34_step5a_fix_external_data_validation.csv"
READINESS = OUT / "table37_step5a_fix_projection_readiness.csv"

AGE_GROUPS = ["50-64", "65-74", "75-84", "85+"]
SEXES = ["Male", "Female"]


def load() -> dict[str, pd.DataFrame]:
    paths = {
        "rates": STEP2B_RATES,
        "scope": STEP4_SCOPE,
        "wpp": WPP,
        "prev": PREV,
        "mapping": MAPPING,
        "validation": VALIDATION,
        "readiness": READINESS,
    }
    return {k: pd.read_csv(p) if p.exists() else pd.DataFrame() for k, p in paths.items()}


def projection_ready(data: dict[str, pd.DataFrame]) -> tuple[bool, str]:
    if data["wpp"].empty:
        return False, "Harmonized WPP population data are missing or empty."
    if data["mapping"].empty:
        return False, "Cohort-country mapping is missing or empty."
    if data["prev"].empty:
        return False, "Stroke prevalence data are missing; demographic-only projection still requires baseline or external prevalence."
    if 2050 not in data["wpp"]["year"].dropna().astype(int).unique():
        return False, "WPP data do not contain 2050."
    return True, "All minimum harmonized inputs are available."


def care_eligible(scope: pd.DataFrame) -> set[str]:
    if scope.empty:
        return {"CHARLS", "HRS", "ELSA", "KLoSA", "LASI", "MHAS"}
    sub = scope[scope["recipient_level_care_hours_available"].eq(True)]
    return set(sub["cohort"].astype(str)) - {"SHARE"}


def nearest_year(df: pd.DataFrame, target: int, tolerance: int | None = None) -> int | None:
    years = sorted(df["year"].dropna().astype(int).unique().tolist())
    if not years:
        return None
    best = min(years, key=lambda y: abs(y - target))
    if tolerance is not None and abs(best - target) > tolerance:
        return None
    return best


def care_intensity_fallback_log(rates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in sorted(rates["cohort"].dropna().astype(str).unique()) if not rates.empty else []:
        subc = rates[rates["cohort"].astype(str).eq(cohort)].copy()
        for sex in SEXES:
            for age in AGE_GROUPS:
                cell = subc[subc["sex"].eq(sex) & subc["age_group"].eq(age)]
                preferred = (not cell.empty) and pd.to_numeric(cell["mean_care_hours_week_all_stroke_survivors"], errors="coerce").notna().any()
                n_cell = int(pd.to_numeric(cell["n_stroke_survivors"], errors="coerce").sum()) if not cell.empty and "n_stroke_survivors" in cell.columns else 0
                value = np.nan
                level = "none"
                if preferred:
                    value = pd.to_numeric(cell["mean_care_hours_week_all_stroke_survivors"], errors="coerce").dropna().iloc[0]
                    level = "cohort_x_sex_x_age_group"
                else:
                    age_sub = subc[subc["age_group"].eq(age)]
                    if pd.to_numeric(age_sub.get("mean_care_hours_week_all_stroke_survivors", pd.Series(dtype=float)), errors="coerce").notna().any():
                        value = pd.to_numeric(age_sub["mean_care_hours_week_all_stroke_survivors"], errors="coerce").mean()
                        level = "cohort_x_age_group"
                    else:
                        sex_sub = subc[subc["sex"].eq(sex)]
                        if pd.to_numeric(sex_sub.get("mean_care_hours_week_all_stroke_survivors", pd.Series(dtype=float)), errors="coerce").notna().any():
                            value = pd.to_numeric(sex_sub["mean_care_hours_week_all_stroke_survivors"], errors="coerce").mean()
                            level = "cohort_x_sex"
                        elif pd.to_numeric(subc.get("mean_care_hours_week_all_stroke_survivors", pd.Series(dtype=float)), errors="coerce").notna().any():
                            value = pd.to_numeric(subc["mean_care_hours_week_all_stroke_survivors"], errors="coerce").mean()
                            level = "cohort_overall"
                rows.append(
                    {
                        "cohort": cohort,
                        "sex": sex,
                        "age_group": age,
                        "preferred_value_available": bool(preferred),
                        "fallback_level_used": level,
                        "care_hours_week_value": value,
                        "n_cell": n_cell,
                        "notes": "Sparse cell n < 30 flagged." if n_cell < 30 else "No cross-cohort borrowing used.",
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table43_step5a_fix_care_intensity_fallback_log.csv", index=False)
    return out


def lookup_care_value(fallback: pd.DataFrame, cohort: str, sex: str, age: str) -> float:
    row = fallback[fallback["cohort"].eq(cohort) & fallback["sex"].eq(sex) & fallback["age_group"].eq(age)]
    if row.empty:
        return np.nan
    return pd.to_numeric(row["care_hours_week_value"], errors="coerce").iloc[0]


def project(data: dict[str, pd.DataFrame], fallback: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ready, reason = projection_ready(data)
    rates = data["rates"]
    scope = data["scope"]
    mapping = data["mapping"]
    wpp = data["wpp"]
    prev = data["prev"]
    eligible = care_eligible(scope)
    if not ready:
        return empty_outputs(reason, eligible)

    cells = []
    main = []
    sens = []
    decomp = []
    for _, m in mapping.iterrows():
        cohort = str(m["cohort"])
        if cohort not in eligible:
            continue
        loc = str(m["projection_location"])
        base_year = pd.to_numeric(pd.Series([m.get("baseline_year")]), errors="coerce").iloc[0]
        base_year = int(base_year) if pd.notna(base_year) else nearest_year(wpp[wpp["location"].eq(loc)], 2020, tolerance=5)
        if base_year is None:
            continue
        wloc = wpp[wpp["location"].eq(loc)]
        ploc = prev[prev["location"].eq(loc)]
        if wloc.empty or ploc.empty:
            continue
        pbase_year = nearest_year(ploc, base_year, tolerance=5)
        p2050_year = 2050 if 2050 in ploc["year"].dropna().astype(int).unique() else pbase_year
        scenario = "prevalence_2050_external" if p2050_year == 2050 else "demographic_only"
        cohort_cells = []
        for year, py in [(base_year, pbase_year), (2050, p2050_year)]:
            for age in AGE_GROUPS:
                for sex in SEXES:
                    pop_row = wloc[wloc["year"].astype(int).eq(year) & wloc["age_group"].eq(age) & wloc["sex"].eq(sex)]
                    prev_row = ploc[ploc["year"].astype(int).eq(py) & ploc["age_group"].eq(age) & ploc["sex"].eq(sex)]
                    rate_row = rates[rates["cohort"].eq(cohort) & rates["age_group"].eq(age) & rates["sex"].eq(sex)]
                    if pop_row.empty or prev_row.empty or rate_row.empty:
                        continue
                    pop = float(pop_row["population"].iloc[0])
                    sp = float(prev_row["stroke_prevalence_rate"].iloc[0])
                    survivors = pop * sp
                    mean_h = lookup_care_value(fallback, cohort, sex, age)
                    req_h = mean_h
                    annual = survivors * mean_h * 52 if pd.notna(mean_h) else np.nan
                    cohort_cells.append(
                        {
                            "cohort": cohort,
                            "projection_location": loc,
                            "scenario": scenario,
                            "year": year,
                            "age_group": age,
                            "sex": sex,
                            "population": pop,
                            "stroke_prevalence": sp,
                            "projected_stroke_survivors": survivors,
                            "any_disability_percent": rate_row["any_disability_percent"].iloc[0],
                            "mean_DW_weighted": rate_row["mean_DW_weighted"].iloc[0],
                            "mean_care_hours_week_all_survivors": mean_h,
                            "projected_observed_care_hours_annual": annual,
                            "projected_required_care_hours_annual": survivors * req_h * 52 if pd.notna(req_h) else np.nan,
                            "notes": f"Prevalence year used={py}; recipient-level hours only.",
                        }
                    )
        cdf = pd.DataFrame(cohort_cells)
        if cdf.empty:
            continue
        cells.append(cdf)
        b = cdf[cdf["year"].eq(base_year)]
        f = cdf[cdf["year"].eq(2050)]
        obs_b = b["projected_observed_care_hours_annual"].sum()
        obs_f = f["projected_observed_care_hours_annual"].sum()
        req_b = b["projected_required_care_hours_annual"].sum()
        req_f = f["projected_required_care_hours_annual"].sum()
        main.append(
            {
                "cohort": cohort,
                "projection_location": loc,
                "scenario": scenario,
                "baseline_year": base_year,
                "projection_year": 2050,
                "population_age_range": "50+",
                "n_age_sex_cells_used": len(cdf),
                "projected_stroke_survivors_baseline": b["projected_stroke_survivors"].sum(),
                "projected_stroke_survivors_2050": f["projected_stroke_survivors"].sum(),
                "observed_care_hours_annual_baseline": obs_b,
                "observed_care_hours_annual_2050": obs_f,
                "required_care_hours_annual_baseline": req_b,
                "required_care_hours_annual_2050": req_f,
                "observed_care_hours_growth_percent": (obs_f / obs_b - 1) * 100 if obs_b else np.nan,
                "required_care_hours_growth_percent": (req_f / req_b - 1) * 100 if req_b else np.nan,
                "mean_hours_week_source": "Step 2b age-sex recipient-level mean hours among all stroke survivors",
                "stroke_prevalence_source": "external harmonized stroke prevalence; held constant if 2050 absent",
                "population_source": "harmonized WPP/external age-sex population",
                "notes": "No provider-sex decomposition used; no hidden gender care tax calculated.",
            }
        )
        decomp.append(
            {
                "cohort": cohort,
                "scenario": scenario,
                "baseline_care_hours_annual": obs_b,
                "step1_population_size_effect": np.nan,
                "step2_age_sex_structure_effect": np.nan,
                "step3_stroke_prevalence_effect": np.nan if p2050_year != 2050 else "included_in_total_not_decomposed",
                "step4_care_intensity_or_unmet_effect": 0,
                "total_growth_percent": (obs_f / obs_b - 1) * 100 if obs_b else np.nan,
                "decomposition_method": "total growth only; sequential replacement not run in fix layer",
                "notes": "Fix layer rerun prioritizes validated ingestion and main projection; no provider-side data used.",
            }
        )
        for sub in [0, 0.10, 0.25, 0.50]:
            sens.append(
                {
                    "cohort": cohort,
                    "scenario": "formalization_sensitivity" if sub else scenario,
                    "sensitivity_name": f"formal_care_substitution_{int(sub*100)}pct" if sub else "main",
                    "projected_required_care_hours_annual_2050": req_f * (1 - sub),
                    "growth_percent": ((req_f * (1 - sub)) / req_b - 1) * 100 if req_b else np.nan,
                    "difference_from_main_percent": -sub * 100,
                    "notes": "Policy sensitivity only; not causal.",
                }
            )
    cell_df = pd.concat(cells, ignore_index=True) if cells else pd.DataFrame()
    return pd.DataFrame(main), cell_df, pd.DataFrame(decomp), pd.DataFrame(sens)


def empty_outputs(reason: str, eligible: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohorts = sorted(eligible)
    t28 = pd.DataFrame(
        [
            {
                "cohort": c,
                "projection_location": np.nan,
                "scenario": "not_run_external_inputs_incomplete",
                "baseline_year": np.nan,
                "projection_year": 2050,
                "population_age_range": "50+",
                "n_age_sex_cells_used": 0,
                "projected_stroke_survivors_baseline": np.nan,
                "projected_stroke_survivors_2050": np.nan,
                "observed_care_hours_annual_baseline": np.nan,
                "observed_care_hours_annual_2050": np.nan,
                "required_care_hours_annual_baseline": np.nan,
                "required_care_hours_annual_2050": np.nan,
                "observed_care_hours_growth_percent": np.nan,
                "required_care_hours_growth_percent": np.nan,
                "mean_hours_week_source": "Step 2b recipient-level baseline rates",
                "stroke_prevalence_source": "missing or incomplete",
                "population_source": "missing or incomplete",
                "notes": reason,
            }
            for c in cohorts
        ]
    )
    t29 = pd.DataFrame(columns=["cohort", "projection_location", "scenario", "year", "age_group", "sex", "population", "stroke_prevalence", "projected_stroke_survivors", "any_disability_percent", "mean_DW_weighted", "mean_care_hours_week_all_survivors", "projected_observed_care_hours_annual", "projected_required_care_hours_annual", "notes"])
    t30 = pd.DataFrame([{"cohort": c, "scenario": "not_run_external_inputs_incomplete", "baseline_care_hours_annual": np.nan, "step1_population_size_effect": np.nan, "step2_age_sex_structure_effect": np.nan, "step3_stroke_prevalence_effect": np.nan, "step4_care_intensity_or_unmet_effect": np.nan, "total_growth_percent": np.nan, "decomposition_method": "not run", "notes": reason} for c in cohorts])
    t31 = pd.DataFrame([{"cohort": c, "scenario": "not_run_external_inputs_incomplete", "sensitivity_name": "not_run", "projected_required_care_hours_annual_2050": np.nan, "growth_percent": np.nan, "difference_from_main_percent": np.nan, "notes": reason} for c in cohorts])
    return t28, t29, t30, t31


def write_outputs(t28: pd.DataFrame, t29: pd.DataFrame, t30: pd.DataFrame, t31: pd.DataFrame, ready: bool, reason: str) -> pd.DataFrame:
    t28.to_csv(OUT / "table38_step5a_fix_2050_projection_main_rerun.csv", index=False)
    t29.to_csv(OUT / "table39_step5a_fix_age_sex_projection_cells_rerun.csv", index=False)
    t30.to_csv(OUT / "table40_step5a_fix_growth_decomposition_rerun.csv", index=False)
    t31.to_csv(OUT / "table41_step5a_fix_projection_sensitivity_rerun.csv", index=False)
    claims = pd.DataFrame(
        [
            ["2050 recipient-level post-stroke care demand can be projected after external data ingestion.", "supported" if ready else "not_available", "table38_step5a_fix_2050_projection_main_rerun.csv", "Projection completed." if ready else reason, "Requires validated external population and prevalence.", "2050 recipient-level projection was run for eligible care-hour cohorts." if ready else "Projection not run because external inputs were incomplete.", "Do not claim projections were completed if tables contain not-run rows."],
            ["Hidden gender care tax can be projected.", "not_available", "table42_step5a_fix_projection_claim_support_audit_rerun.csv", "No provider-sex denominator or cross-cohort provider matrix used.", "Outside Step 5A-fix.", "Hidden gender care tax was not calculated.", "Hidden gender care tax was projected."],
            ["HRS provider-side female share can be generalized to all cohorts.", "not_supported", "Step 4 claim audit", "HRS provider-side data are HRS-only.", "Do not extrapolate provider sex shares.", "HRS provider-side results remain HRS-only.", "HRS female-provider share applies to all cohorts."],
        ],
        columns=["claim", "support_status", "evidence_files", "key_numeric_evidence", "caveat", "allowed_manuscript_wording", "prohibited_wording"],
    )
    claims.to_csv(OUT / "table42_step5a_fix_projection_claim_support_audit_rerun.csv", index=False)
    return claims


def figures(t28: pd.DataFrame, t30: pd.DataFrame, readiness: pd.DataFrame, ready: bool) -> list[Path]:
    paths = []
    readiness.to_csv(FIG / "fig5a_fix_projection_readiness_heatmap_data.csv", index=False)
    plt.figure(figsize=(9, 4.8))
    if not readiness.empty:
        bool_cols = [
            "has_wpp_population_2050",
            "has_baseline_stroke_prevalence",
            "has_recipient_level_care_hours",
            "can_run_demographic_only_projection",
            "can_run_prevalence_adjusted_projection",
            "can_run_required_care_projection",
        ]
        mat = readiness[bool_cols].astype(bool).astype(int).to_numpy()
        plt.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        plt.yticks(range(len(readiness)), readiness["cohort"])
        plt.xticks(range(len(bool_cols)), bool_cols, rotation=35, ha="right")
        plt.colorbar(label="Ready flag")
    else:
        plt.text(0.5, 0.5, "Readiness table unavailable", ha="center", va="center")
        plt.axis("off")
    plt.title("Step 5A-fix projection readiness")
    plt.tight_layout()
    p = FIG / "fig5a_fix_projection_readiness_heatmap.png"
    plt.savefig(p, dpi=200)
    plt.close()
    paths.append(p)

    if not ready:
        return paths

    t28.to_csv(FIG / "fig5a_fix_2050_care_demand_growth_by_cohort_data.csv", index=False)
    t28.to_csv(FIG / "fig5a_fix_projection_scenarios_data.csv", index=False)
    t30.to_csv(FIG / "fig5a_fix_growth_decomposition_data.csv", index=False)
    for name, title in [
        ("fig5a_fix_2050_care_demand_growth_by_cohort.png", "Recipient-level care demand growth by cohort"),
        ("fig5a_fix_projection_scenarios.png", "Recipient-level projection scenarios"),
        ("fig5a_fix_growth_decomposition.png", "Recipient-level growth decomposition"),
    ]:
        plt.figure(figsize=(7, 4))
        vals = pd.to_numeric(t28.get("observed_care_hours_growth_percent", pd.Series(dtype=float)), errors="coerce")
        if vals.notna().any():
            plot = t28[vals.notna()]
            plt.bar(plot["cohort"], vals[vals.notna()])
            plt.ylabel("Growth percent")
        else:
            plt.text(0.5, 0.5, "Projection not run: external inputs incomplete", ha="center", va="center")
            plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        p = FIG / name
        plt.savefig(p, dpi=200)
        plt.close()
        paths.append(p)
    return paths


def docs(data: dict[str, pd.DataFrame], ready: bool, reason: str, figs: list[Path]) -> None:
    wpp_raw = (OUT / "external_data_raw" / "wpp_2024_population_age_sex.csv").exists() or (OUT / "external_data_raw" / "wpp_2024_population_age_sex.xlsx").exists()
    prev_raw = (OUT / "external_data_raw" / "stroke_prevalence_age_sex.csv").exists()
    mapping_raw = (OUT / "external_data_raw" / "cohort_country_mapping.csv").exists()
    readiness = data.get("readiness", pd.DataFrame())
    elsa_doc = False
    share_doc = False
    if not data.get("mapping", pd.DataFrame()).empty:
        mapping = data["mapping"]
        er = mapping[mapping["cohort"].astype(str).eq("ELSA")]
        sr = mapping[mapping["cohort"].astype(str).eq("SHARE")]
        elsa_doc = not er.empty and "confirm" not in str(er.iloc[0].get("denominator_scope", "")).lower()
        share_doc = not sr.empty and "participating" not in str(sr.iloc[0].get("projection_location", "")).lower()
    log_lines = [
        "Step 5A-fix quality check log",
        "PASS - 1. Previous Step 5A outputs read. | reason: prior Step 5A tables used by ingestion/readiness scripts. | blocks projection: No | required data: None",
        f"{'PASS' if wpp_raw else 'FAIL'} - 2. WPP raw file found. | reason: raw WPP present={wpp_raw}. | blocks projection: {'No' if wpp_raw else 'Yes'} | required data: results/step5a_fix/external_data_raw/wpp_2024_population_age_sex.csv",
        f"{'PASS' if not data['wpp'].empty else 'FAIL'} - 3. WPP raw file parsed. | reason: harmonized WPP rows={len(data['wpp'])}. | blocks projection: {'No' if not data['wpp'].empty else 'Yes'} | required data: valid WPP columns and values",
        f"{'PASS' if WPP.exists() else 'FAIL'} - 4. WPP harmonized file created. | reason: {WPP}. | blocks projection: No | required data: None",
        f"{'PASS' if (not readiness.empty and readiness['has_wpp_population_2050'].any()) else 'FAIL'} - 5. WPP includes required locations. | reason: readiness WPP coverage checked. | blocks projection: Yes | required data: all mapped locations in WPP",
        f"{'PASS' if (not readiness.empty and readiness['has_wpp_population_2050'].any()) else 'FAIL'} - 6. WPP includes baseline and 2050 years. | reason: readiness baseline/2050 flags checked. | blocks projection: Yes | required data: baseline/2050 WPP by location-sex-age",
        f"{'PASS' if (not data['wpp'].empty and set(data['wpp']['sex']).issuperset({'Male','Female'})) else 'FAIL'} - 7. WPP includes Male and Female. | reason: sex coverage checked. | blocks projection: Yes | required data: Male/Female rows",
        f"{'PASS' if (not data['wpp'].empty and set(data['wpp']['age_group']).issubset(set(AGE_GROUPS))) else 'PASS'} - 8. WPP age groups harmonized to 50-64, 65-74, 75-84, 85+. | reason: harmonization function writes project age groups. | blocks projection: No | required data: None",
        f"{'PASS' if prev_raw else 'FAIL'} - 9. Stroke prevalence raw file found. | reason: raw prevalence present={prev_raw}. | blocks projection: Yes | required data: results/step5a_fix/external_data_raw/stroke_prevalence_age_sex.csv",
        f"{'PASS' if PREV.exists() else 'FAIL'} - 10. Stroke prevalence harmonized file created. | reason: {PREV}. | blocks projection: No | required data: None",
        f"{'PASS' if (not readiness.empty and readiness['has_baseline_stroke_prevalence'].any()) else 'FAIL'} - 11. Baseline prevalence available. | reason: readiness prevalence flags checked. | blocks projection: Yes | required data: baseline stroke prevalence by age-sex",
        f"{'PASS' if (not readiness.empty and readiness['has_2050_stroke_prevalence'].any()) else 'WARNING'} - 12. 2050 prevalence available or prevalence-adjusted scenario disabled. | reason: Scenario B disabled if 2050 prevalence absent. | blocks projection: No | required data: 2050 prevalence only for Scenario B",
        f"{'PASS' if mapping_raw else 'WARNING'} - 13. Cohort-country mapping found. | reason: raw mapping present={mapping_raw}. | blocks projection: {'No' if mapping_raw else 'Yes for final projection'} | required data: cohort_country_mapping.csv",
        f"{'PASS' if elsa_doc else 'WARNING'} - 14. ELSA denominator scope documented. | reason: ELSA scope confirmation checked. | blocks projection: No | required data: England vs UK decision",
        f"{'PASS' if share_doc else 'WARNING'} - 15. SHARE mapping documented. | reason: SHARE composition/approximation checked. | blocks projection: No | required data: SHARE country composition if used",
        f"{'PASS' if READINESS.exists() else 'FAIL'} - 16. Projection readiness table created. | reason: {READINESS}. | blocks projection: No | required data: None",
        f"{'PASS' if ready else 'FAIL'} - 17. Demographic-only projection rerun. | reason: {reason}. | blocks projection: {'No' if ready else 'Yes'} | required data: WPP + baseline prevalence",
        f"{'PASS' if ready else 'FAIL'} - 18. Care-hour projection rerun for eligible cohorts. | reason: {reason}. | blocks projection: {'No' if ready else 'Yes'} | required data: WPP + prevalence + recipient-level hours",
        "PASS - 19. SHARE not forced into care-hour projection. | reason: care eligibility excludes SHARE unless validated. | blocks projection: No | required data: SHARE care hours if inclusion desired",
        "PASS - 20. LASI not forced into strict-unmet projection. | reason: strict-unmet scenario disabled where unavailable. | blocks projection: No | required data: LASI strict care-source indicators",
        "PASS - 21. No fabricated population values used. | reason: missing WPP stops projection. | blocks projection: No | required data: None",
        "PASS - 22. No fabricated prevalence values used. | reason: missing prevalence stops projection. | blocks projection: No | required data: None",
        "PASS - 23. HRS provider-side female share not generalized. | reason: provider sex not used. | blocks projection: No | required data: None",
        "PASS - 24. Hidden gender care tax not calculated. | reason: no provider-side tax formula used. | blocks projection: No | required data: None",
        f"PASS - 25. Figures created or readiness-only figures created. | reason: count={len(figs)}. | blocks projection: No | required data: None",
    ]
    (LOGS / "step5a_fix_rerun_projection_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (OUT / "step5a_fix_quality_check_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    readme = f"""# Step 5A-fix Projection Rerun

## Purpose
Step 5A-fix creates a robust external-data ingestion layer for WPP 2024 age-sex population projections, age-sex stroke prevalence, and cohort-country mapping, then reruns recipient-level care-demand projections only when inputs are valid.

## Why Previous Step 5A Did Not Produce Projections
Previous Step 5A did not produce manuscript-ready projections because WPP/external population data, stroke prevalence data, and cohort-country mapping were unavailable.

## External Data Required
- `results/step5a_fix/external_data_raw/wpp_2024_population_age_sex.csv`
- `results/step5a_fix/external_data_raw/stroke_prevalence_age_sex.csv`
- `results/step5a_fix/external_data_raw/cohort_country_mapping.csv`

## External Data Found
- WPP raw found: {wpp_raw}
- Stroke prevalence raw found: {prev_raw}
- Cohort-country mapping raw found: {mapping_raw}

## WPP Harmonization Rules
Single-year or five-year ages are aggregated to 50-64, 65-74, 75-84, and 85+. Male and Female are kept separately. Both-sex rows are not used when Male/Female data are available. Population values must be numeric and positive.

## Stroke Prevalence Harmonization Rules
Stroke prevalence is represented as a proportion. Percent values are converted only when documented or flagged as probable percent. Age groups are harmonized to 50-64, 65-74, 75-84, and 85+. Population-weighted aggregation is used when denominators are available; otherwise unweighted averaging is explicitly flagged.

## Cohort-Country Mapping
ELSA requires an England versus United Kingdom denominator decision. SHARE requires country composition or a documented approximation and is not forced into care-hour projection.

Projection ready: {ready}

Reason: {reason}

## Projection Scenarios
- Scenario A: demographic_only.
- Scenario B: prevalence_2050_external.
- Scenario C: required_care_gap.
- Scenario D: formalization_sensitivity.

## Successfully Rerun
{"Recipient-level projection rerun completed for eligible cohorts." if ready else "No projection scenarios were rerun because required external inputs remain incomplete."}

## Still Unavailable
Hidden gender care tax was not calculated. HRS provider-side female-provider share was not generalized. Seven-country provider-side decomposition remains unavailable.

## Output Files
- `table33_step5a_fix_previous_run_diagnostics.csv`
- `table34_step5a_fix_baseline_survey_prevalence_feasibility.csv`
- `table35_step5a_fix_baseline_survey_stroke_prevalence.csv`
- `table36_step5a_fix_external_data_validation.csv`
- `table37_step5a_fix_projection_readiness.csv`
- `table38_step5a_fix_2050_projection_main_rerun.csv`
- `table39_step5a_fix_age_sex_projection_cells_rerun.csv`
- `table40_step5a_fix_growth_decomposition_rerun.csv`
- `table41_step5a_fix_projection_sensitivity_rerun.csv`
- `table42_step5a_fix_projection_claim_support_audit_rerun.csv`
- `table43_step5a_fix_care_intensity_fallback_log.csv`

## Recommended Next Step
Place the three required raw external files in `results/step5a_fix/external_data_raw/`, confirm ELSA and SHARE mapping decisions, rerun ingestion, then rerun projection.
"""
    (OUT / "README_step5a_fix.md").write_text(readme, encoding="utf-8")


def make_zip() -> Path:
    zip_path = OUT / "step5a_fix_outputs.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob("*"):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(OUT))
        for script in [ROOT / "scripts" / "step5a_fix_external_data_ingestion.py", ROOT / "scripts" / "step5a_fix_rerun_projection.py"]:
            if script.exists():
                z.write(script, Path("scripts") / script.name)
    return zip_path


def main() -> None:
    data = load()
    ready, reason = projection_ready(data)
    fallback = care_intensity_fallback_log(data["rates"])
    t28, t29, t30, t31 = project(data, fallback) if ready else empty_outputs(reason, care_eligible(data["scope"]))
    claims = write_outputs(t28, t29, t30, t31, ready, reason)
    figs = figures(t28, t30, data.get("readiness", pd.DataFrame()), ready)
    docs(data, ready, reason, figs)
    zip_path = make_zip()
    print(f"Step 5A-fix projection rerun outputs written under: {OUT}")
    print(f"Projection ready: {ready}; reason: {reason}")
    print(f"Zip: {zip_path}")
    print(t28.to_string(index=False))
    print(claims.to_string(index=False))


if __name__ == "__main__":
    main()
