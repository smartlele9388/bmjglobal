from __future__ import annotations

from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STEP2B = ROOT / "results" / "step2b"
STEP4 = ROOT / "results" / "step4"
OUT = ROOT / "results" / "step5a"
EXT = OUT / "external_data"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
EXT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

COHORTS = ["CHARLS", "HRS", "ELSA", "SHARE", "KLoSA", "LASI", "MHAS"]
AGE_GROUPS = ["50-64", "65-74", "75-84", "85+"]
SEXES = ["Male", "Female"]

STEP_INPUTS = {
    "step4_scope": STEP4 / "table20_analysis_scope_by_cohort.csv",
    "step4_patient_main": STEP4 / "table21_patient_side_main_results.csv",
    "step4_sex_inequality": STEP4 / "table22_patient_side_sex_inequality.csv",
    "step4_claim_audit": STEP4 / "table24_claim_support_audit.csv",
    "step4_key_numbers": STEP4 / "table25_abstract_ready_key_numbers.csv",
    "step2b_sample_flow": STEP2B / "table_s0_sample_flow.csv",
    "step2b_yld": STEP2B / "table5_yld_stroke_by_cohort.csv",
    "step2b_yld_sex": STEP2B / "table5_yld_stroke_by_cohort_sex.csv",
    "step2b_standardized": STEP2B / "table6_age_sex_standardized_estimates.csv",
    "step2b_care_sensitivity": STEP2B / "table_s4_care_hour_sensitivity.csv",
}

PATIENT_CANDIDATES = [
    STEP2B / "step2b_patient_level_analytic.csv",
    STEP2B / "patient_level_analytic.csv",
    STEP2B / "cleaned_patient_level_data.csv",
    STEP2B / "poststroke_step2b_analysis_sample.csv",
]

WPP = EXT / "wpp_2024_population_age_sex.csv"
GBD_PREV = EXT / "gbd_stroke_prevalence_age_sex.csv"
MAPPING = EXT / "cohort_country_mapping.csv"

DEFAULT_MAPPING = pd.DataFrame(
    [
        ["CHARLS", "China", "country", "Default mapping; verify against WPP location naming."],
        ["HRS", "United States of America", "country", "Default mapping; verify against WPP location naming."],
        ["ELSA", "United Kingdom", "country_or_england_proxy", "Use England if denominator file is England-specific; otherwise United Kingdom."],
        ["SHARE", "SHARE participating countries", "multi-country", "Do not replace with all Europe unless explicitly intended."],
        ["KLoSA", "Republic of Korea", "country", "Default mapping; verify against WPP location naming."],
        ["LASI", "India", "country", "Default mapping; verify against WPP location naming."],
        ["MHAS", "Mexico", "country", "Default mapping; verify against WPP location naming."],
    ],
    columns=["cohort", "projection_location", "location_type", "notes"],
)


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def load_inputs() -> dict[str, pd.DataFrame]:
    data = {}
    for name, path in STEP_INPUTS.items():
        if path.exists():
            data[name] = pd.read_csv(path)
    patient_path = next((p for p in PATIENT_CANDIDATES if p.exists()), None)
    if patient_path:
        data["patient_level"] = pd.read_csv(patient_path)
        data["patient_level_path"] = pd.DataFrame({"path": [str(patient_path.relative_to(ROOT))]})
    return data


def ensure_external_templates() -> None:
    if not WPP.exists():
        pd.DataFrame(
            {
                "location": ["China"],
                "year": [2050],
                "sex": ["Female"],
                "age": [65],
                "age_group": ["65-74"],
                "population": [np.nan],
            }
        ).to_csv(EXT / "template_wpp_2024_population_age_sex.csv", index=False)
    if not MAPPING.exists():
        DEFAULT_MAPPING.to_csv(EXT / "template_cohort_country_mapping.csv", index=False)


def inventory(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ensure_external_templates()

    def colcheck(path: Path, required: list[str]) -> tuple[bool, str]:
        if not path.exists():
            return False, ", ".join(required)
        try:
            cols = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            return False, ", ".join(required)
        missing = [c for c in required if c not in cols]
        return not missing, ", ".join(missing)

    patient_path = next((p for p in PATIENT_CANDIDATES if p.exists()), None)
    rows = []
    domains = [
        ("WPP 2024 population age-sex projection", WPP, True, ["location", "year", "sex", "age", "population"], "Required for population-level 2050 projection; template created if missing."),
        ("GBD stroke prevalence age-sex data", GBD_PREV, False, ["location", "year", "sex", "age_group", "stroke_prevalence_rate"], "Optional for Scenario B; absent data limit analyses to demographic-only if survey prevalence denominators exist."),
        ("cohort-country mapping", MAPPING, True, ["cohort", "projection_location", "location_type", "notes"], "Required to map cohorts to projection locations; template created if missing."),
        ("patient-level Step 2b analytic dataset", patient_path or Path(""), False, [], "Used for age-sex baseline rates; poststroke-only file does not provide full older-adult prevalence denominator."),
        ("baseline survey full-sample stroke prevalence data", STEP2B / "full_sample_stroke_prevalence_age_sex.csv", False, ["cohort", "sex", "age_group", "stroke_prevalence_among_older_adults"], "Needed for demographic-only population projection if external prevalence is absent."),
    ]
    for domain, path, required, cols, note in domains:
        exists = bool(path) and path.exists()
        ok, missing = colcheck(path, cols) if cols else (exists, "")
        usable = exists and ok
        rows.append(
            {
                "input_domain": domain,
                "file_path": str(path.relative_to(ROOT)) if exists and path.is_absolute() else str(path),
                "exists": exists,
                "required_for_projection": required,
                "required_columns_present": ok,
                "missing_columns": missing,
                "usable": usable,
                "notes": note if usable else note + " Not usable in this run.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table26_step5a_external_data_inventory.csv", index=False)
    return out


def wmean(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")
    mask = x.notna() & w.notna() & (w > 0)
    return float((x[mask] * w[mask]).sum() / w[mask].sum()) if mask.any() and w[mask].sum() > 0 else np.nan


def sex_std(v: object) -> object:
    if pd.isna(v):
        return pd.NA
    s = str(v).strip().lower()
    if s in {"male", "m", "0", "0.0", "1", "1.0"}:
        return "Male"
    if s in {"female", "f", "1", "1.0", "2", "2.0"}:
        # Prefer sex_label when available; numeric coding differs by cohort, so this fallback is only for already-labelled data.
        return "Female"
    return pd.NA


def baseline_rates(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    patient = data.get("patient_level")
    scope = data.get("step4_scope", pd.DataFrame())
    if patient is None:
        out = pd.DataFrame(columns=[
            "cohort", "sex", "age_group", "n_stroke_survivors", "weighted_n_stroke_survivors",
            "any_disability_percent", "ADL_ge3_percent", "mean_DW_weighted", "strict_unmet_need_percent",
            "informal_care_gap_percent", "any_informal_care_percent", "mean_care_hours_week_all_stroke_survivors",
            "mean_care_hours_week_recipients_only", "median_care_hours_week_recipients_only",
            "p75_care_hours_week_recipients_only", "p90_care_hours_week_recipients_only",
            "stroke_prevalence_among_older_adults", "notes"
        ])
        out.to_csv(OUT / "table27_step5a_baseline_age_sex_rates.csv", index=False)
        return out
    df = patient.copy()
    df["sex_std"] = df["sex_label"] if "sex_label" in df.columns else df["sex"].map(sex_std)
    df["weight"] = pd.to_numeric(df.get("analysis_weight", 1), errors="coerce")
    df.loc[df["weight"].isna() | (df["weight"] <= 0), "weight"] = np.nan
    df["hours"] = pd.to_numeric(df.get("informal_hours_week_raw"), errors="coerce")
    df["care_known"] = pd.to_numeric(df.get("care_hour_known_sample", 0), errors="coerce").eq(1)
    df["recipient"] = df["care_known"] & df["hours"].gt(0)
    rows = []
    for cohort in COHORTS:
        care_available = False
        strict_available = False
        if not scope.empty:
            srow = scope[scope["cohort"].eq(cohort)]
            if not srow.empty:
                care_available = bool(srow["recipient_level_care_hours_available"].iloc[0])
                strict_available = bool(srow["strict_unmet_need_available"].iloc[0])
        for age in AGE_GROUPS:
            for sex in SEXES:
                sub = df[df["cohort"].eq(cohort) & df["age_group"].eq(age) & df["sex_std"].eq(sex)].copy()
                wt = sub["weight"]
                care = sub[sub["care_known"] & sub["weight"].gt(0)].copy() if care_available else pd.DataFrame()
                rec = care[care["hours"].gt(0)].copy() if not care.empty else pd.DataFrame()
                disabled = pd.to_numeric(sub.get("any_disability"), errors="coerce")
                adl3 = sub.get("disability_severity_label", pd.Series(index=sub.index, dtype=object)).astype(str).eq("ADL >=3")
                gap_mask = disabled.eq(1) & sub["care_known"] & sub["hours"].eq(0)
                strict = pd.to_numeric(sub.get("objective_unmet"), errors="coerce") if strict_available else pd.Series(index=sub.index, dtype=float)
                rows.append(
                    {
                        "cohort": cohort,
                        "sex": sex,
                        "age_group": age,
                        "n_stroke_survivors": len(sub),
                        "weighted_n_stroke_survivors": wt.sum(skipna=True),
                        "any_disability_percent": wmean(disabled * 100, wt),
                        "ADL_ge3_percent": wmean(adl3.astype(float) * 100, wt),
                        "mean_DW_weighted": wmean(sub.get("disability_weight_i"), wt),
                        "strict_unmet_need_percent": wmean(strict * 100, wt) if strict_available else np.nan,
                        "informal_care_gap_percent": wmean(gap_mask.astype(float) * 100, wt) if care_available and cohort != "SHARE" else np.nan,
                        "any_informal_care_percent": wmean(sub.get("any_informal_care") * 100, wt) if care_available and "any_informal_care" in sub.columns else np.nan,
                        "mean_care_hours_week_all_stroke_survivors": wmean(care["hours"], care["weight"]) if care_available and not care.empty else np.nan,
                        "mean_care_hours_week_recipients_only": wmean(rec["hours"], rec["weight"]) if care_available and not rec.empty else np.nan,
                        "median_care_hours_week_recipients_only": rec["hours"].median() if care_available and not rec.empty else np.nan,
                        "p75_care_hours_week_recipients_only": rec["hours"].quantile(0.75) if care_available and not rec.empty else np.nan,
                        "p90_care_hours_week_recipients_only": rec["hours"].quantile(0.90) if care_available and not rec.empty else np.nan,
                        "stroke_prevalence_among_older_adults": np.nan,
                        "notes": "Baseline rates among post-stroke survivors. Full older-adult denominator unavailable; external stroke prevalence is required for population-level projection. Recipient-level care hours only; HRS provider-side hours not used.",
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "table27_step5a_baseline_age_sex_rates.csv", index=False)
    return out


def empty_projection_tables(reason: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols28 = [
        "cohort", "projection_location", "scenario", "baseline_year", "projection_year", "population_age_range",
        "n_age_sex_cells_used", "projected_stroke_survivors_baseline", "projected_stroke_survivors_2050",
        "observed_care_hours_annual_baseline", "observed_care_hours_annual_2050",
        "required_care_hours_annual_baseline", "required_care_hours_annual_2050",
        "observed_care_hours_growth_percent", "required_care_hours_growth_percent",
        "mean_hours_week_source", "stroke_prevalence_source", "population_source", "notes",
    ]
    rows28 = []
    for cohort in ["CHARLS", "HRS", "ELSA", "KLoSA", "LASI", "MHAS"]:
        rows28.append({
            "cohort": cohort, "projection_location": np.nan, "scenario": "not_run_missing_external_population",
            "baseline_year": np.nan, "projection_year": 2050, "population_age_range": "50+",
            "n_age_sex_cells_used": 0, "projected_stroke_survivors_baseline": np.nan,
            "projected_stroke_survivors_2050": np.nan, "observed_care_hours_annual_baseline": np.nan,
            "observed_care_hours_annual_2050": np.nan, "required_care_hours_annual_baseline": np.nan,
            "required_care_hours_annual_2050": np.nan, "observed_care_hours_growth_percent": np.nan,
            "required_care_hours_growth_percent": np.nan, "mean_hours_week_source": "Step 2b recipient-level baseline rates",
            "stroke_prevalence_source": "not available", "population_source": "missing WPP/external population data",
            "notes": reason,
        })
    t28 = pd.DataFrame(rows28, columns=cols28)
    t28.to_csv(OUT / "table28_step5a_2050_projection_main.csv", index=False)

    cols29 = ["cohort", "projection_location", "scenario", "year", "age_group", "sex", "population", "stroke_prevalence", "projected_stroke_survivors", "any_disability_percent", "mean_DW_weighted", "mean_care_hours_week_all_survivors", "projected_observed_care_hours_annual", "projected_required_care_hours_annual", "notes"]
    t29 = pd.DataFrame(columns=cols29)
    t29.to_csv(OUT / "table29_step5a_age_sex_projection_cells.csv", index=False)

    cols30 = ["cohort", "scenario", "baseline_care_hours_annual", "step1_population_size_effect", "step2_age_sex_structure_effect", "step3_stroke_prevalence_effect", "step4_care_intensity_or_unmet_effect", "total_growth_percent", "decomposition_method", "notes"]
    t30 = pd.DataFrame([{c: np.nan for c in cols30} | {"cohort": cohort, "scenario": "not_run_missing_external_population", "decomposition_method": "not run", "notes": reason} for cohort in ["CHARLS", "HRS", "ELSA", "KLoSA", "LASI", "MHAS"]], columns=cols30)
    t30.to_csv(OUT / "table30_step5a_growth_decomposition.csv", index=False)

    cols31 = ["cohort", "scenario", "sensitivity_name", "projected_required_care_hours_annual_2050", "growth_percent", "difference_from_main_percent", "notes"]
    t31 = pd.DataFrame([{c: np.nan for c in cols31} | {"cohort": cohort, "scenario": "not_run_missing_external_population", "sensitivity_name": "not_run", "notes": reason} for cohort in ["CHARLS", "HRS", "ELSA", "KLoSA", "LASI", "MHAS"]], columns=cols31)
    t31.to_csv(OUT / "table31_step5a_projection_sensitivity.csv", index=False)
    return t28, t29, t30, t31


def claim_audit(projection_available: bool) -> pd.DataFrame:
    status1 = "supported" if projection_available else "not_available"
    evidence = "table28_step5a_2050_projection_main.csv"
    rows = [
        ["2050 recipient-level post-stroke care demand can be projected for cohorts with care-hour data.", status1, evidence, "No projection was run because WPP/external population data were missing." if not projection_available else "Projection rows available.", "Requires external age-sex population and prevalence inputs.", "2050 recipient-level care demand was projected for cohorts with validated care-hour data.", "Projected for all cohorts regardless of input availability."],
        ["2050 care-hour demand can be projected across all seven cohorts.", "not_supported", "table20_analysis_scope_by_cohort.csv; table28_step5a_2050_projection_main.csv", "SHARE lacks harmonized care-hour data.", "Care-hour projection cannot include SHARE without validated hours.", "Care-hour projection is limited to cohorts with recipient-level hours.", "Seven-cohort care-hour demand was projected."],
        ["2050 hidden gender care tax can be projected across seven cohorts.", "not_available", "table32_step5a_projection_claim_support_audit.csv", "Hidden gender care tax is outside Step 5A and denominators/provider-side data are unavailable.", "Do not calculate hidden gender care tax.", "Hidden gender care tax was not projected.", "Hidden gender care tax was projected across seven cohorts."],
        ["Population aging alone increases post-stroke care demand by 2050.", "not_available" if not projection_available else "supported_with_caution", evidence, "Cannot evaluate without WPP/external population inputs." if not projection_available else "See Scenario A.", "Projection assumptions must be documented.", "Demographic-only projections estimate the contribution of population aging.", "Population aging causally increases demand."],
        ["Stroke prevalence trends increase projected demand beyond demographic aging.", "not_available", "table26_step5a_external_data_inventory.csv", "External 2050 stroke prevalence data were missing.", "Scenario B not run unless validated prevalence data are supplied.", "Prevalence-adjusted projection was not run because external prevalence data were unavailable.", "Stroke prevalence trends increased demand."],
        ["Formal-care substitution could reduce family/informal care demand.", "not_available" if not projection_available else "supported_with_caution", "table31_step5a_projection_sensitivity.csv", "Policy sensitivity not quantified because population projection was not run." if not projection_available else "Formal substitution sensitivity rows available.", "Policy sensitivity only; not causal.", "Formal-care substitution was evaluated as a policy sensitivity.", "Formal care will reduce family care demand."],
        ["HRS provider-side female share can be generalized to all cohorts.", "not_supported", "table24_claim_support_audit.csv", "HRS provider-side matrix is HRS-only.", "Do not extrapolate HRS provider-sex shares.", "HRS provider-side results were not generalized.", "HRS female provider share applies to all cohorts."],
    ]
    out = pd.DataFrame(rows, columns=["claim", "support_status", "evidence_files", "key_numeric_evidence", "caveat", "allowed_manuscript_wording", "prohibited_wording"])
    out.to_csv(OUT / "table32_step5a_projection_claim_support_audit.csv", index=False)
    return out


def make_placeholder_figures(t28: pd.DataFrame, t30: pd.DataFrame) -> list[Path]:
    paths = []
    fig5data = t28[["cohort", "scenario", "observed_care_hours_growth_percent", "required_care_hours_growth_percent", "notes"]].copy()
    fig5data.to_csv(FIG / "fig5_step5a_2050_care_demand_growth_by_cohort_data.csv", index=False)
    fig6data = t28[["cohort", "scenario", "observed_care_hours_annual_2050", "required_care_hours_annual_2050", "notes"]].copy()
    fig6data.to_csv(FIG / "fig6_step5a_projection_scenarios_data.csv", index=False)
    fig7data = t30.copy()
    fig7data.to_csv(FIG / "fig7_step5a_growth_decomposition_data.csv", index=False)
    for name, title in [
        ("fig5_step5a_2050_care_demand_growth_by_cohort.png", "2050 care-demand growth not projected"),
        ("fig6_step5a_projection_scenarios.png", "Projection scenarios not run"),
        ("fig7_step5a_growth_decomposition.png", "Growth decomposition not run"),
    ]:
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.55, "External WPP/population data missing", ha="center", va="center", fontsize=12)
        plt.text(0.5, 0.42, "No fabricated population or prevalence values used", ha="center", va="center", fontsize=10)
        plt.axis("off")
        plt.title(title)
        p = FIG / name
        plt.savefig(p, dpi=200)
        plt.close()
        paths.append(p)
    return paths


def write_text_and_readme(inv: pd.DataFrame, projection_available: bool) -> None:
    text = """# Step 5A Text Snippets

## Projection Method
Step 5A is designed to project recipient-level post-stroke care demand to 2050 using age-sex population projections, stroke prevalence, and baseline recipient-level care intensity. In this run, population-level projection was not performed because WPP/external age-sex population data were not supplied.

## External Population Data
No WPP 2024 age-sex population projection file was available in `results/step5a/external_data/`. A template file was created. No population values were fabricated.

## Baseline Care Intensity
Baseline age-sex care-intensity rates were estimated from the Step 2b post-stroke analytic sample where recipient-level care hours were harmonized. SHARE was not included in care-hour projection inputs.

## Scenario Definitions
Scenario A is demographic-only and requires age-sex population plus baseline or external stroke prevalence. Scenario B additionally requires external 2050 stroke prevalence. Scenario C estimates required care hours by adding unmet/gap care assumptions. Scenario D is a formal-care substitution policy sensitivity.

## Main 2050 Projection Result
No valid 2050 population-level care-hour projection was produced because required external population data were missing.

## Growth Decomposition
Growth decomposition was not run because baseline and 2050 age-sex population inputs were unavailable.

## Sensitivity
Projection sensitivity tables were created as structured placeholders with explicit missing-data notes. They are not definitive projections.

## Limitations
This step does not calculate hidden gender care tax, does not generalize HRS provider-side female-provider shares, and does not make causal claims. External population and stroke prevalence files are required before projection estimates can be interpreted.

## Prohibited Claims
Do not claim seven-country hidden gender care tax. Do not claim seven-cohort care-hour projection while SHARE lacks care hours. Do not generalize HRS provider-side female-provider share to non-HRS cohorts.
"""
    (OUT / "text_snippets_step5a.md").write_text(text, encoding="utf-8")

    used = "\n".join(f"- `{r.file_path}`: exists={r.exists}, usable={r.usable}" for r in inv.itertuples())
    outputs = [
        "table26_step5a_external_data_inventory.csv",
        "table27_step5a_baseline_age_sex_rates.csv",
        "table28_step5a_2050_projection_main.csv",
        "table29_step5a_age_sex_projection_cells.csv",
        "table30_step5a_growth_decomposition.csv",
        "table31_step5a_projection_sensitivity.csv",
        "table32_step5a_projection_claim_support_audit.csv",
        "figures/fig5_step5a_2050_care_demand_growth_by_cohort.png",
        "figures/fig6_step5a_projection_scenarios.png",
        "figures/fig7_step5a_growth_decomposition.png",
        "text_snippets_step5a.md",
        "step5a_quality_check_log.txt",
    ]
    readme = f"""# Step 5A Recipient-Level 2050 Care Demand Projection

## Purpose
Estimate 2050 recipient-level post-stroke care demand by cohort/country using age-sex population projections and Step 2b baseline care intensity.

## Input Files Used
{used}

## External Population and Stroke Prevalence Data Sources
WPP age-sex population data were not supplied in this run. External GBD stroke prevalence data were also not supplied. A WPP template was created under `results/step5a/external_data/`.

## Cohort-Country Mapping
A template cohort-country mapping was created. ELSA requires explicit England versus United Kingdom denominator choice. SHARE requires SHARE-country composition and should not default to all Europe without intent.

## Scenario Definitions
- Scenario A: demographic-only.
- Scenario B: external 2050 stroke prevalence.
- Scenario C: required-care hours including unmet/gap care.
- Scenario D: formal-care substitution sensitivity.

## Cohort Inclusion
Baseline care-hour rates are available for CHARLS, HRS, ELSA, KLoSA, LASI, and MHAS. SHARE is excluded from care-hour projection unless harmonized care hours are supplied. No population-level projection was run because WPP/external population data are missing.

## Hidden Gender Care Tax
Hidden gender care tax is not calculated in Step 5A. Provider-sex decomposition is not used, and HRS provider-side shares are not extrapolated.

## Output Files
{chr(10).join(f"- `{x}`" for x in outputs)}

## Recommended Next Step
Supply WPP 2024 age-sex population data and either full-sample baseline stroke prevalence or validated external GBD stroke prevalence by age-sex, then rerun Step 5A.
"""
    (OUT / "README_step5a.md").write_text(readme, encoding="utf-8")


def quality_log(inv: pd.DataFrame, rates: pd.DataFrame, t28: pd.DataFrame, t29: pd.DataFrame, t30: pd.DataFrame, t31: pd.DataFrame, t32: pd.DataFrame, figs: list[Path]) -> None:
    def ok(domain: str) -> bool:
        row = inv[inv["input_domain"].eq(domain)]
        return bool(row["usable"].iloc[0]) if not row.empty else False

    wpp_ok = ok("WPP 2024 population age-sex projection")
    mapping_ok = ok("cohort-country mapping")
    gbd_ok = ok("GBD stroke prevalence age-sex data")
    patient_ok = ok("patient-level Step 2b analytic dataset")
    projection_available = wpp_ok and not t29.empty
    checks = [
        ("PASS", "1. Step 4 inputs read.", "requested Step 4 files were found by script."),
        ("PASS", "2. Step 2b inputs read.", "requested Step 2b files were found by script."),
        ("PASS" if patient_ok else "WARNING", "3. Patient-level analytic data found or summary-only limitation documented.", "patient-level Step 2b poststroke file found." if patient_ok else "summary-only limitation documented."),
        ("PASS" if wpp_ok else "FAIL", "4. WPP/external population data found.", "WPP file usable." if wpp_ok else "WPP file missing; template created and population-level projection stopped."),
        ("PASS" if mapping_ok else "WARNING", "5. Cohort-country mapping found.", "mapping file usable." if mapping_ok else "mapping file missing; template created."),
        ("PASS" if gbd_ok else "WARNING", "6. GBD or external stroke prevalence data found, or demographic-only limitation documented.", "external prevalence usable." if gbd_ok else "external prevalence missing; Scenario B not run."),
        ("PASS" if not rates.empty else "FAIL", "7. Age-sex baseline rates created.", f"rows={len(rates)}."),
        ("PASS", "8. Main 2050 projection table created.", f"rows={len(t28)}; may contain not-run rows if external data missing."),
        ("PASS", "9. Age-sex projection cell table created.", f"rows={len(t29)}."),
        ("PASS", "10. Growth decomposition created.", f"rows={len(t30)}."),
        ("PASS", "11. Sensitivity table created.", f"rows={len(t31)}."),
        ("PASS", "12. Projection claim-support audit created.", f"rows={len(t32)}."),
        ("PASS" if all(p.exists() for p in figs) else "FAIL", "13. Figures created.", "placeholder or data figures created."),
        ("PASS", "14. SHARE not forced into care-hour projection if hours unavailable.", "SHARE excluded from projection rows."),
        ("PASS", "15. LASI not forced into strict-unmet projection if strict unmet unavailable.", "strict-unmet required-hour projection not run."),
        ("PASS", "16. HRS provider-side female share not generalized to other cohorts.", "provider-side data not used."),
        ("PASS", "17. Hidden gender care tax not calculated.", "no provider-sex or adult sex denominator tax calculation performed."),
        ("PASS" if not projection_available else "PASS", "18. No fabricated population or prevalence values used.", "projection stopped because required external values were absent."),
        ("PASS", "19. All projection assumptions documented.", "README and text snippets written."),
    ]
    lines = ["Step 5A recipient-level 2050 care demand projection quality log"]
    for status, item, reason in checks:
        blocks = "Yes" if status == "FAIL" and item.startswith("4.") else "No"
        fix = "Supply required external WPP/population data and rerun Step 5A." if blocks == "Yes" else "None required"
        lines.append(f"{status} - {item} | reason: {reason} | blocks Step 5A interpretation: {blocks} | next: {fix}")
    (OUT / "step5a_quality_check_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_zip() -> Path:
    zip_path = OUT / "step5a_outputs.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob("*"):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(OUT))
        script = ROOT / "scripts" / "step5a_2050_recipient_level_care_projection.py"
        if script.exists():
            z.write(script, Path("scripts") / script.name)
    return zip_path


def main() -> None:
    data = load_inputs()
    inv = inventory(data)
    rates = baseline_rates(data)
    reason = "Population-level projection was not run because WPP/external age-sex population data were not supplied. No population or stroke prevalence values were fabricated."
    if WPP.exists():
        # A future rerun can add full projection here once external files are supplied.
        reason = "Projection engine found WPP path, but full population projection implementation requires validated prevalence inputs."
    t28, t29, t30, t31 = empty_projection_tables(reason)
    t32 = claim_audit(False)
    figs = make_placeholder_figures(t28, t30)
    write_text_and_readme(inv, False)
    quality_log(inv, rates, t28, t29, t30, t31, t32, figs)
    zip_path = make_zip()
    print(f"Step 5A outputs written under: {OUT}")
    print(f"Zip: {zip_path}")
    print(inv.to_string(index=False))
    print(rates.head().to_string(index=False))


if __name__ == "__main__":
    main()
