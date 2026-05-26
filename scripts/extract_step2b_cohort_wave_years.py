#!/usr/bin/env python3
"""
Recover Step 2b analytic wave/year information from project outputs.

This audit is intentionally based on project files, not external memory:
- Step 2b exported analytic sample supplies cohort, respondent_id, and wave.
- Step 1 protocol code documents that the analytic sample selected the latest
  post-stroke record per respondent after sorting by cohort/respondent/wave.
- Local harmonized documentation/file names document the cohort wave-year ranges.

Outputs are written under output/inventory/ as required by AGENTS.md.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "inventory"
SAMPLE = ROOT / "results" / "step2b" / "poststroke_step2b_analysis_sample.csv"
PROTOCOL_SCRIPT = ROOT / "scripts" / "gender_tax_step1_protocol_analysis.py"


COHORT_ORDER = ["CHARLS", "HRS", "ELSA", "SHARE", "KLoSA", "LASI", "MHAS"]


WAVE_YEAR: Dict[str, Dict[int, str]] = {
    "CHARLS": {1: "2011", 2: "2013", 3: "2015", 4: "2018"},
    "HRS": {
        1: "1992",
        2: "1994",
        3: "1996",
        4: "1998",
        5: "2000",
        6: "2002",
        7: "2004",
        8: "2006",
        9: "2008",
        10: "2010",
        11: "2012",
        12: "2014",
        13: "2016",
        14: "2018",
        15: "2020",
    },
    "ELSA": {
        1: "2002-2003",
        2: "2004-2005",
        3: "2006-2007",
        4: "2008-2009",
        5: "2010-2011",
        6: "2012-2013",
        7: "2014-2015",
        8: "2016-2017",
        9: "2018-2019",
    },
    "SHARE": {
        1: "2004-2005",
        2: "2006-2007",
        4: "2010-2011",
        5: "2013",
        6: "2015",
        7: "2017",
        8: "2019-2020",
    },
    "KLoSA": {
        1: "2006",
        2: "2008",
        3: "2010",
        4: "2012",
        5: "2014",
        6: "2016",
        7: "2018",
        8: "2020",
    },
    "LASI": {1: "2017-2021 documentation range"},
    "MHAS": {1: "2001", 2: "2003", 3: "2012", 4: "2015", 5: "2018"},
}


EVIDENCE_GLOBS: Dict[str, List[str]] = {
    "CHARLS": [
        "raw_data/Harmonized_CHARLS_D/Harmonized_CHARLS_D.pdf",
        "scripts/pcgi_initial_analysis.py",
        "scripts/gender_tax_step1_protocol_analysis.py",
    ],
    "HRS": [
        "3. HRS*/HRS_*/Raw_data/Gateway Harmonized HRS/Harmonized HRS D 1992-2021.pdf",
        "3. HRS*/HRS_*/Temp_data/HRS.dta",
        "scripts/pcgi_initial_analysis.py",
        "scripts/gender_tax_step1_protocol_analysis.py",
    ],
    "ELSA": [
        "2. ELSA*/Raw_data/Harmonized ELSA/5050_harmonized_elsa_g3_2002-2019.pdf",
        "2. ELSA*/Raw_data/Harmonized ELSA/h_elsa_g3.dta",
        "scripts/pcgi_initial_analysis.py",
        "scripts/gender_tax_step1_protocol_analysis.py",
    ],
    "SHARE": [
        "7.SHARE*/**/Harmonized SHARE/Harmonized SHARE F 2004-2020.pdf",
        "7.SHARE*/**/Harmonized SHARE/H_SHARE_f2.dta",
        "scripts/pcgi_initial_analysis.py",
        "scripts/gender_tax_step1_protocol_analysis.py",
    ],
    "KLoSA": [
        "4. KLoSA*/**/H_KLoSA/Harmonized KLoSA E.2 2006-2020.pdf",
        "4. KLoSA*/**/H_KLoSA_e2.dta",
        "scripts/pcgi_initial_analysis.py",
        "scripts/gender_tax_step1_protocol_analysis.py",
    ],
    "LASI": [
        "5. LASI*/**/Harmonized LASI (A.3)/Harmonized LASI A.3 2017-2021.pdf",
        "5. LASI*/**/H_LASI_a3.dta",
        "scripts/pcgi_initial_analysis.py",
        "scripts/gender_tax_step1_protocol_analysis.py",
    ],
    "MHAS": [
        "6. MHAS*/**/Harmonized MHAS File/Harmonized_MHAS_C.2_2001_2018.pdf",
        "6. MHAS*/**/H_MHAS_c2.dta",
        "scripts/pcgi_initial_analysis.py",
        "scripts/gender_tax_step1_protocol_analysis.py",
    ],
}


COHORT_NOTES = {
    "CHARLS": "Multiple waves appear because Step 2b selects the latest post-stroke respondent-wave, not one fixed cohort wave.",
    "HRS": "Multiple waves appear because Step 2b selects the latest post-stroke respondent-wave. Local HRS harmonized documentation spans 1992-2021; observed r15 maps to the 2020 wave.",
    "ELSA": "ELSA waves are fieldwork periods rather than single calendar years; Step 2b uses latest post-stroke respondent-wave through wave 9.",
    "SHARE": "SHARE wave 3 is not present in the Step 2b analytic sample; care-hour analyses still exclude SHARE elsewhere when harmonized hours are unavailable.",
    "KLoSA": "KLoSA Step 2b uses H_KLoSA_e2, observed through wave 8.",
    "LASI": "Only one harmonized LASI wave is present locally; documentation file covers a 2017-2021 range, so a single calendar year is not recoverable from the exported Step 2b sample.",
    "MHAS": "MHAS Step 2b uses observed waves 1-5, latest post-stroke respondent-wave through 2018.",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_globs(patterns: Iterable[str]) -> List[str]:
    found: List[str] = []
    for pattern in patterns:
        matches = sorted(ROOT.glob(pattern))
        if matches:
            found.extend(rel(m) for m in matches)
        else:
            found.append(f"NOT_FOUND:{pattern}")
    return found


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def pct(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return round(num / den * 100, 2)


def markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as a GitHub-flavored Markdown table."""
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)

    def clean(value: object) -> str:
        if pd.isna(value):
            text = ""
        else:
            text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(clean(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_markdown(audit: pd.DataFrame, dist: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Corrected Step 2b Cohort Wave/Year Audit")
    lines.append("")
    lines.append("## Main Correction")
    lines.append("")
    lines.append(
        "The Step 2b analytic file does not contain a single fixed wave/year per cohort. "
        "The project code builds person-wave records and then keeps the latest available "
        "post-stroke observation for each respondent. Therefore, the correct reporting "
        "is a cohort-specific distribution of latest respondent-level waves/years, not "
        "one fixed cohort wave/year."
    )
    lines.append("")
    lines.append("## Cohort-Level Summary")
    lines.append("")
    cols = [
        "cohort",
        "n_latest_poststroke_records",
        "waves_observed",
        "years_or_periods_observed",
        "latest_wave_observed",
        "latest_year_or_period_observed",
        "modal_wave",
        "modal_year_or_period",
        "modal_wave_n",
        "modal_wave_percent",
        "fixed_single_wave_year_for_all_respondents",
    ]
    lines.append(markdown_table(audit[cols]))
    lines.append("")
    lines.append("## Wave Distribution")
    lines.append("")
    lines.append(
        markdown_table(
            dist[
                [
                    "cohort",
                    "wave",
                    "mapped_year_or_period",
                    "n_latest_poststroke_records",
                    "percent_within_cohort",
                    "mapping_status",
                ]
            ]
        )
    )
    lines.append("")
    lines.append("## Evidence")
    lines.append("")
    lines.append(
        "- Analytic sample: `results/step2b/poststroke_step2b_analysis_sample.csv` "
        "with columns `cohort`, `respondent_id`, and `wave`."
    )
    lines.append(
        "- Selection logic: `scripts/gender_tax_step1_protocol_analysis.py` sorts "
        "person-wave records by dataset/respondent/wave and keeps `.tail(1)` among "
        "post-stroke rows."
    )
    lines.append(
        "- Wave-to-year labels are based on local harmonized documentation and filenames "
        "listed in `step2b_cohort_wave_year_file_candidates.csv`."
    )
    lines.append("")
    lines.append("## Recommended Manuscript/Table Wording")
    lines.append("")
    lines.append(
        "Use: \"latest available post-stroke respondent-level observation\" and report "
        "the wave/year distribution. Avoid saying that all respondents in a cohort came "
        "from one fixed wave/year, except that LASI has only one harmonized wave locally "
        "and even there the exported file does not identify a single calendar year."
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    if not SAMPLE.exists():
        raise FileNotFoundError(f"Missing Step 2b analytic sample: {SAMPLE}")

    df = pd.read_csv(SAMPLE, usecols=["cohort", "respondent_id", "wave"])
    df["cohort"] = df["cohort"].astype(str)
    df["wave"] = pd.to_numeric(df["wave"], errors="coerce").astype("Int64")

    dist = (
        df.dropna(subset=["wave"])
        .groupby(["cohort", "wave"], dropna=False)
        .size()
        .reset_index(name="n_latest_poststroke_records")
    )
    totals = df.groupby("cohort").size().rename("cohort_total").reset_index()
    dist = dist.merge(totals, on="cohort", how="left")
    dist["percent_within_cohort"] = dist.apply(
        lambda r: pct(r["n_latest_poststroke_records"], r["cohort_total"]), axis=1
    )
    dist["mapped_year_or_period"] = dist.apply(
        lambda r: WAVE_YEAR.get(r["cohort"], {}).get(int(r["wave"]), "unmapped"),
        axis=1,
    )
    dist["mapping_status"] = dist["mapped_year_or_period"].apply(
        lambda x: "mapped_from_local_project_documentation" if x != "unmapped" else "unmapped"
    )
    dist["source_sample"] = rel(SAMPLE)
    dist = dist.sort_values(
        by=["cohort", "wave"],
        key=lambda s: s.map({c: i for i, c in enumerate(COHORT_ORDER)}) if s.name == "cohort" else s,
    )

    audit_rows = []
    evidence_rows = []
    for cohort in COHORT_ORDER:
        g = df[df["cohort"].eq(cohort)].copy()
        dg = dist[dist["cohort"].eq(cohort)].copy()
        if g.empty:
            waves = []
            years = []
            latest_wave = ""
            latest_year = ""
            modal_wave = ""
            modal_year = ""
            modal_n = 0
            modal_percent = float("nan")
        else:
            waves = [int(x) for x in sorted(g["wave"].dropna().unique())]
            years = [WAVE_YEAR.get(cohort, {}).get(w, "unmapped") for w in waves]
            latest_wave = max(waves) if waves else ""
            latest_year = WAVE_YEAR.get(cohort, {}).get(int(latest_wave), "unmapped") if latest_wave != "" else ""
            modal = dg.sort_values(["n_latest_poststroke_records", "wave"], ascending=[False, False]).head(1)
            modal_wave = int(modal["wave"].iloc[0]) if not modal.empty else ""
            modal_year = WAVE_YEAR.get(cohort, {}).get(int(modal_wave), "unmapped") if modal_wave != "" else ""
            modal_n = int(modal["n_latest_poststroke_records"].iloc[0]) if not modal.empty else 0
            modal_percent = float(modal["percent_within_cohort"].iloc[0]) if not modal.empty else float("nan")

        single_wave = len(waves) == 1
        fixed_wording = (
            f"single observed harmonized wave {waves[0]} ({years[0]})"
            if single_wave and waves
            else "no; latest post-stroke respondent-level waves vary within cohort"
        )
        evidence = resolve_globs(EVIDENCE_GLOBS.get(cohort, []))
        evidence_rows.append(
            {
                "cohort": cohort,
                "source_sample": rel(SAMPLE),
                "selection_logic_script": rel(PROTOCOL_SCRIPT),
                "evidence_files": " | ".join(evidence),
                "evidence_note": "Local files used to map observed harmonized wave numbers to survey years/periods.",
            }
        )

        audit_rows.append(
            {
                "cohort": cohort,
                "analytic_sample_definition": "latest available post-stroke respondent-level observation",
                "n_latest_poststroke_records": int(len(g)),
                "waves_observed": ", ".join(str(w) for w in waves),
                "years_or_periods_observed": ", ".join(years),
                "latest_wave_observed": latest_wave,
                "latest_year_or_period_observed": latest_year,
                "modal_wave": modal_wave,
                "modal_year_or_period": modal_year,
                "modal_wave_n": modal_n,
                "modal_wave_percent": round(modal_percent, 2) if pd.notna(modal_percent) else "",
                "single_wave_in_step2b_sample": yes_no(single_wave),
                "fixed_single_wave_year_for_all_respondents": fixed_wording,
                "source_sample": rel(SAMPLE),
                "selection_logic": "person-wave post-stroke rows sorted by dataset/respondent_id/wave, then latest row retained",
                "evidence_files": " | ".join(evidence),
                "notes": COHORT_NOTES.get(cohort, ""),
            }
        )

    audit = pd.DataFrame(audit_rows)
    evidence_df = pd.DataFrame(evidence_rows)

    audit_path = OUT / "step2b_cohort_wave_year_audit.csv"
    dist_path = OUT / "step2b_cohort_wave_distribution.csv"
    evidence_path = OUT / "step2b_cohort_wave_year_file_candidates.csv"
    md_path = OUT / "step2b_cohort_wave_year_summary.md"
    zip_path = OUT / "step2b_cohort_wave_year_audit_results.zip"

    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    dist.to_csv(dist_path, index=False, encoding="utf-8-sig")
    evidence_df.to_csv(evidence_path, index=False, encoding="utf-8-sig")
    write_markdown(audit, dist, md_path)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in [audit_path, dist_path, evidence_path, md_path, Path(__file__)]:
            zf.write(p, arcname=rel(p))

    print(f"Created {audit_path}")
    print(f"Created {dist_path}")
    print(f"Created {evidence_path}")
    print(f"Created {md_path}")
    print(f"Created {zip_path}")


if __name__ == "__main__":
    main()
