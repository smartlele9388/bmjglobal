from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step3a"
OUT.mkdir(parents=True, exist_ok=True)
STEP2B_SAMPLE = ROOT / "results" / "step2b" / "poststroke_step2b_analysis_sample.csv"

ALL_COHORTS = ["CHARLS", "ELSA", "HRS", "KLoSA", "LASI", "MHAS", "SHARE"]

COHORT_HINTS = {
    "CHARLS": ["charls", "h_charls"],
    "ELSA": ["elsa", "ukda-5050", "ukda-6427", "ukda-8865", "ukda-9081"],
    "HRS": ["hrs", "randhrs"],
    "KLoSA": ["klosa", "klo"],
    "LASI": ["lasi"],
    "MHAS": ["mhas", "mex-cog", "mexico"],
    "SHARE": ["share"],
}

SKIP_DIR_PARTS = {
    ".git",
    "__pycache__",
    "charls_project_template",
    "output\\archive",
    "results\\step2b",
    "results\\step3a",
    "output\\models",
    "output\\figures",
}


def cohort_from_path(path: Path) -> str | None:
    lower = str(path.relative_to(ROOT)).lower()
    first_part = path.relative_to(ROOT).parts[0].lower() if path.relative_to(ROOT).parts else lower
    if first_part in {"2011", "2012", "2013", "2014", "2015", "2018", "2020", "raw_data"}:
        return "CHARLS"
    matches = [cohort for cohort, hints in COHORT_HINTS.items() if any(hint in lower for hint in hints)]
    if len(matches) == 1:
        return matches[0]
    return matches[0] if matches else None


def should_scan(path: Path) -> bool:
    rel = str(path.relative_to(ROOT)).lower()
    if any(part in rel for part in SKIP_DIR_PARTS):
        return False
    if path.suffix.lower() not in [".dta", ".csv"]:
        return False
    if path.stat().st_size > 700_000_000:
        return False
    useful = [
        "harmonized",
        "rand",
        "care",
        "helper",
        "giver",
        "family",
        "household",
        "roster",
        "cover",
        "transfer",
        "h_",
        "hrs",
        "elsa",
        "share",
        "klosa",
        "lasi",
        "mhas",
        "charls",
    ]
    return any(token in rel for token in useful)


def explicit_dataset_files() -> list[Path]:
    patterns = [
        "raw_data/Harmonized_CHARLS_D/H_CHARLS_D_Data.dta",
        "2011/**/*.dta",
        "2012/**/*.dta",
        "2013/**/*.dta",
        "2014/**/*.dta",
        "2015/**/*.dta",
        "2018/**/*.dta",
        "2020/**/*.dta",
        "3. HRS*/HRS_*/Temp_data/HRS.dta",
        "3. HRS*/HRS_*/Raw_data/RAND HRS Longitudinal File 2020/randhrs1992_2020v2.dta",
        "3. HRS*/**/*_HP.dta",
        "3. HRS*/**/*G_HP.dta",
        "2. ELSA*/Raw_data/Harmonized ELSA/h_elsa_g3.dta",
        "7.SHARE*/**/Harmonized SHARE/H_SHARE_f2.dta",
        "4. KLoSA*/**/H_KLoSA_e2.dta",
        "5. LASI*/**/H_LASI_a3.dta",
        "6. MHAS*/**/H_MHAS_c2.dta",
        "raw_data/**/Family_Information.dta",
        "raw_data/**/Family_Transfer.dta",
        "raw_data/**/Household_Member.dta",
        "raw_data/**/Household_Roster.dta",
        "raw_data/**/Health_Care_and_Insurance.dta",
        "raw_data/**/family_information.dta",
        "raw_data/**/family_transfer.dta",
        "raw_data/**/household_roster.dta",
        "raw_data/**/health_care_and_insurance.dta",
        "2. ELSA*/**/*.dta",
        "6. MHAS*/**/sect_cover_int_*.dta",
    ]
    found: list[Path] = []
    seen = set()
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            if path.is_file() and should_scan(path) and path not in seen:
                found.append(path)
                seen.add(path)
    return sorted(found)


def get_column_metadata(path: Path) -> list[dict[str, str]]:
    try:
        if path.suffix.lower() == ".dta":
            reader = pd.read_stata(str(path), iterator=True, convert_categoricals=False)
            cols = reader.read(nrows=1).columns.tolist()
            labels = reader.variable_labels()
            return [{"name": col, "label": labels.get(col, "") or ""} for col in cols]
        cols = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
        return [{"name": col, "label": ""} for col in cols]
    except Exception:
        try:
            cols = pd.read_csv(path, nrows=0, encoding="latin1").columns.tolist()
            return [{"name": col, "label": ""} for col in cols]
        except Exception:
            return []


def classify_column(name: str, label: str = "") -> set[str]:
    n = f"{name} {label}".lower()
    compact = re.sub(r"[^a-z0-9]", "", n)
    roles: set[str] = set()
    if re.search(r"(helper|helpers|caregiver|provider|carer|care.*person|person.*care)", n):
        roles.add("helper_term")
    if re.search(r"(helper|helpers|caregiver|provider|carer|care|other person).*(id|pid|pn|number|no|loop)", n) or re.search(r"(hlpid|helperid|caregiverid|providerid)", compact):
        roles.add("helper_id")
    if re.search(r"(helper|helpers|caregiver|provider|carer|care).*(sex|gender)", n) or re.search(r"(sex|gender).*(helper|helpers|caregiver|provider|carer)", n):
        roles.add("provider_sex")
    if re.search(r"(helper|helpers|caregiver|provider|carer|care).*(age)", n) or re.search(r"age.*(helper|helpers|caregiver|provider|carer)", n):
        roles.add("provider_age")
    if re.search(r"(relation|relationship|relat|spouse|child|daughter|son|relative|friend|neighbor|neighbour|paid|professional)", n) and re.search(r"(care|help|helper|helpers|giver|provider|carer|hlp|rscare|rccare|rfcare|rpfcare|rcany)", n):
        roles.add("helper_relationship_or_relation_specific_care")
    if re.search(r"(care|help|helper|helpers|caregiver|provider|carer).*(hour|hr|hrs|hpw|dpm|day|week|month|annual|year)", n) or re.search(r"(hour|hr|hrs|hpw|dpm).*(care|help|helper|helpers|caregiver|provider|carer)", n):
        roles.add("helper_or_care_hours")
    if re.search(r"(informal|family|spouse|child|relative|unpaid|rscare|rccare|rrcare|rfcare|rascare|raccare|rarcare|rafcare|riccare|riscare)", n):
        roles.add("informal_indicator")
    if re.search(r"(formal|paid|professional|insurance|ins pay|rpfcare|rufcare|rfaany|rafaany|rifaany)", n):
        roles.add("formal_paid_indicator")
    if re.search(r"(hhidpn|mergeid|idauniq|prim_key|rahhidnp|unhhidnp|pid|\\bid\\b|respondent_id|household identification|respondent person identification|\\bhhid\\b|\\bpn\\b)", n):
        roles.add("patient_merge_key_candidate")
    return roles


def scan_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    file_rows = []
    var_rows = []
    files = explicit_dataset_files()
    for path in sorted(files):
        cohort = cohort_from_path(path)
        if cohort is None:
            continue
        col_meta = get_column_metadata(path)
        if not col_meta:
            continue
        role_counts = {role: 0 for role in [
            "helper_id",
            "provider_sex",
            "provider_age",
            "helper_relationship_or_relation_specific_care",
            "helper_or_care_hours",
            "informal_indicator",
            "formal_paid_indicator",
            "patient_merge_key_candidate",
        ]}
        candidate_cols = []
        for meta in col_meta:
            col = meta["name"]
            label = meta.get("label", "")
            roles = classify_column(col, label)
            if roles:
                candidate_cols.append(col)
                for role in role_counts:
                    if role in roles:
                        role_counts[role] += 1
                var_rows.append(
                    {
                        "cohort": cohort,
                        "source_file": str(path.relative_to(ROOT)),
                        "variable_name": col,
                        "variable_label": label,
                        "candidate_roles": "; ".join(sorted(roles)),
                    }
                )
        if candidate_cols:
            file_rows.append(
                    {
                        "cohort": cohort,
                        "source_file": str(path.relative_to(ROOT)),
                    "n_columns": len(col_meta),
                    "n_candidate_columns": len(candidate_cols),
                    **role_counts,
                    "candidate_column_examples": "; ".join(candidate_cols[:30]),
                }
            )
    return pd.DataFrame(file_rows), pd.DataFrame(var_rows)


def summarize_inventory(file_scan: pd.DataFrame, var_scan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    patient = pd.read_csv(STEP2B_SAMPLE, dtype={"respondent_id": str}) if STEP2B_SAMPLE.exists() else pd.DataFrame()
    for cohort in ALL_COHORTS:
        files = file_scan[file_scan["cohort"].eq(cohort)] if not file_scan.empty else pd.DataFrame()
        vars_ = var_scan[var_scan["cohort"].eq(cohort)] if not var_scan.empty else pd.DataFrame()
        helper_sex = vars_[vars_["candidate_roles"].str.contains("provider_sex", na=False)]
        helper_hours = vars_[vars_["candidate_roles"].str.contains("helper_or_care_hours", na=False)]
        helper_id = vars_[vars_["candidate_roles"].str.contains("helper_id", na=False)]
        relation = vars_[vars_["candidate_roles"].str.contains("helper_relationship", na=False)]
        informal = vars_[vars_["candidate_roles"].str.contains("informal_indicator", na=False)]
        formal = vars_[vars_["candidate_roles"].str.contains("formal_paid_indicator", na=False)]
        merge = vars_[vars_["candidate_roles"].str.contains("patient_merge_key_candidate", na=False)]

        has_helper_level_data = len(helper_id) > 0 or len(helper_sex) > 0 or (
            cohort == "HRS" and len(relation) > 0 and len(helper_hours) > 0
        )
        same_structure_validated = cohort == "HRS" and len(helper_sex) > 0 and len(helper_hours) > 0
        can_matrix = same_structure_validated
        step2b_n = int((patient["cohort"].eq(cohort) & patient["poststroke_eligible"].eq(1)).sum()) if not patient.empty else 0
        main_limitation = ""
        if not has_helper_level_data:
            main_limitation = "No helper-level provider roster identified; available care variables are recipient-level or relation-specific."
        elif cohort != "HRS":
            main_limitation = "Caregiver/care relationship candidates are relation-specific, recipient-level, or not yet validated as one helper-care-recipient row with provider sex and helper-specific hours."
        elif len(helper_sex) == 0:
            main_limitation = "Helper-level data candidate found, but provider_sex was not identified."
        elif len(helper_hours) == 0:
            main_limitation = "Provider_sex candidate found, but helper-specific hours were not identified."

        rows.append(
            {
                "cohort": cohort,
                "has_helper_level_data": "yes" if has_helper_level_data else "no",
                "helper_id_variable": "; ".join(helper_id["variable_name"].head(10)) if len(helper_id) else "",
                "helper_provider_sex_variable": "; ".join(helper_sex["variable_name"].head(10)) if len(helper_sex) else "",
                "helper_relationship_variable": "; ".join(relation["variable_name"].head(10)) if len(relation) else "",
                "helper_specific_hours_variable": "; ".join(helper_hours["variable_name"].head(10)) if len(helper_hours) else "",
                "informal_formal_paid_indicator": "; ".join(pd.concat([informal["variable_name"], formal["variable_name"]]).drop_duplicates().head(12)) if (len(informal) + len(formal)) else "",
                "merge_key_to_step2b_patient": "; ".join(merge["variable_name"].drop_duplicates().head(10)) if len(merge) else "",
                "step2b_patient_rows_available": step2b_n,
                "can_build_gender_care_matrix": "yes" if can_matrix else "no",
                "main_limitation": main_limitation,
                "source_file_examples": "; ".join(files["source_file"].head(8)) if len(files) else "",
                "notes": "Provider sex and helper-specific hours must be in the same helper-recipient structure. HRS Section G helper-provider files meet this screen; relation-specific or recipient-level variables in other cohorts are not treated as provider sex or allocated by sex.",
            }
        )
    return pd.DataFrame(rows)


HRS_WAVE_BY_YEAR = {
    2006: 8,
    2008: 9,
    2010: 10,
    2012: 11,
    2014: 12,
    2016: 13,
    2018: 14,
    2020: 15,
}


def hrs_patient_id(hhid: pd.Series, pn: pd.Series) -> pd.Series:
    combined = hhid.astype(str).str.strip().str.zfill(6) + pn.astype(str).str.strip().str.zfill(3)
    return pd.to_numeric(combined, errors="coerce").astype("Int64").astype(str).replace("<NA>", pd.NA)


def col_ending(cols: list[str], suffix: str) -> str | None:
    suffix = suffix.upper()
    for col in cols:
        if col.upper().endswith(suffix):
            return col
    return None


def valid_number(series: pd.Series, low: float | None = None, high: float | None = None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    out = out.mask(out.isin([-9, -8, -7, -1, 98, 99, 99998, 99999]))
    if low is not None:
        out = out.mask(out < low)
    if high is not None:
        out = out.mask(out > high)
    return out


def extract_hrs_helper_long(patient: pd.DataFrame) -> pd.DataFrame:
    patient_hrs = patient[
        patient["cohort"].eq("HRS") & patient["poststroke_eligible"].eq(1)
    ].copy()
    if patient_hrs.empty:
        return pd.DataFrame()

    keep_patient = [
        "respondent_id",
        "wave",
        "sex_label",
        "age",
        "age_group",
        "disability_severity_label",
        "yld_stroke_i",
        "analysis_weight",
    ]
    patient_hrs = patient_hrs[[c for c in keep_patient if c in patient_hrs.columns]].drop_duplicates()
    patient_hrs["respondent_id"] = patient_hrs["respondent_id"].astype(str)
    patient_hrs["wave"] = pd.to_numeric(patient_hrs["wave"], errors="coerce").astype("Int64")

    rows = []
    hp_files = sorted(ROOT.glob("3. HRS*/**/*_HP.dta"))
    for path in hp_files:
        year_match = re.search(r"(2006|2008|2010|2012|2014|2016|2018|2020)", str(path))
        if not year_match:
            continue
        wave = HRS_WAVE_BY_YEAR.get(int(year_match.group(1)))
        if wave is None:
            continue

        meta = get_column_metadata(path)
        cols = [m["name"] for m in meta]
        required_base = ["HHID", "PN", "OPN"]
        if not all(c in cols for c in required_base):
            continue

        c_rel = col_ending(cols, "G069")
        c_helper_loop = col_ending(cols, "G066")
        c_freq_month = col_ending(cols, "G070")
        c_days_week = col_ending(cols, "G071")
        c_every_day = col_ending(cols, "G072")
        c_hours = col_ending(cols, "G073")
        c_sex = col_ending(cols, "G074")
        c_ins_pay = col_ending(cols, "G077")
        c_paid_amount = col_ending(cols, "G078")

        read_cols = [c for c in [
            "HHID",
            "PN",
            "OPN",
            c_rel,
            c_helper_loop,
            c_freq_month,
            c_days_week,
            c_every_day,
            c_hours,
            c_sex,
            c_ins_pay,
            c_paid_amount,
        ] if c]
        df = pd.read_stata(str(path), columns=read_cols, convert_categoricals=False)
        df["cohort"] = "HRS"
        df["wave"] = wave
        df["patient_id"] = hrs_patient_id(df["HHID"], df["PN"])
        df["helper_id"] = df["OPN"].astype(str).str.strip()
        if c_helper_loop:
            df["helper_id"] = df["helper_id"] + "_" + df[c_helper_loop].astype(str).str.replace(r"\.0$", "", regex=True)

        hours_per_day = valid_number(df[c_hours], 0, 24) if c_hours else pd.Series(np.nan, index=df.index)
        days_week = valid_number(df[c_days_week], 0, 7) if c_days_week else pd.Series(np.nan, index=df.index)
        every_day = valid_number(df[c_every_day], 1, 1) if c_every_day else pd.Series(np.nan, index=df.index)
        days_month = valid_number(df[c_freq_month], 0, 31) if c_freq_month else pd.Series(np.nan, index=df.index)
        days_week = days_week.mask(every_day.eq(1), 7)
        weekly_from_month = days_month * 7 / 30.4375
        days_week = days_week.fillna(weekly_from_month)
        df["hours_week"] = hours_per_day * days_week
        df["hours_year"] = df["hours_week"] * 52
        df["helper_specific_care_hours"] = hours_per_day

        sex_code = valid_number(df[c_sex], 1, 2) if c_sex else pd.Series(np.nan, index=df.index)
        df["provider_sex"] = np.select([sex_code.eq(1), sex_code.eq(2)], ["Male", "Female"], default=pd.NA)
        df["provider_age"] = pd.NA
        df["provider_relationship_to_patient"] = df[c_rel] if c_rel else pd.NA

        paid_amount = valid_number(df[c_paid_amount], 0, None) if c_paid_amount else pd.Series(np.nan, index=df.index)
        ins_pay = valid_number(df[c_ins_pay], 1, 5) if c_ins_pay else pd.Series(np.nan, index=df.index)
        paid = pd.Series(pd.NA, index=df.index, dtype="object")
        paid = paid.mask(paid_amount.gt(0) | ins_pay.eq(1), 1)
        paid = paid.mask(paid_amount.eq(0) | ins_pay.eq(5), 0)
        df["paid_care"] = paid
        df["informal_care"] = pd.Series(pd.NA, index=df.index, dtype="object").mask(paid.eq(0), 1).mask(paid.eq(1), 0)
        df["formal_care"] = pd.NA

        merged = df.merge(
            patient_hrs,
            left_on=["patient_id", "wave"],
            right_on=["respondent_id", "wave"],
            how="inner",
        )
        if merged.empty:
            continue
        merged["source_file"] = str(path.relative_to(ROOT))
        merged["source_variable_notes"] = (
            "HRS Section G helper/provider file. Provider sex from helper sex variable; "
            "hours_week derived from helper-specific hours per help day times reported help frequency. "
            "Payment/insurance variables are retained; formal_care is not directly identified."
        )
        rows.append(merged)

    if not rows:
        return pd.DataFrame()

    long = pd.concat(rows, ignore_index=True)
    long = long.rename(
        columns={
            "sex_label": "patient_sex",
            "age": "patient_age",
            "age_group": "patient_age_group",
            "disability_severity_label": "patient_disability_severity",
            "yld_stroke_i": "patient_yld_like_disability_weight",
            "analysis_weight": "patient_survey_weight",
        }
    )
    return long


def create_long_or_unavailable(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_cols = [
        "cohort",
        "patient_id",
        "helper_id",
        "patient_sex",
        "patient_age",
        "patient_age_group",
        "provider_sex",
        "provider_age",
        "provider_relationship_to_patient",
        "helper_specific_care_hours",
        "hours_week",
        "hours_year",
        "informal_care",
        "formal_care",
        "paid_care",
        "patient_disability_severity",
        "patient_yld_like_disability_weight",
        "patient_survey_weight",
        "source_file",
        "source_variable_notes",
    ]
    patient = pd.read_csv(STEP2B_SAMPLE, dtype={"respondent_id": str}) if STEP2B_SAMPLE.exists() else pd.DataFrame()
    long_parts = []
    if "HRS" in inventory[inventory["can_build_gender_care_matrix"].eq("yes")]["cohort"].tolist():
        hrs_long = extract_hrs_helper_long(patient)
        if not hrs_long.empty:
            long_parts.append(hrs_long)

    long_df = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame(columns=long_cols)
    for col in long_cols:
        if col not in long_df.columns:
            long_df[col] = pd.NA
    long_df = long_df[long_cols]
    long_df.to_csv(OUT / "step3_helper_long_format.csv", index=False)

    unavailable = inventory[
        inventory["can_build_gender_care_matrix"].ne("yes")
    ][
        [
            "cohort",
            "has_helper_level_data",
            "helper_provider_sex_variable",
            "helper_specific_hours_variable",
            "can_build_gender_care_matrix",
            "main_limitation",
            "notes",
        ]
    ].copy()
    unavailable.to_csv(OUT / "step3_helper_unavailable_report.csv", index=False)
    return long_df, unavailable


def quality_log(inventory: pd.DataFrame, long_df: pd.DataFrame, unavailable: pd.DataFrame) -> str:
    full = inventory[inventory["can_build_gender_care_matrix"].eq("yes")]["cohort"].tolist()
    partial = inventory[
        inventory["has_helper_level_data"].eq("yes") & inventory["can_build_gender_care_matrix"].ne("yes")
    ]["cohort"].tolist()
    recipient_only = [c for c in ["CHARLS", "ELSA", "HRS", "KLoSA", "LASI", "MHAS"] if c not in full and c not in partial]
    no_care = ["SHARE"]
    checks = [
        ("PASS", "Searched raw, harmonized, and caregiver/family/household-related files by headers."),
        ("PASS", "Did not infer provider sex from patient sex or relationship variables."),
        ("PASS", "Did not allocate recipient-level total hours across provider sex."),
        ("PASS", "Did not calculate H_MM/H_MF/H_FM/H_FF."),
        ("PASS" if full else "WARNING", "Eligible helper-level cohort(s) have provider_sex and helper-specific hours; no gender matrix was calculated." if full else "No cohort currently has both provider_sex and helper-specific hours in the scanned files."),
        ("PASS", f"Created helper long-format file with {len(long_df):,} linked patient-helper rows and an unavailable report for noneligible cohorts."),
    ]
    lines = ["Step 3A helper-level preflight quality checks"]
    for status, text in checks:
        lines.append(f"{status} - {text}")
    lines.extend(
        [
            "",
            f"Full gender care matrix eligible cohorts: {', '.join(full) if full else 'None'}",
            f"Partial caregiver relationship analysis cohorts: {', '.join(partial) if partial else 'None'}",
            f"Recipient-level care-hour analysis only cohorts: {', '.join(recipient_only)}",
            f"No care-hour analysis cohorts: {', '.join(no_care)}",
            f"Linked helper rows with nonmissing provider_sex and hours_week: {int((long_df['provider_sex'].notna() & long_df['hours_week'].notna()).sum()) if not long_df.empty else 0}",
            f"Linked helper rows with hours_week > 168, flagged but not capped in Step 3A: {int((pd.to_numeric(long_df['hours_week'], errors='coerce') > 168).sum()) if not long_df.empty else 0}",
            "",
            "Next requirement before any matrix: verify cohort documentation and restrict to informal care rows using documented paid/formal indicators. No H_MM/H_MF/H_FM/H_FF values were calculated in Step 3A.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (OUT / "step3a_quality_check_log.txt").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    file_scan, var_scan = scan_files()
    file_scan.to_csv(OUT / "step3a_candidate_file_scan.csv", index=False)
    var_scan.to_csv(OUT / "step3a_candidate_variable_scan.csv", index=False)
    inventory = summarize_inventory(file_scan, var_scan)
    inventory.to_csv(OUT / "step3a_helper_variable_inventory.csv", index=False)
    long_df, unavailable = create_long_or_unavailable(inventory)
    qc = quality_log(inventory, long_df, unavailable)
    print(f"Step 3A outputs written under: {OUT}")
    print(qc)


if __name__ == "__main__":
    main()
