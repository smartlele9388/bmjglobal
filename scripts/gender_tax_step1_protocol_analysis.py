from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import pcgi_initial_analysis as pcgi  # noqa: E402


OUT = ROOT / "output" / "tables"
REPORTS = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

WEEKS_PER_MONTH = 52.1775 / 12


ADL_ITEMS = {
    "adl_bath": ["batha", "bathb", "bath"],
    "adl_dress": ["dressa", "dressb", "dress"],
    "adl_eat": ["eata", "eatb", "eat"],
    "adl_bed": ["beda", "bedb", "bedb_k", "bed"],
    "adl_toilet": ["toilta", "toiltb", "toilt"],
    "adl_continence": ["urinb", "urina2y", "urincgh2y"],
}

IADL_ITEMS = {
    "iadl_phone": ["phonea", "phoneb", "phone"],
    "iadl_money": ["moneya", "moneyb", "money"],
    "iadl_meds": ["medsa", "medsb", "meds"],
    "iadl_shop": ["shopa", "shopb", "shop"],
    "iadl_meal": ["mealsa", "mealsb", "meals"],
}

INFORMAL_RECEIVED_STEMS = [
    "rscare",
    "rccare",
    "rrcare",
    "rfcare",
    "rascare",
    "raccare",
    "rarcare",
    "rafcare",
    "riscare",
    "riccare",
    "rircare",
    "rifcare",
]

GBD_LOCATION = pcgi.GBD_LOCATION


def clean_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    out = out.mask(out < 0)
    out = out.mask(out > 1_000_000_000)
    return out


def find_first(columns: set[str], wave: int, suffixes: list[str]) -> str | None:
    for suffix in suffixes:
        candidates = [
            f"r{wave}{suffix}",
            f"r{wave}{suffix}_l",
            f"r{wave}{suffix}_e",
            f"r{wave}{suffix}_m",
            f"r{wave}{suffix}_k",
        ]
        for candidate in candidates:
            if candidate in columns:
                return candidate
    return None


def binary_difficulty(series: pd.Series) -> pd.Series:
    values = clean_numeric(series)
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out.loc[values.eq(0)] = 0
    out.loc[values.ge(1)] = 1
    return out


def scaled_score(frame: pd.DataFrame, target_items: int, min_valid: int) -> pd.Series:
    valid = frame.notna().sum(axis=1)
    score = frame.sum(axis=1, min_count=1)
    out = score * target_items / valid
    out = out.where(valid >= min_valid)
    return out


def care_hours_week_for_wave(df: pd.DataFrame, columns: set[str], wave: int) -> tuple[pd.Series, list[str]]:
    pieces = []
    variables = []

    # Prefer combined ADL/IADL care relation variables where available.
    combined_stems = ["rscare", "rccare", "rrcare", "rfcare"]
    adl_iadl_stems = [stem for stem in INFORMAL_RECEIVED_STEMS if stem not in combined_stems]

    def collect_for_stems(stems: list[str]) -> None:
        for stem in stems:
            base = f"r{wave}{stem}"
            direct_hpw = [f"{base}hpw", f"{base}hpw_e", f"{base}carehpw", f"{base}carehpw_e"]
            for hpw in direct_hpw:
                if hpw in columns:
                    hours = clean_numeric(df[hpw]).mask(lambda s: s > 168)
                    pieces.append(hours)
                    variables.append(hpw)
                    return

            pair_candidates = [
                (f"{base}hr", f"{base}dpm"),
                (f"{base}hr_l", f"{base}dpm_l"),
                (f"{base}hr_c", f"{base}dpm_c"),
                (f"{base}carehr", f"{base}caredpm"),
                (f"{base}carehr_l", f"{base}caredpm_l"),
                (f"{base}carehr_c", f"{base}caredpm_c"),
            ]
            for hr_var, dpm_var in pair_candidates:
                if hr_var in columns and dpm_var in columns:
                    hours_day = clean_numeric(df[hr_var]).mask(lambda s: s > 24)
                    days_month = clean_numeric(df[dpm_var]).mask(lambda s: s > 31)
                    pieces.append((hours_day * days_month / WEEKS_PER_MONTH).mask(lambda s: s > 168, 168))
                    variables.extend([hr_var, dpm_var])
                    return

    collect_for_stems(combined_stems)
    if not pieces:
        collect_for_stems(adl_iadl_stems)

    if not pieces:
        return pd.Series(np.nan, index=df.index, dtype="float"), []

    sub = pd.concat(pieces, axis=1)
    known = sub.notna().any(axis=1)
    total = sub.fillna(0).sum(axis=1).clip(upper=168)
    return total.where(known), variables


def care_intensity(hours: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=hours.index, dtype="object")
    out.loc[hours.eq(0)] = "0 none"
    out.loc[hours.gt(0) & hours.le(13)] = "1 light"
    out.loc[hours.ge(14) & hours.le(40)] = "2 moderate"
    out.loc[hours.gt(40)] = "3 heavy"
    return out


def needed_columns(dataset: str, columns: list[str]) -> list[str]:
    colset = set(columns)
    needed = []
    needed.extend([c for c in pcgi.ID_CANDIDATES[dataset] if c in colset])
    needed.extend([c for c in ["ragender", "raeducl", "raeduc_c", "raeduc_l", "country"] if c in colset])
    for wave in pcgi.wave_numbers(columns):
        for fixed in [
            f"r{wave}stroke",
            f"r{wave}agey",
            f"r{wave}agey_b",
            f"r{wave}agey_e",
            f"r{wave}wtresp",
        ]:
            if fixed in colset:
                needed.append(fixed)
        for suffixes in list(ADL_ITEMS.values()) + list(IADL_ITEMS.values()):
            found = find_first(colset, wave, suffixes)
            if found:
                needed.append(found)
        _, hour_vars = care_hours_week_for_wave(
            pd.DataFrame(index=[0], columns=list(colset)), colset, wave
        )
        needed.extend(hour_vars)
    return sorted(set(needed), key=needed.index)


def build_protocol_long(dataset: str, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = pcgi.get_columns(path)
    colset = set(columns)
    respondent_id_col = pcgi.first_existing(colset, pcgi.ID_CANDIDATES[dataset])
    if respondent_id_col is None:
        raise ValueError(f"No respondent id found for {dataset}")

    read_cols = needed_columns(dataset, columns)
    df = pd.read_stata(str(path), columns=read_cols, convert_categoricals=False)

    frames = []
    manifest = []
    for wave in pcgi.wave_numbers(columns):
        stroke_var = f"r{wave}stroke"
        if stroke_var not in df.columns:
            continue
        stroke = clean_numeric(df[stroke_var])
        if stroke.notna().sum() == 0:
            continue

        adl_data = {}
        adl_vars = {}
        for item, suffixes in ADL_ITEMS.items():
            var = find_first(set(df.columns), wave, suffixes)
            if var:
                adl_data[item] = binary_difficulty(df[var])
                adl_vars[item] = var
            else:
                adl_data[item] = pd.Series(np.nan, index=df.index)
        adl_frame = pd.DataFrame(adl_data)
        adl_score = scaled_score(adl_frame, target_items=6, min_valid=4)

        iadl_data = {}
        iadl_vars = {}
        for item, suffixes in IADL_ITEMS.items():
            var = find_first(set(df.columns), wave, suffixes)
            if var:
                iadl_data[item] = binary_difficulty(df[var])
                iadl_vars[item] = var
            else:
                iadl_data[item] = pd.Series(np.nan, index=df.index)
        iadl_frame = pd.DataFrame(iadl_data)
        iadl_score = scaled_score(iadl_frame, target_items=5, min_valid=4)

        hours_week, hour_vars = care_hours_week_for_wave(df, set(df.columns), wave)

        age = pcgi.value_from_candidates(df, [f"r{wave}agey", f"r{wave}agey_b", f"r{wave}agey_e"])
        age = clean_numeric(age)
        weight = clean_numeric(df[f"r{wave}wtresp"]) if f"r{wave}wtresp" in df.columns else np.nan

        out = pd.DataFrame(
            {
                "dataset": dataset,
                "respondent_id": df[respondent_id_col].astype(str),
                "wave": wave,
                "age": age,
                "sex": clean_numeric(df["ragender"]).map({1: 0, 2: 1}) if "ragender" in df.columns else np.nan,
                "weight_cross": weight,
                "stroke_ever": stroke.eq(1).astype(float).where(stroke.notna()),
                "adl_score": adl_score,
                "iadl_score": iadl_score,
                "care_hours_week": hours_week,
            }
        )
        for col in adl_frame.columns:
            out[col] = adl_frame[col]
        for col in iadl_frame.columns:
            out[col] = iadl_frame[col]

        out["disability_state"] = np.nan
        out.loc[out["adl_score"].eq(0) & out["iadl_score"].le(1), "disability_state"] = 1
        out.loc[
            ((out["adl_score"].ge(1) & out["adl_score"].le(2)) | out["iadl_score"].ge(2))
            & out["disability_state"].isna(),
            "disability_state",
        ] = 2
        out.loc[out["adl_score"].ge(3), "disability_state"] = 3

        out["care_hours_week"] = out["care_hours_week"].mask(out["care_hours_week"] < 0)
        out["care_hours_week"] = out["care_hours_week"].clip(upper=168)
        out.loc[out["adl_score"].eq(0) & out["care_hours_week"].isna(), "care_hours_week"] = 0
        out["care_intensity"] = care_intensity(out["care_hours_week"])
        out["unmet_need_objective"] = np.nan
        known_unmet = out["adl_score"].notna() & out["care_hours_week"].notna()
        out.loc[known_unmet, "unmet_need_objective"] = (
            out.loc[known_unmet, "adl_score"].ge(1) & out.loc[known_unmet, "care_hours_week"].eq(0)
        ).astype(float)

        frames.append(out)
        manifest.append(
            {
                "dataset": dataset,
                "wave": wave,
                "adl_items_found": len(adl_vars),
                "adl_items_missing": 6 - len(adl_vars),
                "iadl_items_found": len(iadl_vars),
                "iadl_items_missing": 5 - len(iadl_vars),
                "care_hour_variables_found": len(hour_vars),
                "adl_variables": "; ".join(adl_vars.values()),
                "iadl_variables": "; ".join(iadl_vars.values()),
                "care_hour_variables": "; ".join(hour_vars),
            }
        )

    return pd.concat(frames, ignore_index=True), pd.DataFrame(manifest)


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    ok = x.notna() & w.notna() & (w > 0)
    if ok.sum() == 0:
        return np.nan
    return float(np.average(x[ok], weights=w[ok]))


def summarize_country(latest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, g in latest.groupby("dataset"):
        hours_known = g[g["care_hours_week"].notna()]
        care_recipients = g[g["care_hours_week"].gt(0)]
        rows.append(
            {
                "dataset": dataset,
                "post_stroke_n": len(g),
                "female_percent": g["sex"].mean() * 100,
                "weighted_female_percent": weighted_mean(g["sex"], g["weight_cross"]) * 100,
                "mean_adl_score": g["adl_score"].mean(),
                "mean_iadl_score": g["iadl_score"].mean(),
                "disability_any_percent": ((g["adl_score"].gt(0)) | (g["iadl_score"].gt(1))).mean() * 100,
                "severe_disability_percent": g["disability_state"].eq(3).mean() * 100,
                "care_hours_known_n": len(hours_known),
                "mean_care_hours_week": hours_known["care_hours_week"].mean(),
                "median_care_hours_week": hours_known["care_hours_week"].median(),
                "care_recipients_n": len(care_recipients),
                "mean_care_hours_week_recipients": care_recipients["care_hours_week"].mean(),
                "p90_care_hours_week_recipients": care_recipients["care_hours_week"].quantile(0.90),
                "p95_care_hours_week_recipients": care_recipients["care_hours_week"].quantile(0.95),
                "heavy_care_percent": g["care_intensity"].eq("3 heavy").mean() * 100,
                "unmet_need_objective_percent": g["unmet_need_objective"].mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def summarize_gender(latest: pd.DataFrame) -> pd.DataFrame:
    labels = {0.0: "Male", 1.0: "Female"}
    rows = []
    for (dataset, sex), g in latest.groupby(["dataset", "sex"], dropna=False):
        hours_known = g[g["care_hours_week"].notna()]
        care_recipients = g[g["care_hours_week"].gt(0)]
        rows.append(
            {
                "dataset": dataset,
                "sex": labels.get(float(sex), "Unknown") if pd.notna(sex) else "Unknown",
                "post_stroke_n": len(g),
                "mean_adl_score": g["adl_score"].mean(),
                "mean_iadl_score": g["iadl_score"].mean(),
                "disability_any_percent": ((g["adl_score"].gt(0)) | (g["iadl_score"].gt(1))).mean() * 100,
                "severe_disability_percent": g["disability_state"].eq(3).mean() * 100,
                "care_hours_known_n": len(hours_known),
                "mean_care_hours_week": hours_known["care_hours_week"].mean(),
                "median_care_hours_week": hours_known["care_hours_week"].median(),
                "care_recipients_n": len(care_recipients),
                "mean_care_hours_week_recipients": care_recipients["care_hours_week"].mean(),
                "p90_care_hours_week_recipients": care_recipients["care_hours_week"].quantile(0.90),
                "p95_care_hours_week_recipients": care_recipients["care_hours_week"].quantile(0.95),
                "heavy_care_percent": g["care_intensity"].eq("3 heavy").mean() * 100,
                "unmet_need_objective_percent": g["unmet_need_objective"].mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def add_gbd(country: pd.DataFrame) -> pd.DataFrame:
    out = country.copy()
    out["gbd_location"] = out["dataset"].map(GBD_LOCATION)
    gbd_path = OUT / "gbd_2023_stroke_60plus_rates_wide.csv"
    if not gbd_path.exists():
        out["stroke_yld_rate_60plus"] = np.nan
        return out
    gbd = pd.read_csv(gbd_path)
    out = out.merge(
        gbd[["location_name", "ylds_rate", "dalys_rate", "prevalence_rate"]],
        left_on="gbd_location",
        right_on="location_name",
        how="left",
    )
    out = out.rename(
        columns={
            "ylds_rate": "stroke_yld_rate_60plus",
            "dalys_rate": "stroke_daly_rate_60plus",
            "prevalence_rate": "stroke_prevalence_rate_60plus",
        }
    )
    return out.drop(columns=["location_name"])


def fmt(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def make_report(country: pd.DataFrame, gender: pd.DataFrame, manifest: pd.DataFrame) -> str:
    lines = [
        "# Gender Tax Step 1 Protocol Analysis",
        "",
        "## Scope",
        "",
        "This reruns Step 1 using the user's protocol definitions where available in the harmonized files: doctor-diagnosed stroke, item-level ADL/IADL scores, weekly informal care hours, care intensity, and objective unmet need.",
        "",
        "## Country Distribution",
        "",
        "| Dataset | Post-stroke N | Mean ADL | Mean IADL | Any disability % | Severe disability % | Mean care h/wk | Heavy care % | Objective unmet need % | Stroke YLD rate 60+ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in country.sort_values("dataset").itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.post_stroke_n:,} | {fmt(row.mean_adl_score)} | {fmt(row.mean_iadl_score)} | "
            f"{fmt(row.disability_any_percent)} | {fmt(row.severe_disability_percent)} | "
            f"{fmt(row.mean_care_hours_week)} | {fmt(row.heavy_care_percent)} | "
            f"{fmt(row.unmet_need_objective_percent)} | {fmt(row.stroke_yld_rate_60plus)} |"
        )
    lines.extend(
        [
            "",
            "## Gender Distribution",
            "",
            "| Dataset | Sex | N | Mean ADL | Mean IADL | Any disability % | Severe disability % | Mean care h/wk | Heavy care % | Objective unmet need % |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in gender.sort_values(["dataset", "sex"]).itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.sex} | {row.post_stroke_n:,} | {fmt(row.mean_adl_score)} | "
            f"{fmt(row.mean_iadl_score)} | {fmt(row.disability_any_percent)} | "
            f"{fmt(row.severe_disability_percent)} | {fmt(row.mean_care_hours_week)} | "
            f"{fmt(row.heavy_care_percent)} | {fmt(row.unmet_need_objective_percent)} |"
        )
    lines.extend(
        [
            "",
            "## Protocol Coverage",
            "",
            "| Dataset | Waves | Median ADL items found | Median IADL items found | Waves with care-hour variables |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset, g in manifest.groupby("dataset"):
        lines.append(
            f"| {dataset} | {g['wave'].nunique()} | {fmt(g['adl_items_found'].median(), 0)} | "
            f"{fmt(g['iadl_items_found'].median(), 0)} | {(g['care_hour_variables_found'] > 0).sum()} |"
        )
    lines.extend(
        [
            "",
            "## Important Deviations",
            "",
            "- ADL continence is not consistently available; when at least four ADL items are valid, the score is scaled to the 0-6 protocol range.",
            "- IADL is similarly scaled to 0-5 when at least four items are valid; this is a transparent extension of the ADL missing-item rule.",
            "- SHARE later waves have informal-care frequency variables but no directly harmonized hour variables in the currently used file, so care-hours are missing for SHARE in this first protocol run.",
            "- This is still descriptive and not multiple-imputed; MICE and mortality-state construction remain for later modelling.",
            "",
            "## Output Files",
            "",
            f"- `{OUT / 'gender_tax_protocol_person_wave.csv'}`",
            f"- `{OUT / 'gender_tax_protocol_latest_poststroke.csv'}`",
            f"- `{OUT / 'gender_tax_protocol_country_distribution.csv'}`",
            f"- `{OUT / 'gender_tax_protocol_gender_distribution.csv'}`",
            f"- `{OUT / 'gender_tax_protocol_variable_manifest.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    frames = []
    manifests = []
    for dataset, pattern in pcgi.DATASETS.items():
        path = pcgi.resolve_dataset(pattern)
        long, manifest = build_protocol_long(dataset, path)
        frames.append(long)
        manifests.append(manifest)

    person_wave = pd.concat(frames, ignore_index=True)
    manifest = pd.concat(manifests, ignore_index=True)
    person_wave = person_wave[person_wave["age"].ge(50) | person_wave["age"].isna()].copy()
    person_wave.to_csv(OUT / "gender_tax_protocol_person_wave.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(OUT / "gender_tax_protocol_variable_manifest.csv", index=False, encoding="utf-8-sig")

    post = person_wave[person_wave["stroke_ever"].eq(1)].copy()
    post = post.sort_values(["dataset", "respondent_id", "wave"])
    latest = post.groupby(["dataset", "respondent_id"], as_index=False, dropna=False).tail(1).copy()
    latest.to_csv(OUT / "gender_tax_protocol_latest_poststroke.csv", index=False, encoding="utf-8-sig")

    no_hour_datasets = set(
        manifest.groupby("dataset")["care_hour_variables_found"]
        .max()
        .loc[lambda s: s.eq(0)]
        .index
    )

    country = add_gbd(summarize_country(latest))
    gender = summarize_gender(latest)
    care_columns = [
        "care_hours_known_n",
        "mean_care_hours_week",
        "median_care_hours_week",
        "care_recipients_n",
        "mean_care_hours_week_recipients",
        "p90_care_hours_week_recipients",
        "p95_care_hours_week_recipients",
        "heavy_care_percent",
        "unmet_need_objective_percent",
    ]
    country.loc[country["dataset"].isin(no_hour_datasets), care_columns] = np.nan
    gender.loc[gender["dataset"].isin(no_hour_datasets), care_columns] = np.nan
    country.to_csv(OUT / "gender_tax_protocol_country_distribution.csv", index=False, encoding="utf-8-sig")
    gender.to_csv(OUT / "gender_tax_protocol_gender_distribution.csv", index=False, encoding="utf-8-sig")
    (REPORTS / "gender_tax_protocol_step1_report.md").write_text(
        make_report(country, gender, manifest), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
