from __future__ import annotations

import sys
from collections import OrderedDict
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

MONTHS_PER_WEEK = 52.1775 / 12

INFORMAL_RELATION_STEMS = {
    "combined": ["rscare", "rccare", "rrcare", "rfcare"],
    "adl_iadl": ["rascare", "raccare", "rarcare", "rafcare", "riscare", "riccare", "rircare", "rifcare"],
}

GBD_LOCATION = pcgi.GBD_LOCATION


def clean_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    out = out.mask(out < 0)
    out = out.mask(out > 1_000_000_000)
    return out


def existing(columns: set[str], names: list[str]) -> list[str]:
    return [name for name in names if name in columns]


def hour_pairs_for_wave(columns: set[str], wave: int) -> tuple[list[tuple[str, str, str]], list[str]]:
    pairs: list[tuple[str, str, str]] = []
    hpw_vars: list[str] = []
    w = f"r{wave}"

    for stem in INFORMAL_RELATION_STEMS["combined"]:
        base = f"{w}{stem}"
        pair_candidates = [
            (f"{base}hr", f"{base}dpm"),
            (f"{base}hr_l", f"{base}dpm_l"),
            (f"{base}carehr", f"{base}caredpm"),
            (f"{base}carehr_l", f"{base}caredpm_l"),
            (f"{base}carehr_c", f"{base}caredpm_c"),
        ]
        for hr, dpm in pair_candidates:
            if hr in columns and dpm in columns:
                pairs.append((stem, hr, dpm))
                break

        for hpw in [f"{base}hpw", f"{base}hpw_e", f"{base}carehpw", f"{base}carehpw_e"]:
            if hpw in columns:
                hpw_vars.append(hpw)
                break

    # If a combined relation exists, prefer it over separate ADL/IADL relation
    # hours to reduce double counting.
    if pairs or hpw_vars:
        return pairs, hpw_vars

    for stem in INFORMAL_RELATION_STEMS["adl_iadl"]:
        base = f"{w}{stem}"
        pair_candidates = [
            (f"{base}hr", f"{base}dpm"),
            (f"{base}carehr", f"{base}caredpm"),
            (f"{base}carehr_l", f"{base}caredpm_l"),
        ]
        for hr, dpm in pair_candidates:
            if hr in columns and dpm in columns:
                pairs.append((stem, hr, dpm))
                break

        for hpw in [f"{base}hpw", f"{base}carehpw", f"{base}carehpw_e"]:
            if hpw in columns:
                hpw_vars.append(hpw)
                break

    return pairs, hpw_vars


def needed_hour_columns(dataset: str, columns: list[str]) -> list[str]:
    colset = set(columns)
    needed: list[str] = []
    needed.extend(existing(colset, pcgi.ID_CANDIDATES[dataset]))
    for wave in pcgi.wave_numbers(columns):
        pairs, hpw_vars = hour_pairs_for_wave(colset, wave)
        for _, hr, dpm in pairs:
            needed.extend([hr, dpm])
        needed.extend(hpw_vars)
    return sorted(set(needed), key=needed.index)


def build_received_hour_long(dataset: str, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = pcgi.get_columns(path)
    colset = set(columns)
    respondent_id_col = pcgi.first_existing(colset, pcgi.ID_CANDIDATES[dataset])
    if respondent_id_col is None:
        raise ValueError(f"No respondent id found for {dataset}")

    read_cols = needed_hour_columns(dataset, columns)
    if len(read_cols) <= 1:
        return pd.DataFrame(), pd.DataFrame(
            [{"dataset": dataset, "wave": np.nan, "hour_variables": 0, "method": "none"}]
        )

    df = pd.read_stata(str(path), columns=read_cols, convert_categoricals=False)
    frames = []
    manifest = []

    for wave in pcgi.wave_numbers(columns):
        pairs, hpw_vars = hour_pairs_for_wave(colset, wave)
        available = [item for pair in pairs for item in pair[1:]] + hpw_vars
        if not available:
            continue

        pieces = []
        for stem, hr_var, dpm_var in pairs:
            hours_day = clean_numeric(df[hr_var]).mask(lambda s: s > 24)
            days_month = clean_numeric(df[dpm_var]).mask(lambda s: s > 31)
            monthly = hours_day * days_month
            pieces.append(monthly.rename(stem))

        for hpw_var in hpw_vars:
            hours_week = clean_numeric(df[hpw_var]).mask(lambda s: s > 168)
            pieces.append((hours_week * MONTHS_PER_WEEK).rename(hpw_var))

        if not pieces:
            continue

        sub = pd.concat(pieces, axis=1)
        known = sub.notna().any(axis=1)
        positive = sub.fillna(0).gt(0).any(axis=1)
        total = sub.fillna(0).sum(axis=1)
        total = total.where(known)

        frames.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "respondent_id": df[respondent_id_col].astype(str),
                    "wave": wave,
                    "received_informal_hours_month": total,
                    "received_informal_hours_known": known.astype(int),
                    "received_informal_hours_positive": positive.astype(int).where(known),
                }
            )
        )
        method = "hours_day_x_days_month" if pairs else "hours_week_x_4.35"
        if pairs and hpw_vars:
            method = "mixed_hours_month_and_hours_week"
        manifest.append(
            {
                "dataset": dataset,
                "wave": wave,
                "hour_variables": len(available),
                "method": method,
                "variables": "; ".join(available),
            }
        )

    if not frames:
        return pd.DataFrame(), pd.DataFrame(manifest)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(manifest)


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if series.notna().any() else np.nan


def safe_quantile(series: pd.Series, q: float) -> float:
    return float(series.quantile(q)) if series.notna().any() else np.nan


def summarize_country(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, g in df.groupby("dataset", dropna=False):
        need = g[g["care_need"].eq(1)]
        hours_known = g[g["received_informal_hours_month"].notna()]
        hours_positive = g[g["received_informal_hours_month"].gt(0)]
        rows.append(
            {
                "dataset": dataset,
                "post_stroke_latest_n": len(g),
                "mean_age": safe_mean(g["age"]),
                "female_percent": safe_mean(g["female"]) * 100,
                "care_need_percent": safe_mean(g["care_need"]) * 100,
                "mean_adl_score": safe_mean(g["adl_score"]),
                "mean_iadl_score": safe_mean(g["iadl_score"]),
                "informal_only_percent": safe_mean(g["pcgi_code"].eq(2).astype(float)) * 100,
                "unmet_care_percent": safe_mean(g["pcgi_code"].eq(3).astype(float)) * 100,
                "received_hour_known_n": len(hours_known),
                "received_hour_known_percent": len(hours_known) / len(g) * 100 if len(g) else np.nan,
                "received_hour_positive_n": len(hours_positive),
                "mean_received_informal_hours_month_known": safe_mean(hours_known["received_informal_hours_month"]),
                "median_received_informal_hours_month_known": safe_quantile(
                    hours_known["received_informal_hours_month"], 0.5
                ),
                "p25_received_informal_hours_month_known": safe_quantile(
                    hours_known["received_informal_hours_month"], 0.25
                ),
                "p75_received_informal_hours_month_known": safe_quantile(
                    hours_known["received_informal_hours_month"], 0.75
                ),
                "mean_received_informal_hours_month_positive": safe_mean(
                    hours_positive["received_informal_hours_month"]
                ),
                "mean_received_informal_hours_month_among_need_known": safe_mean(
                    need["received_informal_hours_month"]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_gender(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = {0.0: "Male", 1.0: "Female"}
    for (dataset, female), g in df.groupby(["dataset", "female"], dropna=False):
        if pd.isna(female):
            sex = "Unknown"
        else:
            sex = labels.get(float(female), str(female))
        hours_known = g[g["received_informal_hours_month"].notna()]
        hours_positive = g[g["received_informal_hours_month"].gt(0)]
        rows.append(
            {
                "dataset": dataset,
                "sex": sex,
                "post_stroke_latest_n": len(g),
                "care_need_percent": safe_mean(g["care_need"]) * 100,
                "mean_adl_score": safe_mean(g["adl_score"]),
                "mean_iadl_score": safe_mean(g["iadl_score"]),
                "informal_only_percent": safe_mean(g["pcgi_code"].eq(2).astype(float)) * 100,
                "unmet_care_percent": safe_mean(g["pcgi_code"].eq(3).astype(float)) * 100,
                "received_hour_known_n": len(hours_known),
                "received_hour_positive_n": len(hours_positive),
                "mean_received_informal_hours_month_known": safe_mean(hours_known["received_informal_hours_month"]),
                "median_received_informal_hours_month_known": safe_quantile(
                    hours_known["received_informal_hours_month"], 0.5
                ),
                "mean_received_informal_hours_month_positive": safe_mean(
                    hours_positive["received_informal_hours_month"]
                ),
            }
        )
    return pd.DataFrame(rows)


def add_gbd_context(country: pd.DataFrame) -> pd.DataFrame:
    out = country.copy()
    out["gbd_location"] = out["dataset"].map(GBD_LOCATION)
    gbd_path = OUT / "gbd_2023_stroke_60plus_rates_wide.csv"
    if not gbd_path.exists():
        out["stroke_yld_rate_60plus"] = np.nan
        out["stroke_daly_rate_60plus"] = np.nan
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


def format_num(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def make_report(country: pd.DataFrame, gender: pd.DataFrame, manifest: pd.DataFrame) -> str:
    lines = []
    lines.append("# Gender Tax Step 1 Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This first step describes post-stroke disability, GBD stroke YLD context, and received informal care hour burden "
        "across seven harmonized aging cohorts. Care hours are currently recipient-side hours, not yet dyadic provider-recipient decomposition."
    )
    lines.append("")
    lines.append("## Country Distribution")
    lines.append("")
    lines.append(
        "| Dataset | Post-stroke N | Female % | Care need % | Informal-only % | Unmet care % | Stroke YLD rate 60+ | Hour-known N | Mean informal hours/mo | Median informal hours/mo |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in country.sort_values("dataset").itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.post_stroke_latest_n:,} | {format_num(row.female_percent)} | "
            f"{format_num(row.care_need_percent)} | {format_num(row.informal_only_percent)} | "
            f"{format_num(row.unmet_care_percent)} | {format_num(row.stroke_yld_rate_60plus)} | "
            f"{int(row.received_hour_known_n):,} | {format_num(row.mean_received_informal_hours_month_known)} | "
            f"{format_num(row.median_received_informal_hours_month_known)} |"
        )
    lines.append("")
    lines.append("## Gender Distribution")
    lines.append("")
    lines.append(
        "| Dataset | Sex | N | Care need % | Informal-only % | Unmet care % | Hour-known N | Mean informal hours/mo |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in gender.sort_values(["dataset", "sex"]).itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.sex} | {row.post_stroke_latest_n:,} | "
            f"{format_num(row.care_need_percent)} | {format_num(row.informal_only_percent)} | "
            f"{format_num(row.unmet_care_percent)} | {int(row.received_hour_known_n):,} | "
            f"{format_num(row.mean_received_informal_hours_month_known)} |"
        )
    lines.append("")
    lines.append("## Hour Variable Coverage")
    lines.append("")
    lines.append("| Dataset | Waves with hour variables | Main methods |")
    lines.append("| --- | ---: | --- |")
    if manifest.empty:
        lines.append("| none | 0 | none |")
    else:
        for dataset, g in manifest.groupby("dataset"):
            methods = ", ".join(sorted(set(g["method"].dropna())))
            waves = ", ".join(str(int(w)) for w in g["wave"].dropna().unique())
            lines.append(f"| {dataset} | {waves} | {methods} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- The disability and PCGI components are available for all seven cohorts; GBD YLD context is linked for China, the United States, the United Kingdom, Korea, India, and Mexico."
    )
    lines.append(
        "- Received informal care hours are directly quantifiable for several cohorts, but SHARE is mainly frequency-based in later waves and is not forced into hour estimates in this first-pass table."
    )
    lines.append(
        "- These results define the descriptive baseline for the next step: separating women's burden as patients from women's burden as care providers."
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for name in [
        "gender_tax_step1_country_distribution.csv",
        "gender_tax_step1_gender_distribution.csv",
        "gender_tax_step1_hour_variable_manifest.csv",
        "gender_tax_step1_latest_with_hours.csv",
    ]:
        lines.append(f"- `{OUT / name}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    latest_path = OUT / "pcgi_latest_poststroke_dataset.csv"
    if not latest_path.exists():
        raise FileNotFoundError(latest_path)
    latest = pd.read_csv(latest_path)
    latest["respondent_id"] = latest["respondent_id"].astype(str)

    hour_frames = []
    manifest_frames = []
    for dataset, pattern in pcgi.DATASETS.items():
        path = pcgi.resolve_dataset(pattern)
        hours, manifest = build_received_hour_long(dataset, path)
        if not hours.empty:
            hour_frames.append(hours)
        if not manifest.empty:
            manifest_frames.append(manifest)

    hours_long = pd.concat(hour_frames, ignore_index=True) if hour_frames else pd.DataFrame()
    manifest = pd.concat(manifest_frames, ignore_index=True) if manifest_frames else pd.DataFrame()

    if hours_long.empty:
        latest_hours = latest.copy()
        latest_hours["received_informal_hours_month"] = np.nan
        latest_hours["received_informal_hours_known"] = np.nan
        latest_hours["received_informal_hours_positive"] = np.nan
    else:
        hours_long["respondent_id"] = hours_long["respondent_id"].astype(str)
        latest_hours = latest.merge(
            hours_long,
            on=["dataset", "respondent_id", "wave"],
            how="left",
        )

    latest_hours.to_csv(OUT / "gender_tax_step1_latest_with_hours.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(OUT / "gender_tax_step1_hour_variable_manifest.csv", index=False, encoding="utf-8-sig")

    country = add_gbd_context(summarize_country(latest_hours))
    gender = summarize_gender(latest_hours)

    country.to_csv(OUT / "gender_tax_step1_country_distribution.csv", index=False, encoding="utf-8-sig")
    gender.to_csv(OUT / "gender_tax_step1_gender_distribution.csv", index=False, encoding="utf-8-sig")
    (REPORTS / "gender_tax_step1_analysis_report.md").write_text(
        make_report(country, gender, manifest),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
