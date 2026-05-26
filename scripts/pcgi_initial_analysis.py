from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "tables"
REPORTS = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


DATASETS = OrderedDict(
    [
        ("CHARLS", "raw_data/Harmonized_CHARLS_D/H_CHARLS_D_Data.dta"),
        # The Gateway HRS file lacks ever-stroke variables locally; the local temp file
        # contains the harmonized stroke and care variables needed for PCGI.
        ("HRS", "3. HRS*/HRS_*/Temp_data/HRS.dta"),
        ("ELSA", "2. ELSA*/Raw_data/Harmonized ELSA/h_elsa_g3.dta"),
        ("SHARE", "7.SHARE*/**/Harmonized SHARE/H_SHARE_f2.dta"),
        ("KLoSA", "4. KLoSA*/**/H_KLoSA_e2.dta"),
        ("LASI", "5. LASI*/**/H_LASI_a3.dta"),
        ("MHAS", "6. MHAS*/**/H_MHAS_c2.dta"),
    ]
)


ID_CANDIDATES = {
    "CHARLS": ["ID"],
    "HRS": ["hhidpn"],
    "ELSA": ["idauniq"],
    "SHARE": ["mergeid"],
    "KLoSA": ["pid"],
    "LASI": ["prim_key"],
    "MHAS": ["rahhidnp", "unhhidnp"],
}

GBD_LOCATION = {
    "CHARLS": "China",
    "HRS": "United States of America",
    "ELSA": "United Kingdom",
    "KLoSA": "Republic of Korea",
    "LASI": "India",
    "MHAS": "Mexico",
    # SHARE contains many European countries; leave unmatched to avoid forcing it
    # onto UK or a regional estimate that was not downloaded.
    "SHARE": None,
}


PCGI_LABELS = {
    0: "No disability-related care need",
    1: "Need met with formal/professional care",
    2: "Need met with informal/family care only",
    3: "Unmet disability-related care need",
}


def resolve_dataset(pattern: str) -> Path:
    direct = ROOT / pattern
    if direct.exists():
        return direct
    matches = sorted(ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]


def get_columns(path: Path) -> list[str]:
    return pd.read_stata(str(path), iterator=True, convert_categoricals=False).read(nrows=1).columns.tolist()


def present(columns: set[str], names: list[str]) -> list[str]:
    return [name for name in names if name in columns]


def wave_numbers(columns: list[str]) -> list[int]:
    waves = set()
    for col in columns:
        match = re.fullmatch(r"r(\d+)stroke", col)
        if match:
            waves.add(int(match.group(1)))
    return sorted(waves)


def first_existing(columns: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def value_from_candidates(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    out = df[existing[0]].copy()
    for col in existing[1:]:
        out = out.where(out.notna(), df[col])
    return out


def any_yes(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    sub = df[existing].apply(pd.to_numeric, errors="coerce")
    yes = sub.eq(1).any(axis=1)
    known = sub.notna().any(axis=1)
    out = pd.Series(np.nan, index=df.index, dtype="float")
    out.loc[known] = 0
    out.loc[yes] = 1
    return out


def clean_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    # Stata extended missings from some harmonized files may appear as large ints.
    out = out.mask(out > 1_000_000_000)
    return out


def wave_candidates(wave: int) -> dict[str, list[str]]:
    w = f"r{wave}"
    h = f"h{wave}"
    return {
        "age": [f"{w}agey", f"{w}agey_b", f"{w}agey_e"],
        "stroke": [f"{w}stroke"],
        "adl": [
            f"{w}adltot6",
            f"{w}adlfive",
            f"{w}adla_c",
            f"{w}adla",
            f"{w}adlwb",
        ],
        "iadl": [
            f"{w}iadlfour",
            f"{w}iadla",
            f"{w}iadlza",
            f"{w}iadlb",
        ],
        "any_care": [
            f"{w}rcany",
            f"{w}rcany_e",
            f"{w}racany",
            f"{w}ricany",
        ],
        "informal_care": [
            f"{w}rcaany",
            f"{w}rcaany_e",
            f"{w}racaany",
            f"{w}ricaany",
            f"{w}rascare",
            f"{w}rscare",
            f"{w}raccare",
            f"{w}rccare",
            f"{w}rrcare",
            f"{w}rfcare",
        ],
        "formal_care": [
            f"{w}rfaany",
            f"{w}rfaany_e",
            f"{w}rafaany",
            f"{w}rifaany",
            f"{w}rpfcare",
            f"{w}rpfcare_e",
            f"{w}rapfcare",
            f"{w}rufcare",
        ],
        "wealth": [
            f"{h}atotb",
            f"{h}atotw",
            f"{h}atotn",
            f"{h}atotf",
            f"{h}itot",
        ],
        "rural": [f"{h}rural", f"{w}rural", f"{w}rural2"],
        "n_children": [f"{h}child", f"{w}child"],
        "coresident_child": [f"{h}coresd", f"{w}coresd"],
        "coupled": [f"{h}cpl"],
    }


def needed_columns(dataset: str, columns: list[str]) -> list[str]:
    colset = set(columns)
    needed = []
    needed.extend(present(colset, ID_CANDIDATES[dataset]))
    needed.extend(present(colset, ["ragender", "raeducl", "raeduc_c", "raeduc_l", "country", "isocountry"]))
    for wave in wave_numbers(columns):
        for candidates in wave_candidates(wave).values():
            needed.extend(present(colset, candidates))
    return sorted(set(needed), key=needed.index)


def build_long_for_dataset(dataset: str, path: Path) -> pd.DataFrame:
    columns = get_columns(path)
    colset = set(columns)
    waves = wave_numbers(columns)
    read_cols = needed_columns(dataset, columns)
    df = pd.read_stata(str(path), columns=read_cols, convert_categoricals=False)
    respondent_id_col = first_existing(colset, ID_CANDIDATES[dataset])
    if respondent_id_col is None:
        raise ValueError(f"No respondent id found for {dataset}")

    frames = []
    for wave in waves:
        cand = wave_candidates(wave)
        stroke = clean_numeric(value_from_candidates(df, cand["stroke"]))
        if stroke.notna().sum() == 0:
            continue
        age = clean_numeric(value_from_candidates(df, cand["age"]))
        adl = clean_numeric(value_from_candidates(df, cand["adl"]))
        iadl = clean_numeric(value_from_candidates(df, cand["iadl"]))
        any_care = any_yes(df, cand["any_care"])
        informal = any_yes(df, cand["informal_care"])
        formal = any_yes(df, cand["formal_care"])
        wealth = clean_numeric(value_from_candidates(df, cand["wealth"]))
        rural = clean_numeric(value_from_candidates(df, cand["rural"]))
        n_children = clean_numeric(value_from_candidates(df, cand["n_children"]))
        coresident_child = clean_numeric(value_from_candidates(df, cand["coresident_child"]))
        coupled = clean_numeric(value_from_candidates(df, cand["coupled"]))

        out = pd.DataFrame(
            {
                "dataset": dataset,
                "respondent_id": df[respondent_id_col].astype(str),
                "wave": wave,
                "age": age,
                "female": clean_numeric(df["ragender"]).eq(2).astype(float) if "ragender" in df.columns else np.nan,
                "education_level": clean_numeric(value_from_candidates(df, ["raeducl", "raeduc_c", "raeduc_l"])),
                "share_country_code": clean_numeric(df["country"]) if "country" in df.columns else np.nan,
                "post_stroke": stroke.eq(1).astype(float).where(stroke.notna(), np.nan),
                "adl_score": adl,
                "iadl_score": iadl,
                "any_care": any_care,
                "informal_care": informal,
                "formal_care": formal,
                "wealth": wealth,
                "rural": rural,
                "n_children": n_children,
                "coresident_child": coresident_child,
                "coupled": coupled,
            }
        )
        out["care_need"] = ((out["adl_score"] > 0) | (out["iadl_score"] > 0)).astype(float)
        out.loc[out["adl_score"].isna() & out["iadl_score"].isna(), "care_need"] = np.nan
        frames.append(out)

    if not frames:
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)
    long = long[long["age"].ge(50) | long["age"].isna()].copy()
    long["pcgi_code"] = np.nan
    long.loc[long["post_stroke"].eq(1) & long["care_need"].eq(0), "pcgi_code"] = 0
    long.loc[
        long["post_stroke"].eq(1) & long["care_need"].eq(1) & long["formal_care"].eq(1),
        "pcgi_code",
    ] = 1
    long.loc[
        long["post_stroke"].eq(1)
        & long["care_need"].eq(1)
        & long["formal_care"].ne(1)
        & ((long["informal_care"].eq(1)) | (long["any_care"].eq(1))),
        "pcgi_code",
    ] = 2
    long.loc[
        long["post_stroke"].eq(1)
        & long["care_need"].eq(1)
        & long["any_care"].eq(0)
        & long["informal_care"].ne(1)
        & long["formal_care"].ne(1),
        "pcgi_code",
    ] = 3
    long["pcgi_label"] = long["pcgi_code"].map(PCGI_LABELS)
    return long


def wilson_ci(count: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0 or pd.isna(n):
        return (np.nan, np.nan)
    p = count / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return center - half, center + half


def summarize_pcgi(latest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, g in latest.groupby("dataset", dropna=False):
        known = g[g["pcgi_code"].notna()]
        denom = len(known)
        for code, label in PCGI_LABELS.items():
            count = int((known["pcgi_code"] == code).sum())
            lo, hi = wilson_ci(count, denom)
            rows.append(
                {
                    "dataset": dataset,
                    "pcgi_code": code,
                    "pcgi_label": label,
                    "n": count,
                    "denominator_known_pcgi": denom,
                    "percent": 100 * count / denom if denom else np.nan,
                    "ci_lower_percent": 100 * lo if pd.notna(lo) else np.nan,
                    "ci_upper_percent": 100 * hi if pd.notna(hi) else np.nan,
                    "unknown_pcgi": int(g["pcgi_code"].isna().sum()),
                    "post_stroke_latest_n": len(g),
                }
            )
    return pd.DataFrame(rows)


def summarize_components(latest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, g in latest.groupby("dataset"):
        rows.append(
            {
                "dataset": dataset,
                "post_stroke_latest_n": len(g),
                "mean_age": g["age"].mean(),
                "female_percent": g["female"].mean() * 100,
                "care_need_percent": g["care_need"].mean() * 100,
                "any_care_percent_among_need": g.loc[g["care_need"].eq(1), "any_care"].mean() * 100,
                "informal_care_percent_among_need": g.loc[g["care_need"].eq(1), "informal_care"].mean() * 100,
                "formal_care_percent_among_need": g.loc[g["care_need"].eq(1), "formal_care"].mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def add_wealth_tertiles(latest: pd.DataFrame) -> pd.DataFrame:
    latest = latest.copy()
    latest["wealth_tertile"] = pd.Series(pd.NA, index=latest.index, dtype="object")
    for dataset, idx in latest.groupby("dataset").groups.items():
        vals = latest.loc[idx, "wealth"]
        if vals.notna().sum() < 50 or vals.nunique(dropna=True) < 3:
            continue
        try:
            latest.loc[idx, "wealth_tertile"] = pd.qcut(vals, 3, labels=["Low", "Middle", "High"], duplicates="drop")
        except ValueError:
            continue
    return latest


def stratified_unmet(latest: pd.DataFrame, stratifier: str) -> pd.DataFrame:
    rows = []
    data = latest[latest["care_need"].eq(1) & latest["pcgi_code"].notna()].copy()
    data["unmet"] = data["pcgi_code"].eq(3).astype(int)
    for (dataset, level), g in data.groupby(["dataset", stratifier], dropna=False):
        if pd.isna(level):
            continue
        n = len(g)
        count = int(g["unmet"].sum())
        lo, hi = wilson_ci(count, n)
        rows.append(
            {
                "dataset": dataset,
                stratifier: level,
                "n_care_need_known_pcgi": n,
                "unmet_n": count,
                "unmet_percent": 100 * count / n if n else np.nan,
                "ci_lower_percent": 100 * lo if pd.notna(lo) else np.nan,
                "ci_upper_percent": 100 * hi if pd.notna(hi) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def merge_gbd(pcgi_dist: pd.DataFrame) -> pd.DataFrame:
    unmet = pcgi_dist[pcgi_dist["pcgi_code"].eq(3)].copy()
    unmet["gbd_location"] = unmet["dataset"].map(GBD_LOCATION)
    gbd_path = OUT / "gbd_2023_stroke_60plus_rates_wide.csv"
    if not gbd_path.exists():
        return unmet
    gbd = pd.read_csv(gbd_path)
    merged = unmet.merge(gbd, left_on="gbd_location", right_on="location_name", how="left")
    merged["care_pressure_index"] = merged["percent"] * merged["ylds_rate"] / 100
    return merged


def make_report(
    latest: pd.DataFrame,
    pcgi_dist: pd.DataFrame,
    components: pd.DataFrame,
    gbd_context: pd.DataFrame,
) -> str:
    lines = []
    lines.append("# PCGI Initial Analysis Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "This first-pass analysis constructs a Post-stroke Care Gap Index (PCGI) from local harmonized aging cohorts. "
        "The primary descriptive sample uses each respondent's latest available wave with post-stroke status at age 50 years or older."
    )
    lines.append("")
    lines.append("## PCGI Definition")
    lines.append("")
    lines.append("| Code | Label |")
    lines.append("| ---: | --- |")
    for code, label in PCGI_LABELS.items():
        lines.append(f"| {code} | {label} |")
    lines.append("")
    lines.append("## Latest-Wave Post-Stroke Sample")
    lines.append("")
    lines.append("| Dataset | N | Mean age | Female % | Care need % | Any care among need % | Informal care among need % | Formal care among need % |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in components.sort_values("dataset").itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.post_stroke_latest_n:,} | {row.mean_age:.1f} | {row.female_percent:.1f} | "
            f"{row.care_need_percent:.1f} | {row.any_care_percent_among_need:.1f} | "
            f"{row.informal_care_percent_among_need:.1f} | {row.formal_care_percent_among_need:.1f} |"
        )
    lines.append("")
    lines.append("## PCGI Distribution")
    lines.append("")
    lines.append("| Dataset | PCGI category | N | % | 95% CI | Known denominator | Unknown PCGI |")
    lines.append("| --- | --- | ---: | ---: | --- | ---: | ---: |")
    for row in pcgi_dist.sort_values(["dataset", "pcgi_code"]).itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.pcgi_label} | {row.n:,} | {row.percent:.1f} | "
            f"{row.ci_lower_percent:.1f}-{row.ci_upper_percent:.1f} | "
            f"{row.denominator_known_pcgi:,} | {row.unknown_pcgi:,} |"
        )
    lines.append("")
    lines.append("## GBD Context Linkage")
    lines.append("")
    show = gbd_context[gbd_context["gbd_location"].notna()].copy()
    if show.empty:
        lines.append("GBD linkage table was not available.")
    else:
        lines.append("| Dataset | GBD location | Unmet care % | Stroke YLD rate 60+ | Stroke DALY rate 60+ | Care Pressure Index |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for row in show.sort_values("care_pressure_index", ascending=False).itertuples(index=False):
            yld = "" if pd.isna(row.ylds_rate) else f"{row.ylds_rate:.1f}"
            daly = "" if pd.isna(row.dalys_rate) else f"{row.dalys_rate:.1f}"
            cpi = "" if pd.isna(row.care_pressure_index) else f"{row.care_pressure_index:.1f}"
            lines.append(f"| {row.dataset} | {row.gbd_location} | {row.percent:.1f} | {yld} | {daly} | {cpi} |")
    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- These are initial descriptive estimates; formal modelling and inequality decomposition should follow.")
    lines.append("- SHARE is kept as a harmonized European cohort in the PCGI analysis, but it is not forced onto a single GBD location in the current context table.")
    lines.append("- LASI is included but has one harmonized wave in the scanned file, so it functions mainly as a cross-sectional comparison.")
    lines.append("- HRS uses the local harmonized temp file because the local Gateway HRS file did not contain ever-stroke variables.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for name in [
        "pcgi_person_wave_long.csv",
        "pcgi_latest_poststroke_dataset.csv",
        "pcgi_latest_distribution.csv",
        "pcgi_latest_components.csv",
        "pcgi_unmet_by_education.csv",
        "pcgi_unmet_by_wealth_tertile.csv",
        "pcgi_gbd_context_table.csv",
    ]:
        lines.append(f"- `{OUT / name}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    frames = []
    manifest_rows = []
    for dataset, pattern in DATASETS.items():
        path = resolve_dataset(pattern)
        long = build_long_for_dataset(dataset, path)
        frames.append(long)
        manifest_rows.append(
            {
                "dataset": dataset,
                "source_file": str(path),
                "person_wave_rows": len(long),
                "post_stroke_person_wave_rows": int(long["post_stroke"].eq(1).sum()) if not long.empty else 0,
            }
        )

    person_wave = pd.concat(frames, ignore_index=True)
    person_wave.to_csv(OUT / "pcgi_person_wave_long.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(manifest_rows).to_csv(OUT / "pcgi_analysis_manifest.csv", index=False, encoding="utf-8-sig")

    post = person_wave[person_wave["post_stroke"].eq(1)].copy()
    post = post.sort_values(["dataset", "respondent_id", "wave"])
    latest = post.groupby(["dataset", "respondent_id"], as_index=False, dropna=False).tail(1).copy()
    latest = add_wealth_tertiles(latest)
    latest.to_csv(OUT / "pcgi_latest_poststroke_dataset.csv", index=False, encoding="utf-8-sig")

    pcgi_dist = summarize_pcgi(latest)
    pcgi_dist.to_csv(OUT / "pcgi_latest_distribution.csv", index=False, encoding="utf-8-sig")

    components = summarize_components(latest)
    components.to_csv(OUT / "pcgi_latest_components.csv", index=False, encoding="utf-8-sig")

    edu = stratified_unmet(latest, "education_level")
    edu.to_csv(OUT / "pcgi_unmet_by_education.csv", index=False, encoding="utf-8-sig")

    wealth = stratified_unmet(latest, "wealth_tertile")
    wealth.to_csv(OUT / "pcgi_unmet_by_wealth_tertile.csv", index=False, encoding="utf-8-sig")

    gbd_context = merge_gbd(pcgi_dist)
    gbd_context.to_csv(OUT / "pcgi_gbd_context_table.csv", index=False, encoding="utf-8-sig")

    (REPORTS / "pcgi_initial_analysis_report.md").write_text(
        make_report(latest, pcgi_dist, components, gbd_context),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
