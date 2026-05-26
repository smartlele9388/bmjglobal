from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step6"
FIG = OUT / "figures"
STEP3B = ROOT / "results" / "step3b"
STEP3C = ROOT / "results" / "step3c"
FIG.mkdir(parents=True, exist_ok=True)

COHORTS = ["CHARLS", "HRS", "ELSA", "SHARE", "KLoSA", "LASI", "MHAS"]
CARE_COHORTS = ["CHARLS", "HRS", "ELSA", "KLoSA", "LASI", "MHAS"]


def read_table(name: str) -> pd.DataFrame:
    path = OUT / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def err(point: pd.Series, lo: pd.Series, hi: pd.Series) -> np.ndarray:
    lower = (point - lo).clip(lower=0).fillna(0).to_numpy()
    upper = (hi - point).clip(lower=0).fillna(0).to_numpy()
    return np.vstack([lower, upper])


def fig1(t33: pd.DataFrame) -> None:
    data = t33[t33["indicator"].eq("mean YLD-like disability weight")].copy()
    data["cohort"] = pd.Categorical(data["cohort"], COHORTS, ordered=True)
    data = data.sort_values("cohort")
    data.to_csv(FIG / "fig1_yld_like_disability_burden_by_cohort_data.csv", index=False)

    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.errorbar(data["estimate"], y, xerr=err(data["estimate"], data["ci_lower"], data["ci_upper"]), fmt="o", color="#1f5a85", ecolor="#6b879a", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(data["cohort"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean YLD-like disability weight (95% CI)")
    ax.set_ylabel("Cohort")
    ax.grid(axis="x", alpha=0.25)
    ax.text(0, -0.18, "Survey-based YLD-like disability burden; not official GBD YLD.", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    save(fig, "fig1_yld_like_disability_burden_by_cohort")


def fig2(t33: pd.DataFrame) -> None:
    strict = t33[t33["indicator"].eq("strict unmet need %")][["cohort", "estimate", "ci_lower", "ci_upper"]].copy()
    strict["measure"] = "Strict unmet need"
    gap = t33[t33["indicator"].eq("informal care gap %")][["cohort", "estimate", "ci_lower", "ci_upper"]].copy()
    gap["measure"] = "Informal care gap"
    data = pd.concat([strict, gap], ignore_index=True)
    data["cohort"] = pd.Categorical(data["cohort"], COHORTS, ordered=True)
    data.to_csv(FIG / "fig2_strict_unmet_vs_informal_care_gap_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(COHORTS))
    offsets = {"Strict unmet need": -0.16, "Informal care gap": 0.16}
    colors = {"Strict unmet need": "#8f3f3f", "Informal care gap": "#2f6f5e"}
    for measure in ["Strict unmet need", "Informal care gap"]:
        sub = data[data["measure"].eq(measure)].set_index("cohort").reindex(COHORTS)
        ax.errorbar(x + offsets[measure], sub["estimate"], yerr=err(sub["estimate"], sub["ci_lower"], sub["ci_upper"]), fmt="o", color=colors[measure], ecolor=colors[measure], capsize=3, label=measure)
        for i, value in enumerate(sub["estimate"]):
            if pd.isna(value):
                ax.text(x[i] + offsets[measure], 2, "NA", ha="center", va="bottom", fontsize=8, color=colors[measure])
    ax.set_xticks(x)
    ax.set_xticklabels(COHORTS, rotation=30, ha="right")
    ax.set_ylabel("Percent (95% CI)")
    ax.set_xlabel("Cohort")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(0, -0.22, "LASI strict unmet need unavailable; SHARE informal care gap unavailable because harmonized care hours are not validated.", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    save(fig, "fig2_strict_unmet_vs_informal_care_gap")


def fig3(t33: pd.DataFrame) -> None:
    all_hours = t33[t33["indicator"].eq("mean informal care hours/week among all stroke survivors")][["cohort", "estimate", "ci_lower", "ci_upper"]].copy()
    all_hours["measure"] = "All stroke survivors"
    rec_hours = t33[t33["indicator"].eq("mean informal care hours/week among informal-care recipients only")][["cohort", "estimate", "ci_lower", "ci_upper"]].copy()
    rec_hours["measure"] = "Recipients only"
    data = pd.concat([all_hours, rec_hours], ignore_index=True)
    data = data[data["cohort"].isin(CARE_COHORTS)].copy()
    data["cohort"] = pd.Categorical(data["cohort"], CARE_COHORTS, ordered=True)
    data.to_csv(FIG / "fig3_informal_care_hours_all_vs_recipients_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(CARE_COHORTS))
    offsets = {"All stroke survivors": -0.16, "Recipients only": 0.16}
    colors = {"All stroke survivors": "#2c5f9e", "Recipients only": "#b45f2f"}
    for measure in ["All stroke survivors", "Recipients only"]:
        sub = data[data["measure"].eq(measure)].set_index("cohort").reindex(CARE_COHORTS)
        ax.errorbar(x + offsets[measure], sub["estimate"], yerr=err(sub["estimate"], sub["ci_lower"], sub["ci_upper"]), fmt="o", color=colors[measure], ecolor=colors[measure], capsize=3, label=measure)
    ax.set_xticks(x)
    ax.set_xticklabels(CARE_COHORTS, rotation=30, ha="right")
    ax.set_ylabel("Hours/week (95% CI)")
    ax.set_xlabel("Care-hour eligible cohort")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(0, -0.20, "SHARE excluded; missing care hours were not recoded to zero.", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    save(fig, "fig3_informal_care_hours_all_vs_recipients")


def fig4() -> None:
    t37 = read_table("table37_yld_like_sensitivity.csv")
    data = t37[t37["sensitivity_mapping"].eq("Main mapping")].copy()
    data["cohort"] = pd.Categorical(data["cohort"], COHORTS, ordered=True)
    data = data.sort_values("cohort")
    data.to_csv(FIG / "fig4_patient_side_sex_difference_mean_DW_data.csv", index=False)

    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.axvline(0, color="#444444", linewidth=1)
    ax.plot(data["female_minus_male_mean_DW"], y, "o", color="#704c8f")
    ax.set_yticks(y)
    ax.set_yticklabels(data["cohort"])
    ax.invert_yaxis()
    ax.set_xlabel("Female minus male mean YLD-like disability weight")
    ax.set_ylabel("Cohort")
    ax.grid(axis="x", alpha=0.25)
    ax.text(0, -0.18, "Patient-side sex difference only; descriptive and not causal.", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    save(fig, "fig4_patient_side_sex_difference_mean_DW")


def supplementary_hrs() -> None:
    shares_path = STEP3B / "table8_step3b_gender_care_shares.csv"
    robust_path = STEP3C / "table19_step3c_robustness_summary.csv"
    shares = pd.read_csv(shares_path) if shares_path.exists() else pd.DataFrame()
    robust = pd.read_csv(robust_path) if robust_path.exists() else pd.DataFrame()
    shares.to_csv(FIG / "supp_fig_hrs_provider_data.csv", index=False)
    if not shares.empty and "analysis" in shares.columns:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        vals = pd.to_numeric(shares.iloc[0].filter(like="share"), errors="coerce").dropna()
        if vals.empty:
            vals = pd.Series({"female_provider_share": pd.to_numeric(shares.get("female_provider_share", pd.Series([np.nan])), errors="coerce").iloc[0]})
        vals.plot(kind="bar", ax=ax, color="#4f7899")
        ax.set_ylabel("Share")
        ax.set_xlabel("HRS-only provider-side measure")
        ax.set_title("HRS-only provider-side gender matrix")
        ax.text(0, -0.30, "HRS-only exploratory provider-side result; not cross-national.", transform=ax.transAxes, fontsize=8, ha="left", va="top")
        save(fig, "supp_fig_hrs_provider_side_gender_matrix")
    else:
        fig, ax = plt.subplots(figsize=(6.2, 3.0))
        ax.text(0.5, 0.5, "HRS-only provider-side data unavailable", ha="center", va="center")
        ax.axis("off")
        save(fig, "supp_fig_hrs_provider_side_gender_matrix")

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    if not robust.empty and {"domain", "main_estimate"}.issubset(robust.columns):
        plot = robust.copy()
        plot["main_estimate"] = pd.to_numeric(plot["main_estimate"], errors="coerce")
        plot = plot.dropna(subset=["main_estimate"]).head(10)
        ax.barh(plot["domain"].astype(str) + ": " + plot["analysis"].astype(str), plot["main_estimate"], color="#6f8f52")
        ax.set_xlabel("Estimate")
    else:
        ax.text(0.5, 0.5, "HRS-only sensitivity data unavailable", ha="center", va="center")
        ax.axis("off")
    ax.set_title("HRS-only provider-side sensitivity")
    ax.text(0, -0.22, "Exploratory HRS-only provider-side sensitivity; no hidden gender care tax calculated.", transform=ax.transAxes, fontsize=8, ha="left", va="top")
    save(fig, "supp_fig_hrs_provider_side_sensitivity")


def main() -> None:
    t33 = read_table("table33_main_estimates_with_95ci.csv")
    fig1(t33)
    fig2(t33)
    fig3(t33)
    fig4()
    supplementary_hrs()


if __name__ == "__main__":
    main()
