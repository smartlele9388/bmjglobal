from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


COLORS = {
    "No disability-related care need": "#5B8DB8",
    "Need met with formal/professional care": "#4C956C",
    "Need met with informal/family care only": "#F2C14E",
    "Unmet disability-related care need": "#C44536",
}


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")


def plot_pcgi_distribution() -> None:
    dist = pd.read_csv(TABLES / "pcgi_latest_distribution.csv")
    order = (
        dist[dist["pcgi_code"].eq(3)]
        .sort_values("percent", ascending=False)["dataset"]
        .tolist()
    )
    labels = [
        "No disability-related care need",
        "Need met with formal/professional care",
        "Need met with informal/family care only",
        "Unmet disability-related care need",
    ]
    pivot = dist.pivot(index="dataset", columns="pcgi_label", values="percent").reindex(order)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))
    for label in labels:
        values = pivot[label].fillna(0).values
        ax.barh(y, values, left=left, color=COLORS[label], edgecolor="white", linewidth=0.8, label=label)
        left += values
    ax.set_yticks(y, pivot.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Distribution among post-stroke respondents with known PCGI (%)")
    ax.set_title("Post-stroke Care Gap Index distribution by harmonized cohort", loc="left", fontsize=12, pad=12)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=2, frameon=False, fontsize=8.5)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    save_figure(fig, "pcgi_distribution_by_cohort")
    plt.close(fig)


def plot_gbd_care_pressure() -> None:
    ctx = pd.read_csv(TABLES / "pcgi_gbd_context_table.csv")
    ctx = ctx[ctx["gbd_location"].notna() & ctx["ylds_rate"].notna()].copy()
    if ctx.empty:
        return
    # Size points by GBD 60+ stroke prevalence rate for a compact pressure map.
    size = 80 + 520 * (ctx["prevalence_rate"] - ctx["prevalence_rate"].min()) / (
        ctx["prevalence_rate"].max() - ctx["prevalence_rate"].min()
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(
        ctx["ylds_rate"],
        ctx["percent"],
        s=size,
        c="#2F5D7C",
        alpha=0.78,
        edgecolor="white",
        linewidth=1.2,
    )
    for row in ctx.itertuples(index=False):
        ax.annotate(
            row.dataset,
            (row.ylds_rate, row.percent),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("GBD 2023 stroke YLD rate, age 60+")
    ax.set_ylabel("Unmet disability-related care need among post-stroke respondents (%)")
    ax.set_title("Stroke disability burden and post-stroke care gaps", loc="left", fontsize=12, pad=12)
    ax.grid(color="#D9D9D9", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    note = "Point size is proportional to GBD 2023 stroke prevalence rate, age 60+."
    ax.text(0, -0.18, note, transform=ax.transAxes, fontsize=8.5, color="#555555")
    save_figure(fig, "pcgi_gbd_care_pressure_scatter")
    plt.close(fig)


def main() -> None:
    plot_pcgi_distribution()
    plot_gbd_care_pressure()


if __name__ == "__main__":
    main()
