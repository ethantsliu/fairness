#!/usr/bin/env python3
"""Extract homogeneous vs heterogeneous training results and robustness checks."""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from comprehensive_feedback_implementation import ComprehensiveFeedbackImplementation

OUT_DIR = "/Users/aakashsuresh/fairness/blood_glucose_project/results"
CSV_PATH = os.path.join(OUT_DIR, "homogeneous_vs_heterogeneous_comparison.csv")
SUMMARY_PATH = os.path.join(OUT_DIR, "homogeneous_vs_heterogeneous_summary.md")


def interpret_delta(delta: float, tol: float = 1.0) -> str:
    if delta < -tol:
        return "pooling_hurts"
    if delta > tol:
        return "pooling_helps"
    return "not_pooling_driven"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    analyzer = ComprehensiveFeedbackImplementation()
    analyzer.load_and_prepare_data()
    result = analyzer.homogeneous_vs_heterogeneous_training_comparison()
    comp_df = result["comparison_df"].copy()

    if comp_df.empty:
        raise RuntimeError("No homogeneous vs heterogeneous comparison rows produced.")

    comp_df["interpretation"] = comp_df["mae_delta_hom_minus_het"].apply(interpret_delta)
    sex_map = {"1.0": "Male", "2.0": "Female"}
    comp_df["subgroup_label"] = comp_df["group_id"].apply(
        lambda g: f"{g.split('__')[0]}, {sex_map.get(g.split('__')[1], g.split('__')[1])}"
    )
    comp_df.to_csv(CSV_PATH, index=False)

    n = len(comp_df)
    n_near_zero = int((comp_df["interpretation"] == "not_pooling_driven").sum())
    n_pooling_hurts = int((comp_df["interpretation"] == "pooling_hurts").sum())
    n_pooling_helps = int((comp_df["interpretation"] == "pooling_helps").sum())

    # subgroup MAE gap from heterogeneous model (spread across groups)
    het_range = comp_df["glucose_mae_heterogeneous"].max() - comp_df["glucose_mae_heterogeneous"].min()
    sd_corr_r, sd_corr_p = stats.pearsonr(
        comp_df["group_glucose_sd"], comp_df["glucose_mae_heterogeneous"]
    )
    delta_sd_r, delta_sd_p = stats.pearsonr(
        comp_df["group_glucose_sd"], comp_df["mae_delta_hom_minus_het"].abs()
    )

    lines = [
        "# Homogeneous vs Heterogeneous Training Results",
        "",
        f"Groups analyzed: **{n}** stable age-by-sex strata (n >= 160 per stratum).",
        "",
        "## Per-group results",
        "",
        comp_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Summary counts (|Δ| <= 1.0 mg/dL treated as near-zero)",
        "",
        f"- Not pooling-driven (|Δ| <= 1.0): **{n_near_zero}/{n}**",
        f"- Pooling hurts (Δ < -1.0): **{n_pooling_hurts}/{n}**",
        f"- Pooling helps / borrow strength (Δ > 1.0): **{n_pooling_helps}/{n}**",
        "",
        f"- Heterogeneous MAE range across groups: **{het_range:.3f} mg/dL**",
        f"- Pearson r(group glucose SD, heterogeneous MAE): **{sd_corr_r:.3f}** (p={sd_corr_p:.4f})",
        f"- Pearson r(group glucose SD, |Δ|): **{delta_sd_r:.3f}** (p={delta_sd_p:.4f})",
        "",
    ]
    SUMMARY_PATH_WRITE = SUMMARY_PATH
    with open(SUMMARY_PATH_WRITE, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved: {CSV_PATH}")
    print(f"Saved: {SUMMARY_PATH}")
    print(comp_df.to_string(index=False))
    print(f"\nSummary: near_zero={n_near_zero}, pooling_hurts={n_pooling_hurts}, pooling_helps={n_pooling_helps}")
    return comp_df


if __name__ == "__main__":
    main()
