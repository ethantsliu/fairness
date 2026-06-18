#!/usr/bin/env python3
"""Regenerate homogeneous_vs_mixed_training.png from saved CSV."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV = "/Users/aakashsuresh/fairness/blood_glucose_project/results/homogeneous_vs_heterogeneous_comparison.csv"
OUT = "/Users/aakashsuresh/fairness/blood_glucose_project/figures/publication/homogeneous_vs_mixed_training.png"

SEX_MAP = {"1.0": "Male", "2.0": "Female"}


def label_group(gid: str) -> str:
    age, sex = gid.split("__")
    return f"{age}, {SEX_MAP.get(sex, sex)}"


def main():
    df = pd.read_csv(CSV)
    df["group_label"] = df["group_id"].map(label_group)
    plot_df = df.melt(
        id_vars=["group_label"],
        value_vars=["glucose_mae_heterogeneous", "glucose_mae_homogeneous"],
        var_name="training_condition",
        value_name="glucose_mae",
    )
    plot_df["training_condition"] = plot_df["training_condition"].map({
        "glucose_mae_heterogeneous": "Heterogeneous (pooled)",
        "glucose_mae_homogeneous": "Homogeneous (within-group)",
    })

    plt.figure(figsize=(11, 7))
    sns.barplot(data=plot_df, x="group_label", y="glucose_mae", hue="training_condition")
    plt.title("Homogeneous vs Heterogeneous Training by Age-by-Sex Stratum")
    plt.xlabel("Subgroup (age bin × sex)")
    plt.ylabel("Glucose MAE (mg/dL)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Training")
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=300)
    plt.close()
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
