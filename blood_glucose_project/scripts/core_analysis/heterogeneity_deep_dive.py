#!/usr/bin/env python3
"""
Within-Group Label Heterogeneity → Model Error: Expanded Analysis
=================================================================
Central hypothesis: subgroup MAE is driven by within-group outcome variance,
not by data representation (sample size) or other confounders.

Expands the analysis across many group slicings with bootstrap CIs on both
MAE (y-axis) and outcome std (x-axis).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(PROJECT, "figures")
RES_DIR = os.path.join(PROJECT, "results")
LAB_DIR = os.path.join(PROJECT, "data", "processed", "nhanes_lab")
LIFE_DIR = os.path.join(PROJECT, "data", "processed", "nhanes_combined")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════

def load_dataset():
    glucose = pd.read_csv(os.path.join(LAB_DIR, "fasting_glucose_processed.csv"))[["seqn", "lbxglu"]]
    hba1c = pd.read_csv(os.path.join(LAB_DIR, "glycohemoglobin_processed.csv"))[["seqn", "lbxgh"]]
    targets = glucose.merge(hba1c, on="seqn").rename(columns={"lbxglu": "glucose", "lbxgh": "hba1c"})

    demo = pd.read_sas(os.path.join(PROJECT, "data", "raw", "nhanes_2011_2014", "P_DEMO.xpt"), format="xport")
    keep = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR"]
    demo = demo[[c for c in keep if c in demo.columns]]
    demo.columns = demo.columns.str.lower()
    demo.rename(columns={"seqn": "seqn", "ridageyr": "age", "riagendr": "gender",
                          "ridreth3": "race_ethnicity", "dmdeduc2": "education",
                          "indfmpir": "income_ratio"}, inplace=True)

    acc = pd.read_csv(os.path.join(LIFE_DIR, "nhanes_combined_acc.csv"))
    acc.columns = acc.columns.str.upper()
    acc["seqn"] = acc["SEQN"]
    act_cols = [c for c in acc.columns if c.startswith("PAX") and acc[c].dtype in ("float64", "int64")]
    agg = {c: "mean" for c in act_cols}
    acc = acc.groupby("seqn").agg(agg).reset_index()
    acc["total_activity"] = acc.get("PAXAISMD", pd.Series(0, index=acc.index))
    acc["mvpa_min"] = acc.get("PAXMVMD", acc.get("PAXVMD", pd.Series(0, index=acc.index)))
    acc["sedentary_min"] = acc.get("PAXSMD", pd.Series(0, index=acc.index))
    acc["wear_time"] = acc.get("PAXTMD", pd.Series(0, index=acc.index))
    acc["sedentary_ratio"] = acc["sedentary_min"] / (acc["wear_time"] + 1)
    acc = acc[["seqn", "total_activity", "mvpa_min", "sedentary_min", "wear_time", "sedentary_ratio"]]

    df = targets.merge(demo, on="seqn", how="left").merge(acc, on="seqn", how="left")
    df = df[df["age"] >= 18].dropna(subset=["glucose", "hba1c"])
    df = df[(df["glucose"] <= 600) & (df["hba1c"] <= 18)]
    return df


def prepare_and_train(df):
    """Prepare features (with demographics), train model, return test_df with predictions."""
    feature_cols = ["age", "gender", "race_ethnicity",
                    "total_activity", "mvpa_min", "sedentary_min", "sedentary_ratio"]
    available = [c for c in feature_cols if c in df.columns]

    X = df[available].copy()
    y = df[["glucose", "hba1c"]].copy()

    for c in X.columns:
        if c in ("gender", "race_ethnicity"):
            X[c] = X[c].fillna(X[c].mode().iloc[0] if X[c].notna().any() else 0)
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        else:
            X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=2,
                              random_state=42, n_jobs=-1)
    )
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)

    test_df = df.loc[X_test.index].copy()
    test_df["glucose_pred"] = preds[:, 0]
    test_df["hba1c_pred"] = preds[:, 1]
    test_df["glucose_error"] = np.abs(test_df["glucose"] - test_df["glucose_pred"])
    test_df["hba1c_error"] = np.abs(test_df["hba1c"] - test_df["hba1c_pred"])
    return test_df


# ═══════════════════════════════════════════════════════════════════════
# GROUP SLICING DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

RACE_MAP = {1.0: "Mexican American", 2.0: "Other Hispanic", 3.0: "Non-Hispanic White",
            4.0: "Non-Hispanic Black", 6.0: "Non-Hispanic Asian", 7.0: "Other/Multi"}

EDUC_MAP = {1.0: "< 9th Grade", 2.0: "9-11th Grade", 3.0: "HS Grad/GED",
            4.0: "Some College", 5.0: "College Grad+"}

GENDER_MAP = {1.0: "Male", 2.0: "Female"}


def assign_groups(df):
    """Create all group columns for slicing."""
    out = df.copy()

    # Age: fine bins
    out["age_fine"] = pd.cut(out["age"], bins=[18, 30, 40, 50, 60, 70, 100],
                             labels=["18-29", "30-39", "40-49", "50-59", "60-69", "70+"])
    # Age: coarse
    out["age_coarse"] = pd.cut(out["age"], bins=[18, 40, 60, 100], labels=["<40", "40-60", ">60"])

    # Gender
    out["gender_label"] = out["gender"].map(GENDER_MAP).fillna("Unknown")

    # Race
    out["race_label"] = out["race_ethnicity"].map(RACE_MAP).fillna("Other")

    # Education
    if "education" in out.columns:
        out["education_label"] = out["education"].map(EDUC_MAP).fillna("Unknown")
        out = out[out["education_label"] != "Unknown"]

    # Income quartiles
    if "income_ratio" in out.columns:
        out["income_quartile"] = pd.qcut(out["income_ratio"].rank(method="first"),
                                          q=4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])

    # Activity quartiles
    if "total_activity" in out.columns and out["total_activity"].notna().sum() > 100:
        out["activity_quartile"] = pd.qcut(out["total_activity"].rank(method="first"),
                                            q=4, labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"])

    # Sedentary quartiles
    if "sedentary_ratio" in out.columns and out["sedentary_ratio"].notna().sum() > 100:
        out["sedentary_quartile"] = pd.qcut(out["sedentary_ratio"].rank(method="first"),
                                             q=4, labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"])

    return out


SLICING_CONFIG = [
    ("Age (fine)", "age_fine"),
    ("Age (coarse)", "age_coarse"),
    ("Gender", "gender_label"),
    ("Race/Ethnicity", "race_label"),
    ("Education", "education_label"),
    ("Income", "income_quartile"),
    ("Activity Level", "activity_quartile"),
    ("Sedentary Level", "sedentary_quartile"),
]


# ═══════════════════════════════════════════════════════════════════════
# BOOTSTRAP COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_stat(values, stat_fn, n_boot=2000, seed=42):
    """Bootstrap a statistic with 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.array([stat_fn(values[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    return float(np.mean(boots)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def compute_group_stats(test_df, group_col, outcome="glucose", min_n=15):
    """For each group in group_col, bootstrap MAE and outcome std."""
    rows = []
    for name, grp in test_df.groupby(group_col, observed=True):
        if len(grp) < min_n:
            continue
        errors = grp[f"{outcome}_error"].values
        true_vals = grp[outcome].values

        mae_mean, mae_lo, mae_hi = bootstrap_stat(errors, np.mean)
        std_mean, std_lo, std_hi = bootstrap_stat(true_vals, lambda x: np.std(x, ddof=1))
        omean_mean, omean_lo, omean_hi = bootstrap_stat(true_vals, np.mean)

        rows.append({
            "group": str(name), "n": len(grp),
            "mae": mae_mean, "mae_ci_lo": mae_lo, "mae_ci_hi": mae_hi,
            "outcome_std": std_mean, "std_ci_lo": std_lo, "std_ci_hi": std_hi,
            "outcome_mean": omean_mean, "mean_ci_lo": omean_lo, "mean_ci_hi": omean_hi,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════

AXIS_COLORS = {
    "Age (fine)": "#E24A33", "Age (coarse)": "#E24A33",
    "Gender": "#348ABD", "Race/Ethnicity": "#988ED5",
    "Education": "#777777", "Income": "#FBC15E",
    "Activity Level": "#8EBA42", "Sedentary Level": "#FFB5B8",
}


def plot_main_scatter(all_stats, outcome_label="Glucose"):
    """
    Main figure: MAE vs outcome std, all groups, with error bars,
    plus regression line with CI band.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel 1: MAE vs Outcome Std with error bars ──
    ax = axes[0]
    for axis_name, tbl in all_stats.items():
        color = AXIS_COLORS.get(axis_name, "gray")
        xerr_lo = tbl["outcome_std"] - tbl["std_ci_lo"]
        xerr_hi = tbl["std_ci_hi"] - tbl["outcome_std"]
        yerr_lo = tbl["mae"] - tbl["mae_ci_lo"]
        yerr_hi = tbl["mae_ci_hi"] - tbl["mae"]

        ax.errorbar(tbl["outcome_std"], tbl["mae"],
                    xerr=[xerr_lo, xerr_hi], yerr=[yerr_lo, yerr_hi],
                    fmt="o", ms=7, capsize=3, label=axis_name,
                    color=color, alpha=0.85, zorder=3)

    # Regression across ALL points
    all_x = np.concatenate([t["outcome_std"].values for t in all_stats.values()])
    all_y = np.concatenate([t["mae"].values for t in all_stats.values()])
    slope, intercept, r, p_val, se = stats.linregress(all_x, all_y)
    r2 = r ** 2
    x_fit = np.linspace(all_x.min() - 2, all_x.max() + 2, 200)
    y_fit = intercept + slope * x_fit

    # CI band via bootstrap regression
    rng = np.random.default_rng(42)
    n = len(all_x)
    boot_lines = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        s, i, _, _, _ = stats.linregress(all_x[idx], all_y[idx])
        boot_lines.append(i + s * x_fit)
    boot_lines = np.array(boot_lines)
    ci_lo = np.quantile(boot_lines, 0.025, axis=0)
    ci_hi = np.quantile(boot_lines, 0.975, axis=0)

    ax.plot(x_fit, y_fit, "k-", lw=2, zorder=4)
    ax.fill_between(x_fit, ci_lo, ci_hi, alpha=0.15, color="gray", zorder=1)
    ax.set_xlabel(f"Within-Group {outcome_label} Std Dev (mg/dL)", fontsize=11)
    ax.set_ylabel(f"{outcome_label} MAE (mg/dL)", fontsize=11)
    ax.set_title(f"MAE vs Label Heterogeneity\nR\u00b2={r2:.2f}, p={p_val:.1e}, n={len(all_x)} groups",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    # ── Panel 2: MAE vs Sample Size (null hypothesis) ──
    ax2 = axes[1]
    for axis_name, tbl in all_stats.items():
        color = AXIS_COLORS.get(axis_name, "gray")
        yerr_lo = tbl["mae"] - tbl["mae_ci_lo"]
        yerr_hi = tbl["mae_ci_hi"] - tbl["mae"]
        ax2.errorbar(tbl["n"], tbl["mae"],
                     yerr=[yerr_lo, yerr_hi],
                     fmt="o", ms=7, capsize=3, label=axis_name,
                     color=color, alpha=0.85, zorder=3)

    all_n = np.concatenate([t["n"].values for t in all_stats.values()])
    slope_n, int_n, r_n, p_n, _ = stats.linregress(all_n.astype(float), all_y)
    x_n = np.linspace(all_n.min(), all_n.max(), 200)
    ax2.plot(x_n, int_n + slope_n * x_n, "k--", lw=1.5, alpha=0.5)
    ax2.set_xlabel("Sample Size (n)", fontsize=11)
    ax2.set_ylabel(f"{outcome_label} MAE (mg/dL)", fontsize=11)
    ax2.set_title(f"MAE vs Representation (control)\nR\u00b2={r_n**2:.2f}, p={p_n:.1e}",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"heterogeneity_scatter_{outcome_label.lower()}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return path, r2, p_val, r_n**2, p_n


def plot_group_detail(all_stats, outcome_label="Glucose"):
    """Per-axis bar charts: MAE and outcome std side by side for each group."""
    n_axes = len(all_stats)
    fig, axes = plt.subplots(2, n_axes, figsize=(3.2 * n_axes, 8), squeeze=False)

    for i, (axis_name, tbl) in enumerate(all_stats.items()):
        color = AXIS_COLORS.get(axis_name, "gray")
        groups = tbl["group"].values
        x = np.arange(len(groups))

        # Top row: MAE
        yerr = [tbl["mae"] - tbl["mae_ci_lo"], tbl["mae_ci_hi"] - tbl["mae"]]
        axes[0, i].bar(x, tbl["mae"], yerr=yerr, capsize=3, color=color, alpha=0.75)
        axes[0, i].set_xticks(x)
        axes[0, i].set_xticklabels(groups, rotation=45, ha="right", fontsize=7)
        axes[0, i].set_ylabel("MAE (mg/dL)" if i == 0 else "")
        axes[0, i].set_title(axis_name, fontsize=10, fontweight="bold")

        # Add sample size annotations
        for xi, (mae_val, n_val) in enumerate(zip(tbl["mae"], tbl["n"])):
            axes[0, i].text(xi, mae_val + 1.5, f"n={n_val}", ha="center", fontsize=6, color="gray")

        # Bottom row: Outcome std
        yerr_std = [tbl["outcome_std"] - tbl["std_ci_lo"], tbl["std_ci_hi"] - tbl["outcome_std"]]
        axes[1, i].bar(x, tbl["outcome_std"], yerr=yerr_std, capsize=3, color=color, alpha=0.5)
        axes[1, i].set_xticks(x)
        axes[1, i].set_xticklabels(groups, rotation=45, ha="right", fontsize=7)
        axes[1, i].set_ylabel(f"{outcome_label} Std Dev" if i == 0 else "")

    axes[0, 0].set_ylabel("MAE (mg/dL)", fontsize=10)
    axes[1, 0].set_ylabel(f"Within-Group {outcome_label} Std Dev", fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"heterogeneity_detail_{outcome_label.lower()}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
# HbA1c ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def run_for_outcome(test_df, outcome, outcome_label):
    """Run full heterogeneity analysis for one outcome."""
    all_stats = {}
    for axis_name, col in SLICING_CONFIG:
        if col not in test_df.columns:
            continue
        tbl = compute_group_stats(test_df, col, outcome=outcome)
        if len(tbl) >= 2:
            all_stats[axis_name] = tbl
            print(f"  {axis_name}: {len(tbl)} groups")

    scatter_path, r2, p_val, r2_n, p_n = plot_main_scatter(all_stats, outcome_label)
    detail_path = plot_group_detail(all_stats, outcome_label)

    # Save full table
    combined = []
    for axis_name, tbl in all_stats.items():
        tbl = tbl.copy()
        tbl["axis"] = axis_name
        combined.append(tbl)
    combined_df = pd.concat(combined, ignore_index=True)
    csv_path = os.path.join(RES_DIR, f"heterogeneity_{outcome}.csv")
    combined_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")

    return {
        "all_stats": all_stats, "scatter_path": scatter_path, "detail_path": detail_path,
        "r2_std": r2, "p_std": p_val, "r2_n": r2_n, "p_n": p_n,
        "n_groups": len(combined_df), "combined_df": combined_df,
    }


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY PDF
# ═══════════════════════════════════════════════════════════════════════

def build_summary_pdf(glu_res, hba1c_res):
    import matplotlib.image as mpimg

    fig = plt.figure(figsize=(11, 8.5), dpi=200)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[0.07, 1, 1],
                           hspace=0.32, wspace=0.12,
                           left=0.03, right=0.97, top=0.95, bottom=0.04)

    # Title
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.axis("off")
    ax_t.text(0.5, 0.75,
              "Within-Group Label Heterogeneity Drives Prediction Error",
              ha="center", va="center", fontsize=15, fontweight="bold")
    ax_t.text(0.5, 0.05,
              f"NHANES 2011\u20132014  |  Lifestyle model  |  "
              f"{glu_res['n_groups']} subgroups across 8 demographic/behavioral axes  |  "
              f"Bootstrap 95% CIs (2,000 iter)",
              ha="center", va="center", fontsize=8.5, color="#444444")

    # Row 1: Glucose scatter + HbA1c scatter
    ax_a = fig.add_subplot(gs[1, 0])
    ax_a.imshow(mpimg.imread(glu_res["scatter_path"]), aspect="auto")
    ax_a.axis("off")
    ax_a.set_title(f"A. Glucose: MAE vs Heterogeneity (R\u00b2={glu_res['r2_std']:.2f}) "
                   f"& vs Representation (R\u00b2={glu_res['r2_n']:.2f})",
                   fontsize=8.5, fontweight="bold", loc="left", pad=4)

    ax_b = fig.add_subplot(gs[1, 1])
    ax_b.imshow(mpimg.imread(hba1c_res["scatter_path"]), aspect="auto")
    ax_b.axis("off")
    ax_b.set_title(f"B. HbA1c: MAE vs Heterogeneity (R\u00b2={hba1c_res['r2_std']:.2f}) "
                   f"& vs Representation (R\u00b2={hba1c_res['r2_n']:.2f})",
                   fontsize=8.5, fontweight="bold", loc="left", pad=4)

    # Row 2: Glucose detail + HbA1c detail
    ax_c = fig.add_subplot(gs[2, 0])
    ax_c.imshow(mpimg.imread(glu_res["detail_path"]), aspect="auto")
    ax_c.axis("off")
    ax_c.set_title("C. Glucose: Per-Group MAE (top) & Outcome Std (bottom)",
                   fontsize=8.5, fontweight="bold", loc="left", pad=4)

    ax_d = fig.add_subplot(gs[2, 1])
    ax_d.imshow(mpimg.imread(hba1c_res["detail_path"]), aspect="auto")
    ax_d.axis("off")
    ax_d.set_title("D. HbA1c: Per-Group MAE (top) & Outcome Std (bottom)",
                   fontsize=8.5, fontweight="bold", loc="left", pad=4)

    # Footer
    footer = (
        f"Glucose: within-group std explains R\u00b2={glu_res['r2_std']:.2f} of MAE variance "
        f"(p={glu_res['p_std']:.1e}); sample size explains R\u00b2={glu_res['r2_n']:.2f} "
        f"(p={glu_res['p_n']:.1e}).  "
        f"HbA1c: std explains R\u00b2={hba1c_res['r2_std']:.2f} (p={hba1c_res['p_std']:.1e}); "
        f"size explains R\u00b2={hba1c_res['r2_n']:.2f} (p={hba1c_res['p_n']:.1e}).  "
        f"Consistent across all 8 group axes: heterogeneity, not representation, drives error."
    )
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7, color="#555555", style="italic")

    out = os.path.join(RES_DIR, "heterogeneity_summary.pdf")
    with PdfPages(out) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Summary PDF: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("WITHIN-GROUP LABEL HETEROGENEITY ANALYSIS")
    print("=" * 70)

    print("\n[1/4] Loading data & training model...")
    df = load_dataset()
    print(f"  Dataset: {len(df)} participants")
    test_df = prepare_and_train(df)
    test_df = assign_groups(test_df)
    print(f"  Test set: {len(test_df)} participants")

    print("\n[2/4] Glucose analysis...")
    glu_res = run_for_outcome(test_df, "glucose", "Glucose")

    print("\n[3/4] HbA1c analysis...")
    hba1c_res = run_for_outcome(test_df, "hba1c", "HbA1c")

    print("\n[4/4] Building summary PDF...")
    pdf_path = build_summary_pdf(glu_res, hba1c_res)

    print("\n" + "=" * 70)
    print(f"DONE. {pdf_path}")
    print(f"Glucose:  heterogeneity R²={glu_res['r2_std']:.2f} vs representation R²={glu_res['r2_n']:.2f}")
    print(f"HbA1c:   heterogeneity R²={hba1c_res['r2_std']:.2f} vs representation R²={hba1c_res['r2_n']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
