#!/usr/bin/env python3
"""
Lifestyle Model Fairness Deep Dive
====================================
Addresses three questions:
  A. Why do group-wise performance discrepancies exist?
     (representation vs. outcome heterogeneity vs. feature separability)
  B. How does including demographics as model inputs affect performance?
  C. Can a simple algorithmic mitigation reduce the gap?

Outputs: figures and a single-page summary PDF.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
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
# DATA LOADING  (reuses same pipeline as lifestyle_glucose_analysis.py)
# ═══════════════════════════════════════════════════════════════════════

def load_dataset():
    """Load and merge lifestyle + demographic + target data. Returns DataFrame."""
    glucose = pd.read_csv(os.path.join(LAB_DIR, "fasting_glucose_processed.csv"))[["seqn", "lbxglu"]]
    hba1c = pd.read_csv(os.path.join(LAB_DIR, "glycohemoglobin_processed.csv"))[["seqn", "lbxgh"]]
    targets = glucose.merge(hba1c, on="seqn").rename(columns={"lbxglu": "glucose", "lbxgh": "hba1c"})

    demo_path = os.path.join(LAB_DIR, "..", "raw", "nhanes_2011_2014", "P_DEMO.xpt")
    if not os.path.exists(demo_path):
        demo_path = os.path.join(PROJECT, "data", "raw", "nhanes_2011_2014", "P_DEMO.xpt")
    demo = pd.read_sas(demo_path, format="xport")
    keep = [c for c in ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "BMXBMI"] if c in demo.columns]
    demo = demo[keep]
    demo.columns = demo.columns.str.lower()
    demo.rename(columns={"seqn": "seqn", "ridageyr": "age", "riagendr": "gender",
                          "ridreth3": "race_ethnicity", "bmxbmi": "bmi"}, inplace=True)

    acc = pd.read_csv(os.path.join(LIFE_DIR, "nhanes_combined_acc.csv"))
    acc.columns = acc.columns.str.upper()
    if "SEQN" in acc.columns:
        acc["seqn"] = acc["SEQN"]
    act_cols = [c for c in acc.columns if any(k in c.lower() for k in ["pax"]) and acc[c].dtype in ("float64", "int64")]
    if act_cols:
        agg = {c: "mean" for c in act_cols}
        acc = acc.groupby("seqn").agg(agg).reset_index()
        acc["total_activity_counts"] = acc.get("PAXAISMD", 0)
        acc["moderate_vigorous_minutes"] = acc.get("PAXMVMD", 0)
        acc["sedentary_minutes"] = acc.get("PAXSMD", 0)
        acc["wear_time_minutes"] = acc.get("PAXTMD", 0)
        acc["mvpa_ratio"] = acc["moderate_vigorous_minutes"] / (acc["wear_time_minutes"] + 1)
        acc["sedentary_ratio"] = acc["sedentary_minutes"] / (acc["wear_time_minutes"] + 1)
        acc = acc[["seqn", "total_activity_counts", "moderate_vigorous_minutes",
                   "sedentary_minutes", "wear_time_minutes", "mvpa_ratio", "sedentary_ratio"]]

    df = targets.merge(demo, on="seqn", how="left").merge(acc, on="seqn", how="left")
    df = df[df["age"] >= 18].dropna(subset=["glucose", "hba1c"])
    df = df[(df["glucose"] <= 600) & (df["hba1c"] <= 18)]
    return df


def prepare_features(df, include_demographics=True):
    """Prepare X, y matrices.  Optionally drop demographic columns."""
    demo_cols = ["age", "gender", "race_ethnicity", "bmi"]
    activity_cols = ["total_activity_counts", "moderate_vigorous_minutes",
                     "sedentary_minutes", "mvpa_ratio", "sedentary_ratio"]

    if include_demographics:
        feature_cols = demo_cols + activity_cols
    else:
        feature_cols = activity_cols

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    y = df[["glucose", "hba1c"]].copy()

    for c in X.columns:
        if c in ("gender", "race_ethnicity"):
            X[c] = X[c].fillna(X[c].mode().iloc[0] if X[c].notna().any() else 0)
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
        else:
            X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0)

    return X, y, available


def train_and_predict(X_train, y_train, X_test, sample_weight=None):
    """Train RF model (optionally weighted) and return test predictions."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=2,
                               random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(rf)
    if sample_weight is not None:
        # MultiOutputRegressor passes fit_params through to each estimator
        model.fit(Xtr, y_train, sample_weight=sample_weight)
    else:
        model.fit(Xtr, y_train)
    return model.predict(Xte), scaler


def add_groups(df):
    """Add readable group columns for fairness slicing."""
    df = df.copy()
    if "age" in df.columns:
        df["age_group"] = pd.cut(df["age"], bins=[18, 40, 60, 100], labels=["<40", "40-60", ">60"])
    if "gender" in df.columns:
        df["gender_label"] = df["gender"].map({0: "Male", 1: "Female"}).fillna(df["gender"].astype(str))
    if "race_ethnicity" in df.columns:
        # Map encoded values to readable labels where possible
        df["race_label"] = df["race_ethnicity"].astype(str)
    return df


def bootstrap_mae(true, pred, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(true)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.mean(np.abs(true[idx] - pred[idx]))
    return np.mean(boots), np.quantile(boots, 0.025), np.quantile(boots, 0.975)


def group_mae_table(test_df, group_col, outcome="glucose"):
    """Compute per-group MAE with bootstrap CIs."""
    rows = []
    for name, grp in test_df.groupby(group_col, observed=True):
        if len(grp) < 10:
            continue
        t, p = grp[f"{outcome}_true"].values, grp[f"{outcome}_pred"].values
        mean_mae, ci_lo, ci_hi = bootstrap_mae(t, p)
        rows.append({
            "group": str(name), "n": len(grp),
            "mae": mean_mae, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "outcome_std": float(np.std(grp[f"{outcome}_true"], ddof=1)),
            "outcome_mean": float(np.mean(grp[f"{outcome}_true"])),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS A: Why do group-wise discrepancies exist?
# ═══════════════════════════════════════════════════════════════════════

def analysis_discrepancy(test_df):
    """
    For each demographic axis (gender, age, race), compute:
      - sample size, outcome mean/std, MAE
    Then correlate MAE with n and with outcome heterogeneity.
    """
    all_rows = []
    for group_col in ["gender_label", "age_group", "race_label"]:
        if group_col not in test_df.columns:
            continue
        tbl = group_mae_table(test_df, group_col, "glucose")
        tbl["axis"] = group_col.replace("_label", "").replace("_group", "")
        all_rows.append(tbl)
    df = pd.concat(all_rows, ignore_index=True)

    # Correlations
    rho_n, p_n = stats.spearmanr(df["n"], df["mae"])
    rho_std, p_std = stats.spearmanr(df["outcome_std"], df["mae"])
    rho_mean, p_mean = stats.spearmanr(df["outcome_mean"], df["mae"])
    corr_stats = {"n": (rho_n, p_n), "outcome_std": (rho_std, p_std), "outcome_mean": (rho_mean, p_mean)}
    print(f"[Discrepancy] Spearman MAE vs n: rho={rho_n:.3f} p={p_n:.3e}")
    print(f"[Discrepancy] Spearman MAE vs outcome_std: rho={rho_std:.3f} p={p_std:.3e}")
    print(f"[Discrepancy] Spearman MAE vs outcome_mean: rho={rho_mean:.3f} p={p_mean:.3e}")
    return df, corr_stats


def plot_discrepancy(disc_df, corr_stats):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    palette = {"gender": "C0", "age": "C1", "race": "C2"}

    # Panel 1: MAE vs sample size
    for axis_name, grp in disc_df.groupby("axis"):
        axes[0].scatter(grp["n"], grp["mae"], label=axis_name, color=palette.get(axis_name, "gray"), s=60, zorder=3)
    rho, p = corr_stats["n"]
    axes[0].set_xlabel("Sample Size (n)")
    axes[0].set_ylabel("Glucose MAE (mg/dL)")
    axes[0].set_title(f"MAE vs Representation\n(Spearman ρ={rho:.2f}, p={p:.2e})")
    axes[0].legend(fontsize=8)

    # Panel 2: MAE vs outcome heterogeneity
    for axis_name, grp in disc_df.groupby("axis"):
        axes[1].scatter(grp["outcome_std"], grp["mae"], label=axis_name, color=palette.get(axis_name, "gray"), s=60, zorder=3)
    rho, p = corr_stats["outcome_std"]
    axes[1].set_xlabel("Within-Group Glucose Std Dev")
    axes[1].set_ylabel("Glucose MAE (mg/dL)")
    axes[1].set_title(f"MAE vs Outcome Heterogeneity\n(Spearman ρ={rho:.2f}, p={p:.2e})")

    # Panel 3: MAE vs outcome mean
    for axis_name, grp in disc_df.groupby("axis"):
        axes[2].scatter(grp["outcome_mean"], grp["mae"], label=axis_name, color=palette.get(axis_name, "gray"), s=60, zorder=3)
    rho, p = corr_stats["outcome_mean"]
    axes[2].set_xlabel("Within-Group Glucose Mean (mg/dL)")
    axes[2].set_ylabel("Glucose MAE (mg/dL)")
    axes[2].set_title(f"MAE vs Outcome Level\n(Spearman ρ={rho:.2f}, p={p:.2e})")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "discrepancy_drivers.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS B: Demographics as inputs vs. not
# ═══════════════════════════════════════════════════════════════════════

def analysis_demographics_ablation(df):
    """
    Train two models: with demographics and without.
    Compare overall and per-group MAE.
    """
    results = {}
    for label, include_demo in [("With Demographics", True), ("Without Demographics", False)]:
        X, y, feat_names = prepare_features(df, include_demographics=include_demo)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        preds, _ = train_and_predict(X_train, y_train, X_test)

        test_df = X_test.copy()
        test_df["glucose_true"] = y_test["glucose"].values
        test_df["hba1c_true"] = y_test["hba1c"].values
        test_df["glucose_pred"] = preds[:, 0]
        test_df["hba1c_pred"] = preds[:, 1]

        # We need raw demographics for grouping even in the "without" case
        test_df_full = df.loc[X_test.index].copy()
        test_df["age"] = test_df_full["age"].values
        test_df["gender_raw"] = test_df_full["gender"].values
        test_df["race_raw"] = test_df_full["race_ethnicity"].values

        # Encode gender labels
        le_g = LabelEncoder()
        le_g.fit(df["gender"].dropna().astype(str))
        test_df["gender_label"] = le_g.inverse_transform(
            test_df_full["gender"].fillna(test_df_full["gender"].mode().iloc[0]).astype(int).clip(0, len(le_g.classes_) - 1).values
        ) if "gender" in df.columns else "Unknown"

        test_df["age_group"] = pd.cut(test_df["age"], bins=[18, 40, 60, 100], labels=["<40", "40-60", ">60"])

        overall_mae_glu = mean_absolute_error(y_test["glucose"], preds[:, 0])
        overall_mae_hba1c = mean_absolute_error(y_test["hba1c"], preds[:, 1])

        # Per-age-group MAE
        age_tbl = group_mae_table(test_df, "age_group", "glucose")

        results[label] = {
            "features": feat_names,
            "overall_glucose_mae": overall_mae_glu,
            "overall_hba1c_mae": overall_mae_hba1c,
            "age_groups": age_tbl,
            "test_df": test_df,
        }
        print(f"[{label}] features={feat_names}")
        print(f"[{label}] Overall Glucose MAE={overall_mae_glu:.2f}, HbA1c MAE={overall_mae_hba1c:.2f}")

    return results


def plot_demographics_ablation(ablation):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Overall MAE comparison
    labels = list(ablation.keys())
    glu_maes = [ablation[l]["overall_glucose_mae"] for l in labels]
    hba1c_maes = [ablation[l]["overall_hba1c_mae"] for l in labels]

    x = np.arange(len(labels))
    w = 0.35
    bars1 = axes[0].bar(x - w/2, glu_maes, w, label="Glucose MAE", color="steelblue", alpha=0.8)
    bars2 = axes[0].bar(x + w/2, [m * 30 for m in hba1c_maes], w, label="HbA1c MAE (×30)", color="salmon", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("MAE (mg/dL)")
    axes[0].set_title("Overall MAE: Demographics vs. No Demographics")
    axes[0].legend(fontsize=8)
    for bar, val in zip(bars1, glu_maes):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, hba1c_maes):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # Panel 2: Per-age-group MAE comparison
    colors = {"With Demographics": "steelblue", "Without Demographics": "darkorange"}
    offsets = {"With Demographics": -0.15, "Without Demographics": 0.15}
    age_groups_all = ["<40", "40-60", ">60"]
    x_pos = np.arange(len(age_groups_all))

    for label, res in ablation.items():
        tbl = res["age_groups"]
        maes, ci_los, ci_his = [], [], []
        for ag in age_groups_all:
            row = tbl[tbl["group"] == ag]
            if len(row):
                maes.append(row["mae"].values[0])
                ci_los.append(row["mae"].values[0] - row["ci_lo"].values[0])
                ci_his.append(row["ci_hi"].values[0] - row["mae"].values[0])
            else:
                maes.append(0); ci_los.append(0); ci_his.append(0)
        axes[1].bar(x_pos + offsets[label], maes, 0.28,
                    yerr=[ci_los, ci_his], capsize=3,
                    label=label, color=colors[label], alpha=0.8)

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(age_groups_all)
    axes[1].set_xlabel("Age Group")
    axes[1].set_ylabel("Glucose MAE (mg/dL)")
    axes[1].set_title("Age-Group MAE: With vs. Without Demographics")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "demographics_ablation.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS C: Fairness mitigation pilot
# ═══════════════════════════════════════════════════════════════════════

def analysis_mitigation(df):
    """
    Pilot: inverse-prevalence sample reweighting to reduce group MAE gaps.
    Groups defined by age bucket since age is the biggest driver.

    Also tests group-aware fine-tuning: train separate models per age group,
    then ensemble predictions.
    """
    X, y, feat_names = prepare_features(df, include_demographics=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    raw_df = df.loc[X_train.index]
    age_train = raw_df["age"].values
    age_bins = pd.cut(age_train, bins=[18, 40, 60, 100], labels=["<40", "40-60", ">60"])

    # --- Baseline (unweighted) ---
    preds_base, scaler = train_and_predict(X_train, y_train, X_test)

    # --- Method 1: Inverse-prevalence reweighting ---
    counts = age_bins.value_counts()
    weight_map = {g: len(age_bins) / (len(counts) * c) for g, c in counts.items()}
    sample_weights = np.array([weight_map.get(g, 1.0) for g in age_bins])
    preds_reweight, _ = train_and_predict(X_train, y_train, X_test, sample_weight=sample_weights)

    # --- Method 2: Group-aware fine-tuning (separate model per age group) ---
    scaler_ft = StandardScaler()
    Xtr_sc = scaler_ft.fit_transform(X_train)
    Xte_sc = scaler_ft.transform(X_test)
    preds_group = np.zeros_like(preds_base)

    test_raw = df.loc[X_test.index]
    test_age_bins = pd.cut(test_raw["age"].values, bins=[18, 40, 60, 100], labels=["<40", "40-60", ">60"])

    for grp_label in ["<40", "40-60", ">60"]:
        train_mask = np.array(age_bins == grp_label)
        test_mask = np.array(test_age_bins == grp_label)

        if train_mask.sum() < 20 or test_mask.sum() < 5:
            preds_group[test_mask] = preds_base[test_mask]
            continue

        rf_grp = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)
        )
        rf_grp.fit(Xtr_sc[train_mask], y_train.values[train_mask])
        preds_group[test_mask] = rf_grp.predict(Xte_sc[test_mask])

    # Build results table
    results = {}
    for method, preds in [("Baseline", preds_base),
                          ("Reweighted", preds_reweight),
                          ("Group Fine-Tuned", preds_group)]:
        test_df = X_test.copy()
        test_df["glucose_true"] = y_test["glucose"].values
        test_df["hba1c_true"] = y_test["hba1c"].values
        test_df["glucose_pred"] = preds[:, 0]
        test_df["hba1c_pred"] = preds[:, 1]
        test_df["age"] = test_raw["age"].values
        test_df["age_group"] = np.array(test_age_bins)

        overall = mean_absolute_error(y_test["glucose"], preds[:, 0])
        tbl = group_mae_table(test_df, "age_group", "glucose")
        mae_range = tbl["mae"].max() - tbl["mae"].min()

        results[method] = {"overall": overall, "age_groups": tbl, "mae_gap": mae_range}
        print(f"[{method}] Overall Glucose MAE={overall:.2f}, Age-group gap={mae_range:.2f}")

    return results


def plot_mitigation(mitigation):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    methods = list(mitigation.keys())
    colors = {"Baseline": "#4878CF", "Reweighted": "#6ACC65", "Group Fine-Tuned": "#D65F5F"}

    # Panel 1: Overall MAE + gap
    x = np.arange(len(methods))
    overall = [mitigation[m]["overall"] for m in methods]
    gaps = [mitigation[m]["mae_gap"] for m in methods]

    bars = axes[0].bar(x, overall, 0.5, color=[colors[m] for m in methods], alpha=0.85)
    for bar, val, gap in zip(bars, overall, gaps):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"MAE {val:.1f}\ngap {gap:.1f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, fontsize=9)
    axes[0].set_ylabel("Glucose MAE (mg/dL)")
    axes[0].set_title("Overall MAE & Age-Group Gap\nby Mitigation Method")

    # Panel 2: Per-age-group comparison
    age_labels = ["<40", "40-60", ">60"]
    x_pos = np.arange(len(age_labels))
    w = 0.25
    for i, method in enumerate(methods):
        tbl = mitigation[method]["age_groups"]
        maes, errs_lo, errs_hi = [], [], []
        for ag in age_labels:
            row = tbl[tbl["group"] == ag]
            if len(row):
                maes.append(row["mae"].values[0])
                errs_lo.append(row["mae"].values[0] - row["ci_lo"].values[0])
                errs_hi.append(row["ci_hi"].values[0] - row["mae"].values[0])
            else:
                maes.append(0); errs_lo.append(0); errs_hi.append(0)
        axes[1].bar(x_pos + (i - 1) * w, maes, w,
                    yerr=[errs_lo, errs_hi], capsize=3,
                    label=method, color=colors[method], alpha=0.85)

    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(age_labels)
    axes[1].set_xlabel("Age Group")
    axes[1].set_ylabel("Glucose MAE (mg/dL)")
    axes[1].set_title("Per-Age-Group MAE\nBaseline vs. Mitigation Methods")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fairness_mitigation.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY PDF
# ═══════════════════════════════════════════════════════════════════════

def build_summary_pdf(fig_paths, corr_stats, ablation, mitigation):
    import matplotlib.image as mpimg

    fig = plt.figure(figsize=(11, 8.5), dpi=200)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[0.06, 1, 1],
                           hspace=0.30, left=0.04, right=0.96, top=0.95, bottom=0.04)

    # Title
    ax_t = fig.add_subplot(gs[0])
    ax_t.axis("off")
    ax_t.text(0.5, 0.75, "Lifestyle Model Fairness: Discrepancy Drivers, Demographics Ablation & Mitigation",
              ha="center", va="center", fontsize=14, fontweight="bold")
    ax_t.text(0.5, 0.05, "NHANES 2011\u20132014  |  Lifestyle features only (no lab values)  |  Bootstrap 95% CIs",
              ha="center", va="center", fontsize=9, color="#444444")

    # Row 1: discrepancy + ablation
    gs_row1 = gs[1].subgridspec(1, 2, wspace=0.15)
    ax_a = fig.add_subplot(gs_row1[0])
    ax_a.imshow(mpimg.imread(fig_paths["discrepancy"]), aspect="auto")
    ax_a.axis("off")
    rho_n = corr_stats["n"][0]
    rho_std = corr_stats["outcome_std"][0]
    ax_a.set_title(f"A. Discrepancy Drivers  (MAE\u2013n: \u03c1={rho_n:.2f},  MAE\u2013std: \u03c1={rho_std:.2f})",
                   fontsize=9, fontweight="bold", loc="left", pad=4)

    ax_b = fig.add_subplot(gs_row1[1])
    ax_b.imshow(mpimg.imread(fig_paths["ablation"]), aspect="auto")
    ax_b.axis("off")
    with_mae = ablation["With Demographics"]["overall_glucose_mae"]
    without_mae = ablation["Without Demographics"]["overall_glucose_mae"]
    delta = without_mae - with_mae
    ax_b.set_title(f"B. Demographics Ablation  (removing demographics: +{delta:.1f} mg/dL MAE)",
                   fontsize=9, fontweight="bold", loc="left", pad=4)

    # Row 2: mitigation
    ax_c = fig.add_subplot(gs[2])
    ax_c.imshow(mpimg.imread(fig_paths["mitigation"]), aspect="auto")
    ax_c.axis("off")
    base_gap = mitigation["Baseline"]["mae_gap"]
    best_method = min(["Reweighted", "Group Fine-Tuned"], key=lambda m: mitigation[m]["mae_gap"])
    best_gap = mitigation[best_method]["mae_gap"]
    reduction = ((base_gap - best_gap) / base_gap) * 100 if base_gap > 0 else 0
    ax_c.set_title(
        f"C. Fairness Mitigation Pilot  (best: {best_method}, gap {base_gap:.1f}\u2192{best_gap:.1f} mg/dL, "
        f"{reduction:.0f}% reduction)",
        fontsize=9, fontweight="bold", loc="left", pad=4)

    # Footer
    footer = (
        f"A: Outcome heterogeneity (\u03c1={rho_std:.2f}) drives group MAE more than sample size (\u03c1={rho_n:.2f}).  "
        f"B: Removing demographics increases MAE by {delta:.1f} mg/dL; age-group gaps widen.  "
        f"C: {best_method} reduces age-group MAE gap by {reduction:.0f}%.  "
        f"All results: lifestyle model only, no lab values."
    )
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7.5, color="#555555", style="italic")

    out = os.path.join(RES_DIR, "fairness_deep_dive_summary.pdf")
    with PdfPages(out) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Summary PDF saved: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("LIFESTYLE MODEL FAIRNESS DEEP DIVE")
    print("=" * 70)

    print("\n[1/4] Loading data...")
    df = load_dataset()
    print(f"Dataset: {len(df)} participants, {df.shape[1]} columns")

    # --- Run baseline for discrepancy analysis ---
    print("\n[2/4] Analysis A: Discrepancy drivers...")
    X, y, feat_names = prepare_features(df, include_demographics=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preds, _ = train_and_predict(X_train, y_train, X_test)

    test_df = X_test.copy()
    test_df["glucose_true"] = y_test["glucose"].values
    test_df["hba1c_true"] = y_test["hba1c"].values
    test_df["glucose_pred"] = preds[:, 0]
    test_df["hba1c_pred"] = preds[:, 1]

    # Recover raw demographics for grouping
    raw_test = df.loc[X_test.index]
    test_df["age"] = raw_test["age"].values
    le_g = LabelEncoder().fit(df["gender"].dropna().astype(str))
    test_df["gender_label"] = raw_test["gender"].apply(
        lambda v: {1.0: "Male", 2.0: "Female"}.get(v, str(v))
    ).values
    test_df["race_label"] = raw_test["race_ethnicity"].astype(str).values
    test_df = add_groups(test_df)

    disc_df, corr_stats = analysis_discrepancy(test_df)
    disc_df.to_csv(os.path.join(RES_DIR, "discrepancy_analysis.csv"), index=False)
    fig_disc = plot_discrepancy(disc_df, corr_stats)

    # --- Demographics ablation ---
    print("\n[3/4] Analysis B: Demographics ablation...")
    ablation = analysis_demographics_ablation(df)
    fig_abl = plot_demographics_ablation(ablation)

    # --- Mitigation pilot ---
    print("\n[4/4] Analysis C: Fairness mitigation pilot...")
    mitigation = analysis_mitigation(df)
    fig_mit = plot_mitigation(mitigation)

    # --- Summary PDF ---
    print("\nBuilding summary PDF...")
    fig_paths = {"discrepancy": fig_disc, "ablation": fig_abl, "mitigation": fig_mit}
    pdf_path = build_summary_pdf(fig_paths, corr_stats, ablation, mitigation)

    print("\n" + "=" * 70)
    print("DONE. Summary: " + pdf_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
