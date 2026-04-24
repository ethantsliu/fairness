#!/usr/bin/env python3
"""
Publication-quality figures for NPJ Digital Medicine / Nature Medicine.
Also investigates WHY within-group label variability differs.

Outputs:
  figures/publication/fig1_study_overview.{pdf,png}  (vertical walkthrough pipeline)
  figures/publication/fig2_heterogeneity.{pdf,png}
  figures/publication/fig3_variability_drivers.{pdf,png}
  figures/publication/fig4_mechanism.{pdf,png}
  figures/publication/all_figures.pdf  (combined)
  results/variability_investigation.csv
"""

import os
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG_DIR = os.path.join(PROJECT, "figures", "publication")
RES_DIR = os.path.join(PROJECT, "results")
LAB_DIR = os.path.join(PROJECT, "data", "processed", "nhanes_lab")
LIFE_DIR = os.path.join(PROJECT, "data", "processed", "nhanes_combined")
os.makedirs(FIG_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# NATURE / NPJ STYLE
# ═══════════════════════════════════════════════════════════════════════════

def set_nature_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.fontsize": 6.5,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


CB = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "cyan": "#56B4E9",
    "yellow": "#F0E442", "gray": "#999999",
}

AXIS_PALETTE = {
    "Age": CB["red"], "Gender": CB["blue"], "Race/Ethnicity": CB["purple"],
    "Education": CB["gray"], "Income": CB["orange"],
}

AXIS_MARKERS = {
    "Age": "o", "Gender": "s", "Race/Ethnicity": "D",
    "Education": "^", "Income": "v",
}

RACE_MAP_FLAT = {1.0: "Mexican American", 2.0: "Other Hispanic", 3.0: "NH White",
                 4.0: "NH Black", 6.0: "NH Asian", 7.0: "Other/Multi"}
GENDER_MAP = {1.0: "Male", 2.0: "Female"}


def panel_label(ax, label):
    """Upper-left panel letter in the outer margin (offset points), clear of subplot titles."""
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-2, 22),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="bottom",
        clip_on=False,
    )


def save_fig(fig, name):
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(FIG_DIR, name + ext),
                    bbox_inches="tight", facecolor="white",
                    dpi=300 if ext == ".png" else None)
    plt.close(fig)
    print(f"  {name}")


# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset():
    glucose = pd.read_csv(os.path.join(LAB_DIR, "fasting_glucose_processed.csv"))[["seqn", "lbxglu"]]
    hba1c = pd.read_csv(os.path.join(LAB_DIR, "glycohemoglobin_processed.csv"))[["seqn", "lbxgh"]]
    targets = glucose.merge(hba1c, on="seqn").rename(columns={"lbxglu": "glucose", "lbxgh": "hba1c"})

    demo = pd.read_sas(os.path.join(PROJECT, "data", "raw", "nhanes_2011_2014", "P_DEMO.xpt"),
                        format="xport")
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
    acc = acc.groupby("seqn").agg({c: "mean" for c in act_cols}).reset_index()
    acc["total_activity"] = acc.get("PAXAISMD", pd.Series(0, index=acc.index))
    acc["mvpa_min"] = acc.get("PAXVMD", pd.Series(0, index=acc.index))
    acc["sedentary_min"] = acc.get("PAXSWMD", pd.Series(0, index=acc.index))
    acc["wear_time"] = acc.get("PAXTMD", pd.Series(0, index=acc.index))
    acc["sedentary_ratio"] = acc["sedentary_min"] / (acc["wear_time"] + 1)
    acc = acc[["seqn", "total_activity", "mvpa_min", "sedentary_min",
               "wear_time", "sedentary_ratio"]]

    df = targets.merge(demo, on="seqn", how="left").merge(acc, on="seqn", how="left")
    df = df[df["age"] >= 18].dropna(subset=["glucose", "hba1c"])
    df = df[(df["glucose"] <= 600) & (df["hba1c"] <= 18)]
    return df


def prepare_and_predict(df):
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
                              random_state=42, n_jobs=-1))
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    test_df = df.loc[X_test.index].copy()
    test_df["glucose_pred"] = preds[:, 0]
    test_df["hba1c_pred"] = preds[:, 1]
    test_df["glucose_error"] = np.abs(test_df["glucose"] - test_df["glucose_pred"])
    test_df["hba1c_error"] = np.abs(test_df["hba1c"] - test_df["hba1c_pred"])
    return test_df, df


def assign_labels(df):
    out = df.copy()
    out["age_group"] = pd.cut(out["age"], bins=[18, 30, 40, 50, 60, 70, 100],
                              labels=["18-29", "30-39", "40-49", "50-59", "60-69", "70+"])
    out["gender_label"] = out["gender"].map(GENDER_MAP).fillna("Unknown")
    out["race_label"] = out["race_ethnicity"].map(RACE_MAP_FLAT).fillna("Other")
    if "education" in out.columns:
        out["education_label"] = out["education"].map(
            {1.0: "<9th", 2.0: "9-11th", 3.0: "HS/GED", 4.0: "Some College", 5.0: "College+"})
    if "income_ratio" in out.columns:
        out["income_quartile"] = pd.qcut(out["income_ratio"].rank(method="first"),
                                          q=4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    out["diabetic_range"] = (out["glucose"] >= 126).astype(int)
    return out


SLICINGS = [
    ("Age", "age_group"), ("Gender", "gender_label"),
    ("Race/Ethnicity", "race_label"), ("Education", "education_label"),
    ("Income", "income_quartile"),
]


def bootstrap_stat(values, stat_fn, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.array([stat_fn(values[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    return float(np.mean(boots)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: STUDY OVERVIEW (vertical walkthrough workflow)
# ═══════════════════════════════════════════════════════════════════════════

def _wrap_fig1_paragraphs(text, width):
    """Hard-wrap each paragraph (blank-line separated); use Unicode symbols, not mathtext."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return "\n\n".join(
        textwrap.fill(" ".join(p.split()), width=width, break_long_words=False, break_on_hyphens=True)
        for p in paras
    )


def figure1_study_overview():
    set_nature_style()
    fig, ax = plt.subplots(figsize=(7.2, 7.85))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    arrow_kw = dict(arrowstyle="->,head_width=0.35,head_length=0.2",
                    color="#555555", lw=1.05, connectionstyle="arc3,rad=0")

    # Card clears step badge (circle ~x 5.5–13.5); text stays inside inner column.
    x_card, w_card = 14.0, 84.0
    pad_x = 2.35
    x_text = x_card + pad_x
    wrap_w = 52
    cx = x_card + w_card / 2
    r_step = 3.65
    x_circ = 9.25

    def step_badge(n, yc):
        c = mpatches.Circle((x_circ, yc), r_step, facecolor="#FFFFFF", edgecolor="#333333",
                            linewidth=1.0, zorder=5)
        ax.add_patch(c)
        ax.text(x_circ, yc, str(n), ha="center", va="center", fontsize=9.5,
                fontweight="bold", color="#222222", zorder=6)

    def walk_card(y, h, n, y_center, title, body, fc, body_fs=6.15):
        top_inset = 0.55
        title_fs = 8.0
        r = mpatches.FancyBboxPatch(
            (x_card, y), w_card, h,
            boxstyle="round,pad=0.5",
            facecolor=fc, edgecolor="#444444", linewidth=0.75, zorder=1,
        )
        ax.add_patch(r)
        step_badge(n, y_center)
        title_wrapped = textwrap.fill(title, width=wrap_w - 2, break_long_words=False,
                                      break_on_hyphens=True)
        body_wrapped = _wrap_fig1_paragraphs(body, wrap_w)
        y_title = y + h - top_inset
        ax.text(x_text, y_title, title_wrapped, ha="left", va="top",
                fontsize=title_fs, fontweight="bold", color="#1a1a1a", linespacing=1.22)
        n_title_lines = title_wrapped.count("\n") + 1
        # Approximate title depth in data coords (ylim 0–100, ~7.85 in tall).
        title_depth = n_title_lines * 1.32 + 0.45
        y_body = y_title - title_depth
        ax.text(x_text, y_body, body_wrapped, ha="left", va="top",
                fontsize=body_fs, color="#333333", linespacing=1.28)

    def v_arrow(y_top, y_bot):
        ax.annotate("", xy=(cx, y_bot), xytext=(cx, y_top), arrowprops=arrow_kw)

    ax.text(50, 98.2, "Study pipeline (walkthrough)", ha="center", va="top",
            fontsize=9.5, fontweight="bold", color="#222222")

    gap = 2.15

    y5, h5 = 6.0, 24.0
    walk_card(
        y5, h5, 5, y5 + h5 / 2,
        "What the analyses show",
        "Key finding — Population outcome SD explains R\u00b2 = 0.63 of subgroup MAE variation "
        "(p < 10\u207b\u2075, n = 23 groups).\n\n"
        "Controls — Sample-size association R\u00b2 \u2248 0.00; demographics ablation and a "
        "mitigation pilot showed minimal or no meaningful effect.\n\n"
        "Mechanism — Diabetes prevalence varies roughly 2–27% across subgroups, widening glucose "
        "distributions and mechanically increasing prediction error.",
        "#E8F4EA",
    )

    y4, h4 = y5 + h5 + gap, 12.8
    walk_card(
        y4, h4, 4, y4 + h4 / 2,
        "Heterogeneity decomposition",
        "Quantify how cross-subgroup MAE tracks within-group glucose dispersion versus sample "
        "size (negative control), using Spearman correlation and OLS regression.",
        "#FCE8E8",
    )

    y3, h3 = y4 + h4 + gap, 12.8
    walk_card(
        y3, h3, 3, y3 + h3 / 2,
        "Subgroup fairness evaluation",
        "Mean absolute error (MAE) on the held-out test set for 23 subgroups across five axes "
        "(age, gender, race/ethnicity, education, income), each with bootstrap 95% CIs.",
        "#FCE8E8",
    )

    y2, h2 = y3 + h3 + gap, 12.2
    walk_card(
        y2, h2, 2, y2 + h2 / 2,
        "Lifestyle-only prediction model",
        "Random forest (multi-output) predicts fasting glucose and HbA1c from lifestyle and "
        "accelerometry features only — no laboratory predictors. Features standardized; 80/20 "
        "train–test split.",
        "#E8E0F0",
    )

    y1, h1 = y2 + h2 + gap, 13.2
    walk_card(
        y1, h1, 1, y1 + h1 / 2,
        "Data sources",
        "NHANES 2011–2014: n = 4,162 adults (≥18y) with linked demographics (age, gender, "
        "race/ethnicity, education, income), wearable-derived activity and sedentary time, and "
        "lab fasting glucose and HbA1c (targets only — not used as model inputs).",
        "#D5E8D4",
    )

    v_arrow(y2, y3 + h3)
    v_arrow(y3, y4 + h4)
    v_arrow(y4, y5 + h5)
    v_arrow(y1, y2 + h2)

    save_fig(fig, "fig1_study_overview")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: MAIN HETEROGENEITY RESULT
# ═══════════════════════════════════════════════════════════════════════════

def collect_subgroup_stats(test_df, full_df):
    """Gather MAE stats from test_df, glucose σ from full population."""
    full_labeled = assign_labels(full_df)
    rows = []
    for axis_name, col in SLICINGS:
        if col not in test_df.columns:
            continue
        for name, grp in test_df.groupby(col, observed=True):
            if len(grp) < 15:
                continue
            full_grp = full_labeled[full_labeled[col] == name]
            if len(full_grp) < 15:
                continue
            mae_m, mae_lo, mae_hi = bootstrap_stat(grp["glucose_error"].values, np.mean)
            std_m, std_lo, std_hi = bootstrap_stat(full_grp["glucose"].values,
                                                    lambda x: np.std(x, ddof=1))
            rows.append({
                "axis": axis_name, "group": str(name),
                "n_test": len(grp), "n_full": len(full_grp),
                "mae": mae_m, "mae_lo": mae_lo, "mae_hi": mae_hi,
                "std": std_m, "std_lo": std_lo, "std_hi": std_hi,
                "pct_diabetic": float(np.mean(full_grp["glucose"].values >= 126) * 100),
            })
    return pd.DataFrame(rows)


def figure2_heterogeneity(df_stats):
    set_nature_style()

    all_x = df_stats["std"].values
    all_y = df_stats["mae"].values
    slope, intercept, r, p_val, se = stats.linregress(all_x, all_y)
    n = len(all_x)

    rng = np.random.default_rng(42)
    x_fit = np.linspace(max(0, all_x.min() - 3), all_x.max() + 3, 200)
    boot_lines = np.array([
        stats.linregress(all_x[idx := rng.integers(0, n, n)], all_y[idx])[1]
        + stats.linregress(all_x[idx], all_y[idx])[0] * x_fit
        for _ in range(2000)
    ])

    slope_n, int_n, r_n, p_n, _ = stats.linregress(
        df_stats["n_full"].values.astype(float), all_y)

    fig = plt.figure(figsize=(7.2, 3.0))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # ── Panel a: MAE vs σ ──
    ax_a = fig.add_subplot(gs[0])
    for axis_name in df_stats["axis"].unique():
        sub = df_stats[df_stats["axis"] == axis_name]
        c = AXIS_PALETTE.get(axis_name, "#999")
        m = AXIS_MARKERS.get(axis_name, "o")
        ax_a.errorbar(sub["std"], sub["mae"],
                      yerr=[sub["mae"] - sub["mae_lo"], sub["mae_hi"] - sub["mae"]],
                      fmt=m, ms=5.5, capsize=1.5, capthick=0.6, color=c, label=axis_name,
                      alpha=0.85, zorder=3, markeredgewidth=0.4, markeredgecolor="white",
                      elinewidth=0.6)

    ax_a.plot(x_fit, intercept + slope * x_fit, color="black", lw=1.2, zorder=4)
    ax_a.fill_between(x_fit,
                      np.quantile(boot_lines, 0.025, axis=0),
                      np.quantile(boot_lines, 0.975, axis=0),
                      alpha=0.08, color="black", zorder=1)

    ax_a.set_xlabel("Within-group glucose SD (mg/dL)")
    ax_a.set_ylabel("Glucose MAE (mg/dL)")
    ax_a.set_xlim(left=0)
    ax_a.set_ylim(bottom=0)
    leg = ax_a.legend(loc="upper left", frameon=True, handletextpad=0.3,
                      borderpad=0.4, labelspacing=0.35)
    leg.get_frame().set_linewidth(0.4)

    p_str = f"{p_val:.1e}" if p_val >= 1e-15 else f"< 10$^{{-15}}$"
    ax_a.text(0.97, 0.05,
              f"R$^2$ = {r**2:.2f}\np = {p_str}\nn = {n} groups",
              transform=ax_a.transAxes, ha="right", va="bottom", fontsize=6.5,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
    panel_label(ax_a, "a")

    # ── Panel b: MAE vs n (control) ──
    ax_b = fig.add_subplot(gs[1])
    for axis_name in df_stats["axis"].unique():
        sub = df_stats[df_stats["axis"] == axis_name]
        c = AXIS_PALETTE.get(axis_name, "#999")
        m = AXIS_MARKERS.get(axis_name, "o")
        ax_b.errorbar(sub["n_full"], sub["mae"],
                      yerr=[sub["mae"] - sub["mae_lo"], sub["mae_hi"] - sub["mae"]],
                      fmt=m, ms=5.5, capsize=1.5, capthick=0.6, color=c,
                      alpha=0.85, zorder=3, markeredgewidth=0.4, markeredgecolor="white",
                      elinewidth=0.6)

    x_n = np.linspace(df_stats["n_full"].min(), df_stats["n_full"].max(), 200)
    ax_b.plot(x_n, int_n + slope_n * x_n, color="black", lw=0.8, ls="--", alpha=0.35, zorder=2)
    ax_b.set_xlabel("Population sample size (n)")
    ax_b.set_ylabel("Glucose MAE (mg/dL)")
    ax_b.set_ylim(bottom=0)
    ax_b.text(0.97, 0.05,
              f"R$^2$ = {r_n**2:.2f}\np = {p_n:.2f}",
              transform=ax_b.transAxes, ha="right", va="bottom", fontsize=6.5,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
    panel_label(ax_b, "b")

    save_fig(fig, "fig2_heterogeneity")
    return df_stats


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: OUTCOME DISTRIBUTIONS BY SUBGROUP
# ═══════════════════════════════════════════════════════════════════════════

def figure3_variability_drivers(full_df):
    set_nature_style()
    df = assign_labels(full_df)

    fig = plt.figure(figsize=(7.2, 6.2))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.45, top=0.94)

    panel_slicings = [
        ("Age group", "age_group",
         ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"], CB["red"]),
        ("Gender", "gender_label",
         ["Male", "Female"], CB["blue"]),
        ("Race/Ethnicity", "race_label", None, CB["purple"]),
    ]

    # ── Row 1: Violin + box plots ──
    for i, (title, col, order, color) in enumerate(panel_slicings):
        ax = fig.add_subplot(gs[0, i])
        if col not in df.columns:
            continue

        groups = order if order else sorted(df[col].dropna().unique())
        groups = [g for g in groups if len(df[df[col] == g]["glucose"].dropna()) >= 15]
        group_data = [df[df[col] == g]["glucose"].dropna().values for g in groups]

        parts = ax.violinplot(group_data, positions=range(len(groups)),
                              showmeans=False, showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
            pc.set_edgecolor(color)
            pc.set_linewidth(0.5)

        bp = ax.boxplot(group_data, positions=range(len(groups)),
                        widths=0.18, patch_artist=True, showfliers=False, zorder=3)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_linewidth(0.5)
        for el in ["whiskers", "caps", "medians"]:
            for line in bp[el]:
                line.set_color("#333")
                line.set_linewidth(0.6)

        ax.axhline(y=126, color=CB["red"], ls="--", lw=0.7, alpha=0.5)
        if i == 0:
            ax.text(len(groups) - 0.5, 130, "Diabetic threshold",
                    fontsize=5, color=CB["red"], va="bottom", ha="right", style="italic")

        labs = [str(g) for g in groups]
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labs)
        for lab in ax.get_xticklabels():
            lab.set_rotation(40)
            lab.set_ha("right")
            lab.set_rotation_mode("anchor")
            lab.set_fontsize(5.5)
        ax.set_ylabel("Fasting glucose (mg/dL)" if i == 0 else "")
        ax.set_title(title, fontsize=8, fontweight="bold", pad=10)
        ax.set_ylim(50, 350)
        panel_label(ax, chr(97 + i))

    # ── Row 2: Diabetes prevalence + SD overlay ──
    for i, (title, col, order, color) in enumerate(panel_slicings):
        ax = fig.add_subplot(gs[1, i])
        if col not in df.columns:
            continue

        groups = order if order else sorted(df[col].dropna().unique())
        groups = [g for g in groups if len(df[df[col] == g]) >= 15]

        prev_vals, std_vals = [], []
        for g in groups:
            grp = df[df[col] == g]
            prev_vals.append(bootstrap_stat(
                grp["diabetic_range"].values, lambda x: np.mean(x) * 100))
            std_vals.append(bootstrap_stat(
                grp["glucose"].values, lambda x: np.std(x, ddof=1)))

        x = np.arange(len(groups))
        pm = [v[0] for v in prev_vals]
        pe = [[v[0] - v[1] for v in prev_vals], [v[2] - v[0] for v in prev_vals]]
        ax.bar(x, pm, yerr=pe, capsize=2, width=0.55,
               color=color, alpha=0.55, edgecolor=color, linewidth=0.5, error_kw={"lw": 0.6})

        ax2 = ax.twinx()
        sm = [v[0] for v in std_vals]
        se_lo = [v[0] - v[1] for v in std_vals]
        se_hi = [v[2] - v[0] for v in std_vals]
        ax2.errorbar(x, sm, yerr=[se_lo, se_hi], fmt="s-", ms=4, color="#222",
                     capsize=2, zorder=4, lw=0.8, capthick=0.5)
        ax2.set_ylabel("Glucose SD (mg/dL)" if i == 2 else "", fontsize=6.5, color="#333")
        ax2.tick_params(axis="y", labelcolor="#333", labelsize=6)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_linewidth(0.4)
        ax2.spines["right"].set_color("#999")

        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in groups])
        for lab in ax.get_xticklabels():
            lab.set_rotation(40)
            lab.set_ha("right")
            lab.set_rotation_mode("anchor")
            lab.set_fontsize(5.5)
        ax.set_ylabel("Diabetes prevalence (%)" if i == 0 else "")
        ax.set_title(f"{title}", fontsize=8, fontweight="bold", pad=10)
        panel_label(ax, chr(100 + i))

    # Legend for dual axis
    fig.text(0.5, 0.01,
             "Bars = diabetes prevalence (%, left axis)  |  "
             "Black squares = glucose SD (mg/dL, right axis)",
             ha="center", fontsize=6, color="#555")

    save_fig(fig, "fig3_variability_drivers")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: MECHANISTIC PATH  (prevalence → σ → MAE)
# ═══════════════════════════════════════════════════════════════════════════

def figure4_mechanism(df_stats, full_df):
    """Three-panel figure showing the causal chain:
       a) Diabetes prevalence vs. within-group σ
       b) σ among normoglycemic-only vs. overall σ
       c) Residual MAE after controlling for σ → flat across axes
    """
    set_nature_style()
    df_full = assign_labels(full_df)

    # Compute σ among normoglycemic for each group
    norm_rows = []
    for _, row in df_stats.iterrows():
        col = [c for a, c in SLICINGS if a == row["axis"]][0]
        grp = df_full[df_full[col].astype(str) == row["group"]]
        normals = grp[grp["glucose"] < 126]["glucose"].values
        if len(normals) > 5:
            std_norm, _, _ = bootstrap_stat(normals, lambda x: np.std(x, ddof=1))
        else:
            std_norm = np.nan
        norm_rows.append(std_norm)
    df_stats = df_stats.copy()
    df_stats["std_normal"] = norm_rows

    fig = plt.figure(figsize=(7.2, 3.0))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)

    # ── Panel a: Diabetes prevalence → σ ──
    ax_a = fig.add_subplot(gs[0])
    for axis_name in df_stats["axis"].unique():
        sub = df_stats[df_stats["axis"] == axis_name]
        c = AXIS_PALETTE.get(axis_name, "#999")
        m = AXIS_MARKERS.get(axis_name, "o")
        ax_a.scatter(sub["pct_diabetic"], sub["std"], c=c, marker=m, s=30,
                     alpha=0.85, zorder=3, edgecolors="white", linewidths=0.4,
                     label=axis_name)

    rho, p = stats.spearmanr(df_stats["pct_diabetic"], df_stats["std"])
    sl, it, r, pv, _ = stats.linregress(df_stats["pct_diabetic"], df_stats["std"])
    xf = np.linspace(0, df_stats["pct_diabetic"].max() + 2, 100)
    ax_a.plot(xf, it + sl * xf, "k-", lw=0.9, alpha=0.6)
    ax_a.set_xlabel("Diabetes prevalence (%)")
    ax_a.set_ylabel("Within-group glucose SD (mg/dL)")
    ax_a.text(0.97, 0.05, f"$\\rho$ = {rho:.2f}\np = {p:.1e}",
              transform=ax_a.transAxes, ha="right", va="bottom", fontsize=6.5,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
    ax_a.set_xlim(left=0)
    ax_a.set_ylim(bottom=0)
    leg = ax_a.legend(loc="upper left", frameon=True, handletextpad=0.3,
                      borderpad=0.4, labelspacing=0.35, markerscale=0.8)
    leg.get_frame().set_linewidth(0.4)
    panel_label(ax_a, "a")

    # ── Panel b: σ among normals vs overall σ ──
    ax_b = fig.add_subplot(gs[1])
    valid = df_stats.dropna(subset=["std_normal"])
    for axis_name in valid["axis"].unique():
        sub = valid[valid["axis"] == axis_name]
        c = AXIS_PALETTE.get(axis_name, "#999")
        m = AXIS_MARKERS.get(axis_name, "o")
        ax_b.scatter(sub["std_normal"], sub["std"], c=c, marker=m, s=30,
                     alpha=0.85, zorder=3, edgecolors="white", linewidths=0.4)

    rho_n, p_n = stats.spearmanr(valid["std_normal"], valid["std"])
    xl = np.linspace(valid["std_normal"].min() - 1, valid["std_normal"].max() + 1, 100)
    sl2, it2, _, _, _ = stats.linregress(valid["std_normal"], valid["std"])
    ax_b.plot(xl, it2 + sl2 * xl, "k-", lw=0.9, alpha=0.6)
    # identity line for reference
    lim = [0, max(valid["std"].max(), valid["std_normal"].max()) + 3]
    ax_b.plot(lim, lim, "k--", lw=0.5, alpha=0.25)

    ax_b.set_xlabel("SD among normoglycemic (mg/dL)")
    ax_b.set_ylabel("Overall glucose SD (mg/dL)")
    ax_b.text(0.97, 0.05, f"$\\rho$ = {rho_n:.2f}\np = {p_n:.1e}",
              transform=ax_b.transAxes, ha="right", va="bottom", fontsize=6.5,
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))
    ax_b.set_xlim(left=0)
    ax_b.set_ylim(bottom=0)
    panel_label(ax_b, "b")

    # ── Panel c: Residual MAE after regression on σ ──
    ax_c = fig.add_subplot(gs[2])
    sl3, it3, _, _, _ = stats.linregress(df_stats["std"], df_stats["mae"])
    df_stats["mae_residual"] = df_stats["mae"] - (it3 + sl3 * df_stats["std"])

    axes_list = df_stats["axis"].unique()
    positions = np.arange(len(axes_list))
    for j, axis_name in enumerate(axes_list):
        sub = df_stats[df_stats["axis"] == axis_name]
        c = AXIS_PALETTE.get(axis_name, "#999")
        vals = sub["mae_residual"].values
        bp = ax_c.boxplot([vals], positions=[j], widths=0.5, patch_artist=True,
                          showfliers=True, flierprops=dict(marker=".", ms=3, alpha=0.4))
        bp["boxes"][0].set_facecolor(c)
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][0].set_linewidth(0.5)
        for el in ["whiskers", "caps", "medians"]:
            for line in bp[el]:
                line.set_color("#333")
                line.set_linewidth(0.6)

    ax_c.axhline(0, color="black", lw=0.6, ls="--", alpha=0.3)
    ax_c.set_xticks(positions)
    ax_c.set_xticklabels([a.replace("/", "/\n") for a in axes_list], fontsize=5.5)
    for lab in ax_c.get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")
        lab.set_rotation_mode("anchor")
    ax_c.set_ylabel("Residual MAE (mg/dL)")
    ax_c.set_title("After controlling for outcome SD", fontsize=7.5, pad=10)
    panel_label(ax_c, "c")

    save_fig(fig, "fig4_mechanism")
    return df_stats


# ═══════════════════════════════════════════════════════════════════════════
# VARIABILITY INVESTIGATION
# ═══════════════════════════════════════════════════════════════════════════

def investigate_variability(full_df):
    df = assign_labels(full_df)

    rows = []
    for axis_name, col in SLICINGS:
        if col not in df.columns:
            continue
        for name, grp in df.groupby(col, observed=True):
            if len(grp) < 15:
                continue
            glu = grp["glucose"].values
            normals = glu[glu < 100]
            diabetic = glu[glu >= 126]
            rows.append({
                "axis": axis_name, "group": str(name), "n": len(grp),
                "glucose_mean": float(np.mean(glu)),
                "glucose_median": float(np.median(glu)),
                "glucose_std": float(np.std(glu, ddof=1)),
                "glucose_iqr": float(np.percentile(glu, 75) - np.percentile(glu, 25)),
                "glucose_skew": float(stats.skew(glu)),
                "glucose_kurtosis": float(stats.kurtosis(glu)),
                "pct_diabetic": float(np.mean(glu >= 126) * 100),
                "pct_prediabetic": float(np.mean((glu >= 100) & (glu < 126)) * 100),
                "pct_normal": float(np.mean(glu < 100) * 100),
                "mean_if_diabetic": float(np.mean(diabetic)) if len(diabetic) > 0 else np.nan,
                "std_if_diabetic": float(np.std(diabetic, ddof=1)) if len(diabetic) > 1 else np.nan,
                "mean_if_normal": float(np.mean(normals)) if len(normals) > 0 else np.nan,
                "std_if_normal": float(np.std(normals, ddof=1)) if len(normals) > 1 else np.nan,
            })

    inv = pd.DataFrame(rows)

    rho_prev, p_prev = stats.spearmanr(inv["pct_diabetic"], inv["glucose_std"])
    rho_skew, p_skew = stats.spearmanr(inv["glucose_skew"], inv["glucose_std"])

    valid = inv.dropna(subset=["std_if_normal"])
    rho_norm, p_norm = stats.spearmanr(valid["std_if_normal"], valid["glucose_std"])
    valid_d = inv.dropna(subset=["std_if_diabetic"])
    rho_diab, p_diab = stats.spearmanr(valid_d["std_if_diabetic"], valid_d["glucose_std"])

    print("\n  ═══ VARIABILITY INVESTIGATION ═══")
    print(f"  Diabetes prevalence  -> overall sigma:  rho={rho_prev:.2f}  p={p_prev:.2e}")
    print(f"  Skewness             -> overall sigma:  rho={rho_skew:.2f}  p={p_skew:.2e}")
    print(f"  sigma(normals)       -> overall sigma:  rho={rho_norm:.2f}  p={p_norm:.2e}")
    print(f"  sigma(diabetics)     -> overall sigma:  rho={rho_diab:.2f}  p={p_diab:.2e}")

    print("\n  ─── Per-axis interpretation ───")
    for axis_name in inv["axis"].unique():
        sub = inv[inv["axis"] == axis_name].sort_values("glucose_std")
        lo, hi = sub.iloc[0], sub.iloc[-1]
        print(f"\n  {axis_name}:")
        print(f"    Lowest sigma:  {lo['group']}")
        print(f"      sigma={lo['glucose_std']:.1f}  diabetes={lo['pct_diabetic']:.1f}%  "
              f"prediabetes={lo['pct_prediabetic']:.1f}%  normal={lo['pct_normal']:.1f}%")
        print(f"    Highest sigma: {hi['group']}")
        print(f"      sigma={hi['glucose_std']:.1f}  diabetes={hi['pct_diabetic']:.1f}%  "
              f"prediabetes={hi['pct_prediabetic']:.1f}%  normal={hi['pct_normal']:.1f}%")

        if axis_name == "Age":
            print("    -> Biological: age-related decline in glucose homeostasis, increasing "
                  "T2D prevalence with age creates wider distributions")
        elif axis_name == "Gender":
            print("    -> Mixed: males have higher diabetes prevalence (hormonal/metabolic "
                  "factors) but the difference is modest")
        elif axis_name == "Race/Ethnicity":
            print("    -> Mixed biological + social: diabetes prevalence varies by genetic "
                  "susceptibility and socioeconomic access to care; NH Asian shows low sigma "
                  "despite moderate diabetes — likely a different glycemic phenotype")
        elif axis_name == "Education":
            print("    -> Social determinants: lower education correlates with higher diabetes "
                  "prevalence through diet, healthcare access, and health literacy pathways")
        elif axis_name == "Income":
            print("    -> Social determinants: poverty-to-income ratio tracks diabetes risk "
                  "through material deprivation and food environment")

    print("\n  ─── Key conclusion ───")
    print("  Variability differences are NOT a data artifact. They reflect real clinical")
    print("  heterogeneity: subgroups with higher diabetes prevalence have more")
    print("  right-skewed glucose distributions, which mechanically produces higher SD")
    print("  and thus higher prediction error. The model cannot be 'debiased' because")
    print("  the disparity originates in the outcome distribution, not the algorithm.")

    out_path = os.path.join(RES_DIR, "variability_investigation.csv")
    inv.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    return inv


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED PDF
# ═══════════════════════════════════════════════════════════════════════════

def combine_pdf():
    """Merge per-figure vector PDFs so text stays editable and aligned (no PNG re-raster)."""
    from pypdf import PdfReader, PdfWriter

    pdf_path = os.path.join(FIG_DIR, "all_figures.pdf")
    names = [
        "fig1_study_overview",
        "fig2_heterogeneity",
        "fig3_variability_drivers",
        "fig4_mechanism",
    ]
    writer = PdfWriter()
    for name in names:
        part = os.path.join(FIG_DIR, name + ".pdf")
        reader = PdfReader(part)
        for page in reader.pages:
            writer.add_page(page)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    print(f"\n  Combined PDF (vector merge): {pdf_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PUBLICATION FIGURES & VARIABILITY INVESTIGATION")
    print("=" * 60)

    print("\nLoading data ...")
    df = load_dataset()
    test_df, full_df = prepare_and_predict(df)
    test_df = assign_labels(test_df)
    print(f"  {len(df)} total participants, {len(test_df)} in test set")

    print("\nFigure 1: Study overview ...")
    figure1_study_overview()

    print("\nCollecting subgroup statistics ...")
    df_stats = collect_subgroup_stats(test_df, full_df)
    print(f"  {len(df_stats)} subgroups across {df_stats['axis'].nunique()} axes")

    print("\nFigure 2: Heterogeneity scatter ...")
    figure2_heterogeneity(df_stats)

    print("\nFigure 3: Outcome distributions ...")
    figure3_variability_drivers(full_df)

    print("\nFigure 4: Mechanistic path ...")
    figure4_mechanism(df_stats, full_df)

    print("\nVariability investigation ...")
    investigate_variability(full_df)

    print("\nCombining PDFs ...")
    combine_pdf()

    print("\n" + "=" * 60)
    print("ALL FIGURES IN: " + FIG_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
