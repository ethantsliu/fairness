"""
Generate a half-page results summary PDF with key figures for the
NHANES blood glucose fairness project.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "results", "results_summary.pdf")


def load_figure(name):
    path = os.path.join(FIG_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing figure: {path}")
    return mpimg.imread(path)


def build_summary():
    fig = plt.figure(figsize=(11, 8.5), dpi=200)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[0.08, 1, 1],
        hspace=0.35,
        wspace=0.25,
        left=0.05, right=0.95, top=0.95, bottom=0.04,
    )

    # ── Title row ──
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.7,
        "Blood Glucose Prediction from Lifestyle Data: Key Results",
        ha="center", va="center", fontsize=16, fontweight="bold",
    )
    ax_title.text(
        0.5, 0.05,
        "NHANES 2011\u20132014  |  n = 4,162 adults  |  Lab-proxy vs. lifestyle-only models  |  Bootstrap 95% CIs",
        ha="center", va="center", fontsize=9, color="#444444",
    )

    # ── Figure A: Model comparison ──
    ax_a = fig.add_subplot(gs[1, 0])
    ax_a.imshow(load_figure("model_comparison.png"), aspect="auto")
    ax_a.axis("off")
    ax_a.set_title(
        "A.  Lab-Proxy vs. Lifestyle Model Performance",
        fontsize=10, fontweight="bold", loc="left", pad=6,
    )

    # ── Figure B: Lifestyle fairness ──
    ax_b = fig.add_subplot(gs[1, 1])
    ax_b.imshow(load_figure("lifestyle_fairness_evaluation.png"), aspect="auto")
    ax_b.axis("off")
    ax_b.set_title(
        "B.  Lifestyle Model \u2014 Fairness by Gender, Age & Race",
        fontsize=10, fontweight="bold", loc="left", pad=6,
    )

    # ── Figure C: Feature importance ──
    ax_c = fig.add_subplot(gs[2, 0])
    ax_c.imshow(load_figure("lifestyle_feature_importance.png"), aspect="auto")
    ax_c.axis("off")
    ax_c.set_title(
        "C.  SHAP Feature Importance (Lifestyle Model)",
        fontsize=10, fontweight="bold", loc="left", pad=6,
    )

    # ── Figure D: Age-error relationship ──
    ax_d = fig.add_subplot(gs[2, 1])
    ax_d.imshow(load_figure("age_error_relationship_lifestyle.png"), aspect="auto")
    ax_d.axis("off")
    ax_d.set_title(
        "D.  Prediction Error vs. Age (Lifestyle Model)",
        fontsize=10, fontweight="bold", loc="left", pad=6,
    )

    # ── Key numbers footer ──
    footer_text = (
        "Lab-proxy model: MAE 1.52 mg/dL, R\u00b2 = 0.868 (clinically meaningless \u2014 circular reasoning).   "
        "Lifestyle model: MAE 10.56 mg/dL, R\u00b2 \u2248 0 (honest baseline).   "
        "Binary risk classifier: 72% accuracy, no lab work required.   "
        "Fairness: gender gap < 3 mg/dL MAE; age is the dominant error driver (quadratic, p < 10\u207b\u2078)."
    )
    fig.text(
        0.5, 0.005, footer_text,
        ha="center", va="bottom", fontsize=7.5, color="#555555",
        style="italic", wrap=True,
    )

    return fig


if __name__ == "__main__":
    fig = build_summary()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with PdfPages(OUTPUT_PATH) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")
