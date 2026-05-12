"""
Generate a methods schematic figure that contrasts the two training paradigms
used in the homogeneous-vs-heterogeneous training comparison:

  - Heterogeneous (pooled) training: one model fit to all participants
    pooled across age-by-sex strata, then evaluated on the held-out portion
    that falls inside each subgroup g.

  - Homogeneous (within-group) training: a separate model fit to the
    train portion of subgroup g only, and evaluated on the held-out portion
    of the same subgroup g.

Output: figures/publication/methods_homogeneous_vs_heterogeneous.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = "/Users/aakashsuresh/fairness/blood_glucose_project/figures/publication"
OUT_PATH = os.path.join(OUT_DIR, "methods_homogeneous_vs_heterogeneous.png")

GROUP_COLORS = {
    "<40 F":   "#7FB3D5",
    "<40 M":   "#5499C7",
    "40-60 F": "#F8C471",
    "40-60 M": "#E59866",
    ">60 F":   "#C39BD3",
    ">60 M":   "#9B59B6",
}
GROUPS = list(GROUP_COLORS.keys())

TRAIN_FACE = "#FFFFFF"
TEST_FACE  = "#F2F2F2"
EDGE       = "#333333"


def rounded_box(ax, x, y, w, h, face, edge=EDGE, lw=1.2, alpha=1.0, zorder=2):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.06",
        linewidth=lw, edgecolor=edge, facecolor=face,
        alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    return box


def arrow(ax, x1, y1, x2, y2, color="#444444", lw=1.6):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        color=color, lw=lw, zorder=3,
    )
    ax.add_patch(a)


def draw_pooled_cohort(ax, x, y, w, h, label="All participants (n=5,488)"):
    rounded_box(ax, x, y, w, h, face="#FAFAFA")
    # 6 colored stripes inside to show all subgroups mixed together
    n = len(GROUPS)
    stripe_w = w / n
    for i, g in enumerate(GROUPS):
        ax.add_patch(mpatches.Rectangle(
            (x + i * stripe_w + 0.01, y + 0.06),
            stripe_w - 0.02, h - 0.12,
            facecolor=GROUP_COLORS[g], edgecolor="none", alpha=0.85, zorder=2,
        ))
    ax.text(x + w / 2, y + h + 0.04, label,
            ha="center", va="bottom", fontsize=10, fontweight="bold")


def draw_group_cohort(ax, x, y, w, h, group):
    rounded_box(ax, x, y, w, h, face=GROUP_COLORS[group], lw=1.0)
    ax.text(x + w / 2, y + h / 2, group,
            ha="center", va="center", fontsize=9, fontweight="bold", color="#1B1B1B")


def panel_heterogeneous(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("A. Heterogeneous (pooled) training",
                 fontsize=13, fontweight="bold", loc="left")

    # Pooled cohort
    draw_pooled_cohort(ax, x=0.6, y=8.0, w=8.8, h=1.1,
                       label="All participants pooled across age \u00d7 sex strata")

    # 80/20 split
    rounded_box(ax, 0.6, 6.4, 6.6, 1.0, face=TRAIN_FACE)
    ax.text(0.6 + 6.6 / 2, 6.9, "Pooled train set (80%)",
            ha="center", va="center", fontsize=10)
    rounded_box(ax, 7.4, 6.4, 2.0, 1.0, face=TEST_FACE)
    ax.text(7.4 + 1.0, 6.9, "Pooled test (20%)",
            ha="center", va="center", fontsize=10)
    arrow(ax, 5.0, 7.95, 5.0, 7.45)

    # One pooled model
    rounded_box(ax, 3.0, 4.6, 4.0, 1.1, face="#222222")
    ax.text(5.0, 5.15, "ONE pooled model\nRandomForest, multi-output",
            ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
    arrow(ax, 3.9, 6.35, 4.5, 5.75)

    # Group-specific evaluation: take pooled test rows that belong to group g
    ax.text(5.0, 4.15, "Evaluate on test rows that fall inside each subgroup g",
            ha="center", va="center", fontsize=9.5, style="italic")

    # 6 group-specific MAE bubbles
    bx = 0.6
    bw = (8.8 - 0.5 * 5) / 6  # 6 boxes with small gaps
    for i, g in enumerate(GROUPS):
        x0 = bx + i * (bw + 0.5)
        draw_group_cohort(ax, x0, 2.5, bw, 0.9, g)
        ax.text(x0 + bw / 2, 2.05, f"MAE$_{{het}}^{{g}}$",
                ha="center", va="center", fontsize=9)
        arrow(ax, 5.0, 4.7, x0 + bw / 2, 3.45)

    # Bottom caption
    rounded_box(ax, 0.6, 0.3, 8.8, 1.2, face="#F4F7FA", edge="#B0BFCC")
    ax.text(5.0, 0.9,
            "Single model trained on heterogeneous (mixed-subgroup) data.\n"
            "Per-subgroup error = how well the population model serves each group.",
            ha="center", va="center", fontsize=9.5)


def panel_homogeneous(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("B. Homogeneous (within-group) training",
                 fontsize=13, fontweight="bold", loc="left")

    # 6 separate subgroup cohorts at the top
    bw = (8.8 - 0.5 * 5) / 6
    for i, g in enumerate(GROUPS):
        x0 = 0.6 + i * (bw + 0.5)
        draw_group_cohort(ax, x0, 8.0, bw, 1.1, g)

    ax.text(5.0, 9.45, "Six stable subgroups (age-bin \u00d7 sex, n \u2265 160)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    # For each subgroup: own 80/20 split -> own model -> own MAE
    for i, g in enumerate(GROUPS):
        x0 = 0.6 + i * (bw + 0.5)
        cx = x0 + bw / 2

        # 80/20 split
        rounded_box(ax, x0, 6.4, bw * 0.7, 0.8, face=TRAIN_FACE, lw=0.9)
        rounded_box(ax, x0 + bw * 0.72, 6.4, bw * 0.28, 0.8, face=TEST_FACE, lw=0.9)
        ax.text(x0 + bw * 0.35, 6.8, "train", ha="center", va="center", fontsize=7.5)
        ax.text(x0 + bw * 0.86, 6.8, "test", ha="center", va="center", fontsize=7.5)
        arrow(ax, cx, 7.95, cx, 7.25)

        # Own model
        rounded_box(ax, x0, 4.9, bw, 0.9, face="#222222", lw=0.9)
        ax.text(cx, 5.35, "model$_g$", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        arrow(ax, cx, 6.35, cx, 5.85)

        # MAE
        ax.text(cx, 4.35, f"MAE$_{{hom}}^{{g}}$",
                ha="center", va="center", fontsize=9)
        arrow(ax, cx, 4.85, cx, 4.6)

        # Re-show subgroup color band as the test bed
        draw_group_cohort(ax, x0, 2.7, bw, 0.9, g)
        arrow(ax, cx, 4.15, cx, 3.65)

    # Bottom caption
    rounded_box(ax, 0.6, 0.3, 8.8, 1.2, face="#F4F7FA", edge="#B0BFCC")
    ax.text(5.0, 0.9,
            "One model per subgroup, trained and tested only inside that subgroup.\n"
            "Removes pooling effects; reflects within-group irreducible difficulty.",
            ha="center", va="center", fontsize=9.5)


def panel_interpretation(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("C. Diagnostic comparison", fontsize=13, fontweight="bold", loc="left")

    # Central delta box
    rounded_box(ax, 1.0, 6.5, 8.0, 2.0, face="#FFFBEA", edge="#C9A227")
    ax.text(5.0, 7.9, "For each subgroup g:", ha="center", va="center",
            fontsize=11, fontweight="bold")
    ax.text(5.0, 7.1,
            r"$\Delta_g \;=\; \mathrm{MAE}_{hom}^{\,g} \;-\; \mathrm{MAE}_{het}^{\,g}$",
            ha="center", va="center", fontsize=14)

    # Three interpretation columns
    box_w = 2.7
    gap = 0.25
    x0 = (10 - (3 * box_w + 2 * gap)) / 2

    # Negative delta
    rounded_box(ax, x0, 2.0, box_w, 4.0, face="#E8F8F0", edge="#1E8449")
    ax.text(x0 + box_w / 2, 5.6, r"$\Delta_g < 0$", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#1E8449")
    ax.text(x0 + box_w / 2, 4.6,
            "Within-group model\nbeats the pooled one",
            ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(x0 + box_w / 2, 3.0,
            "Pooling hurts this group:\nfeature\u2013glucose mapping\ndiffers from the majority.\n"
            "Specializing the model\nis a defensible mitigation.",
            ha="center", va="center", fontsize=8.8)

    # Near-zero
    x1 = x0 + box_w + gap
    rounded_box(ax, x1, 2.0, box_w, 4.0, face="#F4F4F4", edge="#7F8C8D")
    ax.text(x1 + box_w / 2, 5.6, r"$\Delta_g \approx 0$", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#34495E")
    ax.text(x1 + box_w / 2, 4.6,
            "Within and pooled models\nperform similarly",
            ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(x1 + box_w / 2, 3.0,
            "Subgroup error gap is\nNOT explained by pooling.\n"
            "More consistent with\nsubgroup outcome-structure\ndifferences (irreducible).",
            ha="center", va="center", fontsize=8.8)

    # Positive delta
    x2 = x1 + box_w + gap
    rounded_box(ax, x2, 2.0, box_w, 4.0, face="#FDEDEC", edge="#C0392B")
    ax.text(x2 + box_w / 2, 5.6, r"$\Delta_g > 0$", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#C0392B")
    ax.text(x2 + box_w / 2, 4.6,
            "Within-group model is\nworse than the pooled one",
            ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(x2 + box_w / 2, 3.0,
            "Group is too small to learn\nfrom alone; pooled training\n"
            "is borrowing strength.\nDe-pooling would hurt fairness,\n"
            "not help it.",
            ha="center", va="center", fontsize=8.8)

    # Footer
    ax.text(5.0, 1.2,
            "Comparing $\\Delta_g$ across the 6 strata separates pooling-induced "
            "disparities from subgroup-intrinsic difficulty.",
            ha="center", va="center", fontsize=9.5, style="italic")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], hspace=0.18, wspace=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    panel_heterogeneous(ax_a)
    panel_homogeneous(ax_b)
    panel_interpretation(ax_c)

    fig.suptitle(
        "Methods schematic: homogeneous vs heterogeneous training comparison\n"
        "Subgroups: age bins (<40, 40\u201360, >60) \u00d7 sex (F, M)  \u2013  "
        "Model: RandomForest multi-output regressor (glucose, HbA1c)",
        fontsize=13.5, fontweight="bold", y=0.995,
    )

    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
