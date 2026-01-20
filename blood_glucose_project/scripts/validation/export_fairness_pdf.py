#!/usr/bin/env python3
"""
Export fairness figures to a single PDF report.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _first_existing_path(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def export_fairness_pdf(output_path):
    figures = [
        (
            _first_existing_path([
                "/Users/aakashsuresh/fairness/blood_glucose_project/figures/fairness_evaluation.png",
                "/Users/aakashsuresh/fairness/blood_glucose_project/results/figures/fairness_evaluation.png"
            ]),
            "Lab-Proxy Model Fairness (Bootstrap Error Bars)"
        ),
        (
            _first_existing_path([
                "/Users/aakashsuresh/fairness/blood_glucose_project/figures/lifestyle_fairness_evaluation.png",
                "/Users/aakashsuresh/fairness/blood_glucose_project/results/figures/lifestyle_fairness_evaluation.png"
            ]),
            "Lifestyle Model Fairness (Bootstrap Error Bars)"
        ),
        (
            _first_existing_path([
                "/Users/aakashsuresh/fairness/blood_glucose_project/figures/fairness_comparison_lab.png",
                "/Users/aakashsuresh/fairness/blood_glucose_project/figures/fairness_comparison_lifestyle.png",
                "/Users/aakashsuresh/fairness/blood_glucose_project/results/figures/fairness_comparison.png"
            ]),
            "Fairness Comparison (Glucose MAE)"
        )
    ]

    with PdfPages(output_path) as pdf:
        for fig_path, title in figures:
            if not fig_path or not os.path.exists(fig_path):
                print(f"⚠️ Missing figure, skipping: {fig_path}")
                continue

            img = plt.imread(fig_path)
            plt.figure(figsize=(11, 8.5))
            plt.imshow(img)
            plt.axis('off')
            plt.title(title, fontsize=14)
            pdf.savefig()
            plt.close()

    print(f"✅ PDF report saved to {output_path}")


def main():
    output_pdf = "/Users/aakashsuresh/fairness/blood_glucose_project/results/fairness_results_report.pdf"
    export_fairness_pdf(output_pdf)


if __name__ == "__main__":
    main()
