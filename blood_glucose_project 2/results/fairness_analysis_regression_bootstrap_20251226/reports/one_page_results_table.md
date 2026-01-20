# One-Page Results Table (Bootstrapped Fairness & Performance)

All metrics estimated with 1,000 bootstraps (80% sample) unless noted. Fairness thresholds: excellent ≤0.05, acceptable ≤0.10, poor >0.10. MAE disparity: <2 mg/dL excellent, <5 mg/dL acceptable, ≥5 mg/dL poor.

| Factor (wearable metadata) | Key fairness metrics | Performance gap | Regression signal | Notes |
| --- | --- | --- | --- | --- |
| Wear time (avg hours/day) | SP 0.08; EOppo 0.12; EO 0.15; Cal 0.06 | 17% accuracy gap; 21% sensitivity gap (excellent >20h vs low <10h) | Wear time vs correctness: significant; see report R²/p | Low-wear users systematically underperform; focus for mitigation |
| Data quality ratio | SP 0.05; EOppo 0.18; EO 0.22; Cal 0.14 | PPV −24% (poor <70% vs excellent >95%); FPR +16% | Quality ratio vs correctness: significant; see report R²/p | Critical issue; largest fairness violations here |
| Composite user profile (ideal vs problematic) | SP 0.19; EOppo 0.21; EO 0.25; Cal 0.16 | Accuracy −15%; Precision −17%; Recall −17% | Profile indicator vs correctness: significant; see report R²/p | Worst overall disparities; combines wear, quality, consistency |
| Activity level | SP 0.11; EOppo 0.09; EO 0.13; Cal 0.07 | Sedentary: 24% predicted vs 22% actual; High active: 13% predicted vs 14% actual | Activity vs correctness: significant; see report R²/p | Over-prediction for sedentary, under-prediction for active |

Abbreviations: SP = Statistical Parity difference; EOppo = Equal Opportunity difference; EO = Equalized Odds difference; Cal = Calibration difference.

