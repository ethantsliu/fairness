# Robustness Report (Measure Robustness First)

## Temporal robustness (train-on-week-A, test-on-week-B)

- Glucose MAE across temporal splits: 18.550 ± 0.389 mg/dL
- Glucose MAE variance: 0.1509
- HbA1c MAE: 0.635 ± 0.008%

| Split | Glucose MAE (mg/dL) | HbA1c MAE (%) |
|-------|---------------------|---------------|
| Train Week1, Test Week2 | 19.062 | 0.634 |
| Train Week1, Test Week3 | 18.164 | 0.624 |
| Train Week2, Test Week1 | 18.596 | 0.642 |
| Train Week2, Test Week3 | 18.646 | 0.629 |
| Train Week3, Test Week1 | 17.944 | 0.646 |
| Train Week3, Test Week2 | 18.889 | 0.631 |

## Hilden-style synthetic reliability (valid days for ICC ≥ 0.80)

- Synthetic participants: 79, days per person: 365, resamples: 50
- Days needed for ICC ≥ 0.8: **7** (Hilden et al.: 7–10 valid days)

| Valid days | ICC (mean ± std) |
|------------|------------------|
| 3 | 0.702 ± 0.040 |
| 5 | 0.776 ± 0.032 |
| 7 | 0.822 ± 0.025 |
| 10 | 0.865 ± 0.020 |
| 14 | 0.901 ± 0.014 |
| 21 | 0.926 ± 0.012 |

Methodology follows Hilden et al. (2023) resampling approach; within-person variance set ~60% (Jaeschke et al., 2018).
