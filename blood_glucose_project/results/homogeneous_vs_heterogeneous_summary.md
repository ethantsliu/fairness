# Homogeneous vs Heterogeneous Training Results

Groups analyzed: **6** stable age-by-sex strata (n >= 160 per stratum).

## Per-group results

| group_id   |   n_group |   glucose_mae_heterogeneous |   glucose_mae_homogeneous |   mae_delta_hom_minus_het |   group_glucose_sd | interpretation     |
|:-----------|----------:|----------------------------:|--------------------------:|--------------------------:|-------------------:|:-------------------|
| 40-60__1.0 |      1270 |                      17.531 |                    20.644 |                     3.112 |             32.344 | pooling_helps      |
| 40-60__2.0 |      1309 |                      21.040 |                    19.851 |                    -1.189 |             32.115 | pooling_hurts      |
| <40__1.0   |      1032 |                      16.318 |                    19.045 |                     2.727 |             29.821 | pooling_helps      |
| <40__2.0   |      1006 |                      19.213 |                    17.126 |                    -2.086 |             33.557 | pooling_hurts      |
| >60__1.0   |       459 |                      20.114 |                    20.038 |                    -0.076 |             34.162 | not_pooling_driven |
| >60__2.0   |       412 |                      19.296 |                    13.817 |                    -5.479 |             26.001 | pooling_hurts      |

## Summary counts (|Δ| <= 1.0 mg/dL treated as near-zero)

- Not pooling-driven (|Δ| <= 1.0): **1/6**
- Pooling hurts (Δ < -1.0): **3/6**
- Pooling helps / borrow strength (Δ > 1.0): **2/6**

- Heterogeneous MAE range across groups: **4.722 mg/dL**
- Pearson r(group glucose SD, heterogeneous MAE): **0.240** (p=0.6476)
- Pearson r(group glucose SD, |Δ|): **-0.878** (p=0.0213)
