# Email Draft — Homogeneous vs Heterogeneous Analysis

**To:** [Recipient name]  
**Subject:** Clarifying the homogeneous vs heterogeneous training analysis (+ methods figure)

---

## Attachments (send both)

1. **Methods schematic (required):**  
   `/Users/aakashsuresh/fairness/blood_glucose_project/figures/publication/methods_homogeneous_vs_heterogeneous.png`

2. **Empirical results (optional but recommended):**  
   `/Users/aakashsuresh/fairness/blood_glucose_project/figures/publication/homogeneous_vs_mixed_training.png`

---

## Body (copy/paste)

Hi [Name],

Thanks for the question on the homogeneous vs heterogeneous comparison. The two paradigms use the exact same model (RandomForest multi-output regressor for glucose and HbA1c), same features, same seed — the only thing that changes is which rows the model is allowed to learn from. In **heterogeneous (pooled) training**, all participants are pooled, one model is fit on an 80% split, and we report MAE on the subset of the 20% held-out that falls inside each age-by-sex subgroup g (MAE_het(g)) — the standard "one population model, audited per group" number. In **homogeneous (within-group) training**, we restrict to subgroup g's rows only, do an 80/20 split inside the subgroup, fit a model on that within-group train set, and evaluate within-group (MAE_hom(g)) — a specialist model with no cross-group influence, which approximates a per-subgroup difficulty ceiling. I've attached a methods schematic (Figure M1) that draws this out side by side across the six stable age-by-sex strata.

The reason we run both is that their contrast, Δ(g) = MAE_hom(g) − MAE_het(g), is the mechanism signal we care about. In our latest run across six strata, heterogeneous MAE spanned 4.7 mg/dL. One stratum showed Δ ≈ 0 (not pooling-driven), three showed Δ < 0 (within-group beat pooled — pooling hurts that group), and two showed Δ > 0 (pooled beat within-group — those groups benefit from borrowing strength). So the comparison isn't asking which training recipe is globally better; it's a diagnostic that separates pooling-induced disparities from intrinsic subgroup-distribution effects. That mixed pattern is why we argue against reflexive fairness correction before mechanism testing. Happy to discuss further.

Best,  
[Your name]

---

## Quick reference — empirical Δ(g) results

| Subgroup | Δ (hom−het) | Interpretation |
|----------|-------------|----------------|
| <40 Male | +2.7 | Pooling helps |
| <40 Female | −2.1 | Pooling hurts |
| 40–60 Male | +3.1 | Pooling helps |
| 40–60 Female | −1.2 | Pooling hurts |
| >60 Male | −0.1 | Not pooling-driven |
| >60 Female | −5.5 | Pooling hurts |

Full table: `results/homogeneous_vs_heterogeneous_comparison.csv`
