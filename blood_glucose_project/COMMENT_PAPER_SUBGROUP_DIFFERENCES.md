# Misdiagnosing Algorithmic Bias: Subgroup Distribution Differences Explain Performance Disparities in Lifestyle-Based Glucose Prediction

**Target venue:** *Artificial Intelligence in Medicine* (short comment / methods-forward framing)  
**Suggested length:** ~2,000 words main text + 4 figures  
**Terminology:** fairness *diagnosis* (not denial of fairness concerns)

---

## Abstract

Subgroup performance disparities in healthcare machine learning are often interpreted as direct evidence of algorithmic bias. We evaluate an alternative mechanism-focused explanation in lifestyle-based glucose prediction: subgroup differences in outcome structure and training composition. Using NHANES adults with accelerometer-derived lifestyle variables and demographics (n=5,488), we trained multi-output models for fasting glucose and HbA1c and reframed analysis around subgroup contrasts rather than variability alone. Glucose MAE ranged from 16.8 to 18.2 mg/dL across model classes (10-fold CV). Subgroup analyses generated 30 summary profiles and 48 pairwise distribution contrasts. A homogeneous-vs-heterogeneous training diagnostic across six stable age-by-sex strata showed heterogeneous MAE spanned 4.7 mg/dL; in one stratum Δ ≈ 0 (not pooling-driven), in three strata within-group models beat pooled models (pooling hurts), and in two strata pooled models beat within-group models (borrowing strength). Observed subgroup error differences therefore reflect a mix of intrinsic outcome-structure effects and pooling effects — and should not automatically trigger generic fairness corrections. We recommend quantifying disparities, diagnosing subgroup outcome distributions, and running homogeneous-vs-heterogeneous checks before intervention.

---

## Main Text

Performance differences across demographic subgroups are common in health AI. A frequent reaction is to treat these differences as algorithmic bias and immediately apply fairness correction procedures. That sequence can be premature. Error disparities can result from multiple mechanisms, including sample composition effects and subgroup-specific outcome structure. When the prediction target is distributed differently across subgroups, equalizing model behavior alone may not resolve observed error differences.

This comment argues for a mechanism-first framing using lifestyle-based glucose prediction as a concrete case. We use NHANES adults with wearable-derived activity features and demographics to predict fasting glucose and HbA1c. The objective is not to deny fairness concerns, but to improve causal diagnosis before selecting mitigation.

### From Variability to Subgroup Differences

The core narrative shift is from “variability exists” to “subgroups differ in outcome structure in ways that predict model difficulty.” Subgroup characterization is comparative and distributional: mean, standard deviation, interquartile range, skewness, kurtosis, tail mass, and pairwise distribution distances (Wasserstein and Kolmogorov–Smirnov statistics). This operationalizes how subgroup distributions differ from each other, not only how wide each subgroup is in isolation.

In the latest run (n=5,488), this produced 30 subgroup summary rows (age, race/ethnicity, sex, education, income), 48 pairwise contrasts, an age-versus-glucose overlay table, and a reduced core set of 14 stable subgroups for main-text reporting.

### Why This Matters for Fairness Interpretation

If one subgroup’s glucose outcome has broader spread or heavier tails than another, identical models and feature sets face different irreducible difficulty. Under that condition, higher MAE in one subgroup does not automatically imply discriminatory model logic; it can reflect real subgroup differences in target structure given observed inputs.

Our workflow separates three questions:
1. **Do disparities exist?** Measured via subgroup MAE differences.
2. **Do subgroup outcome distributions differ?** Quantified with comparative distribution diagnostics.
3. **Are disparities explained by pooled training?** Tested by homogeneous vs heterogeneous training comparison.

### Homogeneous vs Heterogeneous Training (Methods Summary)

The two paradigms differ only in *which rows the model learns from*; model class, features, targets, scaler, and seed are identical (RandomForest multi-output regressor, glucose + HbA1c). Subgroups are six stable age-bin (<40, 40–60, >60) × sex strata (n ≥ 160).

- **Heterogeneous (pooled):** One 80/20 split on all participants; one model on the pooled train set; MAE on the held-out test rows inside each subgroup g → MAE_het(g).
- **Homogeneous (within-group):** For each g, an 80/20 split inside the subgroup only; a separate model trained and tested within g → MAE_hom(g).

The diagnostic contrast is Δ(g) = MAE_hom(g) − MAE_het(g):
- Δ < −1 mg/dL → pooling hurts g (specialization may help).
- |Δ| ≤ 1 mg/dL → not pooling-driven (intrinsic subgroup difficulty more likely).
- Δ > +1 mg/dL → pooling helps g (de-pooling would hurt).

See Figure M1 (`figures/publication/methods_homogeneous_vs_heterogeneous.png`) for the full schematic.

### Empirical Results

**Overall model performance (10-fold CV, lifestyle-only features):**
- Ridge: glucose MAE 16.789 ± 1.284 mg/dL; HbA1c MAE 0.582 ± 0.050%
- Gradient Boosting: glucose MAE 17.277 ± 1.377 mg/dL; HbA1c MAE 0.600 ± 0.053%
- Random Forest: glucose MAE 18.232 ± 1.305 mg/dL; HbA1c MAE 0.625 ± 0.055%

**Homogeneous vs heterogeneous diagnostic (6 age-by-sex strata):**

| Subgroup | n | MAE_het | MAE_hom | Δ (hom−het) | Interpretation |
|----------|---|---------|---------|-------------|----------------|
| <40, Male | 1,032 | 16.3 | 19.0 | +2.7 | Pooling helps |
| <40, Female | 1,006 | 19.2 | 17.1 | −2.1 | Pooling hurts |
| 40–60, Male | 1,270 | 17.5 | 20.6 | +3.1 | Pooling helps |
| 40–60, Female | 1,309 | 21.0 | 19.9 | −1.2 | Pooling hurts |
| >60, Male | 459 | 20.1 | 20.0 | −0.1 | Not pooling-driven |
| >60, Female | 412 | 19.3 | 13.8 | −5.5 | Pooling hurts |

Across strata, heterogeneous MAE ranged **16.3–21.0 mg/dL** (span **4.7 mg/dL**). Mechanism classification: **1/6** not pooling-driven, **3/6** pooling hurts, **2/6** pooling helps. This mixed pattern is the key finding: a single fairness mitigation (e.g., always de-pooling or always pooling) would help some groups and harm others. Subgroup glucose SD was strongly associated with |Δ| (r = −0.88, p = 0.02), suggesting that groups with tighter outcome spread benefit more from pooled training, while groups where within-group models win may have subgroup-specific feature–outcome structure.

Subgroup outcome-distribution contrasts (Figure 1) and the age–glucose overlay (Figure 2) show that demographic axes differ in spread, tail mass, and pairwise distance — consistent with why irreducible difficulty varies by subgroup. Figure 3 plots homogeneous vs heterogeneous MAE side by side.

### Take-Home

Subgroup error gaps in lifestyle-based glucose prediction are not automatically evidence of algorithmic bias. Before applying fairness corrections, analysts should (1) quantify the gap, (2) compare subgroup outcome distributions, and (3) run a homogeneous-vs-heterogeneous training check. In our data, disparities reflected a *mix* of mechanisms: one stratum showed near-identical pooled and within-group error (outcome-structure dominated), three strata were hurt by pooling (specialization defensible), and two strata benefited from pooling (de-pooling would worsen performance). Generic mitigation without this diagnosis risks fixing the wrong problem — or making some subgroups worse.

### Limitations

- Lifestyle and demographic features only; no glucose-adjacent labs, so absolute MAE remains high and subgroup gaps are expected.
- Cross-sectional NHANES design; no temporal or external validation of the hom/het diagnostic.
- Homogeneous-vs-heterogeneous comparison used a single 80/20 split (not full CV) and only age-by-sex strata with n ≥ 160.
- Sex encoded as NHANES RIAGENDR (1 = Male, 2 = Female); small strata (e.g., >60) have wider uncertainty.
- Pearson correlation of glucose SD with heterogeneous MAE was not significant (r = 0.24, p = 0.65) with only six strata; the |Δ| association should be interpreted cautiously.

### Practical Recommendation

1. Quantify subgroup performance gaps.  
2. Characterize subgroup outcome-distribution contrasts.  
3. Compare homogeneous vs heterogeneous training.  
4. Apply mitigation only when diagnostics support a bias- or pooling-mediated mechanism — and tailor mitigation to the sign of Δ(g).

---

## Figures

| File | Role |
|------|------|
| `figures/publication/outcome_heterogeneity_by_subgroups.png` | Figure 1 — outcome distribution contrasts |
| `figures/publication/overlay_age_vs_glucose.png` | Figure 2 — age vs glucose overlay |
| `figures/publication/homogeneous_vs_mixed_training.png` | Figure 3 — empirical hom vs het MAE |
| `figures/publication/methods_homogeneous_vs_heterogeneous.png` | Figure M1 — methods schematic |

### Suggested Captions

- **Figure M1:** Schematic of homogeneous vs heterogeneous training. Panel A: pooled training and per-subgroup evaluation. Panel B: within-group train/test per stratum. Panel C: interpretation of Δ(g).
- **Figure 1:** Subgroup outcome-distribution contrasts for glucose across demographic axes.
- **Figure 2:** Age versus glucose distribution with subgroup trend divergence.
- **Figure 3:** Group-wise homogeneous and heterogeneous training MAE for six age-by-sex strata.

---

## Supplementary Data

- `results/homogeneous_vs_heterogeneous_comparison.csv` — per-stratum MAE and Δ values  
- `results/homogeneous_vs_heterogeneous_summary.md` — formatted results table and summary stats
