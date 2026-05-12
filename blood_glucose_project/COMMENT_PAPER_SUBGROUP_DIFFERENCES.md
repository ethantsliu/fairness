# Misdiagnosing Algorithmic Bias: Subgroup Distribution Differences Explain Performance Disparities in Lifestyle-Based Glucose Prediction

## Abstract
Subgroup performance disparities in healthcare machine learning are often interpreted as direct evidence of algorithmic bias. We evaluate an alternative mechanism-focused explanation in lifestyle-based glucose prediction: subgroup differences in outcome structure. Using NHANES adults with accelerometer-derived lifestyle variables and demographics (n=5,488), we trained multi-output models for fasting glucose and HbA1c and reframed analysis around subgroup contrasts rather than variability alone. In the latest full run, glucose MAE ranged from 16.789 to 18.232 mg/dL across model classes, and subgroup analyses generated 30 summary profiles and 48 pairwise subgroup contrasts. We additionally compared homogeneous (within-group) and heterogeneous (pooled) training across stable age-by-sex strata (6 groups) and reduced the subgroup reporting space to 14 stable core subgroup entries for main-text interpretability. The results support a diagnosis-first fairness narrative: observed subgroup error differences align with subgroup outcome-distribution differences and should not automatically trigger generic fairness corrections. We recommend a practical workflow of disparity quantification, subgroup distribution diagnosis, and homogeneous-vs-heterogeneous training checks before intervention.

## Main Text
Performance differences across demographic subgroups are a common finding in health AI studies. A frequent reaction is to treat these differences as algorithmic bias and immediately apply fairness correction procedures. That sequence can be premature. Error disparities can result from multiple mechanisms, including sample composition effects and subgroup-specific outcome structure. When the underlying prediction target is distributed differently across subgroups, equalizing model behavior alone may not resolve observed differences in error.

This comment argues for a mechanism-first framing using lifestyle-based glucose prediction as a concrete case. We use NHANES adults with wearable-derived activity features and demographics to predict fasting glucose and HbA1c. The objective is not to deny fairness concerns, but to improve causal diagnosis before selecting mitigation.

### From Variability to Subgroup Differences
The core narrative shift is from “variability exists” to “subgroups differ in outcome structure in ways that predict model difficulty.” In the updated analysis pipeline, subgroup characterization is comparative and distributional, including mean, standard deviation, interquartile range, skewness, kurtosis, tail mass, and pairwise distribution distances (Wasserstein and Kolmogorov-Smirnov statistics). This directly operationalizes how subgroup distributions differ from each other, not only how wide each subgroup is in isolation.

In the latest run (n=5,488), this produced:
- 30 subgroup summary rows across major axes (age bins, race/ethnicity, sex, education, income),
- 48 pairwise subgroup contrasts,
- an overlay-ready age-versus-glucose table for all participants,
- and a reduced, stable core subgroup set of 14 entries for main reporting.

### Why This Matters for Fairness Interpretation
If one subgroup’s glucose outcome has broader spread or heavier tails than another, identical models and feature sets face different irreducible difficulty. Under that condition, higher MAE in one subgroup does not automatically imply discriminatory model logic. It can reflect real subgroup differences in target structure given observed inputs.

Our updated workflow therefore separates three questions:
1. **Do disparities exist?**  
   Yes, measured via subgroup MAE differences.
2. **Do subgroup outcome distributions differ?**  
   Yes, quantified with comparative distribution diagnostics.
3. **Are disparities mostly from pooled-training composition?**  
   Tested by homogeneous vs heterogeneous training comparison.

In this run, homogeneous-vs-heterogeneous comparison was computed for 6 stable age-by-sex groups, providing a direct diagnostic against “pooling is the sole cause” assumptions.

### Methods: Homogeneous vs Heterogeneous Training Comparison
The two paradigms differ only in *which rows the model is allowed to learn from*; the model class, feature set, target set, scaler, and seed are identical (`MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))`, glucose + HbA1c outputs, lifestyle + demographic features). Subgroups are the 6 stable strata defined by age bin (<40, 40–60, >60) × sex (F, M), retained when the stratum has \(n \geq 160\). Figure M1 summarizes the design.

**Heterogeneous (pooled) training.** All participants are pooled regardless of subgroup. We make a single 80/20 split on the pooled data (random_state=42). One model is trained on the 80% pooled train set. For each subgroup g, we then take the intersection of the pooled 20% test rows with subgroup g (requiring at least 30 such rows) and compute glucose MAE on that slice — call it \(\mathrm{MAE}_{het}^{\,g}\). This mirrors the standard real-world practice of training a population model and asking how it performs in each subgroup.

**Homogeneous (within-group) training.** For each subgroup g, we restrict to that subgroup's rows only, do an 80/20 split *inside* the subgroup (random_state=42, requiring at least 20 test rows), fit the same model class on the within-group train set, and evaluate on the within-group test set — call it \(\mathrm{MAE}_{hom}^{\,g}\). This mirrors a "specialist" model trained exclusively for subgroup g and removes any influence of other subgroups on what the model learns.

**Why both, not one or the other.** Each paradigm answers a different question. \(\mathrm{MAE}_{het}^{\,g}\) tells us how a one-size-fits-all model performs on group g; this is the number a typical fairness audit reports. \(\mathrm{MAE}_{hom}^{\,g}\) tells us how well *any* model of this class can do on group g when nothing about other groups distorts the fit. The within-group MAE therefore approximates a per-subgroup difficulty ceiling: it bakes in (a) the subgroup's outcome spread and tail behavior, and (b) the strength of feature–glucose signal inside that group, but excludes pooling effects.

**Diagnostic contrast.** For each subgroup we report
\[
\Delta_g \;=\; \mathrm{MAE}_{hom}^{\,g} \;-\; \mathrm{MAE}_{het}^{\,g}
\]
along with the within-group glucose SD. The sign of \(\Delta_g\) is the mechanism signal:

- \(\Delta_g < 0\) (within-group beats pooled): pooling is genuinely hurting subgroup g — the feature-to-glucose mapping differs enough from the majority that the population model is mis-specified for this group. A specialized or stratified model is a defensible mitigation.
- \(\Delta_g \approx 0\) (within-group ≈ pooled): the per-subgroup error gap is *not* caused by pooling. Even an oracle model trained only on group g can't do meaningfully better. The disparity is more consistent with irreducible subgroup outcome-structure differences (the diagnosis-first interpretation this paper argues for).
- \(\Delta_g > 0\) (within-group worse than pooled): subgroup g is too small to learn from in isolation; the pooled model is borrowing strength from other groups. De-pooling would *increase*, not decrease, error for this group — a useful warning against reflexive "train one model per protected attribute" recipes.

In the latest run this comparison ran across all 6 stable age-by-sex strata; the corresponding empirical bars are in `figures/publication/homogeneous_vs_mixed_training.png`. The methods schematic for the design itself is `figures/publication/methods_homogeneous_vs_heterogeneous.png` (Figure M1 below).

![Figure M1. Methods schematic: homogeneous vs heterogeneous training.](figures/publication/methods_homogeneous_vs_heterogeneous.png)

### Updated Empirical Context from This Run
Model-level prediction results (10-fold CV) remained in the expected lifestyle-only range:
- Random Forest: glucose MAE 18.232 ± 1.305 mg/dL; HbA1c MAE 0.625 ± 0.055%
- Gradient Boosting: glucose MAE 17.277 ± 1.377 mg/dL; HbA1c MAE 0.600 ± 0.053%
- Ridge Regression: glucose MAE 16.789 ± 1.284 mg/dL; HbA1c MAE 0.582 ± 0.050%

These values indicate the task remains hard without glucose-adjacent proxy labs, reinforcing why subgroup difficulty differences should be expected and explicitly diagnosed.

### Figures Supporting the Revised Narrative
The updated run generated three primary narrative figures plus one methods schematic:
- `figures/publication/outcome_heterogeneity_by_subgroups.png`
- `figures/publication/overlay_age_vs_glucose.png`
- `figures/publication/homogeneous_vs_mixed_training.png`
- `figures/publication/methods_homogeneous_vs_heterogeneous.png` (methods schematic, Figure M1)

Together, these move the paper from broad claims about heterogeneity toward an evidence chain anchored in subgroup contrasts and mechanism testing.

## Practical Recommendation
For fairness analysis in healthcare ML, we recommend:
1. quantify subgroup performance gaps,
2. characterize subgroup outcome-distribution contrasts,
3. compare homogeneous vs heterogeneous training,
4. apply mitigation only when diagnostics support a bias-mediated mechanism.

This sequence avoids over-correcting models for disparities that primarily reflect subgroup outcome structure rather than model discrimination.

## Suggested Figure Captions (Short Comment Format)
- **Figure M1 (methods):** Schematic of the homogeneous vs heterogeneous training comparison. *Panel A:* heterogeneous (pooled) training — one model fit on all participants, then evaluated on the held-out test rows belonging to each age-by-sex subgroup. *Panel B:* homogeneous (within-group) training — a separate model fit and evaluated inside each subgroup, with no cross-subgroup data. *Panel C:* interpretation of \(\Delta_g = \mathrm{MAE}_{hom}^{g} - \mathrm{MAE}_{het}^{g}\): negative \(\Rightarrow\) pooling is hurting group g; near-zero \(\Rightarrow\) disparity is not pooling-driven and likely reflects intrinsic subgroup outcome structure; positive \(\Rightarrow\) group is too small to train alone and benefits from pooled borrowing of strength.
- **Figure 1:** Subgroup outcome-distribution contrasts for glucose and HbA1c across demographic and socioeconomic axes.
- **Figure 2:** Overlayed age versus glucose distribution showing subgroup trend divergence and overlap regions.
- **Figure 3:** Group-wise comparison of homogeneous and heterogeneous training MAE for stable age-by-sex strata.

## Notes
- File intentionally formatted as a concise comment-style draft for rapid submission editing.
- Target venue fit remains strongest for *Artificial Intelligence in Medicine* under a methods-forward fairness diagnosis framing.
