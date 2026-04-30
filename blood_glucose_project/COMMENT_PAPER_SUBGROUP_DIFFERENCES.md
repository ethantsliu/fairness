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

### Updated Empirical Context from This Run
Model-level prediction results (10-fold CV) remained in the expected lifestyle-only range:
- Random Forest: glucose MAE 18.232 ± 1.305 mg/dL; HbA1c MAE 0.625 ± 0.055%
- Gradient Boosting: glucose MAE 17.277 ± 1.377 mg/dL; HbA1c MAE 0.600 ± 0.053%
- Ridge Regression: glucose MAE 16.789 ± 1.284 mg/dL; HbA1c MAE 0.582 ± 0.050%

These values indicate the task remains hard without glucose-adjacent proxy labs, reinforcing why subgroup difficulty differences should be expected and explicitly diagnosed.

### Figures Supporting the Revised Narrative
The updated run generated three primary narrative figures:
- `figures/publication/outcome_heterogeneity_by_subgroups.png`
- `figures/publication/overlay_age_vs_glucose.png`
- `figures/publication/homogeneous_vs_mixed_training.png`

Together, these move the paper from broad claims about heterogeneity toward an evidence chain anchored in subgroup contrasts and mechanism testing.

## Practical Recommendation
For fairness analysis in healthcare ML, we recommend:
1. quantify subgroup performance gaps,
2. characterize subgroup outcome-distribution contrasts,
3. compare homogeneous vs heterogeneous training,
4. apply mitigation only when diagnostics support a bias-mediated mechanism.

This sequence avoids over-correcting models for disparities that primarily reflect subgroup outcome structure rather than model discrimination.

## Suggested Figure Captions (Short Comment Format)
- **Figure 1:** Subgroup outcome-distribution contrasts for glucose and HbA1c across demographic and socioeconomic axes.
- **Figure 2:** Overlayed age versus glucose distribution showing subgroup trend divergence and overlap regions.
- **Figure 3:** Group-wise comparison of homogeneous and heterogeneous training MAE for stable age-by-sex strata.

## Notes
- File intentionally formatted as a concise comment-style draft for rapid submission editing.
- Target venue fit remains strongest for *Artificial Intelligence in Medicine* under a methods-forward fairness diagnosis framing.
