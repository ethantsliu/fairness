# Misdiagnosing Algorithmic Bias: Subgroup Distribution Differences Explain Performance Disparities in Lifestyle-Based Glucose Prediction

## Abstract
Subgroup performance disparities in healthcare machine learning are often interpreted as evidence of algorithmic bias. We revisit this assumption in lifestyle-based glucose prediction using NHANES adults and show that subgroup differences in outcome distributions provide a stronger explanation for observed error gaps. We train a multi-output random forest model to predict fasting glucose and HbA1c from lifestyle and demographic features, then evaluate subgroup contrasts across age, race/ethnicity, sex, education, and income dimensions. Rather than focusing on variance in isolation, we profile subgroup differences using distribution summaries (mean, SD, IQR, skewness, kurtosis), tail mass, and pairwise distribution distances. We additionally compare mixed-population (heterogeneous) training against within-group (homogeneous) training to separate training-composition effects from intrinsic outcome structure. The revised analysis supports a diagnosis-first fairness narrative: subgroup error differences are aligned with subgroup outcome-distribution contrasts, while generic fairness corrections are unlikely to resolve disparities when the underlying prediction difficulty differs between groups. These findings suggest that observed subgroup disparities should not automatically trigger bias-mitigation interventions without mechanism-specific diagnostics.

## Main Text
Performance disparities across demographic subgroups are common in clinical machine learning and are frequently treated as direct evidence of algorithmic bias. This framing can be too narrow. In practice, subgroup error gaps can emerge from multiple mechanisms, including sample imbalance, feature quality mismatch, and subgroup-specific outcome distributions. In lifestyle-based glucose prediction, where inputs are behavior and demographics rather than dense laboratory biomarkers, these mechanisms are especially relevant.

We reframe the analysis around subgroup differences in outcome structure. Using NHANES-based lifestyle modeling, we evaluate fasting glucose and HbA1c prediction while emphasizing how subgroup distributions differ from one another, not only how variable each subgroup is internally. For each subgroup axis (age bins, race/ethnicity, sex, education, and income), we summarize outcome distributions with mean, standard deviation, interquartile range, skewness, and kurtosis, and compute pairwise subgroup distances using Kolmogorov-Smirnov statistics and Wasserstein distance. This makes the interpretation directly comparative: some subgroup pairs are distributionally close, while others are separated by heavier tails, wider spread, or shifted central tendency.

This comparative framing clarifies why performance differences are expected. If one subgroup has a broader glucose distribution with higher tail mass, the same model and feature set will face a harder prediction task in that subgroup than in a narrower subgroup. Under this mechanism, disparity does not necessarily indicate discriminatory model behavior; it can reflect unequal outcome difficulty induced by subgroup-specific data-generating structure.

We further add an overlayed age-versus-glucose analysis to visualize structure in continuous space. Instead of only reporting age-bin summary differences, the overlay plot shows how subgroup trends diverge across age and where subgroup clouds overlap versus separate. This figure is central for interpretation because it links subgroup contrasts to plausible model error patterns.

To keep the main narrative interpretable, we reduce the original 26 subgroup breakdown to a stable core set for primary reporting using predefined support and stability criteria (minimum sample size and variance-based stability score). Full subgroup detail is retained for supplement-style reporting, while the core set is used for main text figures and conclusions.

Finally, we test whether disparities are primarily due to mixed-population training composition by comparing heterogeneous (pooled) training against homogeneous (within-group) training on matched subgroup definitions (age-bin by sex). This direct comparison provides a practical diagnostic. If homogeneous training meaningfully reduces subgroup error, training composition likely contributes; if differences persist and align with subgroup outcome structure, intrinsic heterogeneity remains the dominant explanation.

The resulting message is not that fairness is unimportant, but that fairness intervention should follow mechanism diagnosis. In this study context, subgroup distribution differences explain why error disparities arise and why one-size-fits-all fairness corrections may underperform. A more reliable equity workflow is: (1) quantify disparities, (2) characterize subgroup outcome contrasts, (3) test composition effects with homogeneous versus heterogeneous training, and only then (4) select targeted mitigation. This diagnosis-first sequence improves both scientific validity and practical relevance for healthcare ML deployment.

## Figure Plan for Comment Format
- Figure 1: Subgroup distribution contrasts for glucose and HbA1c (faceted by subgroup axis).
- Figure 2: Overlayed age versus glucose distribution with subgroup trend separation.
- Figure 3: Homogeneous versus heterogeneous training MAE comparison for stable core groups.

## Notes for Submission
- Article type: short comment/perspective with empirical support.
- Target length: approximately 1,200 to 1,600 words plus up to 3 compact figures.
- Recommended venue fit: *Artificial Intelligence in Medicine* (methods-forward framing).
