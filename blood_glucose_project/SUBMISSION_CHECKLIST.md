# Submission Checklist — Comment Paper

## Done

- [x] Homogeneous vs heterogeneous analysis run (6 age-by-sex strata)
- [x] Results exported to `results/homogeneous_vs_heterogeneous_comparison.csv`
- [x] Comment paper updated with empirical Δ(g) table and interpretation
- [x] Methods schematic figure generated
- [x] Limitations and take-home paragraphs added
- [x] Email draft prepared (`EMAIL_DRAFT_HOM_VS_HET.md`)
- [x] Terminology aligned to "fairness diagnosis" framing for AIM

## Your action items

- [ ] **Send email** — copy body from `EMAIL_DRAFT_HOM_VS_HET.md`, attach methods PNG (+ optional results PNG)
- [ ] **Proofread** `COMMENT_PAPER_SUBGROUP_DIFFERENCES.md` for voice and co-author names
- [ ] **Check AIM author guidelines** — word limit, figure resolution, reference format
- [ ] **Convert to submission format** (Word/LaTeX per journal template)
- [ ] **Add references** (fairness in healthcare ML, NHANES, Wasserstein/KS if cited)
- [ ] **Confirm co-author approval** before submission

## Files to include in submission package

| Item | Path |
|------|------|
| Manuscript | `COMMENT_PAPER_SUBGROUP_DIFFERENCES.md` |
| Figure 1 | `figures/publication/outcome_heterogeneity_by_subgroups.png` |
| Figure 2 | `figures/publication/overlay_age_vs_glucose.png` |
| Figure 3 | `figures/publication/homogeneous_vs_mixed_training.png` |
| Figure M1 | `figures/publication/methods_homogeneous_vs_heterogeneous.png` |
| Supplementary CSV | `results/homogeneous_vs_heterogeneous_comparison.csv` |

## Optional follow-ups (if reviewers ask)

- Repeat hom/het with 10-fold CV instead of single 80/20 split
- Extend to race/ethnicity strata with sufficient n
- Bootstrap CIs on per-stratum MAE
