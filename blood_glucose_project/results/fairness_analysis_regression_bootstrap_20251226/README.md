# Wearable Fairness Analysis with Regression and Bootstrap Methods

**Analysis Date:** December 26, 2024  
**Analysis Type:** Comprehensive algorithmic fairness analysis with regression analyses and bootstrapped error bars

## Overview

This folder contains results from the enhanced wearable device algorithmic fairness analysis that includes:

1. **Binary Fairness Metrics**: Statistical parity, equalized odds, equal opportunity, and calibration
2. **Regression Analyses**: Continuous factor analyses (wear time, data quality, etc.) with R² and p-values
3. **Bootstrapped Error Bars**: All performance metrics reported as mean ± standard deviation using 1000 bootstrap iterations (80% sample of test set)

## Contents

### Reports
- `wearable_algorithmic_fairness_analysis.md` - Comprehensive fairness analysis report
- `wearable_fairness_executive_summary.md` - Executive summary of key findings

### Figures
- (Figures will be generated when analysis script is run)

## Key Enhancements

### Regression Analyses
- Performance regressed on continuous wearable factors:
  - Average daily wear hours
  - Data quality ratio
  - Wear time variability
  - Weekend proportion
  - Total monitoring days
- Each regression includes:
  - R² (coefficient of determination)
  - p-value (statistical significance)
  - Bootstrapped confidence intervals (1000 iterations, 80% sample)

### Bootstrapped Error Bars
- All performance metrics use bootstrapped sampling:
  - Sample 80% of test data randomly
  - 1000 iterations
  - Report mean ± standard deviation
- Applied to:
  - Classification accuracy, precision, recall, F1
  - Fairness metrics (TPR, FPR, PPV disparities)
  - Regression MAE disparities

### Data Quality Definitions
- **Poor Data Quality**: Data quality ratio < 0.70 (valid wear minutes / total wear minutes)
- **Fair Data Quality**: 0.70 - 0.85
- **Good Data Quality**: 0.85 - 0.95
- **Excellent Data Quality**: > 0.95

### Fairness Thresholds
- **Excellent Fairness**: All metrics ≤ 0.05
- **Acceptable Fairness**: All metrics ≤ 0.10
- **Poor Fairness**: Any metric > 0.10
- **MAE Disparity Thresholds**: < 2 mg/dL (excellent), < 5 mg/dL (acceptable), ≥ 5 mg/dL (poor)

## Running the Analysis

To regenerate these results, run:

```bash
cd blood_glucose_project/scripts/validation
python algorithmic_fairness_wearable_metadata.py
```

## Notes

- Results are based on NHANES 2011-2014 accelerometry and glucose data
- Analysis uses Random Forest classifiers with stratified train/test splits
- All metrics are calculated on held-out test sets only

