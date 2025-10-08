# NHANES Blood Glucose and HbA1c Analysis Project - Progress Report

**Date:** October 6, 2025  
**Project:** Blood Glucose Prediction with Fairness Evaluation  
**Data Source:** NHANES 2017-2020  

## Project Overview

This project implements a comprehensive machine learning pipeline for predicting blood glucose and HbA1c levels using NHANES data, with a focus on fairness evaluation across demographic subgroups.

## ✅ Completed Components

### 3.1 Data Source and Preprocessing ✅
- **Dataset:** NHANES 2017-2020 merged on SEQN
- **Final Dataset:** 3,992 participants, 20 features
- **Inclusion Criteria Applied:**
  - Age ≥ 18 years (excluded 534 participants)
  - Fasting glucose and HbA1c available
  - Outlier removal (glucose > 600 mg/dL or HbA1c > 18%)
- **Features Retained:** Age, Gender, Race/Ethnicity, Triglycerides, Total Cholesterol, LDL Cholesterol, Iron, Uric Acid, Blood Urea Nitrogen, Total Protein, Albumin, Globulin, Glucose Serum, Phosphorus, Calcium, Sodium, Potassium, Chloride, CRP, Cotinine
- **Data Cleaning:** Standardization, missing value imputation (median for numeric, mode for categorical)

### 3.2 Modeling Framework ✅
- **Task:** Multi-output regression for (Glucose, HbA1c)
- **Primary Model:** Random Forest via MultiOutputRegressor
- **Hyperparameter Tuning:** Grid search with 5-fold CV
  - Best parameters: max_depth=20, min_samples_leaf=4, min_samples_split=5, n_estimators=100
- **Baseline:** Ridge regression
- **Performance Metrics:**
  - **Random Forest:** MAE=1.522, MSE=12.453, R²=0.868
  - **Ridge Baseline:** MAE=1.432, MSE=11.298, R²=0.868
- **Data Split:** 80% training (3,193), 20% testing (799)

### 3.3 Feature Importance & Explainability ✅
- **Method:** SHAP values for global and local importance
- **Top Features (Glucose Prediction):**
  1. Glucose_Serum (22.92)
  2. Chloride (0.16)
  3. Sodium (0.13)
  4. Iron (0.10)
  5. Triglycerides (0.09)
- **Visualizations:** Global feature importance plot saved as `glucose_feature_importance.png`
- **Interpretability:** Results align with known physiology (glucose serum as strongest predictor)

### 3.5 Fairness Evaluation ✅
- **Subgroups Evaluated:**
  - **Gender:** Male (n=385) vs Female (n=414)
  - **Age Groups:** <40 (n=270), 40-60 (n=272), >60 (n=241)
  - **Race/Ethnicity:** NHW, NHB, Hispanic, Other
- **Fairness Metrics:** MAE per subgroup for both glucose and HbA1c
- **Key Findings:**
  - **Gender:** Similar performance (Male: 2.66 MAE, Female: 2.65 MAE for glucose)
  - **Age:** Older adults show higher error (>60: 3.01 MAE vs <40: 2.53 MAE)
  - **Race:** Some disparities observed across groups
- **Visualizations:** Fairness evaluation plots saved as `fairness_evaluation.png`

## 🔄 Partially Completed

### 3.4 Dietary Clustering ⚠️
- **Status:** Framework implemented but requires additional dietary data files
- **Planned Implementation:**
  - K-means (k=3) on standardized nutrient intake variables
  - Load NHANES dietary files (DR1TOT_*.xpt, DR2TOT_*.xpt)
  - Compare clusters' average glucose/HbA1c levels
  - Evaluate cluster membership as predictive variable

## 📁 Generated Files

1. **`blood_glucose_analysis.py`** - Complete analysis pipeline (519 lines)
2. **`requirements_glucose.txt`** - Python dependencies
3. **`glucose_feature_importance.png`** - Feature importance visualization
4. **`fairness_evaluation.png`** - Fairness metrics by demographic groups

## 🔧 Technical Implementation

### Key Classes and Methods
- **`NHANESGlucoseAnalyzer`** - Main analysis class
- **Data Pipeline:** `load_and_merge_data()`, `apply_inclusion_exclusion_criteria()`, `prepare_features()`
- **Modeling:** `train_models()`, `evaluate_models()`
- **Analysis:** `analyze_feature_importance()`, `evaluate_fairness()`

### Dependencies Installed
- pandas, numpy, scikit-learn, matplotlib, seaborn
- **shap** (0.48.0) - For explainability analysis
- pyreadstat - For reading NHANES .xpt files

## 📊 Key Results Summary

| Metric | Random Forest | Ridge Baseline |
|--------|---------------|----------------|
| MAE | 1.522 | 1.432 |
| MSE | 12.453 | 11.298 |
| R² | 0.868 | 0.868 |

### Fairness Insights
- **Gender equity:** Minimal performance differences
- **Age bias:** Higher errors for older adults (>60 years)
- **Racial disparities:** Performance varies across ethnic groups

## 🎯 Next Steps (When Resumed)

1. **Complete Dietary Clustering (3.4):**
   - Obtain NHANES dietary data files
   - Implement K-means clustering on nutrient variables
   - Integrate cluster membership as model feature

2. **Enhanced Analysis:**
   - Implement additional fairness metrics (equalized error rates)
   - Add more sophisticated baseline models
   - Conduct sensitivity analysis on hyperparameters

3. **Validation:**
   - Cross-validation across different NHANES cycles
   - External validation on independent datasets
   - Clinical relevance assessment

## 💡 Technical Notes

- **Data Quality:** High-quality merged dataset with comprehensive biomarkers
- **Model Performance:** Strong R² (0.868) indicates good predictive capability
- **Fairness Considerations:** Age-related bias identified for further investigation
- **Scalability:** Pipeline designed for easy extension to additional NHANES cycles

---

**Status:** Core pipeline complete and functional. Ready for dietary clustering completion and enhanced fairness analysis when resumed.
