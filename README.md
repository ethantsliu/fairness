# NHANES Health Prediction & Fairness Analysis

Research repository for machine learning models applied to NHANES (National Health and Nutrition Examination Survey) data, with emphasis on algorithmic fairness and clinical utility.

## Projects

### [`blood_glucose_project/`](blood_glucose_project/)

Blood glucose and HbA1c prediction using NHANES 2011-2020 data. Compares lab-proxy models (which exhibit circular reasoning) against lifestyle-only models suitable for population screening. Includes comprehensive fairness analysis across demographics and wearable metadata.

**Key finding:** A 72% accurate lifestyle-only binary risk classifier is more clinically useful than a 99% accurate lab-proxy model.

### [`dana_nhanes_project/`](dana_nhanes_project/)

NHANES physical activity and cardiovascular disease risk analysis. Compares accelerometer-based vs self-reported activity data for CVD prediction, with bootstrap stability analysis.

## Repository Structure

```
fairness/
├── blood_glucose_project/
│   ├── data/
│   │   ├── raw/                    # Raw NHANES XPT files (2011-2020)
│   │   ├── processed/              # Cleaned and merged datasets
│   │   └── integrated/             # Final integrated datasets
│   ├── notebooks/                  # Jupyter notebooks (EDA, pipelines)
│   ├── scripts/
│   │   ├── preprocessing/          # Data cleaning, XPT conversion, merging
│   │   ├── exploratory/            # Preliminary analysis and prototypes
│   │   ├── core_analysis/          # Main analysis (lab-proxy vs lifestyle)
│   │   ├── feature_analysis/       # Feature importance and SHAP analysis
│   │   ├── modeling/               # Classification and ensemble models
│   │   └── validation/             # Fairness evaluation and clinical validation
│   ├── finetuning/                 # Model fine-tuning experiments
│   ├── figures/                    # Generated visualizations
│   ├── results/                    # Output tables, reports, and PDFs
│   └── documentation/              # Manuscripts, summaries, and methodology
│
└── dana_nhanes_project/
    ├── analysis/                   # Analysis and visualization scripts
    ├── data/                       # Bootstrap and delta AUC results
    ├── figures/                    # Manuscript figures
    └── results/                    # Analysis outputs
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r blood_glucose_project/requirements.txt
```

## Key Dependencies

- pandas, numpy -- data handling
- scikit-learn -- modeling and evaluation
- matplotlib, seaborn -- visualization
- shap -- model explainability
- scipy -- statistical analysis
