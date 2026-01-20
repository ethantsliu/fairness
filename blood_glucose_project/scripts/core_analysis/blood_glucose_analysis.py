#!/usr/bin/env python3
"""
NHANES Blood Glucose and HbA1c Analysis Pipeline
Multi-output regression with fairness evaluation and dietary clustering

Author: Generated for fairness project
Date: October 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import shap
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class NHANESGlucoseAnalyzer:
    """
    Comprehensive analyzer for NHANES glucose and HbA1c prediction with fairness evaluation
    """
    
    def __init__(self, data_dir="/Users/aakashsuresh/fairness/processed_data_nhanes_lab/"):
        self.data_dir = data_dir
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler_X = None
        self.scaler_y = None
        self.model = None
        self.baseline_model = None
        self.feature_names = None
        self.demographic_columns = ['age', 'gender', 'race_ethnicity']
        
    def load_and_merge_data(self):
        """
        3.1 Data Source and Preprocessing
        Load NHANES 2011-2020 datasets and merge on SEQN
        """
        print("=== 3.1 Data Source and Preprocessing ===")
        print("Loading NHANES 2017-2020 datasets...")
        
        # Define lab files to load
        lab_files = [
            "fasting_glucose_processed.csv",
            "glycohemoglobin_processed.csv", 
            "biochemistry_profile_processed.csv",
            "iron_status_processed.csv",
            "c_reactive_protein_processed.csv",
            "cotinine_processed.csv"
        ]
        
        # Load and merge lab data
        labs = {}
        for fn in lab_files:
            if os.path.exists(os.path.join(self.data_dir, fn)):
                key = fn.replace("_processed.csv", "")
                df = pd.read_csv(os.path.join(self.data_dir, fn))
                print(f"→ {key}: {df.shape[0]} rows")
                labs[key] = df
        
        # Inner merge all lab files on SEQN
        merged = None
        for df in labs.values():
            merged = df if merged is None else merged.merge(df, on="seqn", how="inner")
        print(f"Merged labs: {merged.shape}")
        
        # Load demographics
        demo_path = os.path.join(self.data_dir, "P_DEMO.xpt")
        if os.path.exists(demo_path):
            demo = pd.read_sas(demo_path, format="xport")
            demo_cols = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"]  # Add race/ethnicity
            demo = demo[demo_cols]
            demo.columns = demo.columns.str.lower()
            merged.columns = merged.columns.str.lower()
            merged = merged.merge(demo, on="seqn", how="left")
            merged.rename(columns={
                "ridageyr": "age", 
                "riagendr": "gender",
                "ridreth3": "race_ethnicity"
            }, inplace=True)
        
        self.df = merged
        print(f"Final dataset shape: {self.df.shape}")
        return self.df
    
    def apply_inclusion_exclusion_criteria(self):
        """
        Apply inclusion/exclusion criteria:
        - Age >= 18 years
        - Fasting glucose and HbA1c available
        - Remove extreme outliers
        """
        print("\nApplying inclusion/exclusion criteria...")
        initial_count = len(self.df)
        
        # Age >= 18
        self.df = self.df[self.df['age'] >= 18]
        print(f"After age ≥18 filter: {len(self.df)} participants ({initial_count - len(self.df)} excluded)")
        
        # Must have both glucose and HbA1c
        self.df = self.df.dropna(subset=['lbxglu', 'lbxgh'])
        print(f"After glucose/HbA1c requirement: {len(self.df)} participants")
        
        # Remove extreme outliers (glucose > 600 mg/dL or HbA1c > 18%)
        outlier_mask = (self.df['lbxglu'] <= 600) & (self.df['lbxgh'] <= 18)
        self.df = self.df[outlier_mask]
        print(f"After outlier removal: {len(self.df)} participants")
        
        return self.df
    
    def prepare_features(self):
        """
        Prepare feature set and handle missing values
        List retained features: BMI, triglycerides, cholesterol, age, gender, etc.
        """
        print("\nPreparing features...")
        
        # Define feature mapping from NHANES codes to readable names
        feature_mapping = {
            'age': 'Age',
            'gender': 'Gender', 
            'race_ethnicity': 'Race/Ethnicity',
            'lbxstr': 'Triglycerides',
            'lbxsch': 'Total_Cholesterol', 
            'lbxsldsi': 'LDL_Cholesterol',
            'lbxsir': 'Iron',
            'lbxsua': 'Uric_Acid',
            'lbxscrsi': 'Creatinine',
            'lbxsbu': 'Blood_Urea_Nitrogen',
            'lbxstp': 'Total_Protein',
            'lbxsal': 'Albumin',
            'lbxsgb': 'Globulin',
            'lbxsgl': 'Glucose_Serum',
            'lbxsph': 'Phosphorus',
            'lbxsca': 'Calcium',
            'lbxsnasi': 'Sodium',
            'lbxsksi': 'Potassium',
            'lbxsclsi': 'Chloride',
            'lbxhscrp': 'CRP',
            'lbxcot': 'Cotinine'
        }
        
        # Select available features
        available_features = [col for col in feature_mapping.keys() if col in self.df.columns]
        feature_df = self.df[available_features + ['lbxglu', 'lbxgh', 'seqn']].copy()
        
        # Handle missing values
        print("Handling missing values...")
        for col in available_features:
            if col in ['gender', 'race_ethnicity']:
                # Categorical variables - use mode
                feature_df[col] = feature_df[col].fillna(feature_df[col].mode()[0])
            else:
                # Numerical variables - use median
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())
        
        # Encode categorical variables
        le_gender = LabelEncoder()
        le_race = LabelEncoder()
        
        if 'gender' in feature_df.columns:
            feature_df['gender'] = le_gender.fit_transform(feature_df['gender'])
        if 'race_ethnicity' in feature_df.columns:
            feature_df['race_ethnicity'] = le_race.fit_transform(feature_df['race_ethnicity'])
        
        # Separate features and targets
        self.X = feature_df[available_features]
        self.y = feature_df[['lbxglu', 'lbxgh']]
        self.feature_names = [feature_mapping.get(col, col) for col in available_features]
        
        print(f"Resulting dataset: {len(feature_df)} participants, {len(available_features)} features")
        print(f"Features: {self.feature_names}")
        
        return self.X, self.y
    
    def split_and_scale_data(self, test_size=0.2):
        """
        Split data and apply standardization
        """
        print(f"\nSplitting data (test_size={test_size})...")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.X['gender']
        )
        
        # Scale features
        self.scaler_X = StandardScaler()
        self.X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler_X.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """
        3.2 Modeling Framework
        Train Random Forest and baseline models with hyperparameter tuning
        """
        print("\n=== 3.2 Modeling Framework ===")
        print("Training multi-output regression models...")
        
        # Define task: Multi-output regression for (Glucose, HbA1c)
        print("Task: Multi-output regression for (Glucose, HbA1c)")
        
        # Baseline: Ridge regression
        print("Training baseline Ridge regression...")
        self.baseline_model = MultiOutputRegressor(Ridge(alpha=1.0))
        self.baseline_model.fit(self.X_train_scaled, self.y_train)
        
        # Main model: Random Forest with hyperparameter tuning
        print("Training Random Forest with hyperparameter tuning...")
        
        # Grid search parameters
        param_grid = {
            'estimator__n_estimators': [100, 200, 300],
            'estimator__max_depth': [10, 15, 20, None],
            'estimator__min_samples_split': [2, 5, 10],
            'estimator__min_samples_leaf': [1, 2, 4]
        }
        
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_multi = MultiOutputRegressor(rf_base)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            rf_multi, param_grid, 
            cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:.3f}")
        
        return self.model, self.baseline_model
    
    def evaluate_models(self):
        """
        Evaluate models using MAE, MSE, R² metrics
        """
        print("\nEvaluating models...")
        
        # Predictions
        y_pred_rf = self.model.predict(self.X_test_scaled)
        y_pred_baseline = self.baseline_model.predict(self.X_test_scaled)
        
        # Metrics for Random Forest
        mae_rf = mean_absolute_error(self.y_test, y_pred_rf)
        mse_rf = mean_squared_error(self.y_test, y_pred_rf)
        r2_rf = r2_score(self.y_test, y_pred_rf)
        
        # Metrics for Baseline
        mae_baseline = mean_absolute_error(self.y_test, y_pred_baseline)
        mse_baseline = mean_squared_error(self.y_test, y_pred_baseline)
        r2_baseline = r2_score(self.y_test, y_pred_baseline)
        
        # Print results
        print("\n=== Model Performance ===")
        print("Random Forest:")
        print(f"  MAE: {mae_rf:.3f}")
        print(f"  MSE: {mse_rf:.3f}")
        print(f"  R²:  {r2_rf:.3f}")
        
        print("\nBaseline (Ridge):")
        print(f"  MAE: {mae_baseline:.3f}")
        print(f"  MSE: {mse_baseline:.3f}")
        print(f"  R²:  {r2_baseline:.3f}")
        
        return {
            'rf': {'mae': mae_rf, 'mse': mse_rf, 'r2': r2_rf},
            'baseline': {'mae': mae_baseline, 'mse': mse_baseline, 'r2': r2_baseline}
        }
    
    def analyze_feature_importance(self):
        """
        3.3 Feature Importance & Explainability
        Use SHAP values to quantify global and local importance
        """
        print("\n=== 3.3 Feature Importance & Explainability ===")
        print("Computing SHAP values...")
        
        # Extract one of the Random Forest estimators for SHAP analysis
        rf_estimator = self.model.estimators_[0]  # For glucose prediction
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(rf_estimator)
        shap_values = explainer.shap_values(self.X_test_scaled)
        
        # Global feature importance
        feature_importance = np.abs(shap_values).mean(0)
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features (Glucose Prediction):")
        print(importance_df.head(10))
        
        # Create visualizations
        self.create_feature_importance_plots(shap_values, importance_df)
        
        return importance_df, shap_values
    
    def create_feature_importance_plots(self, shap_values, importance_df):
        """
        Create feature importance visualizations
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Global feature importance bar plot
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Global Feature Importance (Glucose Prediction)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/figures/glucose_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # SHAP summary plot would go here (requires proper SHAP setup)
        print("Feature importance plot saved as 'glucose_feature_importance.png'")
    
    def perform_dietary_clustering(self):
        """
        3.4 Dietary Clustering
        K-means clustering on standardized nutrient intake variables
        Note: This is a placeholder as dietary data needs to be loaded separately
        """
        print("\n=== 3.4 Dietary Clustering ===")
        print("Note: Dietary clustering requires separate NHANES dietary files")
        print("This would involve:")
        print("- Loading NHANES dietary files (DR1TOT_*.xpt, DR2TOT_*.xpt)")
        print("- K-means (k=3) on standardized nutrient intake variables")
        print("- Comparing clusters' average glucose/HbA1c levels")
        print("- Evaluating cluster membership as a predictive variable")
        
        # Placeholder for dietary clustering implementation
        return None
    
    def evaluate_fairness(self):
        """
        3.5 Fairness Evaluation
        Evaluate model performance across demographic subgroups
        """
        print("\n=== 3.5 Fairness Evaluation ===")
        print("Evaluating fairness across demographic subgroups...")
        
        # Get predictions
        y_pred = self.model.predict(self.X_test_scaled)
        
        # Create test dataframe with predictions
        test_df = self.X_test.copy()
        test_df['glucose_true'] = self.y_test.iloc[:, 0].values
        test_df['hba1c_true'] = self.y_test.iloc[:, 1].values
        test_df['glucose_pred'] = y_pred[:, 0]
        test_df['hba1c_pred'] = y_pred[:, 1]
        
        # Define subgroups
        fairness_results = {}
        
        # Gender-based evaluation
        if 'gender' in test_df.columns:
            gender_results = self.evaluate_subgroup_fairness(test_df, 'gender', 
                                                           {0: 'Male', 1: 'Female'})
            fairness_results['gender'] = gender_results
        
        # Age-based evaluation
        test_df['age_group'] = pd.cut(test_df['age'], 
                                    bins=[18, 40, 60, 100], 
                                    labels=['<40', '40-60', '>60'])
        age_results = self.evaluate_subgroup_fairness(test_df, 'age_group')
        fairness_results['age'] = age_results
        
        # Race/ethnicity evaluation (if available)
        if 'race_ethnicity' in test_df.columns:
            race_mapping = {0: 'NHW', 1: 'NHB', 2: 'Hispanic', 3: 'Other'}
            race_results = self.evaluate_subgroup_fairness(test_df, 'race_ethnicity', race_mapping)
            fairness_results['race'] = race_results
        
        self.create_fairness_visualizations(fairness_results)
        self.export_fairness_results(
            fairness_results,
            "/Users/aakashsuresh/fairness/blood_glucose_project/results/fairness_lab_bootstrap.csv",
            model_label="Lab-Proxy Model"
        )
        
        return fairness_results
    
    def _bootstrap_group_mae(self, group_data, n_bootstrap=1000, ci=0.95, random_state=42):
        """Bootstrap MAE metrics for a subgroup."""
        n = len(group_data)
        if n < 2:
            return None

        rng = np.random.default_rng(random_state)
        glucose_true = group_data['glucose_true'].to_numpy()
        glucose_pred = group_data['glucose_pred'].to_numpy()
        hba1c_true = group_data['hba1c_true'].to_numpy()
        hba1c_pred = group_data['hba1c_pred'].to_numpy()

        glucose_boot = np.empty(n_bootstrap)
        hba1c_boot = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            glucose_boot[i] = np.mean(np.abs(glucose_true[idx] - glucose_pred[idx]))
            hba1c_boot[i] = np.mean(np.abs(hba1c_true[idx] - hba1c_pred[idx]))

        alpha = (1 - ci) / 2
        glucose_ci = np.quantile(glucose_boot, [alpha, 1 - alpha])
        hba1c_ci = np.quantile(hba1c_boot, [alpha, 1 - alpha])

        return {
            'glucose_mean': float(np.mean(glucose_boot)),
            'glucose_std': float(np.std(glucose_boot, ddof=1)),
            'glucose_ci_low': float(glucose_ci[0]),
            'glucose_ci_high': float(glucose_ci[1]),
            'hba1c_mean': float(np.mean(hba1c_boot)),
            'hba1c_std': float(np.std(hba1c_boot, ddof=1)),
            'hba1c_ci_low': float(hba1c_ci[0]),
            'hba1c_ci_high': float(hba1c_ci[1])
        }

    def evaluate_subgroup_fairness(self, df, group_col, group_mapping=None, n_bootstrap=1000):
        """
        Evaluate fairness metrics for a specific demographic subgroup
        """
        results = {}
        
        for group_val in df[group_col].unique():
            if pd.isna(group_val):
                continue
                
            group_data = df[df[group_col] == group_val]
            group_name = group_mapping.get(group_val, str(group_val)) if group_mapping else str(group_val)
            
            # Calculate MAE for glucose and HbA1c
            glucose_mae = mean_absolute_error(group_data['glucose_true'], group_data['glucose_pred'])
            hba1c_mae = mean_absolute_error(group_data['hba1c_true'], group_data['hba1c_pred'])

            bootstrap = None
            if len(group_data) >= 10:
                bootstrap = self._bootstrap_group_mae(group_data, n_bootstrap=n_bootstrap)
            
            results[group_name] = {
                'n': len(group_data),
                'glucose_mae': glucose_mae,
                'hba1c_mae': hba1c_mae,
                'glucose_mae_mean': bootstrap['glucose_mean'] if bootstrap else glucose_mae,
                'glucose_mae_std': bootstrap['glucose_std'] if bootstrap else 0.0,
                'glucose_mae_ci95_low': bootstrap['glucose_ci_low'] if bootstrap else glucose_mae,
                'glucose_mae_ci95_high': bootstrap['glucose_ci_high'] if bootstrap else glucose_mae,
                'hba1c_mae_mean': bootstrap['hba1c_mean'] if bootstrap else hba1c_mae,
                'hba1c_mae_std': bootstrap['hba1c_std'] if bootstrap else 0.0,
                'hba1c_mae_ci95_low': bootstrap['hba1c_ci_low'] if bootstrap else hba1c_mae,
                'hba1c_mae_ci95_high': bootstrap['hba1c_ci_high'] if bootstrap else hba1c_mae,
                'glucose_mean_true': group_data['glucose_true'].mean(),
                'hba1c_mean_true': group_data['hba1c_true'].mean()
            }
        
        # Print results
        print(f"\nFairness evaluation by {group_col}:")
        for group_name, metrics in results.items():
            print(f"  {group_name} (n={metrics['n']}):")
            print(f"    Glucose MAE: {metrics['glucose_mae_mean']:.3f} ± {metrics['glucose_mae_std']:.3f}")
            print(f"    HbA1c MAE:   {metrics['hba1c_mae_mean']:.3f} ± {metrics['hba1c_mae_std']:.3f}")
        
        return results
    
    def create_fairness_visualizations(self, fairness_results, error_bar='std'):
        """
        Create fairness evaluation visualizations
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, (group_type, results) in enumerate(fairness_results.items()):
            if i >= 2:  # Only plot first 2 group types
                break
                
            groups = list(results.keys())
            glucose_mae = [results[g].get('glucose_mae_mean', results[g]['glucose_mae']) for g in groups]
            hba1c_mae = [results[g].get('hba1c_mae_mean', results[g]['hba1c_mae']) for g in groups]
            if error_bar == 'ci95':
                glucose_err = [
                    [
                        results[g]['glucose_mae_mean'] - results[g].get('glucose_mae_ci95_low', results[g]['glucose_mae_mean']),
                        results[g].get('glucose_mae_ci95_high', results[g]['glucose_mae_mean']) - results[g]['glucose_mae_mean']
                    ]
                    for g in groups
                ]
                hba1c_err = [
                    [
                        results[g]['hba1c_mae_mean'] - results[g].get('hba1c_mae_ci95_low', results[g]['hba1c_mae_mean']),
                        results[g].get('hba1c_mae_ci95_high', results[g]['hba1c_mae_mean']) - results[g]['hba1c_mae_mean']
                    ]
                    for g in groups
                ]
                glucose_err = np.array(glucose_err).T
                hba1c_err = np.array(hba1c_err).T
            else:
                glucose_err = [results[g].get('glucose_mae_std', 0.0) for g in groups]
                hba1c_err = [results[g].get('hba1c_mae_std', 0.0) for g in groups]
            
            # Glucose MAE
            axes[0, i].bar(groups, glucose_mae, yerr=glucose_err, capsize=4, alpha=0.7, color='skyblue')
            axes[0, i].set_title(f'Glucose MAE by {group_type.title()}')
            axes[0, i].set_ylabel('MAE (mg/dL)')
            axes[0, i].tick_params(axis='x', rotation=45)
            
            # HbA1c MAE
            axes[1, i].bar(groups, hba1c_mae, yerr=hba1c_err, capsize=4, alpha=0.7, color='lightcoral')
            axes[1, i].set_title(f'HbA1c MAE by {group_type.title()}')
            axes[1, i].set_ylabel('MAE (%)')
            axes[1, i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/figures/fairness_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Fairness evaluation plot saved as 'fairness_evaluation.png'")

    def export_fairness_results(self, fairness_results, output_path, model_label):
        """Save fairness results with bootstrap summaries to CSV."""
        rows = []
        for group_type, results in fairness_results.items():
            for group_name, metrics in results.items():
                rows.append({
                    'model': model_label,
                    'group_type': group_type,
                    'group': group_name,
                    'n': metrics.get('n', 0),
                    'glucose_mae_mean': metrics.get('glucose_mae_mean', metrics.get('glucose_mae')),
                    'glucose_mae_std': metrics.get('glucose_mae_std', 0.0),
                    'glucose_mae_ci95_low': metrics.get('glucose_mae_ci95_low', metrics.get('glucose_mae')),
                    'glucose_mae_ci95_high': metrics.get('glucose_mae_ci95_high', metrics.get('glucose_mae')),
                    'hba1c_mae_mean': metrics.get('hba1c_mae_mean', metrics.get('hba1c_mae')),
                    'hba1c_mae_std': metrics.get('hba1c_mae_std', 0.0),
                    'hba1c_mae_ci95_low': metrics.get('hba1c_mae_ci95_low', metrics.get('hba1c_mae')),
                    'hba1c_mae_ci95_high': metrics.get('hba1c_mae_ci95_high', metrics.get('hba1c_mae')),
                })

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            print(f"Fairness bootstrap results saved to {output_path}")
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline
        """
        print("NHANES Blood Glucose and HbA1c Analysis Pipeline")
        print("=" * 60)
        
        # Load and preprocess data
        self.load_and_merge_data()
        self.apply_inclusion_exclusion_criteria()
        self.prepare_features()
        self.split_and_scale_data()
        
        # Model training and evaluation
        self.train_models()
        performance_metrics = self.evaluate_models()
        
        # Feature importance analysis
        importance_df, shap_values = self.analyze_feature_importance()
        
        # Dietary clustering (placeholder)
        self.perform_dietary_clustering()
        
        # Fairness evaluation
        fairness_results = self.evaluate_fairness()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
        return {
            'performance': performance_metrics,
            'feature_importance': importance_df,
            'fairness': fairness_results
        }

def main():
    """
    Main execution function
    """
    analyzer = NHANESGlucoseAnalyzer()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    results = main()
