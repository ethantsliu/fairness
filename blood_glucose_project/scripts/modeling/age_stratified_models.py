#!/usr/bin/env python3
"""
Age-Stratified Models to Address Age Dominance
Creates separate models for different age groups to improve performance

Author: Generated for fairness project
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

class AgeStratifiedModels:
    """
    Age-stratified models to address age dominance in feature importance
    """
    
    def __init__(self, dataset_path="/Users/aakashsuresh/fairness/blood_glucose_project/fixed_data/integrated_nhanes_2011_2014.csv"):
        self.dataset_path = dataset_path
        self.df = None
        self.age_groups = {}
        self.age_models = {}
        self.results = {}
        
    def load_and_stratify_data(self):
        """
        Load data and create age-stratified subsets
        """
        print("=== Loading and Stratifying Data by Age ===")
        
        self.df = pd.read_csv(self.dataset_path)
        print(f"Total dataset: {self.df.shape}")
        
        # Define age groups
        age_bins = [18, 35, 50, 65, 100]
        age_labels = ['Young (18-34)', 'Middle-Young (35-49)', 'Middle-Old (50-64)', 'Older (65+)']
        
        self.df['age_group'] = pd.cut(self.df['age'], bins=age_bins, labels=age_labels, right=False)
        
        # Create age-stratified datasets
        for age_label in age_labels:
            age_mask = self.df['age_group'] == age_label
            age_subset = self.df[age_mask].copy()
            
            if len(age_subset) > 100:  # Minimum sample size
                self.age_groups[age_label] = age_subset
                print(f"{age_label}: {len(age_subset):,} participants")
            else:
                print(f"{age_label}: {len(age_subset):,} participants (too small, skipping)")
        
        return self.age_groups
    
    def prepare_age_group_features(self, age_df):
        """
        Prepare features for a specific age group (excluding age)
        """
        # Exclude age-related features since we're stratifying by age
        exclude_cols = ['seqn', 'glucose', 'hba1c', 'age', 'age_group', 'age_activity_interaction']
        feature_cols = [col for col in age_df.columns if col not in exclude_cols]
        
        X = age_df[feature_cols].copy()
        
        # Create targets
        glucose = age_df['glucose']
        hba1c = age_df['hba1c']
        
        y_binary = ((glucose >= 100) | (hba1c >= 5.7)).astype(int)
        y_strict = ((glucose >= 126) | (hba1c >= 6.5)).astype(int)
        
        # Encode categorical variables
        categorical_cols = ['gender', 'race_ethnicity', 'education_level', 'activity_level']
        
        for col in categorical_cols:
            if col in X.columns:
                if X[col].dtype == 'object' or X[col].nunique() < 10:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.median())
        
        return X, y_binary, y_strict
    
    def train_age_stratified_model(self, age_group, age_df, target_type='binary'):
        """
        Train models for a specific age group
        """
        print(f"\n=== Training Models for {age_group} ({target_type.title()} Risk) ===")
        
        # Prepare features and targets
        X, y_binary, y_strict = self.prepare_age_group_features(age_df)
        y_target = y_binary if target_type == 'binary' else y_strict
        
        print(f"Features: {X.shape[1]}")
        print(f"Samples: {len(X)}")
        print(f"Positive cases: {y_target.sum()} ({100*y_target.mean():.1f}%)")
        
        # Check if we have enough positive cases
        if y_target.sum() < 10:
            print(f"Too few positive cases ({y_target.sum()}), skipping...")
            return None
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_target, test_size=0.2, random_state=42, stratify=y_target
            )
        except ValueError:
            # If stratification fails, use regular split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_target, test_size=0.2, random_state=42
            )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        age_results = {}
        
        for model_name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                except ValueError:
                    roc_auc = 0.5
            else:
                roc_auc = 0.5
            
            print(f"  {model_name}:")
            print(f"    Accuracy: {accuracy:.4f}")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall: {recall:.4f}")
            print(f"    F1-score: {f1:.4f}")
            print(f"    ROC AUC: {roc_auc:.4f}")
            
            # Feature importance
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_[0])
            
            age_results[model_name] = {
                'model': model,
                'scaler': scaler,
                'feature_names': X.columns.tolist(),
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc
                },
                'feature_importance': feature_importance,
                'test_data': {
                    'X_test': X_test_scaled,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
            }
        
        return age_results
    
    def analyze_age_specific_features(self, age_group, age_results):
        """
        Analyze which features are most important for each age group
        """
        print(f"\n=== Age-Specific Feature Analysis for {age_group} ===")
        
        if not age_results:
            return None
        
        # Get Random Forest feature importance
        if 'Random Forest' in age_results:
            rf_results = age_results['Random Forest']
            feature_names = rf_results['feature_names']
            importance_scores = rf_results['feature_importance']
            
            if importance_scores is not None:
                # Create feature importance dataframe
                feature_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance_scores
                }).sort_values('Importance', ascending=False)
                
                print(f"Top 10 Features for {age_group}:")
                for idx, row in feature_df.head(10).iterrows():
                    print(f"  {idx+1:2d}. {row['Feature']}: {row['Importance']:.4f}")
                
                return feature_df
        
        return None
    
    def compare_age_group_performance(self):
        """
        Compare performance across age groups
        """
        print("\n" + "="*80)
        print("AGE-STRATIFIED MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        comparison_data = []
        
        for target_type in ['binary', 'strict']:
            print(f"\n{target_type.title()} Risk Classification:")
            
            for age_group in self.age_groups.keys():
                if age_group in self.age_models and target_type in self.age_models[age_group]:
                    age_results = self.age_models[age_group][target_type]
                    
                    if age_results:
                        print(f"\n  {age_group}:")
                        
                        best_model = None
                        best_f1 = 0
                        
                        for model_name, model_data in age_results.items():
                            metrics = model_data['metrics']
                            f1_score = metrics['f1_score']
                            
                            print(f"    {model_name}: F1={f1_score:.4f}, AUC={metrics['roc_auc']:.4f}")
                            
                            if f1_score > best_f1:
                                best_f1 = f1_score
                                best_model = model_name
                            
                            comparison_data.append({
                                'Age_Group': age_group,
                                'Target_Type': target_type,
                                'Model': model_name,
                                'Accuracy': metrics['accuracy'],
                                'F1_Score': metrics['f1_score'],
                                'ROC_AUC': metrics['roc_auc']
                            })
                        
                        print(f"    Best: {best_model} (F1={best_f1:.4f})")
        
        # Save comparison results
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('/Users/aakashsuresh/fairness/blood_glucose_project/age_stratified_comparison.csv', index=False)
        
        return comparison_df
    
    def create_age_stratified_visualizations(self, comparison_df):
        """
        Create visualizations for age-stratified results
        """
        print("\n=== Creating Age-Stratified Visualizations ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. F1-Score by Age Group (Binary Risk)
        binary_data = comparison_df[comparison_df['Target_Type'] == 'binary']
        if not binary_data.empty:
            pivot_binary = binary_data.pivot(index='Age_Group', columns='Model', values='F1_Score')
            pivot_binary.plot(kind='bar', ax=axes[0, 0], rot=45)
            axes[0, 0].set_title('Binary Risk: F1-Score by Age Group')
            axes[0, 0].set_ylabel('F1-Score')
            axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. ROC AUC by Age Group (Binary Risk)
        if not binary_data.empty:
            pivot_auc = binary_data.pivot(index='Age_Group', columns='Model', values='ROC_AUC')
            pivot_auc.plot(kind='bar', ax=axes[0, 1], rot=45)
            axes[0, 1].set_title('Binary Risk: ROC AUC by Age Group')
            axes[0, 1].set_ylabel('ROC AUC')
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. F1-Score by Age Group (Strict Diabetes)
        strict_data = comparison_df[comparison_df['Target_Type'] == 'strict']
        if not strict_data.empty:
            pivot_strict = strict_data.pivot(index='Age_Group', columns='Model', values='F1_Score')
            pivot_strict.plot(kind='bar', ax=axes[1, 0], rot=45)
            axes[1, 0].set_title('Strict Diabetes: F1-Score by Age Group')
            axes[1, 0].set_ylabel('F1-Score')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Model Performance Comparison
        avg_performance = comparison_df.groupby(['Model', 'Target_Type'])['F1_Score'].mean().reset_index()
        pivot_avg = avg_performance.pivot(index='Model', columns='Target_Type', values='F1_Score')
        pivot_avg.plot(kind='bar', ax=axes[1, 1], rot=45)
        axes[1, 1].set_title('Average F1-Score by Model Type')
        axes[1, 1].set_ylabel('Average F1-Score')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/age_stratified_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Age-stratified visualizations saved")
    
    def run_age_stratified_analysis(self):
        """
        Run complete age-stratified analysis
        """
        print("Age-Stratified Models Analysis")
        print("=" * 60)
        
        # Load and stratify data
        self.load_and_stratify_data()
        
        # Train models for each age group and target type
        for age_group, age_df in self.age_groups.items():
            print(f"\n{'='*20} {age_group} {'='*20}")
            
            self.age_models[age_group] = {}
            
            # Train for both target types
            for target_type in ['binary', 'strict']:
                age_results = self.train_age_stratified_model(age_group, age_df, target_type)
                self.age_models[age_group][target_type] = age_results
                
                # Analyze age-specific features
                if age_results:
                    feature_analysis = self.analyze_age_specific_features(age_group, age_results)
        
        # Compare performance across age groups
        comparison_df = self.compare_age_group_performance()
        
        # Create visualizations
        self.create_age_stratified_visualizations(comparison_df)
        
        print("\n" + "="*80)
        print("AGE-STRATIFIED ANALYSIS COMPLETE")
        print("="*80)
        print("Key Insights:")
        print("- Separate models trained for each age group")
        print("- Age-specific feature importance identified")
        print("- Performance comparison across age groups completed")
        print("- Results saved to age_stratified_comparison.csv")
        
        return {
            'age_models': self.age_models,
            'comparison_results': comparison_df,
            'age_groups': self.age_groups
        }

def main():
    """
    Main execution function
    """
    analyzer = AgeStratifiedModels()
    results = analyzer.run_age_stratified_analysis()
    return results

if __name__ == "__main__":
    results = main()
