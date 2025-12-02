#!/usr/bin/env python3
"""
Comprehensive Feature Importance Analysis for Blood Glucose Prediction
Identifies most predictive features and provides interpretable feature names

Author: Generated for fairness project
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.inspection import permutation_importance
import shap
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis for glucose prediction
    """
    
    def __init__(self):
        self.feature_mapping = {
            # Demographic Features
            'age': 'Age (years)',
            'gender': 'Gender (1=Male, 2=Female)',
            'race_ethnicity': 'Race/Ethnicity Category',
            'education_level': 'Education Level',
            
            # Anthropometric Features
            'bmi': 'Body Mass Index (kg/m²)',
            'weight_kg': 'Weight (kg)',
            'height_cm': 'Height (cm)',
            'waist_circumference': 'Waist Circumference (cm)',
            
            # Physical Activity Features
            'total_activity_counts': 'Total Physical Activity Counts (accelerometer)',
            'wear_time_minutes': 'Accelerometer Wear Time (minutes/day)',
            'moderate_activity_minutes': 'Moderate Physical Activity (minutes/day)',
            'vigorous_activity_minutes': 'Vigorous Physical Activity (minutes/day)',
            'light_activity_minutes': 'Light Physical Activity (minutes/day)',
            'sedentary_minutes': 'Sedentary Time (minutes/day)',
            'mvpa_minutes': 'Moderate-to-Vigorous Physical Activity (minutes/day)',
            'mvpa_ratio': 'MVPA Ratio (MVPA/wear time)',
            'sedentary_ratio': 'Sedentary Ratio (sedentary/wear time)',
            'light_activity_ratio': 'Light Activity Ratio (light/wear time)',
            'activity_level': 'Physical Activity Level Category (Low/Moderate/High)',
            'log_total_activity': 'Log-transformed Total Activity Counts',
            
            # Dietary Features
            'DSQTKCAL': 'Total Daily Calories (kcal)',
            'DSQTCARB': 'Total Daily Carbohydrates (g)',
            'DSQTTFAT': 'Total Daily Fat (g)',
            'DSQTSFAT': 'Total Daily Saturated Fat (g)',
            'DSQTMFAT': 'Total Daily Monounsaturated Fat (g)',
            'DSQTPFAT': 'Total Daily Polyunsaturated Fat (g)',
            'DSQTPROT': 'Total Daily Protein (g)',
            'DSQTSODI': 'Total Daily Sodium (mg)',
            'DSQTFIBE': 'Total Daily Fiber (g)',
            'DSQTSUGA': 'Total Daily Sugar (g)',
            
            # Socioeconomic Features
            'household_income': 'Household Income Level',
            'marital_status': 'Marital Status',
            
            # Interaction Features
            'age_activity_interaction': 'Age × Physical Activity Interaction',
            'bmi_mvpa_interaction': 'BMI × MVPA Interaction',
            'gender_sedentary_interaction': 'Gender × Sedentary Time Interaction',
            
            # Target Variables
            'glucose': 'Fasting Glucose (mg/dL)',
            'hba1c': 'Hemoglobin A1c (%)'
        }
        
        self.df = None
        self.X = None
        self.y = None
        self.models = {}
        self.importance_results = {}
        
    def load_improved_dataset(self):
        """
        Load the improved dataset and prepare features
        """
        print("=== Loading Improved Dataset ===")
        
        dataset_path = "/Users/aakashsuresh/fairness/blood_glucose_project/finetuning/improved_dataset.csv"
        self.df = pd.read_csv(dataset_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Available columns: {list(self.df.columns)}")
        
        # Prepare features and targets
        exclude_cols = ['seqn', 'glucose', 'hba1c']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        # Remove features with no variance
        valid_features = []
        for col in feature_cols:
            if self.df[col].notna().sum() > 0 and self.df[col].var() > 0:
                valid_features.append(col)
        
        self.X = self.df[valid_features]
        self.y = self.df[['glucose', 'hba1c']]
        
        print(f"Valid features for analysis: {len(valid_features)}")
        print(f"Features: {valid_features}")
        
        return self.X, self.y
    
    def create_feature_description_table(self):
        """
        Create a comprehensive table of all input features with descriptions
        """
        print("\n=== Input Features Description Table ===")
        
        feature_descriptions = []
        
        for feature in self.X.columns:
            description = self.feature_mapping.get(feature, f"Unknown feature: {feature}")
            
            # Get basic statistics
            if self.X[feature].dtype in ['float64', 'int64']:
                mean_val = self.X[feature].mean()
                std_val = self.X[feature].std()
                min_val = self.X[feature].min()
                max_val = self.X[feature].max()
                missing_pct = (self.X[feature].isna().sum() / len(self.X)) * 100
                
                stats_str = f"Mean: {mean_val:.2f}, Std: {std_val:.2f}, Range: [{min_val:.1f}, {max_val:.1f}]"
                if missing_pct > 0:
                    stats_str += f", Missing: {missing_pct:.1f}%"
            else:
                unique_vals = self.X[feature].nunique()
                missing_pct = (self.X[feature].isna().sum() / len(self.X)) * 100
                stats_str = f"Categories: {unique_vals}"
                if missing_pct > 0:
                    stats_str += f", Missing: {missing_pct:.1f}%"
            
            feature_descriptions.append({
                'Variable_Name': feature,
                'Descriptive_Name': description,
                'Data_Type': str(self.X[feature].dtype),
                'Statistics': stats_str
            })
        
        feature_df = pd.DataFrame(feature_descriptions)
        
        print("\nCOMPLETE INPUT FEATURES LIST:")
        print("=" * 80)
        for idx, row in feature_df.iterrows():
            print(f"{idx+1:2d}. {row['Descriptive_Name']}")
            print(f"    Variable: {row['Variable_Name']}")
            print(f"    Type: {row['Data_Type']}")
            print(f"    Stats: {row['Statistics']}")
            print()
        
        # Save to CSV
        feature_df.to_csv('/Users/aakashsuresh/fairness/blood_glucose_project/input_features_description.csv', index=False)
        print(f"Feature descriptions saved to 'input_features_description.csv'")
        
        return feature_df
    
    def calculate_correlation_importance(self):
        """
        Calculate correlation-based feature importance
        """
        print("\n=== Correlation-Based Feature Importance ===")
        
        # Calculate correlations with glucose and HbA1c
        glucose_corr = []
        hba1c_corr = []
        
        for feature in self.X.columns:
            if self.X[feature].dtype in ['float64', 'int64']:
                glucose_corr_val = self.X[feature].corr(self.y['glucose'])
                hba1c_corr_val = self.X[feature].corr(self.y['hba1c'])
            else:
                glucose_corr_val = 0
                hba1c_corr_val = 0
            
            glucose_corr.append(glucose_corr_val)
            hba1c_corr.append(hba1c_corr_val)
        
        correlation_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Descriptive_Name': [self.feature_mapping.get(f, f) for f in self.X.columns],
            'Glucose_Correlation': glucose_corr,
            'HbA1c_Correlation': hba1c_corr,
            'Abs_Glucose_Corr': [abs(x) if not pd.isna(x) else 0 for x in glucose_corr],
            'Abs_HbA1c_Corr': [abs(x) if not pd.isna(x) else 0 for x in hba1c_corr]
        })
        
        # Sort by absolute glucose correlation
        correlation_df = correlation_df.sort_values('Abs_Glucose_Corr', ascending=False)
        
        print("Top 10 Features by Absolute Correlation with Glucose:")
        top_glucose = correlation_df.head(10)
        for idx, row in top_glucose.iterrows():
            print(f"{row.name+1:2d}. {row['Descriptive_Name']}")
            print(f"    Glucose Correlation: {row['Glucose_Correlation']:.4f}")
            print(f"    HbA1c Correlation: {row['HbA1c_Correlation']:.4f}")
        
        self.importance_results['correlation'] = correlation_df
        return correlation_df
    
    def train_models_for_importance(self):
        """
        Train models for feature importance analysis
        """
        print("\n=== Training Models for Feature Importance ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models_to_train = {
            'Random Forest': MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42)),
            'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, random_state=42)),
            'Ridge Regression': MultiOutputRegressor(Ridge(alpha=1.0))
        }
        
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            self.models[name] = {
                'model': model,
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': self.X.columns.tolist()
            }
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def calculate_tree_based_importance(self):
        """
        Calculate tree-based feature importance
        """
        print("\n=== Tree-Based Feature Importance ===")
        
        tree_importance_results = {}
        
        for model_name in ['Random Forest', 'Gradient Boosting']:
            if model_name in self.models:
                model_data = self.models[model_name]
                model = model_data['model']
                
                # Get feature importance for glucose prediction (first estimator)
                if hasattr(model.estimators_[0], 'feature_importances_'):
                    importance_scores = model.estimators_[0].feature_importances_
                    
                    importance_df = pd.DataFrame({
                        'Feature': model_data['feature_names'],
                        'Descriptive_Name': [self.feature_mapping.get(f, f) for f in model_data['feature_names']],
                        'Importance': importance_scores
                    }).sort_values('Importance', ascending=False)
                    
                    tree_importance_results[model_name] = importance_df
                    
                    print(f"\nTop 10 Features by {model_name} Importance (Glucose Prediction):")
                    for idx, row in importance_df.head(10).iterrows():
                        print(f"{idx+1:2d}. {row['Descriptive_Name']}: {row['Importance']:.4f}")
        
        self.importance_results['tree_based'] = tree_importance_results
        return tree_importance_results
    
    def calculate_permutation_importance(self):
        """
        Calculate permutation-based feature importance
        """
        print("\n=== Permutation-Based Feature Importance ===")
        
        permutation_results = {}
        
        for model_name in ['Random Forest', 'Gradient Boosting']:
            if model_name in self.models:
                model_data = self.models[model_name]
                model = model_data['model']
                
                print(f"Calculating permutation importance for {model_name}...")
                
                # Calculate permutation importance for glucose prediction
                # Create a single-output model for glucose prediction
                glucose_model = model.estimators_[0]  # First estimator is for glucose
                perm_importance = permutation_importance(
                    glucose_model, model_data['X_test'], model_data['y_test'].iloc[:, 0],  # Glucose only
                    n_repeats=10, random_state=42, scoring='neg_mean_squared_error'
                )
                
                importance_df = pd.DataFrame({
                    'Feature': model_data['feature_names'],
                    'Descriptive_Name': [self.feature_mapping.get(f, f) for f in model_data['feature_names']],
                    'Importance_Mean': perm_importance.importances_mean,
                    'Importance_Std': perm_importance.importances_std
                }).sort_values('Importance_Mean', ascending=False)
                
                permutation_results[model_name] = importance_df
                
                print(f"\nTop 10 Features by {model_name} Permutation Importance:")
                for idx, row in importance_df.head(10).iterrows():
                    print(f"{idx+1:2d}. {row['Descriptive_Name']}: {row['Importance_Mean']:.4f} ± {row['Importance_Std']:.4f}")
        
        self.importance_results['permutation'] = permutation_results
        return permutation_results
    
    def calculate_shap_importance(self):
        """
        Calculate SHAP-based feature importance
        """
        print("\n=== SHAP-Based Feature Importance ===")
        
        shap_results = {}
        
        # Use Random Forest for SHAP analysis
        if 'Random Forest' in self.models:
            model_data = self.models['Random Forest']
            rf_model = model_data['model'].estimators_[0]  # Glucose prediction model
            
            print("Calculating SHAP values for Random Forest...")
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(rf_model)
            
            # Calculate SHAP values on a subset for efficiency
            sample_size = min(500, len(model_data['X_test']))
            X_sample = model_data['X_test'][:sample_size]
            shap_values = explainer.shap_values(X_sample)
            
            # Calculate mean absolute SHAP values
            mean_shap_values = np.abs(shap_values).mean(0)
            
            shap_df = pd.DataFrame({
                'Feature': model_data['feature_names'],
                'Descriptive_Name': [self.feature_mapping.get(f, f) for f in model_data['feature_names']],
                'SHAP_Importance': mean_shap_values
            }).sort_values('SHAP_Importance', ascending=False)
            
            shap_results['Random Forest'] = shap_df
            
            print("\nTop 10 Features by SHAP Importance:")
            for idx, row in shap_df.head(10).iterrows():
                print(f"{idx+1:2d}. {row['Descriptive_Name']}: {row['SHAP_Importance']:.4f}")
            
            self.importance_results['shap'] = shap_results
        
        return shap_results
    
    def create_comprehensive_importance_ranking(self):
        """
        Create comprehensive feature importance ranking combining all methods
        """
        print("\n=== Comprehensive Feature Importance Ranking ===")
        
        # Collect all importance scores
        all_features = self.X.columns.tolist()
        importance_summary = pd.DataFrame({
            'Feature': all_features,
            'Descriptive_Name': [self.feature_mapping.get(f, f) for f in all_features]
        })
        
        # Add correlation importance
        if 'correlation' in self.importance_results:
            corr_df = self.importance_results['correlation']
            importance_summary = importance_summary.merge(
                corr_df[['Feature', 'Abs_Glucose_Corr']], on='Feature', how='left'
            )
        
        # Add tree-based importance (Random Forest)
        if 'tree_based' in self.importance_results and 'Random Forest' in self.importance_results['tree_based']:
            rf_df = self.importance_results['tree_based']['Random Forest']
            importance_summary = importance_summary.merge(
                rf_df[['Feature', 'Importance']].rename(columns={'Importance': 'RF_Importance'}),
                on='Feature', how='left'
            )
        
        # Add permutation importance (Random Forest)
        if 'permutation' in self.importance_results and 'Random Forest' in self.importance_results['permutation']:
            perm_df = self.importance_results['permutation']['Random Forest']
            importance_summary = importance_summary.merge(
                perm_df[['Feature', 'Importance_Mean']].rename(columns={'Importance_Mean': 'Perm_Importance'}),
                on='Feature', how='left'
            )
        
        # Add SHAP importance
        if 'shap' in self.importance_results and 'Random Forest' in self.importance_results['shap']:
            shap_df = self.importance_results['shap']['Random Forest']
            importance_summary = importance_summary.merge(
                shap_df[['Feature', 'SHAP_Importance']], on='Feature', how='left'
            )
        
        # Calculate composite importance score (average of normalized scores)
        importance_cols = [col for col in importance_summary.columns if col.endswith('_Importance') or col.endswith('_Corr')]
        
        # Normalize each importance measure to 0-1 scale
        for col in importance_cols:
            if col in importance_summary.columns:
                max_val = importance_summary[col].max()
                if max_val > 0:
                    importance_summary[f'{col}_Normalized'] = importance_summary[col] / max_val
                else:
                    importance_summary[f'{col}_Normalized'] = 0
        
        # Calculate composite score
        normalized_cols = [col for col in importance_summary.columns if col.endswith('_Normalized')]
        if normalized_cols:
            importance_summary['Composite_Score'] = importance_summary[normalized_cols].mean(axis=1, skipna=True)
        else:
            importance_summary['Composite_Score'] = 0
        
        # Sort by composite score
        importance_summary = importance_summary.sort_values('Composite_Score', ascending=False)
        
        print("\nTOP 15 MOST PREDICTIVE FEATURES (Composite Ranking):")
        print("=" * 80)
        for idx, row in importance_summary.head(15).iterrows():
            print(f"{idx+1:2d}. {row['Descriptive_Name']}")
            print(f"    Composite Score: {row['Composite_Score']:.4f}")
            if 'Abs_Glucose_Corr' in row and not pd.isna(row['Abs_Glucose_Corr']):
                print(f"    Correlation: {row['Abs_Glucose_Corr']:.4f}")
            if 'RF_Importance' in row and not pd.isna(row['RF_Importance']):
                print(f"    RF Importance: {row['RF_Importance']:.4f}")
            if 'SHAP_Importance' in row and not pd.isna(row['SHAP_Importance']):
                print(f"    SHAP Importance: {row['SHAP_Importance']:.4f}")
            print()
        
        # Save comprehensive results
        importance_summary.to_csv('/Users/aakashsuresh/fairness/blood_glucose_project/comprehensive_feature_importance.csv', index=False)
        print("Comprehensive importance results saved to 'comprehensive_feature_importance.csv'")
        
        return importance_summary
    
    def create_importance_visualizations(self, importance_summary):
        """
        Create visualizations for feature importance
        """
        print("\n=== Creating Feature Importance Visualizations ===")
        
        # Top 15 features visualization
        plt.figure(figsize=(14, 10))
        top_15 = importance_summary.head(15)
        
        plt.barh(range(len(top_15)), top_15['Composite_Score'], color='skyblue', alpha=0.8)
        plt.yticks(range(len(top_15)), top_15['Descriptive_Name'])
        plt.xlabel('Composite Importance Score')
        plt.title('Top 15 Most Predictive Features for Blood Glucose\n(Composite Score from Multiple Methods)')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(top_15['Composite_Score']):
            plt.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/top_features_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Comparison of different importance methods
        if len([col for col in importance_summary.columns if col.endswith('_Importance') or col.endswith('_Corr')]) > 1:
            plt.figure(figsize=(16, 8))
            
            top_10 = importance_summary.head(10)
            methods = []
            
            if 'Abs_Glucose_Corr' in top_10.columns:
                methods.append(('Abs_Glucose_Corr', 'Correlation'))
            if 'RF_Importance' in top_10.columns:
                methods.append(('RF_Importance', 'Random Forest'))
            if 'SHAP_Importance' in top_10.columns:
                methods.append(('SHAP_Importance', 'SHAP'))
            
            if len(methods) > 1:
                x = np.arange(len(top_10))
                width = 0.25
                
                for i, (col, label) in enumerate(methods):
                    values = top_10[col].fillna(0)
                    plt.bar(x + i*width, values, width, label=label, alpha=0.8)
                
                plt.xlabel('Features (Top 10)')
                plt.ylabel('Importance Score')
                plt.title('Feature Importance Comparison Across Methods')
                plt.xticks(x + width, [name[:30] + '...' if len(name) > 30 else name 
                                     for name in top_10['Descriptive_Name']], rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()
                plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/importance_methods_comparison.png', 
                           dpi=300, bbox_inches='tight')
                plt.show()
        
        print("Visualizations saved as 'top_features_importance.png' and 'importance_methods_comparison.png'")
    
    def run_complete_importance_analysis(self):
        """
        Run complete feature importance analysis
        """
        print("Comprehensive Feature Importance Analysis for Blood Glucose Prediction")
        print("=" * 80)
        
        # Load data
        self.load_improved_dataset()
        
        # Create feature description table
        feature_descriptions = self.create_feature_description_table()
        
        # Calculate different types of importance
        correlation_importance = self.calculate_correlation_importance()
        
        # Train models
        self.train_models_for_importance()
        
        # Calculate model-based importance
        tree_importance = self.calculate_tree_based_importance()
        permutation_importance = self.calculate_permutation_importance()
        shap_importance = self.calculate_shap_importance()
        
        # Create comprehensive ranking
        comprehensive_ranking = self.create_comprehensive_importance_ranking()
        
        # Create visualizations
        self.create_importance_visualizations(comprehensive_ranking)
        
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE ANALYSIS COMPLETE")
        print("=" * 80)
        
        return {
            'feature_descriptions': feature_descriptions,
            'comprehensive_ranking': comprehensive_ranking,
            'correlation_importance': correlation_importance,
            'tree_importance': tree_importance,
            'permutation_importance': permutation_importance,
            'shap_importance': shap_importance
        }

def main():
    """
    Main execution function
    """
    analyzer = FeatureImportanceAnalyzer()
    results = analyzer.run_complete_importance_analysis()
    return results

if __name__ == "__main__":
    results = main()
