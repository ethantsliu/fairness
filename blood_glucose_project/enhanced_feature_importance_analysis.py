#!/usr/bin/env python3
"""
Enhanced Feature Importance Analysis with Complete Dataset
Uses the fixed integrated dataset with all 20 lifestyle features

Author: Generated for fairness project
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureImportanceAnalyzer:
    """
    Enhanced feature importance analysis with complete lifestyle dataset
    """
    
    def __init__(self, dataset_path="/Users/aakashsuresh/fairness/blood_glucose_project/fixed_data/integrated_nhanes_2011_2014.csv"):
        self.dataset_path = dataset_path
        self.feature_mapping = {
            # Demographics
            'age': 'Age (years)',
            'gender': 'Gender (1=Male, 2=Female)',
            'race_ethnicity': 'Race/Ethnicity Category',
            'education_level': 'Education Level',
            'bmi': 'Body Mass Index (kg/m²)',
            
            # Physical Activity
            'total_activity_counts': 'Total Physical Activity (counts/day)',
            'wear_time_minutes': 'Accelerometer Wear Time (min/day)',
            'moderate_activity_minutes': 'Moderate Physical Activity (min/day)',
            'vigorous_activity_minutes': 'Vigorous Physical Activity (min/day)',
            'light_activity_minutes': 'Light Physical Activity (min/day)',
            'sedentary_minutes': 'Sedentary Time (min/day)',
            'mvpa_minutes': 'Moderate-to-Vigorous Activity (min/day)',
            'mvpa_ratio': 'MVPA Ratio (% of wear time)',
            'sedentary_ratio': 'Sedentary Ratio (% of wear time)',
            'light_activity_ratio': 'Light Activity Ratio (% of wear time)',
            'activity_level': 'Physical Activity Level (categorical)',
            'log_total_activity': 'Log-transformed Total Activity',
            
            # Interactions
            'age_activity_interaction': 'Age × Physical Activity Interaction',
            'bmi_mvpa_interaction': 'BMI × MVPA Interaction',
            'gender_sedentary_interaction': 'Gender × Sedentary Time Interaction',
            
            # Targets
            'glucose': 'Fasting Glucose (mg/dL)',
            'hba1c': 'Hemoglobin A1c (%)'
        }
        
        self.df = None
        self.X = None
        self.y = None
        self.models = {}
        self.importance_results = {}
        
    def load_enhanced_dataset(self):
        """
        Load the enhanced integrated dataset
        """
        print("=== Loading Enhanced Integrated Dataset ===")
        
        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Participants: {len(self.df):,}")
        
        # Prepare features and targets
        exclude_cols = ['seqn', 'glucose', 'hba1c']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        self.X = self.df[feature_cols]
        self.y = self.df[['glucose', 'hba1c']]
        
        print(f"Features available: {len(feature_cols)}")
        print(f"Features: {feature_cols}")
        
        # Check for any remaining zero variance features
        zero_var_features = []
        for col in feature_cols:
            if self.X[col].var() == 0:
                zero_var_features.append(col)
        
        if zero_var_features:
            print(f"Warning: Zero variance features found: {zero_var_features}")
            self.X = self.X.drop(columns=zero_var_features)
        
        print(f"Final feature count: {len(self.X.columns)}")
        
        return self.X, self.y
    
    def analyze_dataset_quality(self):
        """
        Analyze the quality of the enhanced dataset
        """
        print("\n=== Dataset Quality Analysis ===")
        
        # Basic statistics
        print(f"Dataset size: {len(self.df):,} participants")
        print(f"Features: {len(self.X.columns)} lifestyle features")
        
        # Glucose and HbA1c statistics
        print(f"\nGlucose statistics:")
        print(f"  Mean: {self.y['glucose'].mean():.1f} mg/dL")
        print(f"  Median: {self.y['glucose'].median():.1f} mg/dL")
        print(f"  Range: {self.y['glucose'].min():.1f} - {self.y['glucose'].max():.1f} mg/dL")
        
        print(f"\nHbA1c statistics:")
        print(f"  Mean: {self.y['hba1c'].mean():.2f}%")
        print(f"  Median: {self.y['hba1c'].median():.2f}%")
        print(f"  Range: {self.y['hba1c'].min():.2f} - {self.y['hba1c'].max():.2f}%")
        
        # Feature categories
        demographic_features = [f for f in self.X.columns if f in ['age', 'gender', 'race_ethnicity', 'education_level', 'bmi']]
        activity_features = [f for f in self.X.columns if any(x in f for x in ['activity', 'mvpa', 'sedentary', 'wear', 'light', 'vigorous', 'moderate'])]
        interaction_features = [f for f in self.X.columns if 'interaction' in f]
        
        print(f"\nFeature categories:")
        print(f"  Demographics: {len(demographic_features)}")
        print(f"  Physical Activity: {len(activity_features)}")
        print(f"  Interactions: {len(interaction_features)}")
        
        return {
            'demographic_features': demographic_features,
            'activity_features': activity_features,
            'interaction_features': interaction_features
        }
    
    def calculate_enhanced_correlations(self):
        """
        Calculate correlations with enhanced feature set
        """
        print("\n=== Enhanced Correlation Analysis ===")
        
        correlations = []
        for feature in self.X.columns:
            glucose_corr = self.X[feature].corr(self.y['glucose'])
            hba1c_corr = self.X[feature].corr(self.y['hba1c'])
            
            correlations.append({
                'Feature': feature,
                'Descriptive_Name': self.feature_mapping.get(feature, feature),
                'Glucose_Correlation': glucose_corr,
                'HbA1c_Correlation': hba1c_corr,
                'Abs_Glucose_Corr': abs(glucose_corr) if not pd.isna(glucose_corr) else 0,
                'Abs_HbA1c_Corr': abs(hba1c_corr) if not pd.isna(hba1c_corr) else 0
            })
        
        correlation_df = pd.DataFrame(correlations)
        correlation_df = correlation_df.sort_values('Abs_Glucose_Corr', ascending=False)
        
        print("Top 10 Features by Glucose Correlation:")
        for idx, row in correlation_df.head(10).iterrows():
            print(f"{idx+1:2d}. {row['Descriptive_Name']}")
            print(f"    Glucose: {row['Glucose_Correlation']:.4f}")
            print(f"    HbA1c: {row['HbA1c_Correlation']:.4f}")
        
        self.importance_results['correlation'] = correlation_df
        return correlation_df
    
    def train_enhanced_models(self):
        """
        Train models with enhanced feature set
        """
        print("\n=== Training Enhanced Models ===")
        
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
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            glucose_mae = mean_absolute_error(y_test.iloc[:, 0], y_pred[:, 0])
            hba1c_mae = mean_absolute_error(y_test.iloc[:, 1], y_pred[:, 1])
            glucose_r2 = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
            hba1c_r2 = r2_score(y_test.iloc[:, 1], y_pred[:, 1])
            
            print(f"  Glucose MAE: {glucose_mae:.3f} mg/dL, R²: {glucose_r2:.3f}")
            print(f"  HbA1c MAE: {hba1c_mae:.3f}%, R²: {hba1c_r2:.3f}")
            
            self.models[name] = {
                'model': model,
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': self.X.columns.tolist(),
                'performance': {
                    'glucose_mae': glucose_mae,
                    'hba1c_mae': hba1c_mae,
                    'glucose_r2': glucose_r2,
                    'hba1c_r2': hba1c_r2
                }
            }
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def calculate_enhanced_tree_importance(self):
        """
        Calculate tree-based importance with enhanced features
        """
        print("\n=== Enhanced Tree-Based Feature Importance ===")
        
        tree_results = {}
        
        for model_name in ['Random Forest', 'Gradient Boosting']:
            if model_name in self.models:
                model_data = self.models[model_name]
                model = model_data['model']
                
                # Get glucose prediction importance
                glucose_importance = model.estimators_[0].feature_importances_
                
                importance_df = pd.DataFrame({
                    'Feature': model_data['feature_names'],
                    'Descriptive_Name': [self.feature_mapping.get(f, f) for f in model_data['feature_names']],
                    'Importance': glucose_importance
                }).sort_values('Importance', ascending=False)
                
                tree_results[model_name] = importance_df
                
                print(f"\nTop 10 Features by {model_name} Importance:")
                for idx, row in importance_df.head(10).iterrows():
                    print(f"{idx+1:2d}. {row['Descriptive_Name']}: {row['Importance']:.4f}")
        
        self.importance_results['tree_based'] = tree_results
        return tree_results
    
    def calculate_enhanced_shap_importance(self):
        """
        Calculate SHAP importance with enhanced features
        """
        print("\n=== Enhanced SHAP Analysis ===")
        
        if 'Random Forest' in self.models:
            model_data = self.models['Random Forest']
            rf_model = model_data['model'].estimators_[0]  # Glucose prediction
            
            print("Calculating SHAP values...")
            
            # Create explainer
            explainer = shap.TreeExplainer(rf_model)
            
            # Calculate SHAP values on sample
            sample_size = min(500, len(model_data['X_test']))
            X_sample = model_data['X_test'][:sample_size]
            shap_values = explainer.shap_values(X_sample)
            
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(0)
            
            shap_df = pd.DataFrame({
                'Feature': model_data['feature_names'],
                'Descriptive_Name': [self.feature_mapping.get(f, f) for f in model_data['feature_names']],
                'SHAP_Importance': mean_shap
            }).sort_values('SHAP_Importance', ascending=False)
            
            print("\nTop 10 Features by SHAP Importance:")
            for idx, row in shap_df.head(10).iterrows():
                print(f"{idx+1:2d}. {row['Descriptive_Name']}: {row['SHAP_Importance']:.4f}")
            
            self.importance_results['shap'] = {'Random Forest': shap_df}
            return shap_df
        
        return None
    
    def create_comprehensive_ranking(self):
        """
        Create comprehensive feature ranking with enhanced dataset
        """
        print("\n=== Comprehensive Enhanced Feature Ranking ===")
        
        # Start with all features
        all_features = self.X.columns.tolist()
        ranking_df = pd.DataFrame({
            'Feature': all_features,
            'Descriptive_Name': [self.feature_mapping.get(f, f) for f in all_features]
        })
        
        # Add correlation scores
        if 'correlation' in self.importance_results:
            corr_df = self.importance_results['correlation']
            ranking_df = ranking_df.merge(
                corr_df[['Feature', 'Abs_Glucose_Corr']], on='Feature', how='left'
            )
        
        # Add Random Forest importance
        if 'tree_based' in self.importance_results and 'Random Forest' in self.importance_results['tree_based']:
            rf_df = self.importance_results['tree_based']['Random Forest']
            ranking_df = ranking_df.merge(
                rf_df[['Feature', 'Importance']].rename(columns={'Importance': 'RF_Importance'}),
                on='Feature', how='left'
            )
        
        # Add SHAP importance
        if 'shap' in self.importance_results and 'Random Forest' in self.importance_results['shap']:
            shap_df = self.importance_results['shap']['Random Forest']
            ranking_df = ranking_df.merge(
                shap_df[['Feature', 'SHAP_Importance']], on='Feature', how='left'
            )
        
        # Calculate composite score
        importance_cols = [col for col in ranking_df.columns if col.endswith('_Importance') or col.endswith('_Corr')]
        
        # Normalize each importance measure
        for col in importance_cols:
            if col in ranking_df.columns:
                max_val = ranking_df[col].max()
                if max_val > 0:
                    ranking_df[f'{col}_Normalized'] = ranking_df[col] / max_val
                else:
                    ranking_df[f'{col}_Normalized'] = 0
        
        # Calculate composite score
        normalized_cols = [col for col in ranking_df.columns if col.endswith('_Normalized')]
        if normalized_cols:
            ranking_df['Composite_Score'] = ranking_df[normalized_cols].mean(axis=1, skipna=True)
        else:
            ranking_df['Composite_Score'] = 0
        
        # Sort by composite score
        ranking_df = ranking_df.sort_values('Composite_Score', ascending=False)
        
        print("\nTOP 15 MOST PREDICTIVE FEATURES (Enhanced Dataset):")
        print("=" * 80)
        for idx, row in ranking_df.head(15).iterrows():
            print(f"{idx+1:2d}. {row['Descriptive_Name']}")
            print(f"    Composite Score: {row['Composite_Score']:.4f}")
            if 'Abs_Glucose_Corr' in row and not pd.isna(row['Abs_Glucose_Corr']):
                print(f"    Correlation: {row['Abs_Glucose_Corr']:.4f}")
            if 'RF_Importance' in row and not pd.isna(row['RF_Importance']):
                print(f"    RF Importance: {row['RF_Importance']:.4f}")
            if 'SHAP_Importance' in row and not pd.isna(row['SHAP_Importance']):
                print(f"    SHAP Importance: {row['SHAP_Importance']:.4f}")
            print()
        
        # Save results
        output_path = "/Users/aakashsuresh/fairness/blood_glucose_project/enhanced_feature_importance.csv"
        ranking_df.to_csv(output_path, index=False)
        print(f"Enhanced feature importance saved to: {output_path}")
        
        return ranking_df
    
    def create_enhanced_visualizations(self, ranking_df):
        """
        Create visualizations for enhanced feature importance
        """
        print("\n=== Creating Enhanced Visualizations ===")
        
        # Top 15 features
        plt.figure(figsize=(14, 10))
        top_15 = ranking_df.head(15)
        
        plt.barh(range(len(top_15)), top_15['Composite_Score'], color='skyblue', alpha=0.8)
        plt.yticks(range(len(top_15)), [name[:50] + '...' if len(name) > 50 else name for name in top_15['Descriptive_Name']])
        plt.xlabel('Composite Importance Score')
        plt.title('Top 15 Most Predictive Features for Fasting Glucose\n(Enhanced Dataset with Complete Lifestyle Features)')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(top_15['Composite_Score']):
            plt.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/enhanced_top_features.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature categories comparison
        demographic_features = [f for f in ranking_df['Feature'] if f in ['age', 'gender', 'race_ethnicity', 'education_level', 'bmi']]
        activity_features = [f for f in ranking_df['Feature'] if any(x in f for x in ['activity', 'mvpa', 'sedentary', 'wear', 'light', 'vigorous', 'moderate'])]
        interaction_features = [f for f in ranking_df['Feature'] if 'interaction' in f]
        
        category_scores = {
            'Demographics': ranking_df[ranking_df['Feature'].isin(demographic_features)]['Composite_Score'].mean(),
            'Physical Activity': ranking_df[ranking_df['Feature'].isin(activity_features)]['Composite_Score'].mean(),
            'Interactions': ranking_df[ranking_df['Feature'].isin(interaction_features)]['Composite_Score'].mean()
        }
        
        plt.figure(figsize=(10, 6))
        categories = list(category_scores.keys())
        scores = list(category_scores.values())
        
        plt.bar(categories, scores, color=['lightblue', 'lightgreen', 'orange'], alpha=0.8)
        plt.ylabel('Average Composite Importance Score')
        plt.title('Feature Importance by Category\n(Enhanced Dataset)')
        plt.xticks(rotation=45)
        
        for i, v in enumerate(scores):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/enhanced_category_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Enhanced visualizations saved")
    
    def run_enhanced_analysis(self):
        """
        Run complete enhanced feature importance analysis
        """
        print("Enhanced Feature Importance Analysis with Complete Lifestyle Dataset")
        print("=" * 80)
        
        # Load enhanced dataset
        self.load_enhanced_dataset()
        
        # Analyze dataset quality
        dataset_analysis = self.analyze_dataset_quality()
        
        # Calculate correlations
        correlation_results = self.calculate_enhanced_correlations()
        
        # Train models
        self.train_enhanced_models()
        
        # Calculate importance measures
        tree_importance = self.calculate_enhanced_tree_importance()
        shap_importance = self.calculate_enhanced_shap_importance()
        
        # Create comprehensive ranking
        comprehensive_ranking = self.create_comprehensive_ranking()
        
        # Create visualizations
        self.create_enhanced_visualizations(comprehensive_ranking)
        
        print("\n" + "=" * 80)
        print("ENHANCED FEATURE IMPORTANCE ANALYSIS COMPLETE")
        print("=" * 80)
        
        # Performance comparison
        print("\nModel Performance Summary (Enhanced Dataset):")
        for model_name, model_data in self.models.items():
            perf = model_data['performance']
            print(f"{model_name}:")
            print(f"  Glucose MAE: {perf['glucose_mae']:.3f} mg/dL (R²: {perf['glucose_r2']:.3f})")
            print(f"  HbA1c MAE: {perf['hba1c_mae']:.3f}% (R²: {perf['hba1c_r2']:.3f})")
        
        return {
            'dataset_analysis': dataset_analysis,
            'comprehensive_ranking': comprehensive_ranking,
            'model_performance': {name: data['performance'] for name, data in self.models.items()},
            'correlation_results': correlation_results
        }

def main():
    """
    Main execution function
    """
    analyzer = EnhancedFeatureImportanceAnalyzer()
    results = analyzer.run_enhanced_analysis()
    return results

if __name__ == "__main__":
    results = main()
