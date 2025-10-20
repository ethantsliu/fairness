#!/usr/bin/env python3
"""
NHANES Lifestyle-Based Blood Glucose and HbA1c Analysis Pipeline
Multi-output regression using only lifestyle, dietary, and demographic features

This version removes lab value proxies to create a clinically meaningful model
that could be used for screening when lab values are not available.

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

class LifestyleGlucoseAnalyzer:
    """
    Lifestyle-focused analyzer for NHANES glucose and HbA1c prediction with fairness evaluation
    Uses only demographic, physical activity, and dietary features - no lab value proxies
    """
    
    def __init__(self, 
                 lab_data_dir="/Users/aakashsuresh/fairness/processed_data_nhanes_lab/",
                 lifestyle_data_dir="/Users/aakashsuresh/fairness/processed_data_new/"):
        self.lab_data_dir = lab_data_dir
        self.lifestyle_data_dir = lifestyle_data_dir
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler_X = None
        self.model = None
        self.baseline_model = None
        self.feature_names = None
        
    def load_glucose_targets(self):
        """
        Load glucose and HbA1c targets from lab data
        """
        print("=== Loading Glucose and HbA1c Targets ===")
        
        # Load glucose data
        glucose_file = os.path.join(self.lab_data_dir, "fasting_glucose_processed.csv")
        hba1c_file = os.path.join(self.lab_data_dir, "glycohemoglobin_processed.csv")
        
        if os.path.exists(glucose_file) and os.path.exists(hba1c_file):
            glucose_df = pd.read_csv(glucose_file)[['seqn', 'lbxglu']]
            hba1c_df = pd.read_csv(hba1c_file)[['seqn', 'lbxgh']]
            
            # Merge targets
            targets_df = glucose_df.merge(hba1c_df, on='seqn', how='inner')
            targets_df.columns = ['seqn', 'glucose', 'hba1c']
            
            print(f"Loaded targets for {len(targets_df)} participants")
            return targets_df
        else:
            raise FileNotFoundError("Glucose or HbA1c target files not found")
    
    def load_demographics(self):
        """
        Load demographic data including age, gender, race/ethnicity, BMI
        """
        print("Loading demographic data...")
        
        demo_path = os.path.join(self.lab_data_dir, "P_DEMO.xpt")
        if os.path.exists(demo_path):
            demo = pd.read_sas(demo_path, format="xport")
            
            # Select relevant demographic columns
            demo_cols = ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "BMXBMI"]
            available_cols = [col for col in demo_cols if col in demo.columns]
            demo = demo[available_cols]
            
            demo.columns = demo.columns.str.lower()
            demo.rename(columns={
                'seqn': 'seqn',
                'ridageyr': 'age', 
                'riagendr': 'gender',
                'ridreth3': 'race_ethnicity',
                'bmxbmi': 'bmi'
            }, inplace=True)
            
            print(f"Loaded demographics for {len(demo)} participants")
            return demo
        else:
            print("Demographics file not found, using minimal demo data")
            return None
    
    def load_physical_activity_data(self):
        """
        Load and process physical activity/accelerometry data
        """
        print("Loading physical activity data...")
        
        acc_file = os.path.join(self.lifestyle_data_dir, "nhanes_combined_acc.csv")
        if os.path.exists(acc_file):
            acc_df = pd.read_csv(acc_file)
            
            # Clean column names and convert SEQN
            acc_df.columns = acc_df.columns.str.upper()
            if 'SEQN' in acc_df.columns:
                acc_df['seqn'] = acc_df['SEQN']
            
            # Select meaningful activity features and aggregate by participant
            activity_features = []
            for col in acc_df.columns:
                if any(x in col.lower() for x in ['pax', 'activity', 'step', 'mvpa', 'sed']):
                    if acc_df[col].dtype in ['float64', 'int64']:
                        activity_features.append(col)
            
            if activity_features:
                # Aggregate activity data by participant (mean across days)
                agg_dict = {col: 'mean' for col in activity_features}
                acc_summary = acc_df.groupby('seqn').agg(agg_dict).reset_index()
                
                # Create meaningful activity variables
                acc_summary['total_activity_counts'] = acc_summary.get('PAXAISMD', 0)
                acc_summary['moderate_vigorous_minutes'] = acc_summary.get('PAXMVMD', 0) 
                acc_summary['sedentary_minutes'] = acc_summary.get('PAXSMD', 0)
                acc_summary['wear_time_minutes'] = acc_summary.get('PAXTMD', 0)
                
                # Calculate activity ratios
                acc_summary['mvpa_ratio'] = (acc_summary['moderate_vigorous_minutes'] / 
                                           (acc_summary['wear_time_minutes'] + 1))
                acc_summary['sedentary_ratio'] = (acc_summary['sedentary_minutes'] / 
                                                (acc_summary['wear_time_minutes'] + 1))
                
                activity_cols = ['seqn', 'total_activity_counts', 'moderate_vigorous_minutes', 
                               'sedentary_minutes', 'wear_time_minutes', 'mvpa_ratio', 'sedentary_ratio']
                
                available_activity_cols = [col for col in activity_cols if col in acc_summary.columns]
                acc_final = acc_summary[available_activity_cols]
                
                print(f"Loaded activity data for {len(acc_final)} participants")
                return acc_final
            else:
                print("No suitable activity features found")
                return None
        else:
            print("Activity data file not found")
            return None
    
    def load_dietary_data(self):
        """
        Load and process dietary data
        """
        print("Loading dietary data...")
        
        diet_file = os.path.join(self.lifestyle_data_dir, "filled_nhanes_combined_diet.csv")
        if os.path.exists(diet_file):
            diet_df = pd.read_csv(diet_file)
            
            # Clean column names
            diet_df.columns = diet_df.columns.str.upper()
            if 'SEQN' in diet_df.columns:
                diet_df['seqn'] = diet_df['SEQN']
            
            # Select key dietary variables (nutrients that affect glucose)
            dietary_features = []
            nutrient_keywords = ['carb', 'sugar', 'fiber', 'fat', 'protein', 'calorie', 'energy',
                               'sodium', 'potassium', 'calcium', 'iron', 'vitc', 'vitd']
            
            for col in diet_df.columns:
                if any(keyword in col.lower() for keyword in nutrient_keywords):
                    if diet_df[col].dtype in ['float64', 'int64']:
                        dietary_features.append(col)
            
            if dietary_features and len(dietary_features) > 0:
                dietary_cols = ['seqn'] + dietary_features[:15]  # Limit to top 15 features
                available_dietary_cols = [col for col in dietary_cols if col in diet_df.columns]
                diet_final = diet_df[available_dietary_cols]
                
                print(f"Loaded dietary data for {len(diet_final)} participants with {len(available_dietary_cols)-1} nutrients")
                return diet_final
            else:
                print("No suitable dietary features found")
                return None
        else:
            print("Dietary data file not found")
            return None
    
    def merge_lifestyle_data(self):
        """
        3.1 Data Source and Preprocessing (Revised)
        Merge lifestyle, demographic, and target data - NO LAB VALUE PROXIES
        """
        print("\n=== 3.1 Data Source and Preprocessing (Lifestyle Focus) ===")
        print("Creating clinically meaningful model using only lifestyle and demographic data")
        
        # Load all data sources
        targets_df = self.load_glucose_targets()
        demo_df = self.load_demographics()
        activity_df = self.load_physical_activity_data()
        diet_df = self.load_dietary_data()
        
        # Start with targets as base
        merged_df = targets_df.copy()
        
        # Merge demographics
        if demo_df is not None:
            merged_df = merged_df.merge(demo_df, on='seqn', how='left')
            print(f"After demographics merge: {len(merged_df)} participants")
        
        # Merge physical activity
        if activity_df is not None:
            merged_df = merged_df.merge(activity_df, on='seqn', how='left')
            print(f"After activity merge: {len(merged_df)} participants")
        
        # Merge dietary data
        if diet_df is not None:
            merged_df = merged_df.merge(diet_df, on='seqn', how='left')
            print(f"After dietary merge: {len(merged_df)} participants")
        
        self.df = merged_df
        print(f"Final merged dataset: {self.df.shape}")
        return self.df
    
    def apply_inclusion_exclusion_criteria(self):
        """
        Apply inclusion/exclusion criteria for lifestyle model
        """
        print("\nApplying inclusion/exclusion criteria...")
        initial_count = len(self.df)
        
        # Age >= 18 (if age available)
        if 'age' in self.df.columns:
            self.df = self.df[self.df['age'] >= 18]
            print(f"After age ≥18 filter: {len(self.df)} participants ({initial_count - len(self.df)} excluded)")
        
        # Must have both glucose and HbA1c targets
        self.df = self.df.dropna(subset=['glucose', 'hba1c'])
        print(f"After glucose/HbA1c requirement: {len(self.df)} participants")
        
        # Remove extreme outliers
        outlier_mask = (self.df['glucose'] <= 600) & (self.df['hba1c'] <= 18)
        self.df = self.df[outlier_mask]
        print(f"After outlier removal: {len(self.df)} participants")
        
        return self.df
    
    def prepare_lifestyle_features(self):
        """
        Prepare lifestyle and demographic features - NO LAB VALUES
        """
        print("\nPreparing lifestyle features (NO lab value proxies)...")
        
        # Define feature categories
        demographic_features = ['age', 'gender', 'race_ethnicity', 'bmi']
        activity_features = ['total_activity_counts', 'moderate_vigorous_minutes', 
                           'sedentary_minutes', 'mvpa_ratio', 'sedentary_ratio']
        
        # Get dietary features (all columns that aren't targets or demographics/activity)
        exclude_cols = ['seqn', 'glucose', 'hba1c'] + demographic_features + activity_features
        dietary_features = [col for col in self.df.columns 
                          if col not in exclude_cols and self.df[col].dtype in ['float64', 'int64']]
        
        # Combine all lifestyle features
        all_features = demographic_features + activity_features + dietary_features
        available_features = [col for col in all_features if col in self.df.columns]
        
        print(f"Available feature categories:")
        print(f"  Demographics: {[f for f in demographic_features if f in self.df.columns]}")
        print(f"  Physical Activity: {[f for f in activity_features if f in self.df.columns]}")
        print(f"  Dietary: {len([f for f in dietary_features if f in self.df.columns])} nutrients")
        
        # Create feature and target datasets
        feature_df = self.df[['seqn'] + available_features + ['glucose', 'hba1c']].copy()
        
        # Handle missing values more robustly
        print("Handling missing values...")
        for col in available_features:
            if col in ['gender', 'race_ethnicity']:
                # Categorical variables - use mode or default
                if feature_df[col].notna().sum() > 0:
                    mode_val = feature_df[col].mode()
                    if len(mode_val) > 0:
                        feature_df[col] = feature_df[col].fillna(mode_val[0])
                    else:
                        feature_df[col] = feature_df[col].fillna(0)
                else:
                    feature_df[col] = feature_df[col].fillna(0)
            else:
                # Numerical variables - use median or 0
                if feature_df[col].notna().sum() > 0:
                    feature_df[col] = feature_df[col].fillna(feature_df[col].median())
                else:
                    feature_df[col] = feature_df[col].fillna(0)
        
        # Encode categorical variables
        if 'gender' in feature_df.columns:
            le_gender = LabelEncoder()
            feature_df['gender'] = le_gender.fit_transform(feature_df['gender'].astype(str))
        
        if 'race_ethnicity' in feature_df.columns:
            le_race = LabelEncoder()
            feature_df['race_ethnicity'] = le_race.fit_transform(feature_df['race_ethnicity'].astype(str))
        
        # Final feature selection - remove any remaining non-numeric and check for NaNs
        numeric_features = []
        for col in available_features:
            if col in feature_df.columns and feature_df[col].dtype in ['float64', 'int64']:
                # Double-check for any remaining NaNs
                if feature_df[col].isna().sum() > 0:
                    print(f"Warning: {col} still has {feature_df[col].isna().sum()} NaN values, filling with 0")
                    feature_df[col] = feature_df[col].fillna(0)
                numeric_features.append(col)
        
        self.X = feature_df[numeric_features]
        self.y = feature_df[['glucose', 'hba1c']]
        self.feature_names = numeric_features
        
        print(f"Final feature set: {len(numeric_features)} features")
        print(f"Features: {numeric_features}")
        print(f"Dataset size: {len(feature_df)} participants")
        
        return self.X, self.y
    
    def split_and_scale_data(self, test_size=0.2):
        """
        Split data and apply standardization
        """
        print(f"\nSplitting data (test_size={test_size})...")
        
        # Train-test split
        stratify_col = None
        if 'gender' in self.X.columns:
            stratify_col = self.X['gender']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=stratify_col
        )
        
        # Scale features
        self.scaler_X = StandardScaler()
        self.X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler_X.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_lifestyle_models(self):
        """
        3.2 Modeling Framework (Lifestyle Focus)
        Train models using only lifestyle features
        """
        print("\n=== 3.2 Modeling Framework (Lifestyle Features Only) ===")
        print("Training models with NO lab value proxies...")
        print("Use case: Screening when lab values are not available")
        
        # Baseline: Ridge regression
        print("Training baseline Ridge regression...")
        self.baseline_model = MultiOutputRegressor(Ridge(alpha=1.0))
        self.baseline_model.fit(self.X_train_scaled, self.y_train)
        
        # Main model: Random Forest with hyperparameter tuning
        print("Training Random Forest with hyperparameter tuning...")
        
        # Simplified grid search for lifestyle model
        param_grid = {
            'estimator__n_estimators': [100, 200],
            'estimator__max_depth': [10, 15, None],
            'estimator__min_samples_split': [2, 5],
            'estimator__min_samples_leaf': [1, 2]
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
    
    def evaluate_lifestyle_models(self):
        """
        Evaluate lifestyle-focused models
        """
        print("\nEvaluating lifestyle models...")
        
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
        print("\n=== Lifestyle Model Performance ===")
        print("Random Forest (Lifestyle Features):")
        print(f"  MAE: {mae_rf:.3f} mg/dL")
        print(f"  MSE: {mse_rf:.3f}")
        print(f"  R²:  {r2_rf:.3f}")
        
        print("\nBaseline Ridge (Lifestyle Features):")
        print(f"  MAE: {mae_baseline:.3f} mg/dL")
        print(f"  MSE: {mse_baseline:.3f}")
        print(f"  R²:  {r2_baseline:.3f}")
        
        print(f"\nNote: Higher MAE expected since we removed lab value proxies")
        print(f"This model is clinically meaningful for screening purposes")
        
        return {
            'rf': {'mae': mae_rf, 'mse': mse_rf, 'r2': r2_rf},
            'baseline': {'mae': mae_baseline, 'mse': mse_baseline, 'r2': r2_baseline}
        }
    
    def analyze_lifestyle_feature_importance(self):
        """
        3.3 Feature Importance & Explainability (Lifestyle Focus)
        """
        print("\n=== 3.3 Feature Importance & Explainability (Lifestyle Features) ===")
        print("Computing SHAP values for lifestyle features...")
        
        # Extract Random Forest estimator for glucose prediction
        rf_estimator = self.model.estimators_[0]
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(rf_estimator)
        shap_values = explainer.shap_values(self.X_test_scaled)
        
        # Global feature importance
        feature_importance = np.abs(shap_values).mean(0)
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Lifestyle Features (Glucose Prediction):")
        print(importance_df.head(10))
        
        # Create visualizations
        self.create_lifestyle_importance_plots(shap_values, importance_df)
        
        return importance_df, shap_values
    
    def create_lifestyle_importance_plots(self, shap_values, importance_df):
        """
        Create lifestyle feature importance visualizations
        """
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Global feature importance bar plot
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('Lifestyle Feature Importance (Glucose Prediction)\nNo Lab Value Proxies')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/figures/lifestyle_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Lifestyle feature importance plot saved as 'lifestyle_feature_importance.png'")
    
    def perform_dietary_clustering(self):
        """
        3.4 Dietary Clustering (Enhanced Implementation)
        K-means clustering on standardized nutrient intake variables
        """
        print("\n=== 3.4 Dietary Clustering ===")
        
        # Get dietary features from the dataset
        dietary_cols = [col for col in self.feature_names 
                       if any(nutrient in col.lower() for nutrient in 
                             ['carb', 'sugar', 'fiber', 'fat', 'protein', 'calorie', 'energy'])]
        
        if len(dietary_cols) >= 3:
            print(f"Performing K-means clustering on {len(dietary_cols)} dietary variables")
            
            # Extract dietary data
            dietary_data = self.X[dietary_cols].copy()
            
            # Standardize dietary variables
            scaler_diet = StandardScaler()
            dietary_scaled = scaler_diet.fit_transform(dietary_data)
            
            # K-means clustering (k=3)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(dietary_scaled)
            
            # Add clusters to dataframe
            cluster_df = self.X.copy()
            cluster_df['dietary_cluster'] = clusters
            cluster_df['glucose'] = self.y['glucose']
            cluster_df['hba1c'] = self.y['hba1c']
            
            # Analyze clusters
            print("\nDietary Cluster Analysis:")
            for cluster_id in range(3):
                cluster_data = cluster_df[cluster_df['dietary_cluster'] == cluster_id]
                print(f"\nCluster {cluster_id} (n={len(cluster_data)}):")
                print(f"  Mean Glucose: {cluster_data['glucose'].mean():.1f} mg/dL")
                print(f"  Mean HbA1c:  {cluster_data['hba1c'].mean():.2f}%")
                
                # Show top dietary characteristics
                for col in dietary_cols[:5]:  # Top 5 dietary features
                    print(f"  {col}: {cluster_data[col].mean():.1f}")
            
            # Visualize clusters
            self.visualize_dietary_clusters(cluster_df, dietary_cols)
            
            return cluster_df
        else:
            print("Insufficient dietary features for clustering")
            return None
    
    def visualize_dietary_clusters(self, cluster_df, dietary_cols):
        """
        Visualize dietary clusters and their glucose/HbA1c associations
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cluster glucose levels
        cluster_glucose = cluster_df.groupby('dietary_cluster')['glucose'].mean()
        axes[0, 0].bar(range(3), cluster_glucose.values, alpha=0.7, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0, 0].set_xlabel('Dietary Cluster')
        axes[0, 0].set_ylabel('Mean Glucose (mg/dL)')
        axes[0, 0].set_title('Mean Glucose by Dietary Cluster')
        axes[0, 0].set_xticks(range(3))
        
        # Cluster HbA1c levels
        cluster_hba1c = cluster_df.groupby('dietary_cluster')['hba1c'].mean()
        axes[0, 1].bar(range(3), cluster_hba1c.values, alpha=0.7, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0, 1].set_xlabel('Dietary Cluster')
        axes[0, 1].set_ylabel('Mean HbA1c (%)')
        axes[0, 1].set_title('Mean HbA1c by Dietary Cluster')
        axes[0, 1].set_xticks(range(3))
        
        # Glucose distribution by cluster
        for cluster_id in range(3):
            cluster_data = cluster_df[cluster_df['dietary_cluster'] == cluster_id]
            axes[1, 0].hist(cluster_data['glucose'], alpha=0.5, label=f'Cluster {cluster_id}', bins=20)
        axes[1, 0].set_xlabel('Glucose (mg/dL)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Glucose Distribution by Dietary Cluster')
        axes[1, 0].legend()
        
        # Cluster sizes
        cluster_sizes = cluster_df['dietary_cluster'].value_counts().sort_index()
        axes[1, 1].pie(cluster_sizes.values, labels=[f'Cluster {i}' for i in range(3)], 
                      autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'salmon'])
        axes[1, 1].set_title('Dietary Cluster Distribution')
        
        plt.tight_layout()
        plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/figures/dietary_clustering.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Dietary clustering visualization saved as 'dietary_clustering.png'")
    
    def evaluate_lifestyle_fairness(self):
        """
        3.5 Fairness Evaluation (Lifestyle Model)
        Evaluate fairness with lifestyle-focused model
        """
        print("\n=== 3.5 Fairness Evaluation (Lifestyle Model) ===")
        print("Evaluating fairness with lifestyle features only...")
        
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
        if 'age' in test_df.columns:
            test_df['age_group'] = pd.cut(test_df['age'], 
                                        bins=[18, 40, 60, 100], 
                                        labels=['<40', '40-60', '>60'])
            age_results = self.evaluate_subgroup_fairness(test_df, 'age_group')
            fairness_results['age'] = age_results
        
        # BMI-based evaluation
        if 'bmi' in test_df.columns:
            test_df['bmi_group'] = pd.cut(test_df['bmi'], 
                                        bins=[0, 25, 30, 100], 
                                        labels=['Normal', 'Overweight', 'Obese'])
            bmi_results = self.evaluate_subgroup_fairness(test_df, 'bmi_group')
            fairness_results['bmi'] = bmi_results
        
        # Race/ethnicity evaluation (if available)
        if 'race_ethnicity' in test_df.columns:
            race_mapping = {0: 'Group_0', 1: 'Group_1', 2: 'Group_2', 3: 'Group_3'}
            race_results = self.evaluate_subgroup_fairness(test_df, 'race_ethnicity', race_mapping)
            fairness_results['race'] = race_results
        
        self.create_lifestyle_fairness_visualizations(fairness_results)
        
        return fairness_results
    
    def evaluate_subgroup_fairness(self, df, group_col, group_mapping=None):
        """
        Evaluate fairness metrics for a specific demographic subgroup
        """
        results = {}
        
        for group_val in df[group_col].unique():
            if pd.isna(group_val):
                continue
                
            group_data = df[df[group_col] == group_val]
            group_name = group_mapping.get(group_val, str(group_val)) if group_mapping else str(group_val)
            
            if len(group_data) < 10:  # Skip small groups
                continue
            
            # Calculate MAE for glucose and HbA1c
            glucose_mae = mean_absolute_error(group_data['glucose_true'], group_data['glucose_pred'])
            hba1c_mae = mean_absolute_error(group_data['hba1c_true'], group_data['hba1c_pred'])
            
            results[group_name] = {
                'n': len(group_data),
                'glucose_mae': glucose_mae,
                'hba1c_mae': hba1c_mae,
                'glucose_mean_true': group_data['glucose_true'].mean(),
                'hba1c_mean_true': group_data['hba1c_true'].mean()
            }
        
        # Print results
        print(f"\nLifestyle Model Fairness by {group_col}:")
        for group_name, metrics in results.items():
            print(f"  {group_name} (n={metrics['n']}):")
            print(f"    Glucose MAE: {metrics['glucose_mae']:.3f} mg/dL")
            print(f"    HbA1c MAE:   {metrics['hba1c_mae']:.3f}%")
        
        return results
    
    def create_lifestyle_fairness_visualizations(self, fairness_results):
        """
        Create lifestyle model fairness visualizations
        """
        n_groups = len(fairness_results)
        fig, axes = plt.subplots(2, n_groups, figsize=(5*n_groups, 10))
        
        if n_groups == 1:
            axes = axes.reshape(2, 1)
        
        for i, (group_type, results) in enumerate(fairness_results.items()):
            if i >= n_groups:
                break
                
            groups = list(results.keys())
            glucose_mae = [results[g]['glucose_mae'] for g in groups]
            hba1c_mae = [results[g]['hba1c_mae'] for g in groups]
            
            # Glucose MAE
            axes[0, i].bar(groups, glucose_mae, alpha=0.7, color='skyblue')
            axes[0, i].set_title(f'Glucose MAE by {group_type.title()}\n(Lifestyle Model)')
            axes[0, i].set_ylabel('MAE (mg/dL)')
            axes[0, i].tick_params(axis='x', rotation=45)
            
            # HbA1c MAE
            axes[1, i].bar(groups, hba1c_mae, alpha=0.7, color='lightcoral')
            axes[1, i].set_title(f'HbA1c MAE by {group_type.title()}\n(Lifestyle Model)')
            axes[1, i].set_ylabel('MAE (%)')
            axes[1, i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/figures/lifestyle_fairness_evaluation.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Lifestyle fairness evaluation plot saved as 'lifestyle_fairness_evaluation.png'")
    
    def run_complete_lifestyle_analysis(self):
        """
        Run the complete lifestyle-focused analysis pipeline
        """
        print("NHANES Lifestyle-Based Blood Glucose Analysis Pipeline")
        print("=" * 70)
        print("CLINICALLY MEANINGFUL MODEL - NO LAB VALUE PROXIES")
        print("Use case: Screening when lab values are not available")
        print("=" * 70)
        
        # Load and preprocess data
        self.merge_lifestyle_data()
        self.apply_inclusion_exclusion_criteria()
        self.prepare_lifestyle_features()
        self.split_and_scale_data()
        
        # Model training and evaluation
        self.train_lifestyle_models()
        performance_metrics = self.evaluate_lifestyle_models()
        
        # Feature importance analysis
        importance_df, shap_values = self.analyze_lifestyle_feature_importance()
        
        # Dietary clustering
        cluster_results = self.perform_dietary_clustering()
        
        # Fairness evaluation
        fairness_results = self.evaluate_lifestyle_fairness()
        
        print("\n" + "=" * 70)
        print("LIFESTYLE ANALYSIS COMPLETE")
        print("=" * 70)
        print("Key Insights:")
        print("- Model uses only lifestyle, dietary, and demographic features")
        print("- No lab value proxies that would make the model clinically meaningless")
        print("- Higher MAE expected but model is useful for screening purposes")
        print("- Fairness evaluation shows performance across demographic groups")
        
        return {
            'performance': performance_metrics,
            'feature_importance': importance_df,
            'dietary_clusters': cluster_results,
            'fairness': fairness_results
        }

def main():
    """
    Main execution function
    """
    analyzer = LifestyleGlucoseAnalyzer()
    results = analyzer.run_complete_lifestyle_analysis()
    return results

if __name__ == "__main__":
    results = main()
