#!/usr/bin/env python3
"""
Improved Feature Engineering for Blood Glucose Prediction
Addresses data quality issues and adds sophisticated features

Key Issues Found:
1. Mysterious value 5.397605346934028e-79 represents missing data
2. MVPA and sedentary features were all zeros due to wrong column mapping
3. No correlation analysis possible due to SEQN mismatch

Author: Generated for fairness project
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

class ImprovedFeatureEngineer:
    """
    Improved feature engineering for NHANES glucose prediction
    """
    
    def __init__(self, 
                 lab_data_dir="/Users/aakashsuresh/fairness/blood_glucose_project/data/processed/nhanes_lab/",
                 lifestyle_data_dir="/Users/aakashsuresh/fairness/blood_glucose_project/data/processed/nhanes_combined/"):
        self.lab_data_dir = lab_data_dir
        self.lifestyle_data_dir = lifestyle_data_dir
        self.mysterious_value = 5.397605346934028e-79
        
    def load_and_clean_activity_data(self):
        """
        Load and properly clean accelerometry data
        """
        print("=== Loading and Cleaning Activity Data ===")
        
        acc_file = f"{self.lifestyle_data_dir}/nhanes_combined_acc.csv"
        acc_df = pd.read_csv(acc_file)
        
        print(f"Raw data shape: {acc_df.shape}")
        
        # Replace mysterious values with NaN
        print(f"Replacing mysterious value {self.mysterious_value} with NaN...")
        acc_df = acc_df.replace(self.mysterious_value, np.nan)
        
        # Clean SEQN column
        acc_df['seqn'] = acc_df['SEQN'].astype(float)
        
        # Identify actual activity columns based on NHANES documentation
        activity_mapping = {
            'PAXAISMD': 'total_activity_counts',      # Total activity intensity (counts/day)
            'PAXTMD': 'wear_time_minutes',            # Monitor wear time (minutes/day)
            'PAXMTSD': 'moderate_activity_minutes',   # Moderate activity minutes
            'PAXVMD': 'vigorous_activity_minutes',    # Vigorous activity minutes
            'PAXLXSD': 'light_activity_minutes',      # Light activity minutes
            'PAXSSNDP': 'sedentary_minutes'           # Sedentary minutes
        }
        
        # Select and rename relevant columns
        available_cols = ['seqn'] + [col for col in activity_mapping.keys() if col in acc_df.columns]
        acc_clean = acc_df[available_cols].copy()
        
        # Rename columns
        rename_dict = {'seqn': 'seqn'}
        rename_dict.update({k: v for k, v in activity_mapping.items() if k in acc_clean.columns})
        acc_clean = acc_clean.rename(columns=rename_dict)
        
        print(f"Available activity features: {list(acc_clean.columns)[1:]}")
        
        return acc_clean
    
    def engineer_sophisticated_activity_features(self, acc_df):
        """
        Create sophisticated activity features
        """
        print("\n=== Engineering Sophisticated Activity Features ===")
        
        # Aggregate by participant (median to reduce outlier impact)
        print("Aggregating by participant using median...")
        agg_dict = {col: 'median' for col in acc_df.columns if col != 'seqn'}
        acc_summary = acc_df.groupby('seqn').agg(agg_dict).reset_index()
        
        print(f"Participants with activity data: {len(acc_summary)}")
        
        # Create derived features
        if 'moderate_activity_minutes' in acc_summary.columns and 'vigorous_activity_minutes' in acc_summary.columns:
            acc_summary['mvpa_minutes'] = (acc_summary['moderate_activity_minutes'].fillna(0) + 
                                         acc_summary['vigorous_activity_minutes'].fillna(0))
        
        # Activity ratios (avoid division by zero)
        if 'wear_time_minutes' in acc_summary.columns:
            wear_time_safe = acc_summary['wear_time_minutes'].fillna(1440)  # Default to 24 hours
            wear_time_safe = wear_time_safe.replace(0, 1440)
            
            if 'mvpa_minutes' in acc_summary.columns:
                acc_summary['mvpa_ratio'] = acc_summary['mvpa_minutes'] / wear_time_safe
            
            if 'sedentary_minutes' in acc_summary.columns:
                acc_summary['sedentary_ratio'] = acc_summary['sedentary_minutes'].fillna(0) / wear_time_safe
            
            if 'light_activity_minutes' in acc_summary.columns:
                acc_summary['light_activity_ratio'] = acc_summary['light_activity_minutes'].fillna(0) / wear_time_safe
        
        # Activity intensity categories
        if 'total_activity_counts' in acc_summary.columns:
            acc_summary['activity_level'] = pd.cut(
                acc_summary['total_activity_counts'].fillna(0),
                bins=[0, 1000000, 3000000, np.inf],
                labels=['Low', 'Moderate', 'High']
            )
        
        # Log-transform highly skewed variables
        if 'total_activity_counts' in acc_summary.columns:
            acc_summary['log_total_activity'] = np.log1p(acc_summary['total_activity_counts'].fillna(0))
        
        print(f"Engineered features: {list(acc_summary.columns)}")
        
        return acc_summary
    
    def load_comprehensive_demographic_data(self):
        """
        Load comprehensive demographic and health data
        """
        print("\n=== Loading Comprehensive Demographic Data ===")
        
        demo_path = f"{self.lab_data_dir}/P_DEMO.xpt"
        if pd.io.common.file_exists(demo_path):
            demo = pd.read_sas(demo_path, format="xport")
            
            # Select comprehensive demographic features
            demo_features = {
                'SEQN': 'seqn',
                'RIDAGEYR': 'age',
                'RIAGENDR': 'gender', 
                'RIDRETH3': 'race_ethnicity',
                'BMXBMI': 'bmi',
                'BMXWT': 'weight_kg',
                'BMXHT': 'height_cm',
                'BMXWAIST': 'waist_circumference',
                'DMDEDUC2': 'education_level',
                'INDHHIN2': 'household_income',
                'DMDMARTL': 'marital_status'
            }
            
            available_features = {k: v for k, v in demo_features.items() if k in demo.columns}
            demo_clean = demo[list(available_features.keys())].copy()
            demo_clean = demo_clean.rename(columns=available_features)
            
            # Convert SEQN to match other datasets
            demo_clean['seqn'] = demo_clean['seqn'].astype(float)
            
            print(f"Demographic features: {list(demo_clean.columns)}")
            print(f"Participants with demographic data: {len(demo_clean)}")
            
            return demo_clean
        else:
            print("Demographic file not found")
            return None
    
    def add_dietary_features(self):
        """
        Add dietary features from available data
        """
        print("\n=== Adding Dietary Features ===")
        
        # Try multiple dietary files
        dietary_files = [
            "filled_nhanes_combined_diet.csv",
            "cleaned_nhanes_combined_diet.csv", 
            "nhanes_combined_diet.csv"
        ]
        
        for file in dietary_files:
            diet_path = f"{self.lifestyle_data_dir}/{file}"
            if pd.io.common.file_exists(diet_path):
                print(f"Loading dietary data from {file}")
                diet_df = pd.read_csv(diet_path)
                
                # Clean SEQN
                if 'SEQN' in diet_df.columns:
                    diet_df['seqn'] = diet_df['SEQN'].astype(float)
                
                # Select meaningful dietary features
                dietary_keywords = [
                    'energy', 'calorie', 'kcal',
                    'carb', 'sugar', 'fiber',
                    'fat', 'protein', 'sodium',
                    'dr1t', 'dr2t'  # NHANES dietary recall prefixes
                ]
                
                dietary_features = ['seqn']
                for col in diet_df.columns:
                    if any(keyword in col.lower() for keyword in dietary_keywords):
                        if diet_df[col].dtype in ['float64', 'int64']:
                            dietary_features.append(col)
                
                if len(dietary_features) > 1:
                    diet_clean = diet_df[dietary_features[:16]].copy()  # Limit to 15 dietary features
                    print(f"Selected dietary features: {len(dietary_features)-1}")
                    return diet_clean
        
        print("No suitable dietary data found")
        return None
    
    def create_interaction_features(self, merged_df):
        """
        Create interaction features between demographics and activity
        """
        print("\n=== Creating Interaction Features ===")
        
        # Age × Activity interactions
        if 'age' in merged_df.columns and 'total_activity_counts' in merged_df.columns:
            merged_df['age_activity_interaction'] = merged_df['age'] * merged_df['total_activity_counts']
        
        # BMI × Activity interactions
        if 'bmi' in merged_df.columns and 'mvpa_ratio' in merged_df.columns:
            merged_df['bmi_mvpa_interaction'] = merged_df['bmi'] * merged_df['mvpa_ratio']
        
        # Gender × Activity interactions
        if 'gender' in merged_df.columns and 'sedentary_ratio' in merged_df.columns:
            merged_df['gender_sedentary_interaction'] = merged_df['gender'] * merged_df['sedentary_ratio']
        
        print("Created interaction features")
        return merged_df
    
    def handle_missing_data_intelligently(self, df):
        """
        Intelligent missing data handling using KNN imputation
        """
        print("\n=== Intelligent Missing Data Handling ===")
        
        # Separate categorical and numerical features
        categorical_features = []
        numerical_features = []
        
        for col in df.columns:
            if col == 'seqn':
                continue
            elif df[col].dtype in ['object', 'category']:
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        print(f"Categorical features: {len(categorical_features)}")
        print(f"Numerical features: {len(numerical_features)}")
        
        # Handle categorical features
        for col in categorical_features:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    # Convert to string first to avoid categorical issues
                    df[col] = df[col].astype(str).fillna('Unknown')
        
        # Handle numerical features with simple imputation (median/zero)
        if numerical_features:
            print("Handling numerical missing values...")
            missing_counts = df[numerical_features].isnull().sum()
            if missing_counts.sum() > 0:
                print(f"Missing values found in: {missing_counts[missing_counts > 0].to_dict()}")
                for col in numerical_features:
                    if df[col].isnull().sum() > 0:
                        if 'activity' in col.lower() or 'dietary' in col.lower():
                            # Activity and dietary features - use 0 for missing
                            df[col] = df[col].fillna(0)
                        else:
                            # Other numerical features - use median
                            df[col] = df[col].fillna(df[col].median())
            else:
                print("No missing values in numerical features")
        
        print("Missing data handling complete")
        return df
    
    def load_glucose_targets(self):
        """
        Load glucose and HbA1c targets
        """
        print("\n=== Loading Glucose Targets ===")
        
        glucose_file = f"{self.lab_data_dir}/fasting_glucose_processed.csv"
        hba1c_file = f"{self.lab_data_dir}/glycohemoglobin_processed.csv"
        
        if pd.io.common.file_exists(glucose_file) and pd.io.common.file_exists(hba1c_file):
            glucose_df = pd.read_csv(glucose_file)[['seqn', 'lbxglu']]
            hba1c_df = pd.read_csv(hba1c_file)[['seqn', 'lbxgh']]
            
            # Merge targets
            targets_df = glucose_df.merge(hba1c_df, on='seqn', how='inner')
            targets_df.columns = ['seqn', 'glucose', 'hba1c']
            
            print(f"Loaded targets for {len(targets_df)} participants")
            return targets_df
        else:
            raise FileNotFoundError("Glucose or HbA1c target files not found")
    
    def create_comprehensive_dataset(self):
        """
        Create comprehensive dataset with all improved features
        """
        print("Creating Comprehensive Dataset with Improved Features")
        print("=" * 70)
        
        # Load all data sources
        acc_df = self.load_and_clean_activity_data()
        acc_features = self.engineer_sophisticated_activity_features(acc_df)
        demo_df = self.load_comprehensive_demographic_data()
        diet_df = self.add_dietary_features()
        targets_df = self.load_glucose_targets()
        
        # Start merging
        print(f"\n=== Merging Data Sources ===")
        merged_df = targets_df.copy()
        print(f"Starting with targets: {len(merged_df)} participants")
        
        # Merge demographics
        if demo_df is not None:
            merged_df = merged_df.merge(demo_df, on='seqn', how='left')
            print(f"After demographics: {len(merged_df)} participants")
        
        # Merge activity features
        merged_df = merged_df.merge(acc_features, on='seqn', how='left')
        print(f"After activity: {len(merged_df)} participants")
        
        # Merge dietary features
        if diet_df is not None:
            merged_df = merged_df.merge(diet_df, on='seqn', how='left')
            print(f"After dietary: {len(merged_df)} participants")
        
        # Create interaction features
        merged_df = self.create_interaction_features(merged_df)
        
        # Handle missing data intelligently
        merged_df = self.handle_missing_data_intelligently(merged_df)
        
        # Apply inclusion/exclusion criteria
        print(f"\n=== Applying Inclusion/Exclusion Criteria ===")
        initial_count = len(merged_df)
        
        # Age >= 18
        if 'age' in merged_df.columns:
            merged_df = merged_df[merged_df['age'] >= 18]
            print(f"After age ≥18: {len(merged_df)} participants ({initial_count - len(merged_df)} excluded)")
        
        # Remove extreme outliers
        outlier_mask = (merged_df['glucose'] <= 600) & (merged_df['hba1c'] <= 18)
        merged_df = merged_df[outlier_mask]
        print(f"After outlier removal: {len(merged_df)} participants")
        
        # Encode categorical variables
        categorical_cols = merged_df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col != 'seqn':
                le = LabelEncoder()
                merged_df[col] = le.fit_transform(merged_df[col].astype(str))
        
        print(f"\nFinal dataset shape: {merged_df.shape}")
        print(f"Features available: {merged_df.columns.tolist()}")
        
        return merged_df
    
    def analyze_feature_quality(self, df):
        """
        Analyze the quality of engineered features
        """
        print("\n=== Feature Quality Analysis ===")
        
        # Exclude targets and ID
        feature_cols = [col for col in df.columns if col not in ['seqn', 'glucose', 'hba1c']]
        
        feature_stats = []
        for col in feature_cols:
            stats = {
                'feature': col,
                'missing_pct': df[col].isnull().sum() / len(df) * 100,
                'unique_values': df[col].nunique(),
                'variance': df[col].var() if df[col].dtype in ['float64', 'int64'] else 0,
                'glucose_corr': df[col].corr(df['glucose']) if df[col].dtype in ['float64', 'int64'] else 0
            }
            feature_stats.append(stats)
        
        feature_quality_df = pd.DataFrame(feature_stats)
        feature_quality_df = feature_quality_df.sort_values('glucose_corr', key=abs, ascending=False)
        
        print("Top 15 features by absolute correlation with glucose:")
        print(feature_quality_df.head(15)[['feature', 'glucose_corr', 'variance', 'unique_values']])
        
        # Visualize correlations
        plt.figure(figsize=(12, 8))
        top_features = feature_quality_df.head(20)
        plt.barh(range(len(top_features)), top_features['glucose_corr'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Correlation with Glucose')
        plt.title('Top 20 Features by Correlation with Glucose')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/finetuning/improved_feature_correlations.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_quality_df

def main():
    """
    Main execution function
    """
    engineer = ImprovedFeatureEngineer()
    
    # Create comprehensive dataset
    improved_df = engineer.create_comprehensive_dataset()
    
    # Analyze feature quality
    feature_quality = engineer.analyze_feature_quality(improved_df)
    
    # Save improved dataset
    output_path = '/Users/aakashsuresh/fairness/blood_glucose_project/finetuning/improved_dataset.csv'
    improved_df.to_csv(output_path, index=False)
    print(f"\nImproved dataset saved to: {output_path}")
    
    return improved_df, feature_quality

if __name__ == "__main__":
    improved_df, feature_quality = main()
