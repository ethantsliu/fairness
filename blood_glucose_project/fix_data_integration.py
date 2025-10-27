#!/usr/bin/env python3
"""
Fix Data Integration Issues for Complete Feature Set
Addresses the critical data quality problems identified in feature importance analysis

Issues to Fix:
1. Physical activity features have zero variance (12 features)
2. Dietary features missing/zero variance (6 features) 
3. BMI and anthropometric data not properly integrated
4. SEQN matching problems between datasets

Author: Generated for fairness project
Date: October 2025
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

class DataIntegrationFixer:
    """
    Comprehensive data integration fixer for NHANES glucose prediction
    """
    
    def __init__(self):
        self.lab_data_dir = "/Users/aakashsuresh/fairness/processed_data_nhanes_lab/"
        self.lifestyle_data_dir = "/Users/aakashsuresh/fairness/processed_data_new/"
        self.mysterious_value = 5.397605346934028e-79
        
    def load_glucose_targets_properly(self):
        """
        Load glucose and HbA1c targets with proper SEQN handling
        """
        print("=== Loading Glucose Targets (Fixed) ===")
        
        glucose_file = os.path.join(self.lab_data_dir, "fasting_glucose_processed.csv")
        hba1c_file = os.path.join(self.lab_data_dir, "glycohemoglobin_processed.csv")
        
        if os.path.exists(glucose_file) and os.path.exists(hba1c_file):
            glucose_df = pd.read_csv(glucose_file)
            hba1c_df = pd.read_csv(hba1c_file)
            
            print(f"Glucose data shape: {glucose_df.shape}")
            print(f"HbA1c data shape: {hba1c_df.shape}")
            print(f"Glucose SEQN range: {glucose_df['seqn'].min():.0f} - {glucose_df['seqn'].max():.0f}")
            print(f"HbA1c SEQN range: {hba1c_df['seqn'].min():.0f} - {hba1c_df['seqn'].max():.0f}")
            
            # Merge targets
            targets_df = glucose_df[['seqn', 'lbxglu']].merge(
                hba1c_df[['seqn', 'lbxgh']], on='seqn', how='inner'
            )
            targets_df.columns = ['seqn', 'glucose', 'hba1c']
            
            print(f"Merged targets: {len(targets_df)} participants")
            return targets_df
        else:
            raise FileNotFoundError("Glucose or HbA1c files not found")
    
    def load_demographics_properly(self):
        """
        Load comprehensive demographic data including BMI
        """
        print("\n=== Loading Demographics (Fixed) ===")
        
        demo_path = os.path.join(self.lab_data_dir, "P_DEMO.xpt")
        if os.path.exists(demo_path):
            demo = pd.read_sas(demo_path, format="xport")
            print(f"Raw demographics shape: {demo.shape}")
            print(f"Available columns: {list(demo.columns)}")
            
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
            
            print(f"Demographics SEQN range: {demo_clean['seqn'].min():.0f} - {demo_clean['seqn'].max():.0f}")
            print(f"Available demographic features: {list(available_features.values())}")
            print(f"BMI available: {'bmi' in demo_clean.columns}")
            
            if 'bmi' in demo_clean.columns:
                print(f"BMI stats: Mean={demo_clean['bmi'].mean():.1f}, Missing={demo_clean['bmi'].isna().sum()}")
            else:
                print("BMI not found in demographics file - will need to load from examination data")
            
            return demo_clean
        else:
            print("Demographics file not found")
            return None
    
    def load_examination_data(self):
        """
        Load examination data including BMI from separate examination file
        """
        print("\n=== Loading Examination Data for BMI ===")
        
        # Try to find examination data file
        exam_files = ['P_BMX.xpt', 'P_EXAM.xpt', 'EXAM_*.xpt']
        
        for exam_file in exam_files:
            exam_path = os.path.join(self.lab_data_dir, exam_file)
            if os.path.exists(exam_path):
                print(f"Loading examination data from {exam_file}")
                exam_df = pd.read_sas(exam_path, format="xport")
                print(f"Examination data shape: {exam_df.shape}")
                print(f"Available columns: {list(exam_df.columns)}")
                
                # Look for BMI-related columns
                bmi_cols = [col for col in exam_df.columns if 'BMX' in col or 'bmi' in col.lower()]
                if bmi_cols:
                    print(f"BMI-related columns found: {bmi_cols}")
                    
                    exam_features = {
                        'SEQN': 'seqn',
                        'BMXBMI': 'bmi',
                        'BMXWT': 'weight_kg',
                        'BMXHT': 'height_cm',
                        'BMXWAIST': 'waist_circumference'
                    }
                    
                    available_exam = {k: v for k, v in exam_features.items() if k in exam_df.columns}
                    if available_exam:
                        exam_clean = exam_df[list(available_exam.keys())].copy()
                        exam_clean = exam_clean.rename(columns=available_exam)
                        exam_clean['seqn'] = exam_clean['seqn'].astype(float)
                        
                        print(f"Examination features: {list(available_exam.values())}")
                        if 'bmi' in exam_clean.columns:
                            print(f"BMI stats: Mean={exam_clean['bmi'].mean():.1f}, Missing={exam_clean['bmi'].isna().sum()}")
                        
                        return exam_clean
                
        print("No examination data with BMI found")
        return None
    
    def load_activity_data_properly(self):
        """
        Load and properly process physical activity data
        """
        print("\n=== Loading Physical Activity Data (Fixed) ===")
        
        acc_file = os.path.join(self.lifestyle_data_dir, "nhanes_combined_acc.csv")
        if os.path.exists(acc_file):
            acc_df = pd.read_csv(acc_file)
            print(f"Raw activity data shape: {acc_df.shape}")
            
            # Check SEQN format and range
            print(f"Activity SEQN sample: {acc_df['SEQN'].head().tolist()}")
            print(f"Activity SEQN range: {acc_df['SEQN'].min():.0f} - {acc_df['SEQN'].max():.0f}")
            print(f"Unique participants: {acc_df['SEQN'].nunique()}")
            
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
            
            # Check data quality before aggregation
            for col in acc_clean.columns[1:]:  # Skip seqn
                non_null = acc_clean[col].notna().sum()
                unique_vals = acc_clean[col].nunique()
                print(f"{col}: {non_null} non-null values, {unique_vals} unique values")
            
            # Aggregate by participant using median (more robust than mean)
            print("Aggregating by participant using median...")
            agg_dict = {col: 'median' for col in acc_clean.columns if col != 'seqn'}
            acc_summary = acc_clean.groupby('seqn').agg(agg_dict).reset_index()
            
            # Create derived features
            if 'moderate_activity_minutes' in acc_summary.columns and 'vigorous_activity_minutes' in acc_summary.columns:
                acc_summary['mvpa_minutes'] = (
                    acc_summary['moderate_activity_minutes'].fillna(0) + 
                    acc_summary['vigorous_activity_minutes'].fillna(0)
                )
            
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
                    labels=[0, 1, 2]  # Low, Moderate, High
                )
            
            # Log-transform highly skewed variables
            if 'total_activity_counts' in acc_summary.columns:
                acc_summary['log_total_activity'] = np.log1p(acc_summary['total_activity_counts'].fillna(0))
            
            print(f"Final activity data: {acc_summary.shape}")
            print(f"Activity SEQN range: {acc_summary['seqn'].min():.0f} - {acc_summary['seqn'].max():.0f}")
            
            # Check final data quality
            for col in acc_summary.columns[1:]:  # Skip seqn
                non_null = acc_summary[col].notna().sum()
                if acc_summary[col].dtype in ['float64', 'int64']:
                    variance = acc_summary[col].var()
                    print(f"{col}: {non_null} non-null, variance={variance:.3f}")
                else:
                    unique_vals = acc_summary[col].nunique()
                    print(f"{col}: {non_null} non-null, {unique_vals} unique categories")
            
            return acc_summary
        else:
            print("Activity data file not found")
            return None
    
    def load_dietary_data_properly(self):
        """
        Load and properly process dietary data
        """
        print("\n=== Loading Dietary Data (Fixed) ===")
        
        # Try multiple dietary files
        dietary_files = [
            "filled_nhanes_combined_diet.csv",
            "cleaned_nhanes_combined_diet.csv", 
            "nhanes_combined_diet.csv"
        ]
        
        for file in dietary_files:
            diet_path = os.path.join(self.lifestyle_data_dir, file)
            if os.path.exists(diet_path):
                print(f"Loading dietary data from {file}")
                diet_df = pd.read_csv(diet_path)
                print(f"Raw dietary data shape: {diet_df.shape}")
                
                # Clean SEQN
                if 'SEQN' in diet_df.columns:
                    diet_df['seqn'] = diet_df['SEQN'].astype(float)
                    print(f"Dietary SEQN range: {diet_df['seqn'].min():.0f} - {diet_df['seqn'].max():.0f}")
                
                # Select meaningful dietary features
                dietary_keywords = [
                    'kcal', 'energy', 'calorie',
                    'carb', 'sugar', 'fiber',
                    'fat', 'protein', 'sodium',
                    'dr1t', 'dr2t', 'dsqt'  # NHANES dietary recall prefixes
                ]
                
                dietary_features = ['seqn']
                for col in diet_df.columns:
                    if any(keyword in col.lower() for keyword in dietary_keywords):
                        if diet_df[col].dtype in ['float64', 'int64']:
                            dietary_features.append(col)
                
                if len(dietary_features) > 1:
                    diet_clean = diet_df[dietary_features[:16]].copy()  # Limit to 15 dietary features
                    
                    # Check data quality
                    print(f"Selected dietary features: {len(dietary_features)-1}")
                    for col in diet_clean.columns[1:]:  # Skip seqn
                        non_null = diet_clean[col].notna().sum()
                        variance = diet_clean[col].var()
                        print(f"{col}: {non_null} non-null, variance={variance:.3f}")
                    
                    return diet_clean
        
        print("No suitable dietary data found")
        return None
    
    def create_interaction_features(self, merged_df):
        """
        Create interaction features between demographics and lifestyle
        """
        print("\n=== Creating Interaction Features (Fixed) ===")
        
        interactions_created = []
        
        # Age × Activity interactions
        if 'age' in merged_df.columns and 'total_activity_counts' in merged_df.columns:
            merged_df['age_activity_interaction'] = merged_df['age'] * merged_df['total_activity_counts']
            interactions_created.append('age_activity_interaction')
        
        # BMI × Activity interactions
        if 'bmi' in merged_df.columns and 'mvpa_ratio' in merged_df.columns:
            merged_df['bmi_mvpa_interaction'] = merged_df['bmi'] * merged_df['mvpa_ratio']
            interactions_created.append('bmi_mvpa_interaction')
        
        # Gender × Activity interactions
        if 'gender' in merged_df.columns and 'sedentary_ratio' in merged_df.columns:
            merged_df['gender_sedentary_interaction'] = merged_df['gender'] * merged_df['sedentary_ratio']
            interactions_created.append('gender_sedentary_interaction')
        
        # Age × BMI interaction
        if 'age' in merged_df.columns and 'bmi' in merged_df.columns:
            merged_df['age_bmi_interaction'] = merged_df['age'] * merged_df['bmi']
            interactions_created.append('age_bmi_interaction')
        
        print(f"Created interaction features: {interactions_created}")
        return merged_df
    
    def handle_missing_data_intelligently(self, df):
        """
        Intelligent missing data handling
        """
        print("\n=== Intelligent Missing Data Handling (Fixed) ===")
        
        # Separate categorical and numerical features
        categorical_features = []
        numerical_features = []
        
        for col in df.columns:
            if col in ['seqn', 'glucose', 'hba1c']:
                continue
            elif df[col].dtype in ['object', 'category'] or col in ['gender', 'race_ethnicity', 'education_level', 'activity_level']:
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
                    df[col] = df[col].fillna(0)
        
        # Handle numerical features with domain-specific imputation
        for col in numerical_features:
            if df[col].isnull().sum() > 0:
                if 'activity' in col.lower() or 'dietary' in col.lower() or 'dsqt' in col.lower():
                    # Activity and dietary features - use 0 for missing (no activity/intake)
                    df[col] = df[col].fillna(0)
                elif col in ['bmi', 'weight_kg', 'height_cm', 'waist_circumference']:
                    # Anthropometric features - use median
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # Other numerical features - use median
                    df[col] = df[col].fillna(df[col].median())
        
        print("Missing data handling complete")
        return df
    
    def create_fixed_comprehensive_dataset(self):
        """
        Create comprehensive dataset with all data integration issues fixed
        """
        print("Creating Fixed Comprehensive Dataset")
        print("=" * 70)
        
        # Load all data sources with fixes
        targets_df = self.load_glucose_targets_properly()
        demo_df = self.load_demographics_properly()
        exam_df = self.load_examination_data()  # For BMI
        activity_df = self.load_activity_data_properly()
        diet_df = self.load_dietary_data_properly()
        
        # Start merging with detailed tracking
        print(f"\n=== Merging Data Sources (Fixed) ===")
        merged_df = targets_df.copy()
        print(f"Starting with targets: {len(merged_df)} participants")
        print(f"Target SEQN range: {merged_df['seqn'].min():.0f} - {merged_df['seqn'].max():.0f}")
        
        # Merge demographics
        if demo_df is not None:
            print(f"Merging demographics...")
            print(f"Demo SEQN overlap: {len(set(merged_df['seqn']) & set(demo_df['seqn']))} participants")
            merged_df = merged_df.merge(demo_df, on='seqn', how='left')
            print(f"After demographics: {len(merged_df)} participants, {merged_df.shape[1]} features")
        
        # Merge examination data (BMI)
        if exam_df is not None:
            print(f"Merging examination data...")
            print(f"Exam SEQN overlap: {len(set(merged_df['seqn']) & set(exam_df['seqn']))} participants")
            merged_df = merged_df.merge(exam_df, on='seqn', how='left')
            print(f"After examination: {len(merged_df)} participants, {merged_df.shape[1]} features")
        
        # Merge activity features
        if activity_df is not None:
            print(f"Merging activity data...")
            print(f"Activity SEQN overlap: {len(set(merged_df['seqn']) & set(activity_df['seqn']))} participants")
            merged_df = merged_df.merge(activity_df, on='seqn', how='left')
            print(f"After activity: {len(merged_df)} participants, {merged_df.shape[1]} features")
        
        # Merge dietary features
        if diet_df is not None:
            print(f"Merging dietary data...")
            print(f"Dietary SEQN overlap: {len(set(merged_df['seqn']) & set(diet_df['seqn']))} participants")
            merged_df = merged_df.merge(diet_df, on='seqn', how='left')
            print(f"After dietary: {len(merged_df)} participants, {merged_df.shape[1]} features")
        
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
        categorical_cols = ['gender', 'race_ethnicity', 'education_level', 'activity_level', 'marital_status']
        for col in categorical_cols:
            if col in merged_df.columns:
                if merged_df[col].dtype in ['object', 'category']:
                    le = LabelEncoder()
                    merged_df[col] = le.fit_transform(merged_df[col].astype(str))
        
        print(f"\nFinal fixed dataset shape: {merged_df.shape}")
        
        # Analyze feature variance
        print(f"\n=== Feature Variance Analysis ===")
        exclude_cols = ['seqn', 'glucose', 'hba1c']
        feature_cols = [col for col in merged_df.columns if col not in exclude_cols]
        
        valid_features = []
        zero_variance_features = []
        
        for col in feature_cols:
            if merged_df[col].notna().sum() > 0 and merged_df[col].var() > 0:
                valid_features.append(col)
            else:
                zero_variance_features.append(col)
        
        print(f"Features with variance: {len(valid_features)}")
        print(f"Features with zero variance: {len(zero_variance_features)}")
        
        if zero_variance_features:
            print(f"Zero variance features: {zero_variance_features}")
        
        print(f"Valid features: {valid_features}")
        
        return merged_df, valid_features, zero_variance_features
    
    def save_fixed_dataset(self, merged_df, valid_features, zero_variance_features):
        """
        Save the fixed dataset and analysis
        """
        print(f"\n=== Saving Fixed Dataset ===")
        
        # Save complete dataset
        output_path = '/Users/aakashsuresh/fairness/blood_glucose_project/fixed_comprehensive_dataset.csv'
        merged_df.to_csv(output_path, index=False)
        print(f"Fixed dataset saved to: {output_path}")
        
        # Save feature analysis
        feature_analysis = pd.DataFrame({
            'Feature': merged_df.columns,
            'Has_Variance': [col in valid_features for col in merged_df.columns],
            'Data_Type': [str(merged_df[col].dtype) for col in merged_df.columns],
            'Non_Null_Count': [merged_df[col].notna().sum() for col in merged_df.columns],
            'Variance': [merged_df[col].var() if merged_df[col].dtype in ['float64', 'int64'] else 0 for col in merged_df.columns]
        })
        
        feature_analysis_path = '/Users/aakashsuresh/fairness/blood_glucose_project/fixed_feature_analysis.csv'
        feature_analysis.to_csv(feature_analysis_path, index=False)
        print(f"Feature analysis saved to: {feature_analysis_path}")
        
        return output_path, feature_analysis_path

def main():
    """
    Main execution function
    """
    fixer = DataIntegrationFixer()
    
    # Create fixed comprehensive dataset
    merged_df, valid_features, zero_variance_features = fixer.create_fixed_comprehensive_dataset()
    
    # Save results
    dataset_path, analysis_path = fixer.save_fixed_dataset(merged_df, valid_features, zero_variance_features)
    
    print("\n" + "=" * 70)
    print("DATA INTEGRATION FIXES COMPLETE")
    print("=" * 70)
    print(f"Fixed dataset: {dataset_path}")
    print(f"Feature analysis: {analysis_path}")
    print(f"Valid features: {len(valid_features)} out of {merged_df.shape[1]-3} total")
    
    return merged_df, valid_features, zero_variance_features

if __name__ == "__main__":
    results = main()
