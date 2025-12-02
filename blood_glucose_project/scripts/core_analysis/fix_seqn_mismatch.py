#!/usr/bin/env python3
"""
Fix SEQN Mismatch by Using Matching NHANES Cycles
Load glucose data from 2011-2014 to match existing activity/dietary data

Critical Issue: Current glucose data (2017-2020) has SEQN 109264-124822
                Activity/dietary data (2011-2014) has SEQN 62161-83731
                ZERO overlap!

Solution: Load glucose data from 2011-2014 cycles to match activity data

Author: Generated for fairness project
Date: October 2025
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SEQNMismatchFixer:
    """
    Fix SEQN mismatch by loading matching NHANES cycles
    """
    
    def __init__(self):
        self.processed_data_new = "/Users/aakashsuresh/fairness/processed_data_new/"
        self.processed_data_lab = "/Users/aakashsuresh/fairness/processed_data_nhanes_lab/"
        self.mysterious_value = 5.397605346934028e-79
        
    def load_matching_glucose_data(self):
        """
        Load glucose and HbA1c data from 2011-2014 to match activity data
        """
        print("=== Loading Matching Glucose Data (2011-2014) ===")
        
        # Load glucose data from matching cycles
        glucose_files = [
            "2011-2012_GLU_G.csv",
            "2013-2014_GLU_H.csv"
        ]
        
        hba1c_files = [
            "2011-2012_GHB_G.csv", 
            "2013-2014_GHB_H.csv"
        ]
        
        glucose_dfs = []
        hba1c_dfs = []
        
        # Load glucose files
        for file in glucose_files:
            file_path = os.path.join(self.processed_data_new, file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Loaded {file}: {df.shape[0]} rows")
                print(f"  SEQN range: {df['SEQN'].min():.0f} - {df['SEQN'].max():.0f}")
                glucose_dfs.append(df)
            else:
                print(f"File not found: {file}")
        
        # Load HbA1c files  
        for file in hba1c_files:
            file_path = os.path.join(self.processed_data_new, file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Loaded {file}: {df.shape[0]} rows")
                print(f"  SEQN range: {df['SEQN'].min():.0f} - {df['SEQN'].max():.0f}")
                hba1c_dfs.append(df)
            else:
                print(f"File not found: {file}")
        
        if not glucose_dfs or not hba1c_dfs:
            print("ERROR: Could not load matching glucose/HbA1c data")
            return None
        
        # Combine glucose data
        glucose_combined = pd.concat(glucose_dfs, ignore_index=True)
        hba1c_combined = pd.concat(hba1c_dfs, ignore_index=True)
        
        print(f"\nCombined glucose data: {glucose_combined.shape[0]} rows")
        print(f"Combined HbA1c data: {hba1c_combined.shape[0]} rows")
        print(f"Glucose SEQN range: {glucose_combined['SEQN'].min():.0f} - {glucose_combined['SEQN'].max():.0f}")
        print(f"HbA1c SEQN range: {hba1c_combined['SEQN'].min():.0f} - {hba1c_combined['SEQN'].max():.0f}")
        
        # Merge glucose and HbA1c
        glucose_clean = glucose_combined[['SEQN', 'LBXGLU']].copy()
        hba1c_clean = hba1c_combined[['SEQN', 'LBXGH']].copy()
        
        glucose_clean.columns = ['seqn', 'glucose']
        hba1c_clean.columns = ['seqn', 'hba1c']
        
        targets_df = glucose_clean.merge(hba1c_clean, on='seqn', how='inner')
        
        print(f"Merged targets: {len(targets_df)} participants")
        print(f"Target SEQN range: {targets_df['seqn'].min():.0f} - {targets_df['seqn'].max():.0f}")
        
        return targets_df
    
    def load_matching_demographics(self, target_seqn_range):
        """
        Load demographics data that matches the target SEQN range
        """
        print("\n=== Loading Matching Demographics ===")
        
        # Try to find demographics data for 2011-2014
        demo_files = [
            "2011-2012_Demographics.csv",
            "2013-2014_Demographics.csv",
            "DEMO_G.csv",  # 2011-2012
            "DEMO_H.csv"   # 2013-2014
        ]
        
        demo_dfs = []
        
        for file in demo_files:
            file_path = os.path.join(self.processed_data_new, file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Found {file}: {df.shape[0]} rows")
                if 'SEQN' in df.columns:
                    print(f"  SEQN range: {df['SEQN'].min():.0f} - {df['SEQN'].max():.0f}")
                    demo_dfs.append(df)
        
        if demo_dfs:
            demo_combined = pd.concat(demo_dfs, ignore_index=True)
            print(f"Combined demographics: {demo_combined.shape[0]} rows")
            
            # Select relevant columns
            demo_cols = ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2']
            available_cols = [col for col in demo_cols if col in demo_combined.columns]
            
            if available_cols:
                demo_clean = demo_combined[available_cols].copy()
                demo_clean.columns = ['seqn', 'age', 'gender', 'race_ethnicity', 'education_level'][:len(available_cols)]
                
                # Filter to matching SEQN range
                min_seqn, max_seqn = target_seqn_range
                demo_filtered = demo_clean[
                    (demo_clean['seqn'] >= min_seqn) & (demo_clean['seqn'] <= max_seqn)
                ]
                
                print(f"Demographics in target range: {len(demo_filtered)} participants")
                return demo_filtered
        
        print("No matching demographics data found")
        return None
    
    def load_activity_data_fixed(self):
        """
        Load activity data (already confirmed to be 2011-2014)
        """
        print("\n=== Loading Activity Data (2011-2014) ===")
        
        acc_file = os.path.join(self.processed_data_new, "nhanes_combined_acc.csv")
        if os.path.exists(acc_file):
            acc_df = pd.read_csv(acc_file)
            print(f"Activity data: {acc_df.shape[0]} rows")
            print(f"SEQN range: {acc_df['SEQN'].min():.0f} - {acc_df['SEQN'].max():.0f}")
            print(f"Unique participants: {acc_df['SEQN'].nunique()}")
            
            # Clean data
            acc_df = acc_df.replace(self.mysterious_value, np.nan)
            acc_df['seqn'] = acc_df['SEQN'].astype(float)
            
            # Activity mapping
            activity_mapping = {
                'PAXAISMD': 'total_activity_counts',
                'PAXTMD': 'wear_time_minutes',
                'PAXMTSD': 'moderate_activity_minutes',
                'PAXVMD': 'vigorous_activity_minutes',
                'PAXLXSD': 'light_activity_minutes',
                'PAXSSNDP': 'sedentary_minutes'
            }
            
            available_cols = ['seqn'] + [col for col in activity_mapping.keys() if col in acc_df.columns]
            acc_clean = acc_df[available_cols].copy()
            
            rename_dict = {'seqn': 'seqn'}
            rename_dict.update({k: v for k, v in activity_mapping.items() if k in acc_clean.columns})
            acc_clean = acc_clean.rename(columns=rename_dict)
            
            # Aggregate by participant
            agg_dict = {col: 'median' for col in acc_clean.columns if col != 'seqn'}
            acc_summary = acc_clean.groupby('seqn').agg(agg_dict).reset_index()
            
            # Create derived features
            if 'moderate_activity_minutes' in acc_summary.columns and 'vigorous_activity_minutes' in acc_summary.columns:
                acc_summary['mvpa_minutes'] = (
                    acc_summary['moderate_activity_minutes'].fillna(0) + 
                    acc_summary['vigorous_activity_minutes'].fillna(0)
                )
            
            if 'wear_time_minutes' in acc_summary.columns:
                wear_time_safe = acc_summary['wear_time_minutes'].fillna(1440).replace(0, 1440)
                
                if 'mvpa_minutes' in acc_summary.columns:
                    acc_summary['mvpa_ratio'] = acc_summary['mvpa_minutes'] / wear_time_safe
                
                if 'sedentary_minutes' in acc_summary.columns:
                    acc_summary['sedentary_ratio'] = acc_summary['sedentary_minutes'].fillna(0) / wear_time_safe
                
                if 'light_activity_minutes' in acc_summary.columns:
                    acc_summary['light_activity_ratio'] = acc_summary['light_activity_minutes'].fillna(0) / wear_time_safe
            
            if 'total_activity_counts' in acc_summary.columns:
                acc_summary['activity_level'] = pd.cut(
                    acc_summary['total_activity_counts'].fillna(0),
                    bins=[0, 1000000, 3000000, np.inf],
                    labels=[0, 1, 2]
                )
                acc_summary['log_total_activity'] = np.log1p(acc_summary['total_activity_counts'].fillna(0))
            
            print(f"Processed activity data: {acc_summary.shape}")
            return acc_summary
        
        print("Activity data not found")
        return None
    
    def load_dietary_data_fixed(self):
        """
        Load dietary data (already confirmed to be 2011-2014)
        """
        print("\n=== Loading Dietary Data (2011-2014) ===")
        
        diet_files = [
            "2011-2012_Dietary.csv",
            "2013-2014_Dietary.csv",
            "nhanes_combined_diet.csv"
        ]
        
        for file in diet_files:
            diet_path = os.path.join(self.processed_data_new, file)
            if os.path.exists(diet_path):
                print(f"Loading {file}")
                diet_df = pd.read_csv(diet_path)
                print(f"Dietary data: {diet_df.shape}")
                
                if 'SEQN' in diet_df.columns:
                    diet_df['seqn'] = diet_df['SEQN'].astype(float)
                    print(f"SEQN range: {diet_df['seqn'].min():.0f} - {diet_df['seqn'].max():.0f}")
                    
                    # Select dietary features
                    dietary_keywords = ['kcal', 'energy', 'carb', 'sugar', 'fiber', 'fat', 'protein', 'sodium']
                    dietary_features = ['seqn']
                    
                    for col in diet_df.columns:
                        if any(keyword in col.lower() for keyword in dietary_keywords):
                            if diet_df[col].dtype in ['float64', 'int64']:
                                dietary_features.append(col)
                    
                    if len(dietary_features) > 1:
                        diet_clean = diet_df[dietary_features[:16]].copy()  # Limit to 15 features
                        print(f"Selected {len(dietary_features)-1} dietary features")
                        return diet_clean
        
        print("No suitable dietary data found")
        return None
    
    def create_complete_dataset_with_matching_seqn(self):
        """
        Create complete dataset with matching SEQN ranges
        """
        print("Creating Complete Dataset with Matching SEQN")
        print("=" * 60)
        
        # Load matching glucose data (2011-2014)
        targets_df = self.load_matching_glucose_data()
        if targets_df is None:
            print("ERROR: Could not load matching glucose data")
            return None, [], []
        
        target_seqn_range = (targets_df['seqn'].min(), targets_df['seqn'].max())
        
        # Load other data sources
        demo_df = self.load_matching_demographics(target_seqn_range)
        activity_df = self.load_activity_data_fixed()
        diet_df = self.load_dietary_data_fixed()
        
        # Start merging
        print(f"\n=== Merging with Matching SEQN ===")
        merged_df = targets_df.copy()
        print(f"Starting with targets: {len(merged_df)} participants")
        
        # Merge demographics
        if demo_df is not None:
            overlap = len(set(merged_df['seqn']) & set(demo_df['seqn']))
            print(f"Demographics overlap: {overlap} participants")
            merged_df = merged_df.merge(demo_df, on='seqn', how='left')
            print(f"After demographics: {len(merged_df)} participants, {merged_df.shape[1]} features")
        
        # Merge activity
        if activity_df is not None:
            overlap = len(set(merged_df['seqn']) & set(activity_df['seqn']))
            print(f"Activity overlap: {overlap} participants")
            merged_df = merged_df.merge(activity_df, on='seqn', how='left')
            print(f"After activity: {len(merged_df)} participants, {merged_df.shape[1]} features")
        
        # Merge dietary
        if diet_df is not None:
            overlap = len(set(merged_df['seqn']) & set(diet_df['seqn']))
            print(f"Dietary overlap: {overlap} participants")
            merged_df = merged_df.merge(diet_df, on='seqn', how='left')
            print(f"After dietary: {len(merged_df)} participants, {merged_df.shape[1]} features")
        
        # Create interaction features
        if 'age' in merged_df.columns and 'total_activity_counts' in merged_df.columns:
            merged_df['age_activity_interaction'] = merged_df['age'] * merged_df['total_activity_counts']
        
        if 'gender' in merged_df.columns and 'sedentary_ratio' in merged_df.columns:
            merged_df['gender_sedentary_interaction'] = merged_df['gender'] * merged_df['sedentary_ratio']
        
        # Handle missing data
        print(f"\n=== Handling Missing Data ===")
        for col in merged_df.columns:
            if col in ['seqn', 'glucose', 'hba1c']:
                continue
            
            if merged_df[col].dtype in ['object', 'category']:
                mode_val = merged_df[col].mode()
                if len(mode_val) > 0:
                    merged_df[col] = merged_df[col].fillna(mode_val[0])
            else:
                if 'activity' in col.lower() or 'dietary' in col.lower():
                    merged_df[col] = merged_df[col].fillna(0)
                else:
                    merged_df[col] = merged_df[col].fillna(merged_df[col].median())
        
        # Apply inclusion criteria
        print(f"\n=== Applying Inclusion Criteria ===")
        initial_count = len(merged_df)
        
        if 'age' in merged_df.columns:
            merged_df = merged_df[merged_df['age'] >= 18]
            print(f"After age ≥18: {len(merged_df)} participants")
        
        outlier_mask = (merged_df['glucose'] <= 600) & (merged_df['hba1c'] <= 18)
        merged_df = merged_df[outlier_mask]
        print(f"After outlier removal: {len(merged_df)} participants")
        
        # Encode categorical variables
        categorical_cols = ['gender', 'race_ethnicity', 'education_level', 'activity_level']
        for col in categorical_cols:
            if col in merged_df.columns and merged_df[col].dtype in ['object', 'category']:
                le = LabelEncoder()
                merged_df[col] = le.fit_transform(merged_df[col].astype(str))
        
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
        print(f"Valid features: {valid_features}")
        
        if zero_variance_features:
            print(f"Zero variance features: {zero_variance_features}")
        
        return merged_df, valid_features, zero_variance_features
    
    def save_complete_dataset(self, merged_df, valid_features, zero_variance_features):
        """
        Save the complete dataset with matching SEQN
        """
        if merged_df is None:
            return None, None
        
        print(f"\n=== Saving Complete Dataset ===")
        
        # Save complete dataset
        output_path = '/Users/aakashsuresh/fairness/blood_glucose_project/complete_lifestyle_dataset.csv'
        merged_df.to_csv(output_path, index=False)
        print(f"Complete dataset saved to: {output_path}")
        
        # Save feature analysis
        feature_analysis = pd.DataFrame({
            'Feature': merged_df.columns,
            'Has_Variance': [col in valid_features for col in merged_df.columns],
            'Data_Type': [str(merged_df[col].dtype) for col in merged_df.columns],
            'Non_Null_Count': [merged_df[col].notna().sum() for col in merged_df.columns],
            'Variance': [merged_df[col].var() if merged_df[col].dtype in ['float64', 'int64'] else 0 for col in merged_df.columns]
        })
        
        analysis_path = '/Users/aakashsuresh/fairness/blood_glucose_project/complete_feature_analysis.csv'
        feature_analysis.to_csv(analysis_path, index=False)
        print(f"Feature analysis saved to: {analysis_path}")
        
        return output_path, analysis_path

def main():
    """
    Main execution function
    """
    fixer = SEQNMismatchFixer()
    
    # Create complete dataset with matching SEQN
    merged_df, valid_features, zero_variance_features = fixer.create_complete_dataset_with_matching_seqn()
    
    if merged_df is not None:
        # Save results
        dataset_path, analysis_path = fixer.save_complete_dataset(merged_df, valid_features, zero_variance_features)
        
        print("\n" + "=" * 60)
        print("SEQN MISMATCH FIX COMPLETE")
        print("=" * 60)
        print(f"Complete dataset: {dataset_path}")
        print(f"Feature analysis: {analysis_path}")
        print(f"Valid features: {len(valid_features)} out of {merged_df.shape[1]-3} total")
        print(f"Dataset size: {merged_df.shape[0]} participants")
        
        if len(valid_features) > 4:
            print("SUCCESS: More than 4 features with variance!")
            print("Ready for comprehensive feature importance analysis")
        else:
            print("Still limited features - may need additional data sources")
    
    return merged_df, valid_features, zero_variance_features

if __name__ == "__main__":
    results = main()
