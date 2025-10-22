#!/usr/bin/env python3
"""
Investigation: Why Physical Activity Has Zero Importance
Analyze NHANES accelerometry data quality and feature engineering issues

Author: Generated for fairness project
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ActivityDataInvestigator:
    """
    Investigate physical activity data quality and feature engineering
    """
    
    def __init__(self, lifestyle_data_dir="/Users/aakashsuresh/fairness/processed_data_new/"):
        self.lifestyle_data_dir = lifestyle_data_dir
        self.acc_df = None
        self.glucose_df = None
        
    def load_raw_activity_data(self):
        """
        Load and examine raw accelerometry data
        """
        print("=== Loading Raw Accelerometry Data ===")
        
        acc_file = f"{self.lifestyle_data_dir}/nhanes_combined_acc.csv"
        self.acc_df = pd.read_csv(acc_file)
        
        print(f"Raw accelerometry data shape: {self.acc_df.shape}")
        print(f"Columns: {list(self.acc_df.columns)}")
        print(f"Unique participants: {self.acc_df['SEQN'].nunique()}")
        
        return self.acc_df
    
    def examine_data_quality(self):
        """
        Examine data quality issues in accelerometry data
        """
        print("\n=== Data Quality Analysis ===")
        
        # Check for missing values
        print("Missing value analysis:")
        missing_stats = self.acc_df.isnull().sum()
        print(missing_stats[missing_stats > 0])
        
        # Check for unusual values
        print("\nUnusual value patterns:")
        for col in self.acc_df.columns:
            if self.acc_df[col].dtype in ['float64', 'int64']:
                unique_vals = self.acc_df[col].nunique()
                if unique_vals < 10:
                    print(f"{col}: {unique_vals} unique values - {self.acc_df[col].unique()}")
        
        # Check for the mysterious 5.397605346934028e-79 value
        mysterious_value = 5.397605346934028e-79
        print(f"\nChecking for mysterious value {mysterious_value}:")
        for col in self.acc_df.columns:
            if self.acc_df[col].dtype in ['float64', 'int64']:
                count = (self.acc_df[col] == mysterious_value).sum()
                if count > 0:
                    print(f"{col}: {count} instances of mysterious value")
    
    def analyze_activity_distributions(self):
        """
        Analyze distributions of activity variables
        """
        print("\n=== Activity Variable Distributions ===")
        
        # Key activity columns
        activity_cols = ['PAXAISMD', 'PAXMVMD', 'PAXSMD', 'PAXTMD']  # Total activity, MVPA, sedentary, total time
        available_cols = [col for col in activity_cols if col in self.acc_df.columns]
        
        if not available_cols:
            print("Standard activity columns not found. Available columns:")
            numeric_cols = [col for col in self.acc_df.columns if self.acc_df[col].dtype in ['float64', 'int64']]
            print(numeric_cols[:10])  # Show first 10 numeric columns
            available_cols = numeric_cols[:4]  # Use first 4 for analysis
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(available_cols[:4]):
            if col in self.acc_df.columns:
                # Remove mysterious values for visualization
                clean_data = self.acc_df[col].replace(5.397605346934028e-79, np.nan)
                clean_data = clean_data.dropna()
                
                if len(clean_data) > 0:
                    axes[i].hist(clean_data, bins=50, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{col} Distribution')
                    axes[i].set_xlabel('Value')
                    axes[i].set_ylabel('Frequency')
                    
                    # Add statistics
                    mean_val = clean_data.mean()
                    median_val = clean_data.median()
                    axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
                    axes[i].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.1f}')
                    axes[i].legend()
                    
                    print(f"{col} statistics:")
                    print(f"  Mean: {mean_val:.2f}")
                    print(f"  Median: {median_val:.2f}")
                    print(f"  Std: {clean_data.std():.2f}")
                    print(f"  Valid values: {len(clean_data)}/{len(self.acc_df)}")
        
        plt.tight_layout()
        plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/finetuning/activity_distributions.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def investigate_feature_engineering_issues(self):
        """
        Investigate how features were engineered in the original analysis
        """
        print("\n=== Feature Engineering Investigation ===")
        
        # Recreate the feature engineering from lifestyle_glucose_analysis.py
        acc_df_copy = self.acc_df.copy()
        
        # Clean column names and convert SEQN
        acc_df_copy.columns = acc_df_copy.columns.str.upper()
        if 'SEQN' in acc_df_copy.columns:
            acc_df_copy['seqn'] = acc_df_copy['SEQN']
        
        # Select meaningful activity features
        activity_features = []
        for col in acc_df_copy.columns:
            if any(x in col.lower() for x in ['pax', 'activity', 'step', 'mvpa', 'sed']):
                if acc_df_copy[col].dtype in ['float64', 'int64']:
                    activity_features.append(col)
        
        print(f"Identified activity features: {activity_features}")
        
        if activity_features:
            # Aggregate by participant
            agg_dict = {col: 'mean' for col in activity_features}
            acc_summary = acc_df_copy.groupby('seqn').agg(agg_dict).reset_index()
            
            # Create meaningful activity variables (as done in original)
            acc_summary['total_activity_counts'] = acc_summary.get('PAXAISMD', 0)
            acc_summary['moderate_vigorous_minutes'] = acc_summary.get('PAXMVMD', 0) 
            acc_summary['sedentary_minutes'] = acc_summary.get('PAXSMD', 0)
            acc_summary['wear_time_minutes'] = acc_summary.get('PAXTMD', 0)
            
            # Calculate activity ratios
            acc_summary['mvpa_ratio'] = (acc_summary['moderate_vigorous_minutes'] / 
                                       (acc_summary['wear_time_minutes'] + 1))
            acc_summary['sedentary_ratio'] = (acc_summary['sedentary_minutes'] / 
                                            (acc_summary['wear_time_minutes'] + 1))
            
            print(f"\nEngineered features summary:")
            engineered_features = ['total_activity_counts', 'moderate_vigorous_minutes', 
                                 'sedentary_minutes', 'wear_time_minutes', 'mvpa_ratio', 'sedentary_ratio']
            
            for feature in engineered_features:
                if feature in acc_summary.columns:
                    values = acc_summary[feature].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(values) > 0:
                        print(f"{feature}:")
                        print(f"  Mean: {values.mean():.3f}")
                        print(f"  Std: {values.std():.3f}")
                        print(f"  Min: {values.min():.3f}")
                        print(f"  Max: {values.max():.3f}")
                        print(f"  Zero values: {(values == 0).sum()}/{len(values)}")
                        print(f"  Variance: {values.var():.3f}")
            
            return acc_summary
        else:
            print("No activity features found!")
            return None
    
    def load_glucose_data_for_correlation(self):
        """
        Load glucose data to examine correlations with activity
        """
        print("\n=== Loading Glucose Data for Correlation Analysis ===")
        
        # Load glucose targets
        glucose_file = "/Users/aakashsuresh/fairness/processed_data_nhanes_lab/fasting_glucose_processed.csv"
        hba1c_file = "/Users/aakashsuresh/fairness/processed_data_nhanes_lab/glycohemoglobin_processed.csv"
        
        if pd.io.common.file_exists(glucose_file) and pd.io.common.file_exists(hba1c_file):
            glucose_df = pd.read_csv(glucose_file)[['seqn', 'lbxglu']]
            hba1c_df = pd.read_csv(hba1c_file)[['seqn', 'lbxgh']]
            
            # Merge targets
            self.glucose_df = glucose_df.merge(hba1c_df, on='seqn', how='inner')
            self.glucose_df.columns = ['seqn', 'glucose', 'hba1c']
            
            print(f"Loaded glucose data for {len(self.glucose_df)} participants")
            return self.glucose_df
        else:
            print("Glucose data files not found")
            return None
    
    def analyze_activity_glucose_correlations(self, acc_summary):
        """
        Analyze correlations between activity and glucose
        """
        if acc_summary is None or self.glucose_df is None:
            print("Cannot perform correlation analysis - missing data")
            return
        
        print("\n=== Activity-Glucose Correlation Analysis ===")
        
        # Merge activity and glucose data
        merged_df = acc_summary.merge(self.glucose_df, on='seqn', how='inner')
        print(f"Merged dataset: {len(merged_df)} participants")
        
        # Calculate correlations
        activity_features = ['total_activity_counts', 'moderate_vigorous_minutes', 
                           'sedentary_minutes', 'wear_time_minutes', 'mvpa_ratio', 'sedentary_ratio']
        
        correlations = {}
        for feature in activity_features:
            if feature in merged_df.columns:
                # Clean data (remove inf, nan, zeros if they dominate)
                clean_activity = merged_df[feature].replace([np.inf, -np.inf], np.nan)
                clean_glucose = merged_df['glucose']
                
                # Remove rows where either is NaN
                mask = ~(clean_activity.isna() | clean_glucose.isna())
                if mask.sum() > 10:  # Need at least 10 valid pairs
                    corr_glucose = clean_activity[mask].corr(clean_glucose[mask])
                    corr_hba1c = clean_activity[mask].corr(merged_df['hba1c'][mask])
                    
                    correlations[feature] = {
                        'glucose_corr': corr_glucose,
                        'hba1c_corr': corr_hba1c,
                        'valid_pairs': mask.sum()
                    }
                    
                    print(f"{feature}:")
                    print(f"  Glucose correlation: {corr_glucose:.4f}")
                    print(f"  HbA1c correlation: {corr_hba1c:.4f}")
                    print(f"  Valid pairs: {mask.sum()}")
        
        # Create correlation heatmap
        if correlations:
            corr_matrix = pd.DataFrame({
                feature: [data['glucose_corr'], data['hba1c_corr']] 
                for feature, data in correlations.items()
            }, index=['Glucose', 'HbA1c'])
            
            plt.figure(figsize=(12, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                       fmt='.3f', cbar_kws={'label': 'Correlation'})
            plt.title('Activity-Glucose Correlations')
            plt.tight_layout()
            plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/finetuning/activity_glucose_correlations.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
        
        return correlations
    
    def propose_improved_features(self):
        """
        Propose improved feature engineering approaches
        """
        print("\n=== Proposed Feature Engineering Improvements ===")
        
        improvements = [
            "1. Handle mysterious values (5.397605346934028e-79) properly - likely missing data codes",
            "2. Use median aggregation instead of mean to reduce outlier impact",
            "3. Create activity intensity categories (low/medium/high) instead of continuous",
            "4. Calculate weekly patterns (weekday vs weekend activity)",
            "5. Create activity consistency metrics (standard deviation across days)",
            "6. Use log-transformation for highly skewed activity variables",
            "7. Create interaction features (age × activity, BMI × activity)",
            "8. Implement proper missing data imputation instead of filling with 0"
        ]
        
        for improvement in improvements:
            print(improvement)
        
        return improvements
    
    def run_complete_investigation(self):
        """
        Run complete activity data investigation
        """
        print("Physical Activity Data Quality Investigation")
        print("=" * 60)
        
        # Load and examine raw data
        self.load_raw_activity_data()
        self.examine_data_quality()
        
        # Analyze distributions
        self.analyze_activity_distributions()
        
        # Investigate feature engineering
        acc_summary = self.investigate_feature_engineering_issues()
        
        # Load glucose data and analyze correlations
        self.load_glucose_data_for_correlation()
        correlations = self.analyze_activity_glucose_correlations(acc_summary)
        
        # Propose improvements
        improvements = self.propose_improved_features()
        
        print("\n" + "=" * 60)
        print("INVESTIGATION COMPLETE")
        print("=" * 60)
        
        return {
            'acc_summary': acc_summary,
            'correlations': correlations,
            'improvements': improvements
        }

def main():
    """
    Main execution function
    """
    investigator = ActivityDataInvestigator()
    results = investigator.run_complete_investigation()
    return results

if __name__ == "__main__":
    results = main()
