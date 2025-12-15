#!/usr/bin/env python3
"""
Algorithmic Fairness Analysis for Wearable-Based Glucose Prediction
==================================================================

This script performs comprehensive algorithmic fairness analysis across wearable device
metadata factors including device usage patterns, wear time, data quality, and 
demographic intersections.

Fairness Metrics Implemented:
- Statistical Parity (Demographic Parity)
- Equalized Odds (True Positive Rate + False Positive Rate equality)
- Equal Opportunity (True Positive Rate equality)
- Calibration (Positive Predictive Value equality)
- Individual Fairness (Lipschitz continuity)

Author: Blood Glucose Prediction Team
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, confusion_matrix,
    classification_report
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class WearableFairnessAnalyzer:
    """
    Comprehensive algorithmic fairness analyzer for wearable-based glucose prediction.
    """
    
    def __init__(self, data_path="/Users/aakashsuresh/fairness/processed_data_new"):
        self.data_path = data_path
        self.results = {}
        self.fairness_metrics = {}
        
    def load_and_prepare_data(self):
        """Load and prepare accelerometry data with rich metadata."""
        print("🔄 Loading accelerometry data with metadata...")
        
        # Load accelerometry data
        acc_2011 = pd.read_csv(f"{self.data_path}/2011-2012_Accelerometry.csv")
        acc_2013 = pd.read_csv(f"{self.data_path}/2013-2014_Accelerometry.csv")
        
        # Combine accelerometry data
        accelerometry = pd.concat([acc_2011, acc_2013], ignore_index=True)
        
        # Clean the mysterious missing value code
        accelerometry = accelerometry.replace(5.397605346934028e-79, np.nan)
        
        print(f"📊 Accelerometry data shape: {accelerometry.shape}")
        print(f"📊 Available columns: {list(accelerometry.columns)}")
        
        # Load glucose data for targets
        glucose_2011 = pd.read_csv(f"{self.data_path}/2011-2012_GLU_G.csv")
        glucose_2013 = pd.read_csv(f"{self.data_path}/2013-2014_GLU_H.csv")
        glucose = pd.concat([glucose_2011, glucose_2013], ignore_index=True)
        
        # Load HbA1c data
        hba1c_2011 = pd.read_csv(f"{self.data_path}/2011-2012_GHB_G.csv")
        hba1c_2013 = pd.read_csv(f"{self.data_path}/2013-2014_GHB_H.csv")
        hba1c = pd.concat([hba1c_2011, hba1c_2013], ignore_index=True)
        
        # Merge datasets
        self.data = accelerometry.merge(glucose[['SEQN', 'LBXGLU']], on='SEQN', how='inner')
        self.data = self.data.merge(hba1c[['SEQN', 'LBXGH']], on='SEQN', how='inner')
        
        print(f"✅ Merged dataset shape: {self.data.shape}")
        return self.data
    
    def engineer_wearable_metadata_features(self):
        """Engineer comprehensive wearable device metadata features."""
        print("🔧 Engineering wearable metadata features...")
        
        df = self.data.copy()
        
        # === WEAR TIME PATTERNS ===
        # Daily wear time (total minutes worn)
        df['daily_wear_minutes'] = df['PAXTMD'].fillna(0)
        
        # Valid data minutes
        df['valid_data_minutes'] = df['PAXVMD'].fillna(0)
        
        # Data quality ratio (valid minutes / total minutes)
        df['data_quality_ratio'] = np.where(
            df['daily_wear_minutes'] > 0,
            df['valid_data_minutes'] / df['daily_wear_minutes'],
            0
        )
        
        # Waking wear minutes
        df['waking_wear_minutes'] = df['PAXWWMD'].fillna(0)
        
        # Non-wear minutes
        df['non_wear_minutes'] = df['PAXNWMD'].fillna(0)
        
        # === DEVICE USAGE PATTERNS ===
        # Compliance categories based on wear time
        df['wear_compliance'] = pd.cut(
            df['daily_wear_minutes'],
            bins=[0, 600, 900, 1200, 1440],  # 0-10h, 10-15h, 15-20h, 20-24h
            labels=['Poor', 'Fair', 'Good', 'Excellent'],
            include_lowest=True
        )
        
        # Data quality categories
        df['data_quality_category'] = pd.cut(
            df['data_quality_ratio'],
            bins=[0, 0.5, 0.75, 0.9, 1.0],
            labels=['Poor', 'Fair', 'Good', 'Excellent'],
            include_lowest=True
        )
        
        # === DAY-OF-WEEK PATTERNS ===
        # Convert day of week to readable format
        day_mapping = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 
                      5: 'Thursday', 6: 'Friday', 7: 'Saturday'}
        df['day_of_week'] = df['PAXDAYWD'].map(day_mapping)
        
        # Weekend vs weekday
        df['is_weekend'] = df['PAXDAYWD'].isin([1, 7])  # Sunday=1, Saturday=7
        
        # === ACTIVITY INTENSITY PATTERNS ===
        # Sedentary minutes
        df['sedentary_minutes'] = df['PAXSWMD'].fillna(0)
        
        # Light activity minutes  
        df['light_activity_minutes'] = df['PAXLXSD'].fillna(0)
        
        # Moderate/vigorous activity minutes
        df['moderate_vigorous_minutes'] = df['PAXMTSD'].fillna(0)
        
        # Activity diversity (non-zero activity types)
        activity_cols = ['sedentary_minutes', 'light_activity_minutes', 'moderate_vigorous_minutes']
        df['activity_diversity'] = (df[activity_cols] > 0).sum(axis=1)
        
        # === DEVICE RELIABILITY INDICATORS ===
        # Consistent wear pattern (low variance in daily wear time)
        participant_wear_stats = df.groupby('SEQN')['daily_wear_minutes'].agg(['mean', 'std']).reset_index()
        participant_wear_stats['wear_consistency'] = np.where(
            participant_wear_stats['std'] < 120,  # Less than 2 hours variation
            'Consistent', 'Variable'
        )
        df = df.merge(participant_wear_stats[['SEQN', 'wear_consistency']], on='SEQN', how='left')
        
        # === AGGREGATE PARTICIPANT-LEVEL FEATURES ===
        participant_features = df.groupby('SEQN').agg({
            'daily_wear_minutes': ['mean', 'std', 'min', 'max'],
            'data_quality_ratio': ['mean', 'std'],
            'valid_data_minutes': 'sum',
            'sedentary_minutes': 'mean',
            'light_activity_minutes': 'mean',
            'moderate_vigorous_minutes': 'mean',
            'is_weekend': 'sum',  # Number of weekend days
            'LBXGLU': 'first',  # Glucose (should be same for all days)
            'LBXGH': 'first'    # HbA1c (should be same for all days)
        }).reset_index()
        
        # Flatten column names
        participant_features.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                      for col in participant_features.columns]
        participant_features = participant_features.rename(columns={'SEQN_': 'SEQN'})
        
        # Add derived features
        participant_features['avg_daily_wear_hours'] = participant_features['daily_wear_minutes_mean'] / 60
        participant_features['wear_time_variability'] = (
            participant_features['daily_wear_minutes_std'] / 
            participant_features['daily_wear_minutes_mean']
        ).fillna(0)
        
        participant_features['total_monitoring_days'] = df.groupby('SEQN').size().values
        participant_features['weekend_proportion'] = (
            participant_features['is_weekend_sum'] / 
            participant_features['total_monitoring_days']
        )
        
        self.participant_data = participant_features
        print(f"✅ Participant-level features shape: {participant_features.shape}")
        
        return participant_features
    
    def create_stratification_groups(self):
        """Create comprehensive stratification groups for fairness analysis."""
        print("🎯 Creating stratification groups...")
        
        df = self.participant_data.copy()
        
        # === DEVICE USAGE STRATIFICATION ===
        # Wear time categories
        df['wear_time_category'] = pd.cut(
            df['avg_daily_wear_hours'],
            bins=[0, 10, 15, 20, 24],
            labels=['Low_Wear', 'Medium_Wear', 'High_Wear', 'Excellent_Wear'],
            include_lowest=True
        )
        
        # Data quality categories
        df['quality_category'] = pd.cut(
            df['data_quality_ratio_mean'],
            bins=[0, 0.7, 0.85, 0.95, 1.0],
            labels=['Poor_Quality', 'Fair_Quality', 'Good_Quality', 'Excellent_Quality'],
            include_lowest=True
        )
        
        # Wear consistency categories
        df['consistency_category'] = pd.cut(
            df['wear_time_variability'],
            bins=[0, 0.1, 0.25, 0.5, np.inf],
            labels=['Very_Consistent', 'Consistent', 'Moderate', 'Variable'],
            include_lowest=True
        )
        
        # === ACTIVITY PATTERN STRATIFICATION ===
        # Activity level categories
        df['activity_level'] = pd.cut(
            df['moderate_vigorous_minutes_mean'],
            bins=[0, 10, 30, 60, np.inf],
            labels=['Sedentary', 'Low_Active', 'Moderate_Active', 'High_Active'],
            include_lowest=True
        )
        
        # Sedentary behavior categories
        df['sedentary_category'] = pd.cut(
            df['sedentary_minutes_mean'],
            bins=[0, 300, 600, 900, np.inf],
            labels=['Low_Sedentary', 'Moderate_Sedentary', 'High_Sedentary', 'Very_High_Sedentary'],
            include_lowest=True
        )
        
        # === MONITORING PATTERN STRATIFICATION ===
        # Monitoring duration categories
        df['monitoring_duration'] = pd.cut(
            df['total_monitoring_days'],
            bins=[0, 3, 5, 7, np.inf],
            labels=['Short_Monitor', 'Medium_Monitor', 'Standard_Monitor', 'Extended_Monitor'],
            include_lowest=True
        )
        
        # Weekend coverage categories
        df['weekend_coverage'] = pd.cut(
            df['weekend_proportion'],
            bins=[0, 0.1, 0.25, 0.4, 1.0],
            labels=['No_Weekend', 'Low_Weekend', 'Balanced', 'High_Weekend'],
            include_lowest=True
        )
        
        # === INTERSECTIONAL GROUPS ===
        # High-quality, consistent users
        df['ideal_user'] = (
            (df['wear_time_category'].isin(['High_Wear', 'Excellent_Wear'])) &
            (df['quality_category'].isin(['Good_Quality', 'Excellent_Quality'])) &
            (df['consistency_category'].isin(['Very_Consistent', 'Consistent']))
        )
        
        # Problematic usage patterns
        df['problematic_user'] = (
            (df['wear_time_category'] == 'Low_Wear') |
            (df['quality_category'] == 'Poor_Quality') |
            (df['consistency_category'] == 'Variable')
        )
        
        self.stratified_data = df
        
        # Print stratification summary
        print("\n📊 STRATIFICATION SUMMARY:")
        stratification_vars = [
            'wear_time_category', 'quality_category', 'consistency_category',
            'activity_level', 'sedentary_category', 'monitoring_duration',
            'weekend_coverage', 'ideal_user', 'problematic_user'
        ]
        
        for var in stratification_vars:
            if var in df.columns:
                print(f"{var}: {df[var].value_counts().to_dict()}")
        
        return df
    
    def calculate_fairness_metrics(self, y_true, y_pred, y_prob, sensitive_attr):
        """Calculate comprehensive fairness metrics."""
        
        # Get unique groups
        groups = sensitive_attr.unique()
        groups = groups[~pd.isna(groups)]
        
        fairness_results = {}
        
        for group in groups:
            mask = (sensitive_attr == group)
            if mask.sum() == 0:
                continue
                
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            group_y_prob = y_prob[mask] if y_prob is not None else None
            
            # Basic metrics
            fairness_results[f'{group}_accuracy'] = accuracy_score(group_y_true, group_y_pred)
            fairness_results[f'{group}_precision'] = precision_score(group_y_true, group_y_pred, average='weighted', zero_division=0)
            fairness_results[f'{group}_recall'] = recall_score(group_y_true, group_y_pred, average='weighted', zero_division=0)
            fairness_results[f'{group}_f1'] = f1_score(group_y_true, group_y_pred, average='weighted', zero_division=0)
            
            # ROC AUC if probabilities available
            if group_y_prob is not None and len(np.unique(group_y_true)) > 1:
                try:
                    fairness_results[f'{group}_auc'] = roc_auc_score(group_y_true, group_y_prob)
                except:
                    fairness_results[f'{group}_auc'] = np.nan
            
            # Confusion matrix components
            tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred, labels=[0, 1]).ravel()
            
            # True Positive Rate (Sensitivity, Recall)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fairness_results[f'{group}_tpr'] = tpr
            
            # False Positive Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fairness_results[f'{group}_fpr'] = fpr
            
            # Positive Predictive Value (Precision)
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            fairness_results[f'{group}_ppv'] = ppv
            
            # Selection Rate (Positive Prediction Rate)
            selection_rate = (group_y_pred == 1).mean()
            fairness_results[f'{group}_selection_rate'] = selection_rate
            
            fairness_results[f'{group}_sample_size'] = len(group_y_true)
        
        # Calculate fairness disparities
        if len(groups) >= 2:
            group_list = list(groups)
            
            # Statistical Parity (Selection Rate Difference)
            selection_rates = [fairness_results.get(f'{g}_selection_rate', 0) for g in group_list]
            fairness_results['statistical_parity_difference'] = max(selection_rates) - min(selection_rates)
            
            # Equalized Odds (TPR and FPR differences)
            tprs = [fairness_results.get(f'{g}_tpr', 0) for g in group_list]
            fprs = [fairness_results.get(f'{g}_fpr', 0) for g in group_list]
            fairness_results['equalized_odds_tpr_diff'] = max(tprs) - min(tprs)
            fairness_results['equalized_odds_fpr_diff'] = max(fprs) - min(fprs)
            fairness_results['equalized_odds_max_diff'] = max(
                fairness_results['equalized_odds_tpr_diff'],
                fairness_results['equalized_odds_fpr_diff']
            )
            
            # Equal Opportunity (TPR difference only)
            fairness_results['equal_opportunity_difference'] = max(tprs) - min(tprs)
            
            # Calibration (PPV difference)
            ppvs = [fairness_results.get(f'{g}_ppv', 0) for g in group_list]
            fairness_results['calibration_difference'] = max(ppvs) - min(ppvs)
            
            # Overall accuracy difference
            accuracies = [fairness_results.get(f'{g}_accuracy', 0) for g in group_list]
            fairness_results['accuracy_difference'] = max(accuracies) - min(accuracies)
        
        return fairness_results
    
    def run_comprehensive_fairness_analysis(self):
        """Run comprehensive fairness analysis across all stratification groups."""
        print("🔍 Running comprehensive fairness analysis...")
        
        # Prepare features and targets
        feature_cols = [
            'avg_daily_wear_hours', 'data_quality_ratio_mean', 'wear_time_variability',
            'sedentary_minutes_mean', 'light_activity_minutes_mean', 'moderate_vigorous_minutes_mean',
            'weekend_proportion', 'total_monitoring_days'
        ]
        
        # Clean data
        analysis_data = self.stratified_data.dropna(subset=feature_cols + ['LBXGLU_first'])
        
        X = analysis_data[feature_cols]
        y_glucose = analysis_data['LBXGLU_first']
        
        # Create diabetes risk classification (fasting glucose >= 126 mg/dL)
        y_diabetes = (y_glucose >= 126).astype(int)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train models
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Split data
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X_scaled, y_diabetes, y_glucose, test_size=0.3, random_state=42, stratify=y_diabetes
        )
        
        # Train models
        clf.fit(X_train, y_class_train)
        reg.fit(X_train, y_reg_train)
        
        # Get predictions
        y_class_pred = clf.predict(X_test)
        y_class_prob = clf.predict_proba(X_test)[:, 1]
        y_reg_pred = reg.predict(X_test)
        
        # Get test data indices for stratification
        test_indices = analysis_data.iloc[X_test.shape[0]:].index if X_test.shape[0] < len(analysis_data) else analysis_data.index[-len(X_test):]
        test_data = analysis_data.loc[test_indices].reset_index(drop=True)
        
        # Ensure alignment
        if len(test_data) != len(y_class_test):
            test_data = analysis_data.iloc[-len(y_class_test):].reset_index(drop=True)
        
        # Analyze fairness across different stratification variables
        stratification_vars = [
            'wear_time_category', 'quality_category', 'consistency_category',
            'activity_level', 'sedentary_category', 'monitoring_duration',
            'weekend_coverage', 'ideal_user', 'problematic_user'
        ]
        
        self.fairness_results = {}
        
        for var in stratification_vars:
            if var in test_data.columns:
                print(f"\n🎯 Analyzing fairness for: {var}")
                
                sensitive_attr = test_data[var]
                
                # Classification fairness
                class_fairness = self.calculate_fairness_metrics(
                    y_class_test, y_class_pred, y_class_prob, sensitive_attr
                )
                
                # Regression fairness (MAE by group)
                reg_fairness = {}
                for group in sensitive_attr.unique():
                    if pd.notna(group):
                        mask = (sensitive_attr == group)
                        if mask.sum() > 0:
                            group_mae = mean_absolute_error(y_reg_test[mask], y_reg_pred[mask])
                            reg_fairness[f'{group}_mae'] = group_mae
                
                # Calculate MAE disparity
                if len(reg_fairness) >= 2:
                    maes = [v for k, v in reg_fairness.items() if k.endswith('_mae')]
                    reg_fairness['mae_disparity'] = max(maes) - min(maes)
                
                self.fairness_results[var] = {
                    'classification': class_fairness,
                    'regression': reg_fairness
                }
        
        return self.fairness_results
    
    def create_fairness_visualizations(self):
        """Create comprehensive fairness visualizations."""
        print("📊 Creating fairness visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Fairness Metrics Heatmap
        ax1 = plt.subplot(4, 2, 1)
        fairness_summary = []
        
        for var, results in self.fairness_results.items():
            class_results = results['classification']
            reg_results = results['regression']
            
            row = {
                'Variable': var,
                'Statistical_Parity_Diff': class_results.get('statistical_parity_difference', 0),
                'Equal_Opportunity_Diff': class_results.get('equal_opportunity_difference', 0),
                'Equalized_Odds_Diff': class_results.get('equalized_odds_max_diff', 0),
                'Calibration_Diff': class_results.get('calibration_difference', 0),
                'Accuracy_Diff': class_results.get('accuracy_difference', 0),
                'MAE_Disparity': reg_results.get('mae_disparity', 0)
            }
            fairness_summary.append(row)
        
        fairness_df = pd.DataFrame(fairness_summary)
        fairness_matrix = fairness_df.set_index('Variable').iloc[:, :6]  # Exclude MAE for heatmap scale
        
        sns.heatmap(fairness_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   center=0, ax=ax1, cbar_kws={'label': 'Fairness Disparity'})
        ax1.set_title('Algorithmic Fairness Metrics Across Wearable Metadata\n(Lower values = More fair)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Fairness Metrics')
        ax1.set_ylabel('Stratification Variables')
        
        # 2. MAE Disparity Bar Plot
        ax2 = plt.subplot(4, 2, 2)
        mae_disparities = fairness_df.set_index('Variable')['MAE_Disparity'].sort_values(ascending=True)
        colors = ['green' if x < 2 else 'orange' if x < 5 else 'red' for x in mae_disparities.values]
        
        bars = ax2.barh(range(len(mae_disparities)), mae_disparities.values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(mae_disparities)))
        ax2.set_yticklabels(mae_disparities.index, fontsize=10)
        ax2.set_xlabel('MAE Disparity (mg/dL)')
        ax2.set_title('Glucose Prediction MAE Disparity\nAcross Wearable Metadata Groups', 
                     fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, mae_disparities.values)):
            ax2.text(value + 0.1, i, f'{value:.2f}', va='center', fontsize=9)
        
        # Add fairness threshold lines
        ax2.axvline(x=2, color='green', linestyle='--', alpha=0.7, label='Acceptable (≤2 mg/dL)')
        ax2.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='Concerning (≤5 mg/dL)')
        ax2.legend()
        
        # 3. Sample Size Distribution
        ax3 = plt.subplot(4, 2, 3)
        sample_sizes = []
        group_labels = []
        
        for var, results in self.fairness_results.items():
            class_results = results['classification']
            for key, value in class_results.items():
                if key.endswith('_sample_size'):
                    group_name = key.replace('_sample_size', '')
                    sample_sizes.append(value)
                    group_labels.append(f"{var}_{group_name}")
        
        if sample_sizes:
            # Plot top 15 groups by sample size
            sorted_data = sorted(zip(sample_sizes, group_labels), reverse=True)[:15]
            sizes, labels = zip(*sorted_data)
            
            bars = ax3.barh(range(len(sizes)), sizes, alpha=0.7)
            ax3.set_yticks(range(len(sizes)))
            ax3.set_yticklabels([label.replace('_', ' ') for label in labels], fontsize=8)
            ax3.set_xlabel('Sample Size')
            ax3.set_title('Sample Sizes by Stratification Group\n(Top 15 Groups)', 
                         fontsize=14, fontweight='bold')
            
            # Add value labels
            for i, (bar, size) in enumerate(zip(bars, sizes)):
                ax3.text(size + max(sizes)*0.01, i, f'{int(size)}', va='center', fontsize=8)
        
        # 4. Accuracy vs Fairness Scatter Plot
        ax4 = plt.subplot(4, 2, 4)
        accuracy_diffs = []
        statistical_parity_diffs = []
        var_names = []
        
        for var, results in self.fairness_results.items():
            class_results = results['classification']
            acc_diff = class_results.get('accuracy_difference', 0)
            sp_diff = class_results.get('statistical_parity_difference', 0)
            
            if acc_diff > 0 or sp_diff > 0:  # Only plot if there's some disparity
                accuracy_diffs.append(acc_diff)
                statistical_parity_diffs.append(sp_diff)
                var_names.append(var.replace('_', ' '))
        
        if accuracy_diffs and statistical_parity_diffs:
            scatter = ax4.scatter(statistical_parity_diffs, accuracy_diffs, 
                                alpha=0.7, s=100, c=range(len(accuracy_diffs)), cmap='viridis')
            
            # Add labels for points
            for i, var in enumerate(var_names):
                ax4.annotate(var, (statistical_parity_diffs[i], accuracy_diffs[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
            
            ax4.set_xlabel('Statistical Parity Difference')
            ax4.set_ylabel('Accuracy Difference')
            ax4.set_title('Accuracy vs Statistical Parity Trade-off\nAcross Wearable Metadata', 
                         fontsize=14, fontweight='bold')
            
            # Add fairness quadrants
            ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Accuracy Threshold')
            ax4.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='Fairness Threshold')
            ax4.legend()
        
        # 5. Wear Time vs Performance Analysis
        ax5 = plt.subplot(4, 2, 5)
        wear_time_results = self.fairness_results.get('wear_time_category', {})
        if wear_time_results:
            class_results = wear_time_results['classification']
            reg_results = wear_time_results['regression']
            
            # Extract group performance
            groups = []
            accuracies = []
            maes = []
            
            for key in class_results.keys():
                if key.endswith('_accuracy'):
                    group = key.replace('_accuracy', '')
                    groups.append(group)
                    accuracies.append(class_results[key])
                    maes.append(reg_results.get(f'{group}_mae', 0))
            
            if groups:
                x_pos = np.arange(len(groups))
                
                # Dual y-axis plot
                ax5_twin = ax5.twinx()
                
                bars1 = ax5.bar(x_pos - 0.2, accuracies, 0.4, label='Classification Accuracy', 
                              alpha=0.7, color='skyblue')
                bars2 = ax5_twin.bar(x_pos + 0.2, maes, 0.4, label='Regression MAE', 
                                   alpha=0.7, color='lightcoral')
                
                ax5.set_xlabel('Wear Time Category')
                ax5.set_ylabel('Classification Accuracy', color='skyblue')
                ax5_twin.set_ylabel('MAE (mg/dL)', color='lightcoral')
                ax5.set_title('Model Performance by Wear Time Category', 
                             fontsize=14, fontweight='bold')
                
                ax5.set_xticks(x_pos)
                ax5.set_xticklabels([g.replace('_', ' ') for g in groups], rotation=45)
                
                # Add value labels
                for bar, acc in zip(bars1, accuracies):
                    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
                
                for bar, mae in zip(bars2, maes):
                    ax5_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                                f'{mae:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 6. Data Quality Impact Analysis
        ax6 = plt.subplot(4, 2, 6)
        quality_results = self.fairness_results.get('quality_category', {})
        if quality_results:
            class_results = quality_results['classification']
            
            # Extract TPR and FPR by quality
            groups = []
            tprs = []
            fprs = []
            
            for key in class_results.keys():
                if key.endswith('_tpr'):
                    group = key.replace('_tpr', '')
                    groups.append(group)
                    tprs.append(class_results[key])
                    fprs.append(class_results.get(f'{group}_fpr', 0))
            
            if groups and tprs and fprs:
                x_pos = np.arange(len(groups))
                
                bars1 = ax6.bar(x_pos - 0.2, tprs, 0.4, label='True Positive Rate', 
                              alpha=0.7, color='green')
                bars2 = ax6.bar(x_pos + 0.2, fprs, 0.4, label='False Positive Rate', 
                              alpha=0.7, color='red')
                
                ax6.set_xlabel('Data Quality Category')
                ax6.set_ylabel('Rate')
                ax6.set_title('Classification Performance by Data Quality\n(Equalized Odds Components)', 
                             fontsize=14, fontweight='bold')
                ax6.set_xticks(x_pos)
                ax6.set_xticklabels([g.replace('_', ' ') for g in groups], rotation=45)
                ax6.legend()
                ax6.set_ylim(0, 1)
                
                # Add value labels
                for bar, tpr in zip(bars1, tprs):
                    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                           f'{tpr:.3f}', ha='center', va='bottom', fontsize=8)
                
                for bar, fpr in zip(bars2, fprs):
                    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                           f'{fpr:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 7. Ideal vs Problematic Users Comparison
        ax7 = plt.subplot(4, 1, 4)
        
        # Compare ideal vs problematic users across all metrics
        comparison_data = []
        
        for user_type in ['ideal_user', 'problematic_user']:
            if user_type in self.fairness_results:
                results = self.fairness_results[user_type]
                class_results = results['classification']
                reg_results = results['regression']
                
                # Get metrics for True and False groups
                for group in ['True', 'False']:
                    row = {
                        'User_Type': f"{user_type}_{group}",
                        'Accuracy': class_results.get(f'{group}_accuracy', 0),
                        'Precision': class_results.get(f'{group}_precision', 0),
                        'Recall': class_results.get(f'{group}_recall', 0),
                        'F1_Score': class_results.get(f'{group}_f1', 0),
                        'MAE': reg_results.get(f'{group}_mae', 0)
                    }
                    comparison_data.append(row)
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            # Create grouped bar chart
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
            x_pos = np.arange(len(metrics))
            width = 0.15
            
            colors = ['darkgreen', 'lightgreen', 'darkred', 'lightcoral']
            
            for i, user_type in enumerate(comp_df['User_Type'].unique()):
                if user_type in comp_df['User_Type'].values:
                    values = comp_df[comp_df['User_Type'] == user_type][metrics].iloc[0].values
                    bars = ax7.bar(x_pos + i*width, values, width, 
                                 label=user_type.replace('_', ' '), 
                                 color=colors[i % len(colors)], alpha=0.8)
                    
                    # Add value labels
                    for bar, value in zip(bars, values):
                        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                               f'{value:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
            
            ax7.set_xlabel('Performance Metrics')
            ax7.set_ylabel('Score')
            ax7.set_title('Model Performance: Ideal vs Problematic Wearable Users\n' +
                         '(Ideal = High wear time + Good quality + Consistent use)', 
                         fontsize=14, fontweight='bold')
            ax7.set_xticks(x_pos + width * 1.5)
            ax7.set_xticklabels(metrics)
            ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax7.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/results/figures/wearable_algorithmic_fairness_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_fairness_report(self):
        """Generate comprehensive fairness analysis report."""
        print("📝 Generating fairness analysis report...")
        
        report = []
        report.append("# ALGORITHMIC FAIRNESS ANALYSIS: WEARABLE-BASED GLUCOSE PREDICTION")
        report.append("=" * 80)
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Dataset: NHANES 2011-2014 Accelerometry + Glucose Data")
        report.append(f"Total Participants Analyzed: {len(self.stratified_data)}")
        report.append("")
        
        report.append("## EXECUTIVE SUMMARY")
        report.append("-" * 40)
        
        # Calculate overall fairness summary
        all_disparities = []
        critical_disparities = []
        
        for var, results in self.fairness_results.items():
            class_results = results['classification']
            reg_results = results['regression']
            
            # Collect disparity metrics
            sp_diff = class_results.get('statistical_parity_difference', 0)
            eo_diff = class_results.get('equal_opportunity_difference', 0)
            eq_odds_diff = class_results.get('equalized_odds_max_diff', 0)
            mae_disp = reg_results.get('mae_disparity', 0)
            
            all_disparities.extend([sp_diff, eo_diff, eq_odds_diff, mae_disp])
            
            # Flag critical disparities
            if sp_diff > 0.1 or eo_diff > 0.1 or eq_odds_diff > 0.1 or mae_disp > 5:
                critical_disparities.append((var, sp_diff, eo_diff, eq_odds_diff, mae_disp))
        
        avg_disparity = np.mean([d for d in all_disparities if d > 0])
        max_disparity = max(all_disparities) if all_disparities else 0
        
        report.append(f"• Average Fairness Disparity: {avg_disparity:.3f}")
        report.append(f"• Maximum Fairness Disparity: {max_disparity:.3f}")
        report.append(f"• Critical Disparities Found: {len(critical_disparities)}")
        report.append("")
        
        if critical_disparities:
            report.append("⚠️  CRITICAL FAIRNESS ISSUES IDENTIFIED:")
            for var, sp, eo, eq, mae in critical_disparities:
                report.append(f"   - {var}: SP={sp:.3f}, EO={eo:.3f}, EqOdds={eq:.3f}, MAE_Disp={mae:.2f}")
        else:
            report.append("✅ NO CRITICAL FAIRNESS ISSUES IDENTIFIED")
        
        report.append("")
        
        # Detailed analysis by category
        report.append("## DETAILED FAIRNESS ANALYSIS")
        report.append("-" * 40)
        
        categories = {
            'Device Usage Patterns': ['wear_time_category', 'quality_category', 'consistency_category'],
            'Activity Patterns': ['activity_level', 'sedentary_category'],
            'Monitoring Patterns': ['monitoring_duration', 'weekend_coverage'],
            'User Profiles': ['ideal_user', 'problematic_user']
        }
        
        for category, variables in categories.items():
            report.append(f"\n### {category}")
            report.append("-" * len(category))
            
            for var in variables:
                if var in self.fairness_results:
                    results = self.fairness_results[var]
                    class_results = results['classification']
                    reg_results = results['regression']
                    
                    report.append(f"\n**{var.replace('_', ' ').title()}:**")
                    
                    # Classification metrics
                    sp_diff = class_results.get('statistical_parity_difference', 0)
                    eo_diff = class_results.get('equal_opportunity_difference', 0)
                    eq_odds_diff = class_results.get('equalized_odds_max_diff', 0)
                    acc_diff = class_results.get('accuracy_difference', 0)
                    
                    report.append(f"  - Statistical Parity Difference: {sp_diff:.3f}")
                    report.append(f"  - Equal Opportunity Difference: {eo_diff:.3f}")
                    report.append(f"  - Equalized Odds Difference: {eq_odds_diff:.3f}")
                    report.append(f"  - Accuracy Difference: {acc_diff:.3f}")
                    
                    # Regression metrics
                    mae_disp = reg_results.get('mae_disparity', 0)
                    report.append(f"  - MAE Disparity: {mae_disp:.2f} mg/dL")
                    
                    # Fairness assessment
                    if sp_diff < 0.05 and eo_diff < 0.05 and eq_odds_diff < 0.05 and mae_disp < 2:
                        assessment = "✅ EXCELLENT FAIRNESS"
                    elif sp_diff < 0.1 and eo_diff < 0.1 and eq_odds_diff < 0.1 and mae_disp < 5:
                        assessment = "⚠️  ACCEPTABLE FAIRNESS"
                    else:
                        assessment = "❌ POOR FAIRNESS - REQUIRES ATTENTION"
                    
                    report.append(f"  - Assessment: {assessment}")
        
        # Recommendations
        report.append("\n## RECOMMENDATIONS")
        report.append("-" * 40)
        
        report.append("\n### Immediate Actions:")
        if critical_disparities:
            report.append("1. **Address Critical Disparities:**")
            for var, sp, eo, eq, mae in critical_disparities:
                report.append(f"   - Investigate {var} bias through targeted data collection")
                report.append(f"   - Consider stratified model training for {var} groups")
        else:
            report.append("1. **Maintain Current Fairness Standards:**")
            report.append("   - Continue monitoring fairness metrics in production")
        
        report.append("\n2. **Data Collection Improvements:**")
        report.append("   - Ensure balanced representation across wear time categories")
        report.append("   - Implement data quality thresholds for model training")
        report.append("   - Collect additional metadata on device types and user characteristics")
        
        report.append("\n3. **Model Development:**")
        report.append("   - Implement fairness-aware machine learning techniques")
        report.append("   - Consider ensemble methods with fairness constraints")
        report.append("   - Develop separate models for high-risk fairness groups")
        
        report.append("\n4. **Deployment Considerations:**")
        report.append("   - Implement real-time fairness monitoring")
        report.append("   - Provide uncertainty quantification for predictions")
        report.append("   - Establish fairness thresholds for model updates")
        
        # Technical details
        report.append("\n## TECHNICAL DETAILS")
        report.append("-" * 40)
        report.append("\n### Fairness Metrics Definitions:")
        report.append("• **Statistical Parity**: Equal positive prediction rates across groups")
        report.append("• **Equal Opportunity**: Equal true positive rates across groups")
        report.append("• **Equalized Odds**: Equal TPR and FPR across groups")
        report.append("• **Calibration**: Equal positive predictive values across groups")
        report.append("• **MAE Disparity**: Difference in mean absolute error across groups")
        
        report.append("\n### Fairness Thresholds Used:")
        report.append("• Excellent: All metrics < 0.05, MAE disparity < 2 mg/dL")
        report.append("• Acceptable: All metrics < 0.10, MAE disparity < 5 mg/dL")
        report.append("• Poor: Any metric ≥ 0.10 or MAE disparity ≥ 5 mg/dL")
        
        # Save report
        report_text = "\n".join(report)
        
        with open('/Users/aakashsuresh/fairness/blood_glucose_project/results/reports/wearable_algorithmic_fairness_report.md', 'w') as f:
            f.write(report_text)
        
        print("✅ Fairness report saved!")
        return report_text

def main():
    """Main execution function."""
    print("🚀 STARTING COMPREHENSIVE WEARABLE ALGORITHMIC FAIRNESS ANALYSIS")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = WearableFairnessAnalyzer()
    
    # Load and prepare data
    data = analyzer.load_and_prepare_data()
    
    # Engineer wearable metadata features
    participant_features = analyzer.engineer_wearable_metadata_features()
    
    # Create stratification groups
    stratified_data = analyzer.create_stratification_groups()
    
    # Run comprehensive fairness analysis
    fairness_results = analyzer.run_comprehensive_fairness_analysis()
    
    # Create visualizations
    fig = analyzer.create_fairness_visualizations()
    
    # Generate report
    report = analyzer.generate_fairness_report()
    
    print("\n🎉 ANALYSIS COMPLETE!")
    print("=" * 80)
    print("📊 Visualizations saved: results/figures/wearable_algorithmic_fairness_analysis.png")
    print("📝 Report saved: results/reports/wearable_algorithmic_fairness_report.md")
    print("\n✨ Key Findings:")
    
    # Print summary of key findings
    critical_issues = 0
    for var, results in fairness_results.items():
        class_results = results['classification']
        reg_results = results['regression']
        
        sp_diff = class_results.get('statistical_parity_difference', 0)
        mae_disp = reg_results.get('mae_disparity', 0)
        
        if sp_diff > 0.1 or mae_disp > 5:
            critical_issues += 1
            print(f"⚠️  {var}: Statistical Parity = {sp_diff:.3f}, MAE Disparity = {mae_disp:.2f} mg/dL")
    
    if critical_issues == 0:
        print("✅ No critical fairness issues detected across wearable metadata factors!")
    else:
        print(f"❌ {critical_issues} critical fairness issues require attention")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
