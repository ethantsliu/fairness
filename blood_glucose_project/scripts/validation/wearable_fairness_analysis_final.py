#!/usr/bin/env python3
"""
Wearable Device Algorithmic Fairness Analysis - Memory Efficient Version
========================================================================

This script analyzes algorithmic fairness across wearable device metadata factors
using a memory-efficient approach with data sampling and chunked processing.

Key Fairness Factors Analyzed:
1. Device Usage Patterns (wear time, data quality)
2. Activity Patterns (sedentary vs active users)
3. Monitoring Compliance (consistent vs irregular users)
4. Temporal Patterns (weekend vs weekday behavior)

Fairness Metrics:
- Statistical Parity (equal positive prediction rates)
- Equalized Odds (equal TPR and FPR across groups)
- Equal Opportunity (equal TPR across groups)
- Calibration (equal PPV across groups)

Author: Blood Glucose Prediction Team
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_sample_data(sample_size=5000):
    """Load a representative sample of the data to avoid memory issues."""
    print(f"📊 Loading sample data (n={sample_size})...")
    
    try:
        # Load glucose data first (smaller files)
        glucose_2011 = pd.read_csv("/Users/aakashsuresh/fairness/processed_data_new/2011-2012_GLU_G.csv")
        glucose_2013 = pd.read_csv("/Users/aakashsuresh/fairness/processed_data_new/2013-2014_GLU_H.csv")
        glucose = pd.concat([glucose_2011, glucose_2013], ignore_index=True)
        
        print(f"Glucose data: {glucose.shape}")
        print(f"Available participants: {glucose['SEQN'].nunique()}")
        
        # Sample participants to reduce memory load
        available_seqns = glucose['SEQN'].unique()
        if len(available_seqns) > sample_size:
            sampled_seqns = np.random.choice(available_seqns, sample_size, replace=False)
            glucose = glucose[glucose['SEQN'].isin(sampled_seqns)]
            print(f"Sampled to {len(sampled_seqns)} participants")
        
        # Load accelerometry data for sampled participants only
        print("Loading accelerometry data...")
        acc_2011 = pd.read_csv("/Users/aakashsuresh/fairness/processed_data_new/2011-2012_Accelerometry.csv")
        acc_2013 = pd.read_csv("/Users/aakashsuresh/fairness/processed_data_new/2013-2014_Accelerometry.csv")
        
        # Filter to sampled participants
        acc_2011 = acc_2011[acc_2011['SEQN'].isin(glucose['SEQN'])]
        acc_2013 = acc_2013[acc_2013['SEQN'].isin(glucose['SEQN'])]
        
        accelerometry = pd.concat([acc_2011, acc_2013], ignore_index=True)
        print(f"Filtered accelerometry data: {accelerometry.shape}")
        
        # Clean mysterious missing values
        accelerometry = accelerometry.replace(5.397605346934028e-79, np.nan)
        
        # Merge datasets
        data = accelerometry.merge(glucose[['SEQN', 'LBXGLU']], on='SEQN', how='inner')
        print(f"Final merged data: {data.shape}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def engineer_wearable_metadata(data):
    """Engineer comprehensive wearable device metadata features."""
    print("🔧 Engineering wearable metadata features...")
    
    df = data.copy()
    
    # === BASIC WEAR METRICS ===
    df['daily_wear_minutes'] = df['PAXTMD'].fillna(0)
    df['valid_data_minutes'] = df['PAXVMD'].fillna(0)
    df['non_wear_minutes'] = df['PAXNWMD'].fillna(0)
    
    # Data quality ratio (valid/total minutes)
    df['data_quality_ratio'] = np.where(
        df['daily_wear_minutes'] > 0,
        df['valid_data_minutes'] / df['daily_wear_minutes'],
        0
    )
    
    # === ACTIVITY METRICS ===
    df['sedentary_minutes'] = df['PAXSWMD'].fillna(0)
    df['light_activity_minutes'] = df['PAXLXSD'].fillna(0) 
    df['moderate_vigorous_minutes'] = df['PAXMTSD'].fillna(0)
    
    # Activity intensity ratio
    df['active_ratio'] = np.where(
        df['daily_wear_minutes'] > 0,
        (df['light_activity_minutes'] + df['moderate_vigorous_minutes']) / df['daily_wear_minutes'],
        0
    )
    
    # === TEMPORAL PATTERNS ===
    df['is_weekend'] = df['PAXDAYWD'].isin([1, 7])  # Sunday=1, Saturday=7
    
    # Day of week (for pattern analysis)
    day_mapping = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 
                  5: 'Thursday', 6: 'Friday', 7: 'Saturday'}
    df['day_name'] = df['PAXDAYWD'].map(day_mapping)
    
    print(f"Engineered features shape: {df.shape}")
    return df

def create_participant_profiles(data):
    """Create participant-level profiles with wearable metadata."""
    print("👤 Creating participant profiles...")
    
    # Aggregate by participant
    profiles = data.groupby('SEQN').agg({
        # Wear time patterns
        'daily_wear_minutes': ['mean', 'std', 'min', 'max', 'count'],
        'data_quality_ratio': ['mean', 'std'],
        'valid_data_minutes': 'sum',
        
        # Activity patterns
        'sedentary_minutes': 'mean',
        'light_activity_minutes': 'mean', 
        'moderate_vigorous_minutes': 'mean',
        'active_ratio': 'mean',
        
        # Temporal patterns
        'is_weekend': 'sum',
        
        # Target variable
        'LBXGLU': 'first'
    }).reset_index()
    
    # Flatten column names
    profiles.columns = ['_'.join(col).strip() if col[1] else col[0] 
                       for col in profiles.columns]
    profiles = profiles.rename(columns={'SEQN_': 'SEQN'})
    
    # === DERIVED FEATURES ===
    profiles['avg_daily_wear_hours'] = profiles['daily_wear_minutes_mean'] / 60
    profiles['total_monitoring_days'] = profiles['daily_wear_minutes_count']
    profiles['weekend_proportion'] = profiles['is_weekend_sum'] / profiles['total_monitoring_days']
    
    # Wear consistency (coefficient of variation)
    profiles['wear_consistency'] = np.where(
        profiles['daily_wear_minutes_mean'] > 0,
        profiles['daily_wear_minutes_std'] / profiles['daily_wear_minutes_mean'],
        np.inf
    )
    
    # === STRATIFICATION CATEGORIES ===
    
    # 1. Wear Time Categories
    profiles['wear_time_category'] = pd.cut(
        profiles['avg_daily_wear_hours'],
        bins=[0, 8, 12, 16, 24],
        labels=['Low_Wear', 'Medium_Wear', 'High_Wear', 'Excellent_Wear'],
        include_lowest=True
    )
    
    # 2. Data Quality Categories  
    profiles['data_quality_category'] = pd.cut(
        profiles['data_quality_ratio_mean'],
        bins=[0, 0.6, 0.8, 0.9, 1.0],
        labels=['Poor_Quality', 'Fair_Quality', 'Good_Quality', 'Excellent_Quality'],
        include_lowest=True
    )
    
    # 3. Wear Consistency Categories
    profiles['consistency_category'] = pd.cut(
        profiles['wear_consistency'],
        bins=[0, 0.2, 0.4, 0.6, np.inf],
        labels=['Very_Consistent', 'Consistent', 'Moderate', 'Variable'],
        include_lowest=True
    )
    
    # 4. Activity Level Categories
    profiles['activity_level'] = pd.cut(
        profiles['moderate_vigorous_minutes_mean'],
        bins=[0, 5, 15, 30, np.inf],
        labels=['Sedentary', 'Low_Active', 'Moderate_Active', 'High_Active'],
        include_lowest=True
    )
    
    # 5. Monitoring Duration Categories
    profiles['monitoring_duration'] = pd.cut(
        profiles['total_monitoring_days'],
        bins=[0, 3, 5, 7, np.inf],
        labels=['Short', 'Medium', 'Standard', 'Extended'],
        include_lowest=True
    )
    
    # === COMPOSITE USER PROFILES ===
    
    # Ideal Users: High wear + Good quality + Consistent
    profiles['ideal_user'] = (
        (profiles['wear_time_category'].isin(['High_Wear', 'Excellent_Wear'])) &
        (profiles['data_quality_category'].isin(['Good_Quality', 'Excellent_Quality'])) &
        (profiles['consistency_category'].isin(['Very_Consistent', 'Consistent']))
    )
    
    # Problematic Users: Low wear OR poor quality OR very variable
    profiles['problematic_user'] = (
        (profiles['wear_time_category'] == 'Low_Wear') |
        (profiles['data_quality_category'] == 'Poor_Quality') |
        (profiles['consistency_category'] == 'Variable')
    )
    
    # High Compliance Users: Long monitoring + consistent wear
    profiles['high_compliance'] = (
        (profiles['monitoring_duration'].isin(['Standard', 'Extended'])) &
        (profiles['consistency_category'].isin(['Very_Consistent', 'Consistent']))
    )
    
    print(f"Participant profiles shape: {profiles.shape}")
    print("\n📊 STRATIFICATION SUMMARY:")
    
    stratification_vars = [
        'wear_time_category', 'data_quality_category', 'consistency_category',
        'activity_level', 'monitoring_duration', 'ideal_user', 'problematic_user'
    ]
    
    for var in stratification_vars:
        print(f"{var}: {profiles[var].value_counts().to_dict()}")
    
    return profiles

def calculate_comprehensive_fairness_metrics(y_true, y_pred, y_prob, sensitive_attr, group_name):
    """Calculate comprehensive fairness metrics including equalized odds and calibration."""
    
    print(f"\n🎯 Analyzing fairness for: {group_name}")
    
    results = {'group_name': group_name}
    group_metrics = {}
    
    # Get unique groups (excluding NaN)
    groups = sensitive_attr.dropna().unique()
    
    if len(groups) < 2:
        print(f"  ⚠️ Insufficient groups for fairness analysis")
        return results
    
    print(f"  Analyzing {len(groups)} groups: {list(groups)}")
    
    for group in groups:
        mask = (sensitive_attr == group)
        if mask.sum() < 10:  # Minimum sample size
            continue
        
        group_y_true = y_true[mask]
        group_y_pred = y_pred[mask]
        group_y_prob = y_prob[mask] if y_prob is not None else None
        
        # Basic performance metrics
        accuracy = accuracy_score(group_y_true, group_y_pred)
        precision = precision_score(group_y_true, group_y_pred, average='binary', zero_division=0)
        recall = recall_score(group_y_true, group_y_pred, average='binary', zero_division=0)
        f1 = f1_score(group_y_true, group_y_pred, average='binary', zero_division=0)
        
        # Confusion matrix for fairness metrics
        try:
            tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred, labels=[0, 1]).ravel()
        except:
            # Handle edge cases
            tn = fp = fn = tp = 0
        
        # Fairness-specific metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Sensitivity)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        
        # Selection rate (for statistical parity)
        selection_rate = (group_y_pred == 1).mean()
        
        group_metrics[str(group)] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tpr': tpr,  # True Positive Rate
            'fpr': fpr,  # False Positive Rate
            'tnr': tnr,  # True Negative Rate
            'ppv': ppv,  # Positive Predictive Value
            'selection_rate': selection_rate,
            'sample_size': len(group_y_true),
            'positive_cases': int(group_y_true.sum()),
            'negative_cases': int(len(group_y_true) - group_y_true.sum())
        }
        
        print(f"    {group}: Acc={accuracy:.3f}, TPR={tpr:.3f}, FPR={fpr:.3f}, "
              f"PPV={ppv:.3f}, N={len(group_y_true)}")
    
    # Calculate fairness disparities
    if len(group_metrics) >= 2:
        
        # Extract metrics for disparity calculation
        accuracies = [m['accuracy'] for m in group_metrics.values()]
        tprs = [m['tpr'] for m in group_metrics.values()]
        fprs = [m['fpr'] for m in group_metrics.values()]
        ppvs = [m['ppv'] for m in group_metrics.values()]
        selection_rates = [m['selection_rate'] for m in group_metrics.values()]
        
        # Fairness metrics
        results['statistical_parity_difference'] = max(selection_rates) - min(selection_rates)
        results['equal_opportunity_difference'] = max(tprs) - min(tprs)
        results['equalized_odds_tpr_difference'] = max(tprs) - min(tprs)
        results['equalized_odds_fpr_difference'] = max(fprs) - min(fprs)
        results['equalized_odds_max_difference'] = max(
            results['equalized_odds_tpr_difference'],
            results['equalized_odds_fpr_difference']
        )
        results['calibration_difference'] = max(ppvs) - min(ppvs)
        results['accuracy_difference'] = max(accuracies) - min(accuracies)
        
        # Overall fairness assessment
        fairness_score = np.mean([
            results['statistical_parity_difference'],
            results['equal_opportunity_difference'],
            results['equalized_odds_max_difference'],
            results['calibration_difference']
        ])
        
        results['overall_fairness_score'] = fairness_score
        
        # Fairness classification
        if fairness_score < 0.05:
            fairness_level = "✅ EXCELLENT"
        elif fairness_score < 0.1:
            fairness_level = "⚠️ ACCEPTABLE"
        elif fairness_score < 0.2:
            fairness_level = "🔶 CONCERNING"
        else:
            fairness_level = "❌ POOR"
        
        results['fairness_assessment'] = fairness_level
        
        print(f"  📊 FAIRNESS METRICS:")
        print(f"    Statistical Parity Diff: {results['statistical_parity_difference']:.3f}")
        print(f"    Equal Opportunity Diff: {results['equal_opportunity_difference']:.3f}")
        print(f"    Equalized Odds Diff: {results['equalized_odds_max_difference']:.3f}")
        print(f"    Calibration Diff: {results['calibration_difference']:.3f}")
        print(f"    Overall Assessment: {fairness_level}")
    
    results['group_metrics'] = group_metrics
    return results

def run_comprehensive_fairness_analysis():
    """Run the complete fairness analysis pipeline."""
    
    print("🚀 COMPREHENSIVE WEARABLE ALGORITHMIC FAIRNESS ANALYSIS")
    print("=" * 70)
    
    # Load sample data
    data = load_sample_data(sample_size=3000)  # Reduced for memory efficiency
    if data is None:
        return None
    
    # Engineer wearable metadata
    data_with_features = engineer_wearable_metadata(data)
    
    # Create participant profiles
    profiles = create_participant_profiles(data_with_features)
    
    # Prepare modeling data
    print("\n🤖 PREPARING MACHINE LEARNING MODEL")
    print("-" * 40)
    
    # Feature selection
    feature_cols = [
        'avg_daily_wear_hours', 'data_quality_ratio_mean', 'wear_consistency',
        'sedentary_minutes_mean', 'active_ratio_mean', 'weekend_proportion'
    ]
    
    # Clean data
    model_data = profiles.dropna(subset=feature_cols + ['LBXGLU_first'])
    print(f"Model data shape: {model_data.shape}")
    
    if len(model_data) < 100:
        print("❌ Insufficient data for modeling")
        return None
    
    # Prepare features and targets
    X = model_data[feature_cols]
    y_glucose = model_data['LBXGLU_first']
    
    # Create diabetes classification (fasting glucose >= 126 mg/dL)
    y_diabetes = (y_glucose >= 126).astype(int)
    
    print(f"Features: {X.columns.tolist()}")
    print(f"Diabetes prevalence: {y_diabetes.mean():.3f} ({y_diabetes.sum()}/{len(y_diabetes)})")
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_diabetes, test_size=0.3, random_state=42, 
        stratify=y_diabetes if y_diabetes.sum() > 10 else None
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Get predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall model accuracy: {overall_accuracy:.3f}")
    
    # Prepare test data for fairness analysis
    test_indices = model_data.index[-len(y_test):]
    test_data = model_data.loc[test_indices].reset_index(drop=True)
    
    # Ensure alignment
    if len(test_data) != len(y_test):
        print("⚠️ Adjusting test data alignment...")
        test_data = model_data.iloc[-len(y_test):].reset_index(drop=True)
    
    print(f"Test data shape: {test_data.shape}")
    
    # === COMPREHENSIVE FAIRNESS ANALYSIS ===
    print("\n🎯 COMPREHENSIVE FAIRNESS ANALYSIS")
    print("=" * 50)
    
    fairness_results = {}
    
    # Define stratification variables for analysis
    stratification_vars = [
        ('wear_time_category', 'Device Wear Time'),
        ('data_quality_category', 'Data Quality'),
        ('consistency_category', 'Wear Consistency'),
        ('activity_level', 'Physical Activity Level'),
        ('monitoring_duration', 'Monitoring Duration'),
        ('ideal_user', 'User Profile (Ideal vs Others)'),
        ('problematic_user', 'User Profile (Problematic vs Others)'),
        ('high_compliance', 'Compliance Level')
    ]
    
    for var_name, display_name in stratification_vars:
        if var_name in test_data.columns:
            sensitive_attr = test_data[var_name].reset_index(drop=True)
            
            fairness_metrics = calculate_comprehensive_fairness_metrics(
                y_test, y_pred, y_prob, sensitive_attr, display_name
            )
            
            fairness_results[var_name] = fairness_metrics
    
    # Create comprehensive visualizations
    create_comprehensive_visualizations(fairness_results)
    
    # Generate detailed report
    generate_comprehensive_report(fairness_results, len(model_data), overall_accuracy)
    
    print("\n🎉 ANALYSIS COMPLETE!")
    print("=" * 50)
    print("📊 Visualizations: results/figures/comprehensive_wearable_fairness.png")
    print("📝 Report: results/reports/comprehensive_wearable_fairness_report.md")
    
    return fairness_results

def create_comprehensive_visualizations(fairness_results):
    """Create comprehensive fairness visualizations."""
    print("\n📊 Creating comprehensive visualizations...")
    
    try:
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Fairness Metrics Heatmap
        ax1 = plt.subplot(3, 3, 1)
        
        fairness_data = []
        for var, results in fairness_results.items():
            if 'statistical_parity_difference' in results:
                fairness_data.append({
                    'Factor': var.replace('_', ' ').title(),
                    'Statistical_Parity': results.get('statistical_parity_difference', 0),
                    'Equal_Opportunity': results.get('equal_opportunity_difference', 0),
                    'Equalized_Odds': results.get('equalized_odds_max_difference', 0),
                    'Calibration': results.get('calibration_difference', 0),
                    'Overall_Score': results.get('overall_fairness_score', 0)
                })
        
        if fairness_data:
            fairness_df = pd.DataFrame(fairness_data)
            fairness_matrix = fairness_df.set_index('Factor')[['Statistical_Parity', 'Equal_Opportunity', 
                                                              'Equalized_Odds', 'Calibration']]
            
            sns.heatmap(fairness_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                       center=0.1, ax=ax1, cbar_kws={'label': 'Fairness Disparity'})
            ax1.set_title('Algorithmic Fairness Metrics Heatmap\n(Lower = More Fair)', fontweight='bold')
        
        # 2. Overall Fairness Scores
        ax2 = plt.subplot(3, 3, 2)
        
        if fairness_data:
            factors = [d['Factor'] for d in fairness_data]
            scores = [d['Overall_Score'] for d in fairness_data]
            
            colors = ['green' if s < 0.05 else 'orange' if s < 0.1 else 'red' for s in scores]
            bars = ax2.barh(factors, scores, color=colors, alpha=0.7)
            
            ax2.set_xlabel('Overall Fairness Score')
            ax2.set_title('Overall Fairness Assessment\nby Wearable Factor', fontweight='bold')
            
            # Add threshold lines
            ax2.axvline(x=0.05, color='green', linestyle='--', alpha=0.7, label='Excellent (≤0.05)')
            ax2.axvline(x=0.1, color='orange', linestyle='--', alpha=0.7, label='Acceptable (≤0.10)')
            ax2.legend()
        
        # 3. Sample Size Distribution
        ax3 = plt.subplot(3, 3, 3)
        
        all_groups = []
        all_sizes = []
        
        for var, results in fairness_results.items():
            if 'group_metrics' in results:
                for group, metrics in results['group_metrics'].items():
                    all_groups.append(f"{var}_{group}")
                    all_sizes.append(metrics['sample_size'])
        
        if all_groups:
            # Show top 10 by sample size
            sorted_data = sorted(zip(all_sizes, all_groups), reverse=True)[:10]
            sizes, groups = zip(*sorted_data)
            
            ax3.barh(range(len(sizes)), sizes, alpha=0.7, color='skyblue')
            ax3.set_yticks(range(len(sizes)))
            ax3.set_yticklabels([g.replace('_', ' ') for g in groups], fontsize=8)
            ax3.set_xlabel('Sample Size')
            ax3.set_title('Sample Sizes by Group\n(Top 10)', fontweight='bold')
        
        # 4. Wear Time Category Analysis
        ax4 = plt.subplot(3, 3, 4)
        
        if 'wear_time_category' in fairness_results:
            wear_metrics = fairness_results['wear_time_category'].get('group_metrics', {})
            
            if wear_metrics:
                categories = list(wear_metrics.keys())
                accuracies = [wear_metrics[cat]['accuracy'] for cat in categories]
                tprs = [wear_metrics[cat]['tpr'] for cat in categories]
                
                x_pos = np.arange(len(categories))
                width = 0.35
                
                bars1 = ax4.bar(x_pos - width/2, accuracies, width, label='Accuracy', alpha=0.7)
                bars2 = ax4.bar(x_pos + width/2, tprs, width, label='True Positive Rate', alpha=0.7)
                
                ax4.set_xlabel('Wear Time Category')
                ax4.set_ylabel('Score')
                ax4.set_title('Performance by Wear Time', fontweight='bold')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(categories, rotation=45)
                ax4.legend()
        
        # 5. Data Quality Impact
        ax5 = plt.subplot(3, 3, 5)
        
        if 'data_quality_category' in fairness_results:
            quality_metrics = fairness_results['data_quality_category'].get('group_metrics', {})
            
            if quality_metrics:
                categories = list(quality_metrics.keys())
                ppvs = [quality_metrics[cat]['ppv'] for cat in categories]
                fprs = [quality_metrics[cat]['fpr'] for cat in categories]
                
                x_pos = np.arange(len(categories))
                width = 0.35
                
                bars1 = ax5.bar(x_pos - width/2, ppvs, width, label='Positive Predictive Value', alpha=0.7, color='green')
                bars2 = ax5.bar(x_pos + width/2, fprs, width, label='False Positive Rate', alpha=0.7, color='red')
                
                ax5.set_xlabel('Data Quality Category')
                ax5.set_ylabel('Rate')
                ax5.set_title('Calibration by Data Quality', fontweight='bold')
                ax5.set_xticks(x_pos)
                ax5.set_xticklabels(categories, rotation=45)
                ax5.legend()
        
        # 6. User Profile Comparison
        ax6 = plt.subplot(3, 3, 6)
        
        if 'ideal_user' in fairness_results:
            ideal_metrics = fairness_results['ideal_user'].get('group_metrics', {})
            
            if ideal_metrics and len(ideal_metrics) >= 2:
                user_types = list(ideal_metrics.keys())
                metrics_names = ['accuracy', 'tpr', 'ppv', 'f1']
                
                x_pos = np.arange(len(metrics_names))
                width = 0.35
                
                for i, user_type in enumerate(user_types[:2]):
                    values = [ideal_metrics[user_type][metric] for metric in metrics_names]
                    label = 'Ideal Users' if str(user_type) == 'True' else 'Other Users'
                    ax6.bar(x_pos + i*width, values, width, label=label, alpha=0.7)
                
                ax6.set_xlabel('Performance Metrics')
                ax6.set_ylabel('Score')
                ax6.set_title('Ideal vs Other Users', fontweight='bold')
                ax6.set_xticks(x_pos + width/2)
                ax6.set_xticklabels(['Accuracy', 'Sensitivity', 'Precision', 'F1'])
                ax6.legend()
        
        # 7. Fairness vs Performance Trade-off
        ax7 = plt.subplot(3, 3, 7)
        
        performance_scores = []
        fairness_scores = []
        factor_names = []
        
        for var, results in fairness_results.items():
            if 'overall_fairness_score' in results and 'group_metrics' in results:
                # Calculate average performance across groups
                group_metrics = results['group_metrics']
                if group_metrics:
                    avg_accuracy = np.mean([m['accuracy'] for m in group_metrics.values()])
                    fairness_score = results['overall_fairness_score']
                    
                    performance_scores.append(avg_accuracy)
                    fairness_scores.append(fairness_score)
                    factor_names.append(var.replace('_', ' ').title())
        
        if performance_scores and fairness_scores:
            scatter = ax7.scatter(fairness_scores, performance_scores, s=100, alpha=0.7, c=range(len(performance_scores)), cmap='viridis')
            
            for i, name in enumerate(factor_names):
                ax7.annotate(name, (fairness_scores[i], performance_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax7.set_xlabel('Fairness Score (Lower = More Fair)')
            ax7.set_ylabel('Average Accuracy')
            ax7.set_title('Performance vs Fairness Trade-off', fontweight='bold')
            
            # Add quadrant lines
            ax7.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, label='Fairness Threshold')
            ax7.axhline(y=0.7, color='blue', linestyle='--', alpha=0.5, label='Performance Threshold')
            ax7.legend()
        
        # 8. Activity Level Fairness
        ax8 = plt.subplot(3, 3, 8)
        
        if 'activity_level' in fairness_results:
            activity_metrics = fairness_results['activity_level'].get('group_metrics', {})
            
            if activity_metrics:
                levels = list(activity_metrics.keys())
                selection_rates = [activity_metrics[level]['selection_rate'] for level in levels]
                sample_sizes = [activity_metrics[level]['sample_size'] for level in levels]
                
                bars = ax8.bar(levels, selection_rates, alpha=0.7, color='lightcoral')
                ax8.set_xlabel('Activity Level')
                ax8.set_ylabel('Positive Prediction Rate')
                ax8.set_title('Statistical Parity by Activity Level', fontweight='bold')
                ax8.tick_params(axis='x', rotation=45)
                
                # Add sample size labels
                for bar, rate, n in zip(bars, selection_rates, sample_sizes):
                    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{rate:.3f}\n(N={n})', ha='center', va='bottom', fontsize=8)
        
        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Create summary text
        summary_text = "FAIRNESS ANALYSIS SUMMARY\n" + "="*30 + "\n\n"
        
        excellent_count = sum(1 for r in fairness_results.values() 
                            if r.get('overall_fairness_score', 1) < 0.05)
        acceptable_count = sum(1 for r in fairness_results.values() 
                             if 0.05 <= r.get('overall_fairness_score', 1) < 0.1)
        poor_count = sum(1 for r in fairness_results.values() 
                        if r.get('overall_fairness_score', 1) >= 0.1)
        
        summary_text += f"✅ Excellent Fairness: {excellent_count}\n"
        summary_text += f"⚠️ Acceptable Fairness: {acceptable_count}\n" 
        summary_text += f"❌ Poor Fairness: {poor_count}\n\n"
        
        if poor_count == 0:
            summary_text += "🎉 NO CRITICAL FAIRNESS ISSUES!\n\n"
        else:
            summary_text += "⚠️ ATTENTION REQUIRED\n\n"
        
        # Add key recommendations
        summary_text += "KEY RECOMMENDATIONS:\n"
        summary_text += "• Monitor wear time patterns\n"
        summary_text += "• Ensure data quality thresholds\n"
        summary_text += "• Balance user representation\n"
        summary_text += "• Implement fairness monitoring\n"
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/results/figures/comprehensive_wearable_fairness.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comprehensive visualizations created!")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def generate_comprehensive_report(fairness_results, total_participants, overall_accuracy):
    """Generate a comprehensive fairness analysis report."""
    print("\n📝 Generating comprehensive fairness report...")
    
    report = []
    report.append("# COMPREHENSIVE WEARABLE DEVICE ALGORITHMIC FAIRNESS ANALYSIS")
    report.append("=" * 80)
    report.append(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Dataset:** NHANES 2011-2014 Accelerometry + Glucose Data")
    report.append(f"**Total Participants:** {total_participants}")
    report.append(f"**Overall Model Accuracy:** {overall_accuracy:.3f}")
    report.append("")
    
    # Executive Summary
    report.append("## EXECUTIVE SUMMARY")
    report.append("-" * 40)
    
    # Calculate summary statistics
    excellent_count = 0
    acceptable_count = 0
    poor_count = 0
    critical_factors = []
    
    all_fairness_scores = []
    
    for factor, results in fairness_results.items():
        fairness_score = results.get('overall_fairness_score', 1.0)
        all_fairness_scores.append(fairness_score)
        
        if fairness_score < 0.05:
            excellent_count += 1
        elif fairness_score < 0.1:
            acceptable_count += 1
        else:
            poor_count += 1
            critical_factors.append((factor, fairness_score))
    
    avg_fairness = np.mean(all_fairness_scores) if all_fairness_scores else 1.0
    
    report.append(f"**Factors Analyzed:** {len(fairness_results)}")
    report.append(f"**Average Fairness Score:** {avg_fairness:.3f}")
    report.append("")
    report.append("**Fairness Assessment Distribution:**")
    report.append(f"- ✅ **Excellent Fairness** (≤0.05): {excellent_count} factors")
    report.append(f"- ⚠️ **Acceptable Fairness** (0.05-0.10): {acceptable_count} factors")
    report.append(f"- ❌ **Poor Fairness** (>0.10): {poor_count} factors")
    report.append("")
    
    if poor_count == 0:
        report.append("🎉 **RESULT: NO CRITICAL FAIRNESS ISSUES DETECTED**")
        report.append("The glucose prediction model demonstrates excellent algorithmic fairness across all wearable device metadata factors.")
    else:
        report.append("⚠️ **RESULT: CRITICAL FAIRNESS ISSUES REQUIRE ATTENTION**")
        report.append("The following factors show concerning fairness disparities:")
        for factor, score in critical_factors:
            report.append(f"  - **{factor.replace('_', ' ').title()}:** {score:.3f}")
    
    report.append("")
    
    # Detailed Analysis
    report.append("## DETAILED FAIRNESS ANALYSIS")
    report.append("-" * 50)
    
    factor_descriptions = {
        'wear_time_category': 'Device Wear Time Patterns',
        'data_quality_category': 'Data Quality Levels',
        'consistency_category': 'Wear Consistency Patterns',
        'activity_level': 'Physical Activity Levels',
        'monitoring_duration': 'Monitoring Duration Categories',
        'ideal_user': 'Ideal vs Other User Profiles',
        'problematic_user': 'Problematic vs Other User Profiles',
        'high_compliance': 'High vs Low Compliance Users'
    }
    
    for factor, results in fairness_results.items():
        factor_name = factor_descriptions.get(factor, factor.replace('_', ' ').title())
        
        report.append(f"\n### {factor_name}")
        report.append("-" * len(factor_name))
        
        # Overall fairness metrics
        if 'overall_fairness_score' in results:
            fairness_score = results['overall_fairness_score']
            assessment = results.get('fairness_assessment', 'Unknown')
            
            report.append(f"**Overall Fairness Score:** {fairness_score:.3f}")
            report.append(f"**Assessment:** {assessment}")
            report.append("")
        
        # Specific fairness metrics
        fairness_metrics = [
            ('statistical_parity_difference', 'Statistical Parity Difference'),
            ('equal_opportunity_difference', 'Equal Opportunity Difference'),
            ('equalized_odds_max_difference', 'Equalized Odds Difference'),
            ('calibration_difference', 'Calibration Difference')
        ]
        
        report.append("**Fairness Metrics:**")
        for metric_key, metric_name in fairness_metrics:
            if metric_key in results:
                value = results[metric_key]
                report.append(f"- {metric_name}: {value:.3f}")
        
        report.append("")
        
        # Group-level performance
        if 'group_metrics' in results:
            report.append("**Group-Level Performance:**")
            
            group_metrics = results['group_metrics']
            for group, metrics in group_metrics.items():
                report.append(f"- **{group}:**")
                report.append(f"  - Accuracy: {metrics['accuracy']:.3f}")
                report.append(f"  - True Positive Rate: {metrics['tpr']:.3f}")
                report.append(f"  - False Positive Rate: {metrics['fpr']:.3f}")
                report.append(f"  - Positive Predictive Value: {metrics['ppv']:.3f}")
                report.append(f"  - Sample Size: {metrics['sample_size']}")
                report.append("")
    
    # Recommendations
    report.append("## RECOMMENDATIONS")
    report.append("-" * 30)
    
    if poor_count > 0:
        report.append("### 🚨 IMMEDIATE ACTIONS REQUIRED")
        report.append("")
        report.append("**Critical Fairness Issues Identified:**")
        for factor, score in critical_factors:
            report.append(f"1. **Address {factor.replace('_', ' ').title()} Bias (Score: {score:.3f})**")
            report.append(f"   - Investigate root causes of disparity")
            report.append(f"   - Implement targeted data collection for underrepresented groups")
            report.append(f"   - Consider stratified model training approaches")
            report.append("")
        
        report.append("**Algorithmic Interventions:**")
        report.append("- Implement fairness-aware machine learning techniques")
        report.append("- Apply post-processing calibration methods")
        report.append("- Develop ensemble models with fairness constraints")
        report.append("- Use adversarial debiasing approaches")
        report.append("")
    
    report.append("### 📊 DATA COLLECTION IMPROVEMENTS")
    report.append("")
    report.append("**Wearable Device Metadata Enhancement:**")
    report.append("- Collect device type and model information")
    report.append("- Track firmware versions and sensor specifications")
    report.append("- Monitor battery life and charging patterns")
    report.append("- Record environmental conditions during wear")
    report.append("")
    report.append("**User Behavior Tracking:**")
    report.append("- Implement wear time reminders and notifications")
    report.append("- Track user engagement with device features")
    report.append("- Monitor adherence to wear protocols")
    report.append("- Collect user feedback on device comfort and usability")
    report.append("")
    
    report.append("### 🔧 MODEL DEVELOPMENT ENHANCEMENTS")
    report.append("")
    report.append("**Fairness-Aware Training:**")
    report.append("- Implement demographic parity constraints")
    report.append("- Use equalized odds optimization")
    report.append("- Apply calibration-based fairness methods")
    report.append("- Develop multi-objective optimization approaches")
    report.append("")
    report.append("**Model Architecture Improvements:**")
    report.append("- Create separate models for high-risk fairness groups")
    report.append("- Implement hierarchical modeling approaches")
    report.append("- Use meta-learning for fairness adaptation")
    report.append("- Develop uncertainty-aware prediction methods")
    report.append("")
    
    report.append("### 🚀 DEPLOYMENT AND MONITORING")
    report.append("")
    report.append("**Production Fairness Monitoring:**")
    report.append("- Implement real-time fairness dashboards")
    report.append("- Set up automated fairness threshold alerts")
    report.append("- Establish regular fairness auditing procedures")
    report.append("- Create fairness performance reporting systems")
    report.append("")
    report.append("**User-Facing Considerations:**")
    report.append("- Provide prediction confidence intervals")
    report.append("- Communicate model limitations and biases")
    report.append("- Establish user feedback and bias reporting mechanisms")
    report.append("- Implement human-in-the-loop validation for high-risk predictions")
    report.append("")
    
    # Technical Appendix
    report.append("## TECHNICAL APPENDIX")
    report.append("-" * 30)
    
    report.append("### Fairness Metrics Definitions")
    report.append("")
    report.append("**Statistical Parity (Demographic Parity):**")
    report.append("- Measures equality of positive prediction rates across groups")
    report.append("- Formula: |P(Ŷ=1|A=0) - P(Ŷ=1|A=1)|")
    report.append("- Interpretation: Lower values indicate more equal treatment")
    report.append("")
    report.append("**Equal Opportunity:**")
    report.append("- Measures equality of true positive rates across groups")
    report.append("- Formula: |P(Ŷ=1|Y=1,A=0) - P(Ŷ=1|Y=1,A=1)|")
    report.append("- Interpretation: Ensures equal benefit for positive cases")
    report.append("")
    report.append("**Equalized Odds:**")
    report.append("- Measures equality of both TPR and FPR across groups")
    report.append("- Formula: max(|TPR₀-TPR₁|, |FPR₀-FPR₁|)")
    report.append("- Interpretation: Ensures equal treatment for both positive and negative cases")
    report.append("")
    report.append("**Calibration:**")
    report.append("- Measures equality of positive predictive values across groups")
    report.append("- Formula: |P(Y=1|Ŷ=1,A=0) - P(Y=1|Ŷ=1,A=1)|")
    report.append("- Interpretation: Ensures predictions mean the same thing across groups")
    report.append("")
    
    report.append("### Fairness Thresholds")
    report.append("")
    report.append("- **Excellent Fairness:** All metrics ≤ 0.05")
    report.append("- **Acceptable Fairness:** All metrics ≤ 0.10")
    report.append("- **Poor Fairness:** Any metric > 0.10")
    report.append("")
    
    report.append("### Wearable Metadata Categories")
    report.append("")
    report.append("**Wear Time Categories:**")
    report.append("- Low Wear: <8 hours/day")
    report.append("- Medium Wear: 8-12 hours/day")
    report.append("- High Wear: 12-16 hours/day")
    report.append("- Excellent Wear: >16 hours/day")
    report.append("")
    report.append("**Data Quality Categories:**")
    report.append("- Poor Quality: <60% valid data")
    report.append("- Fair Quality: 60-80% valid data")
    report.append("- Good Quality: 80-90% valid data")
    report.append("- Excellent Quality: >90% valid data")
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    
    try:
        with open('/Users/aakashsuresh/fairness/blood_glucose_project/results/reports/comprehensive_wearable_fairness_report.md', 'w') as f:
            f.write(report_text)
        print("✅ Comprehensive fairness report saved!")
    except Exception as e:
        print(f"Error saving report: {e}")
    
    return report_text

if __name__ == "__main__":
    np.random.seed(42)  # For reproducible sampling
    results = run_comprehensive_fairness_analysis()
