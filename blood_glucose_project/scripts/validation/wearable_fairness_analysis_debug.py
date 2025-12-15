#!/usr/bin/env python3
"""
Algorithmic Fairness Analysis for Wearable-Based Glucose Prediction - Debug Version
==================================================================================

This script performs comprehensive algorithmic fairness analysis across wearable device
metadata factors with enhanced error handling and debugging.

Author: Blood Glucose Prediction Team
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and examine the data structure."""
    print("🔄 Loading data...")
    
    data_path = "/Users/aakashsuresh/fairness/processed_data_new"
    
    try:
        # Load accelerometry data
        print("Loading accelerometry data...")
        acc_2011 = pd.read_csv(f"{data_path}/2011-2012_Accelerometry.csv")
        acc_2013 = pd.read_csv(f"{data_path}/2013-2014_Accelerometry.csv")
        
        print(f"2011-2012 Accelerometry shape: {acc_2011.shape}")
        print(f"2013-2014 Accelerometry shape: {acc_2013.shape}")
        print(f"Columns: {list(acc_2011.columns)}")
        
        # Combine accelerometry data
        accelerometry = pd.concat([acc_2011, acc_2013], ignore_index=True)
        print(f"Combined accelerometry shape: {accelerometry.shape}")
        
        # Clean the mysterious missing value code
        accelerometry = accelerometry.replace(5.397605346934028e-79, np.nan)
        
        # Load glucose data
        print("Loading glucose data...")
        glucose_2011 = pd.read_csv(f"{data_path}/2011-2012_GLU_G.csv")
        glucose_2013 = pd.read_csv(f"{data_path}/2013-2014_GLU_H.csv")
        glucose = pd.concat([glucose_2011, glucose_2013], ignore_index=True)
        
        print(f"Glucose data shape: {glucose.shape}")
        print(f"Glucose columns: {list(glucose.columns)}")
        
        # Load HbA1c data
        print("Loading HbA1c data...")
        hba1c_2011 = pd.read_csv(f"{data_path}/2011-2012_GHB_G.csv")
        hba1c_2013 = pd.read_csv(f"{data_path}/2013-2014_GHB_H.csv")
        hba1c = pd.concat([hba1c_2011, hba1c_2013], ignore_index=True)
        
        print(f"HbA1c data shape: {hba1c.shape}")
        
        # Merge datasets
        print("Merging datasets...")
        merged_data = accelerometry.merge(glucose[['SEQN', 'LBXGLU']], on='SEQN', how='inner')
        merged_data = merged_data.merge(hba1c[['SEQN', 'LBXGH']], on='SEQN', how='inner')
        
        print(f"Final merged data shape: {merged_data.shape}")
        print(f"Unique participants: {merged_data['SEQN'].nunique()}")
        
        return merged_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def engineer_wearable_features(data):
    """Engineer wearable device metadata features."""
    print("🔧 Engineering wearable features...")
    
    df = data.copy()
    
    # Basic wear time features
    df['daily_wear_minutes'] = df['PAXTMD'].fillna(0)
    df['valid_data_minutes'] = df['PAXVMD'].fillna(0)
    
    # Data quality ratio
    df['data_quality_ratio'] = np.where(
        df['daily_wear_minutes'] > 0,
        df['valid_data_minutes'] / df['daily_wear_minutes'],
        0
    )
    
    # Wear compliance categories
    df['wear_compliance'] = pd.cut(
        df['daily_wear_minutes'],
        bins=[0, 600, 900, 1200, 1440],
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
    
    # Activity features (with safe handling)
    df['sedentary_minutes'] = df['PAXSWMD'].fillna(0)
    df['light_activity_minutes'] = df['PAXLXSD'].fillna(0)
    df['moderate_vigorous_minutes'] = df['PAXMTSD'].fillna(0)
    
    # Weekend indicator
    df['is_weekend'] = df['PAXDAYWD'].isin([1, 7])  # Sunday=1, Saturday=7
    
    print(f"Engineered features shape: {df.shape}")
    print("Sample wear compliance distribution:")
    print(df['wear_compliance'].value_counts())
    
    return df

def create_participant_level_data(data):
    """Aggregate to participant level for analysis."""
    print("📊 Creating participant-level data...")
    
    # Aggregate by participant
    participant_data = data.groupby('SEQN').agg({
        'daily_wear_minutes': ['mean', 'std', 'count'],
        'data_quality_ratio': 'mean',
        'valid_data_minutes': 'sum',
        'sedentary_minutes': 'mean',
        'light_activity_minutes': 'mean',
        'moderate_vigorous_minutes': 'mean',
        'is_weekend': 'sum',
        'LBXGLU': 'first',
        'LBXGH': 'first'
    }).reset_index()
    
    # Flatten column names
    participant_data.columns = ['_'.join(col).strip() if col[1] else col[0] 
                              for col in participant_data.columns]
    participant_data = participant_data.rename(columns={'SEQN_': 'SEQN'})
    
    # Create derived features
    participant_data['avg_daily_wear_hours'] = participant_data['daily_wear_minutes_mean'] / 60
    participant_data['total_monitoring_days'] = participant_data['daily_wear_minutes_count']
    participant_data['weekend_proportion'] = (
        participant_data['is_weekend_sum'] / participant_data['total_monitoring_days']
    )
    
    # Create stratification categories
    participant_data['wear_time_category'] = pd.cut(
        participant_data['avg_daily_wear_hours'],
        bins=[0, 10, 15, 20, 24],
        labels=['Low_Wear', 'Medium_Wear', 'High_Wear', 'Excellent_Wear'],
        include_lowest=True
    )
    
    participant_data['quality_category'] = pd.cut(
        participant_data['data_quality_ratio_mean'],
        bins=[0, 0.7, 0.85, 0.95, 1.0],
        labels=['Poor_Quality', 'Fair_Quality', 'Good_Quality', 'Excellent_Quality'],
        include_lowest=True
    )
    
    participant_data['activity_level'] = pd.cut(
        participant_data['moderate_vigorous_minutes_mean'],
        bins=[0, 10, 30, 60, np.inf],
        labels=['Sedentary', 'Low_Active', 'Moderate_Active', 'High_Active'],
        include_lowest=True
    )
    
    # Create ideal vs problematic user categories
    participant_data['ideal_user'] = (
        (participant_data['wear_time_category'].isin(['High_Wear', 'Excellent_Wear'])) &
        (participant_data['quality_category'].isin(['Good_Quality', 'Excellent_Quality']))
    )
    
    participant_data['problematic_user'] = (
        (participant_data['wear_time_category'] == 'Low_Wear') |
        (participant_data['quality_category'] == 'Poor_Quality')
    )
    
    print(f"Participant data shape: {participant_data.shape}")
    print("\nStratification summary:")
    print(f"Wear time categories: {participant_data['wear_time_category'].value_counts().to_dict()}")
    print(f"Quality categories: {participant_data['quality_category'].value_counts().to_dict()}")
    print(f"Ideal users: {participant_data['ideal_user'].sum()}")
    print(f"Problematic users: {participant_data['problematic_user'].sum()}")
    
    return participant_data

def calculate_fairness_metrics(y_true, y_pred, sensitive_attr, group_name):
    """Calculate fairness metrics for a specific grouping."""
    print(f"Calculating fairness for: {group_name}")
    
    results = {}
    
    # Get unique groups
    groups = sensitive_attr.unique()
    groups = groups[~pd.isna(groups)]
    
    group_metrics = {}
    
    for group in groups:
        mask = (sensitive_attr == group)
        if mask.sum() < 5:  # Skip groups with too few samples
            continue
            
        group_y_true = y_true[mask]
        group_y_pred = y_pred[mask]
        
        # Calculate metrics
        try:
            accuracy = accuracy_score(group_y_true, group_y_pred)
            precision = precision_score(group_y_true, group_y_pred, average='weighted', zero_division=0)
            recall = recall_score(group_y_true, group_y_pred, average='weighted', zero_division=0)
            f1 = f1_score(group_y_true, group_y_pred, average='weighted', zero_division=0)
            
            group_metrics[group] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'sample_size': len(group_y_true)
            }
            
        except Exception as e:
            print(f"Error calculating metrics for group {group}: {e}")
            continue
    
    # Calculate disparities
    if len(group_metrics) >= 2:
        accuracies = [metrics['accuracy'] for metrics in group_metrics.values()]
        results['accuracy_disparity'] = max(accuracies) - min(accuracies)
        
        precisions = [metrics['precision'] for metrics in group_metrics.values()]
        results['precision_disparity'] = max(precisions) - min(precisions)
        
        recalls = [metrics['recall'] for metrics in group_metrics.values()]
        results['recall_disparity'] = max(recalls) - min(recalls)
    
    results['group_metrics'] = group_metrics
    return results

def run_fairness_analysis():
    """Run the main fairness analysis."""
    print("🚀 Starting Wearable Algorithmic Fairness Analysis")
    print("=" * 60)
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Engineer features
    data_with_features = engineer_wearable_features(data)
    
    # Create participant-level data
    participant_data = create_participant_level_data(data_with_features)
    
    # Prepare for modeling
    feature_cols = [
        'avg_daily_wear_hours', 'data_quality_ratio_mean', 
        'sedentary_minutes_mean', 'light_activity_minutes_mean', 
        'moderate_vigorous_minutes_mean', 'weekend_proportion'
    ]
    
    # Clean data
    analysis_data = participant_data.dropna(subset=feature_cols + ['LBXGLU_first'])
    print(f"Analysis data shape after cleaning: {analysis_data.shape}")
    
    if len(analysis_data) < 50:
        print("❌ Insufficient data for analysis")
        return
    
    # Prepare features and targets
    X = analysis_data[feature_cols]
    y_glucose = analysis_data['LBXGLU_first']
    y_diabetes = (y_glucose >= 126).astype(int)  # Diabetes classification
    
    print(f"Features shape: {X.shape}")
    print(f"Diabetes prevalence: {y_diabetes.mean():.3f}")
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_diabetes, test_size=0.3, random_state=42, stratify=y_diabetes
    )
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Get predictions
    y_pred = clf.predict(X_test)
    
    print(f"Overall accuracy: {accuracy_score(y_test, y_pred):.3f}")
    
    # Get test data for stratification
    test_indices = analysis_data.index[-len(y_test):]
    test_data = analysis_data.loc[test_indices].reset_index(drop=True)
    
    # Analyze fairness across different factors
    stratification_vars = ['wear_time_category', 'quality_category', 'activity_level', 'ideal_user']
    
    fairness_results = {}
    
    for var in stratification_vars:
        if var in test_data.columns:
            print(f"\n🎯 Analyzing fairness for: {var}")
            
            sensitive_attr = test_data[var].reset_index(drop=True)
            
            # Ensure alignment
            if len(sensitive_attr) != len(y_test):
                print(f"Length mismatch: {len(sensitive_attr)} vs {len(y_test)}")
                continue
            
            fairness_metrics = calculate_fairness_metrics(y_test, y_pred, sensitive_attr, var)
            fairness_results[var] = fairness_metrics
            
            # Print results
            if 'accuracy_disparity' in fairness_metrics:
                print(f"  Accuracy disparity: {fairness_metrics['accuracy_disparity']:.3f}")
            
            for group, metrics in fairness_metrics['group_metrics'].items():
                print(f"  {group}: Accuracy={metrics['accuracy']:.3f}, N={metrics['sample_size']}")
    
    # Create summary visualization
    create_fairness_summary(fairness_results)
    
    # Generate summary report
    generate_summary_report(fairness_results, len(analysis_data))
    
    print("\n🎉 Analysis Complete!")
    return fairness_results

def create_fairness_summary(fairness_results):
    """Create a summary visualization of fairness results."""
    print("📊 Creating fairness summary visualization...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Wearable Device Algorithmic Fairness Analysis', fontsize=16, fontweight='bold')
        
        # 1. Accuracy Disparity by Factor
        ax1 = axes[0, 0]
        factors = []
        disparities = []
        
        for factor, results in fairness_results.items():
            if 'accuracy_disparity' in results:
                factors.append(factor.replace('_', ' '))
                disparities.append(results['accuracy_disparity'])
        
        if factors and disparities:
            colors = ['green' if d < 0.05 else 'orange' if d < 0.1 else 'red' for d in disparities]
            bars = ax1.bar(factors, disparities, color=colors, alpha=0.7)
            ax1.set_title('Accuracy Disparity by Wearable Factor')
            ax1.set_ylabel('Accuracy Disparity')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, disp in zip(bars, disparities):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                        f'{disp:.3f}', ha='center', va='bottom')
            
            # Add fairness thresholds
            ax1.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='Acceptable (≤0.05)')
            ax1.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Concerning (≤0.10)')
            ax1.legend()
        
        # 2. Sample Size Distribution
        ax2 = axes[0, 1]
        all_groups = []
        all_sizes = []
        
        for factor, results in fairness_results.items():
            for group, metrics in results['group_metrics'].items():
                all_groups.append(f"{factor}_{group}")
                all_sizes.append(metrics['sample_size'])
        
        if all_groups and all_sizes:
            # Show top 10 groups by size
            sorted_data = sorted(zip(all_sizes, all_groups), reverse=True)[:10]
            sizes, groups = zip(*sorted_data)
            
            ax2.barh(range(len(sizes)), sizes, alpha=0.7)
            ax2.set_yticks(range(len(sizes)))
            ax2.set_yticklabels([g.replace('_', ' ') for g in groups], fontsize=8)
            ax2.set_xlabel('Sample Size')
            ax2.set_title('Sample Sizes by Group (Top 10)')
        
        # 3. Wear Time Category Performance
        ax3 = axes[1, 0]
        if 'wear_time_category' in fairness_results:
            wear_results = fairness_results['wear_time_category']['group_metrics']
            
            categories = list(wear_results.keys())
            accuracies = [wear_results[cat]['accuracy'] for cat in categories]
            
            bars = ax3.bar(categories, accuracies, alpha=0.7, color='skyblue')
            ax3.set_title('Model Accuracy by Wear Time Category')
            ax3.set_ylabel('Accuracy')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, acc in zip(bars, accuracies):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}', ha='center', va='bottom')
        
        # 4. Ideal vs Problematic Users
        ax4 = axes[1, 1]
        if 'ideal_user' in fairness_results:
            ideal_results = fairness_results['ideal_user']['group_metrics']
            
            user_types = list(ideal_results.keys())
            metrics_names = ['accuracy', 'precision', 'recall', 'f1']
            
            x_pos = np.arange(len(metrics_names))
            width = 0.35
            
            for i, user_type in enumerate(user_types):
                if user_type in ideal_results:
                    values = [ideal_results[user_type][metric] for metric in metrics_names]
                    ax4.bar(x_pos + i*width, values, width, 
                           label=f'Ideal User: {user_type}', alpha=0.7)
            
            ax4.set_xlabel('Metrics')
            ax4.set_ylabel('Score')
            ax4.set_title('Performance: Ideal vs Problematic Users')
            ax4.set_xticks(x_pos + width/2)
            ax4.set_xticklabels(metrics_names)
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/results/figures/wearable_fairness_summary.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization saved!")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

def generate_summary_report(fairness_results, total_participants):
    """Generate a summary report of fairness findings."""
    print("📝 Generating summary report...")
    
    report = []
    report.append("# WEARABLE DEVICE ALGORITHMIC FAIRNESS ANALYSIS SUMMARY")
    report.append("=" * 60)
    report.append(f"Total Participants Analyzed: {total_participants}")
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Overall fairness assessment
    critical_issues = 0
    all_disparities = []
    
    for factor, results in fairness_results.items():
        if 'accuracy_disparity' in results:
            disparity = results['accuracy_disparity']
            all_disparities.append(disparity)
            if disparity > 0.1:
                critical_issues += 1
    
    avg_disparity = np.mean(all_disparities) if all_disparities else 0
    max_disparity = max(all_disparities) if all_disparities else 0
    
    report.append("## EXECUTIVE SUMMARY")
    report.append("-" * 30)
    report.append(f"Average Fairness Disparity: {avg_disparity:.3f}")
    report.append(f"Maximum Fairness Disparity: {max_disparity:.3f}")
    report.append(f"Critical Issues (>0.10 disparity): {critical_issues}")
    
    if critical_issues == 0:
        report.append("✅ NO CRITICAL FAIRNESS ISSUES DETECTED")
    else:
        report.append("⚠️ CRITICAL FAIRNESS ISSUES REQUIRE ATTENTION")
    
    report.append("")
    
    # Detailed results
    report.append("## DETAILED RESULTS")
    report.append("-" * 30)
    
    for factor, results in fairness_results.items():
        report.append(f"\n### {factor.replace('_', ' ').title()}")
        
        if 'accuracy_disparity' in results:
            disparity = results['accuracy_disparity']
            report.append(f"Accuracy Disparity: {disparity:.3f}")
            
            if disparity < 0.05:
                assessment = "✅ Excellent Fairness"
            elif disparity < 0.1:
                assessment = "⚠️ Acceptable Fairness"
            else:
                assessment = "❌ Poor Fairness - Requires Attention"
            
            report.append(f"Assessment: {assessment}")
        
        # Group-level results
        report.append("Group Performance:")
        for group, metrics in results['group_metrics'].items():
            report.append(f"  - {group}: Accuracy={metrics['accuracy']:.3f}, N={metrics['sample_size']}")
    
    # Recommendations
    report.append("\n## RECOMMENDATIONS")
    report.append("-" * 30)
    
    if critical_issues > 0:
        report.append("1. **Immediate Actions Required:**")
        report.append("   - Investigate high-disparity factors through targeted analysis")
        report.append("   - Consider stratified model training for problematic groups")
        report.append("   - Implement fairness-aware machine learning techniques")
    else:
        report.append("1. **Maintain Current Standards:**")
        report.append("   - Continue monitoring fairness in production deployment")
        report.append("   - Establish regular fairness auditing procedures")
    
    report.append("\n2. **Data Collection Improvements:**")
    report.append("   - Ensure balanced representation across wear time categories")
    report.append("   - Implement data quality thresholds for model training")
    report.append("   - Collect additional device metadata for enhanced analysis")
    
    report.append("\n3. **Model Development:**")
    report.append("   - Implement real-time fairness monitoring")
    report.append("   - Develop ensemble methods with fairness constraints")
    report.append("   - Provide uncertainty quantification for high-risk groups")
    
    # Save report
    report_text = "\n".join(report)
    
    with open('/Users/aakashsuresh/fairness/blood_glucose_project/results/reports/wearable_fairness_summary_report.md', 'w') as f:
        f.write(report_text)
    
    print("✅ Summary report saved!")
    return report_text

if __name__ == "__main__":
    results = run_fairness_analysis()
