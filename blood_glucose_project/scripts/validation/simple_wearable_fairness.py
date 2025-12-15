#!/usr/bin/env python3
"""
Simple Wearable Device Algorithmic Fairness Analysis
===================================================

A focused analysis of algorithmic fairness across wearable device metadata factors.

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 WEARABLE DEVICE ALGORITHMIC FAIRNESS ANALYSIS")
    print("=" * 60)
    
    # Load data with error handling
    try:
        print("📊 Loading accelerometry data...")
        acc_2011 = pd.read_csv("/Users/aakashsuresh/fairness/processed_data_new/2011-2012_Accelerometry.csv")
        acc_2013 = pd.read_csv("/Users/aakashsuresh/fairness/processed_data_new/2013-2014_Accelerometry.csv")
        
        print(f"2011-2012 shape: {acc_2011.shape}")
        print(f"2013-2014 shape: {acc_2013.shape}")
        
        # Combine data
        accelerometry = pd.concat([acc_2011, acc_2013], ignore_index=True)
        print(f"Combined accelerometry shape: {accelerometry.shape}")
        
        # Load glucose data
        print("📊 Loading glucose data...")
        glucose_2011 = pd.read_csv("/Users/aakashsuresh/fairness/processed_data_new/2011-2012_GLU_G.csv")
        glucose_2013 = pd.read_csv("/Users/aakashsuresh/fairness/processed_data_new/2013-2014_GLU_H.csv")
        glucose = pd.concat([glucose_2011, glucose_2013], ignore_index=True)
        
        print(f"Glucose data shape: {glucose.shape}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Clean and merge data
    print("🔧 Processing data...")
    
    # Replace mysterious missing values
    accelerometry = accelerometry.replace(5.397605346934028e-79, np.nan)
    
    # Merge datasets
    data = accelerometry.merge(glucose[['SEQN', 'LBXGLU']], on='SEQN', how='inner')
    print(f"Merged data shape: {data.shape}")
    print(f"Unique participants: {data['SEQN'].nunique()}")
    
    # Engineer wearable metadata features
    print("🔧 Engineering wearable features...")
    
    # Basic wear time features
    data['daily_wear_minutes'] = data['PAXTMD'].fillna(0)
    data['valid_data_minutes'] = data['PAXVMD'].fillna(0)
    
    # Data quality ratio
    data['data_quality_ratio'] = np.where(
        data['daily_wear_minutes'] > 0,
        data['valid_data_minutes'] / data['daily_wear_minutes'],
        0
    )
    
    # Create stratification categories
    print("🎯 Creating stratification groups...")
    
    # Wear time categories
    data['wear_time_category'] = pd.cut(
        data['daily_wear_minutes'],
        bins=[0, 600, 900, 1200, 1440],  # 0-10h, 10-15h, 15-20h, 20-24h
        labels=['Low_Wear', 'Medium_Wear', 'High_Wear', 'Excellent_Wear'],
        include_lowest=True
    )
    
    # Data quality categories
    data['quality_category'] = pd.cut(
        data['data_quality_ratio'],
        bins=[0, 0.7, 0.85, 0.95, 1.0],
        labels=['Poor_Quality', 'Fair_Quality', 'Good_Quality', 'Excellent_Quality'],
        include_lowest=True
    )
    
    # Weekend vs weekday
    data['is_weekend'] = data['PAXDAYWD'].isin([1, 7])  # Sunday=1, Saturday=7
    
    print("Wear time distribution:")
    print(data['wear_time_category'].value_counts())
    print("\nQuality distribution:")
    print(data['quality_category'].value_counts())
    
    # Create participant-level data
    print("📊 Aggregating to participant level...")
    
    participant_data = data.groupby('SEQN').agg({
        'daily_wear_minutes': 'mean',
        'data_quality_ratio': 'mean',
        'is_weekend': 'sum',
        'LBXGLU': 'first'
    }).reset_index()
    
    participant_data['avg_daily_wear_hours'] = participant_data['daily_wear_minutes'] / 60
    participant_data['monitoring_days'] = data.groupby('SEQN').size().values
    participant_data['weekend_proportion'] = (
        participant_data['is_weekend'] / participant_data['monitoring_days']
    )
    
    # Create participant-level categories
    participant_data['wear_category'] = pd.cut(
        participant_data['avg_daily_wear_hours'],
        bins=[0, 10, 15, 20, 24],
        labels=['Low', 'Medium', 'High', 'Excellent'],
        include_lowest=True
    )
    
    participant_data['quality_category'] = pd.cut(
        participant_data['data_quality_ratio'],
        bins=[0, 0.7, 0.85, 0.95, 1.0],
        labels=['Poor', 'Fair', 'Good', 'Excellent'],
        include_lowest=True
    )
    
    # Create ideal vs problematic users
    participant_data['ideal_user'] = (
        (participant_data['wear_category'].isin(['High', 'Excellent'])) &
        (participant_data['quality_category'].isin(['Good', 'Excellent']))
    )
    
    print(f"Participant data shape: {participant_data.shape}")
    print(f"Ideal users: {participant_data['ideal_user'].sum()}")
    
    # Prepare for modeling
    print("🤖 Preparing machine learning model...")
    
    # Clean data for modeling
    model_data = participant_data.dropna(subset=[
        'avg_daily_wear_hours', 'data_quality_ratio', 'weekend_proportion', 'LBXGLU'
    ])
    
    print(f"Model data shape: {model_data.shape}")
    
    if len(model_data) < 50:
        print("❌ Insufficient data for modeling")
        return
    
    # Features and target
    X = model_data[['avg_daily_wear_hours', 'data_quality_ratio', 'weekend_proportion']]
    y_glucose = model_data['LBXGLU']
    y_diabetes = (y_glucose >= 126).astype(int)  # Diabetes classification
    
    print(f"Features shape: {X.shape}")
    print(f"Diabetes prevalence: {y_diabetes.mean():.3f}")
    
    # Train model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_diabetes, test_size=0.3, random_state=42, stratify=y_diabetes
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    overall_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Overall model accuracy: {overall_accuracy:.3f}")
    
    # Fairness analysis
    print("\n🎯 ALGORITHMIC FAIRNESS ANALYSIS")
    print("-" * 40)
    
    # Get test data for stratification
    test_indices = model_data.index[-len(y_test):]
    test_data = model_data.loc[test_indices].reset_index(drop=True)
    
    fairness_results = {}
    
    # Analyze fairness by wear time category
    print("\n📊 Fairness by Wear Time Category:")
    wear_attr = test_data['wear_category'].reset_index(drop=True)
    
    wear_fairness = analyze_group_fairness(y_test, y_pred, wear_attr, "Wear Time")
    fairness_results['wear_time'] = wear_fairness
    
    # Analyze fairness by data quality
    print("\n📊 Fairness by Data Quality Category:")
    quality_attr = test_data['quality_category'].reset_index(drop=True)
    
    quality_fairness = analyze_group_fairness(y_test, y_pred, quality_attr, "Data Quality")
    fairness_results['data_quality'] = quality_fairness
    
    # Analyze fairness by ideal user status
    print("\n📊 Fairness by User Type (Ideal vs Others):")
    ideal_attr = test_data['ideal_user'].reset_index(drop=True)
    
    ideal_fairness = analyze_group_fairness(y_test, y_pred, ideal_attr, "User Type")
    fairness_results['user_type'] = ideal_fairness
    
    # Create visualizations
    create_fairness_visualizations(fairness_results)
    
    # Generate report
    generate_fairness_report(fairness_results, len(model_data), overall_accuracy)
    
    print("\n🎉 ANALYSIS COMPLETE!")
    print("📊 Visualizations saved to: results/figures/")
    print("📝 Report saved to: results/reports/")
    
    return fairness_results

def analyze_group_fairness(y_true, y_pred, sensitive_attr, group_name):
    """Analyze fairness metrics for a specific grouping variable."""
    
    results = {}
    group_metrics = {}
    
    # Get unique groups
    groups = sensitive_attr.unique()
    groups = groups[~pd.isna(groups)]
    
    print(f"Analyzing {len(groups)} groups for {group_name}")
    
    for group in groups:
        mask = (sensitive_attr == group)
        if mask.sum() < 5:  # Skip small groups
            continue
        
        group_y_true = y_true[mask]
        group_y_pred = y_pred[mask]
        
        # Calculate metrics
        accuracy = accuracy_score(group_y_true, group_y_pred)
        precision = precision_score(group_y_true, group_y_pred, average='weighted', zero_division=0)
        recall = recall_score(group_y_true, group_y_pred, average='weighted', zero_division=0)
        f1 = f1_score(group_y_true, group_y_pred, average='weighted', zero_division=0)
        
        group_metrics[str(group)] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sample_size': len(group_y_true)
        }
        
        print(f"  {group}: Accuracy={accuracy:.3f}, Precision={precision:.3f}, N={len(group_y_true)}")
    
    # Calculate disparities
    if len(group_metrics) >= 2:
        accuracies = [metrics['accuracy'] for metrics in group_metrics.values()]
        precisions = [metrics['precision'] for metrics in group_metrics.values()]
        recalls = [metrics['recall'] for metrics in group_metrics.values()]
        
        results['accuracy_disparity'] = max(accuracies) - min(accuracies)
        results['precision_disparity'] = max(precisions) - min(precisions)
        results['recall_disparity'] = max(recalls) - min(recalls)
        
        print(f"  Accuracy Disparity: {results['accuracy_disparity']:.3f}")
        print(f"  Precision Disparity: {results['precision_disparity']:.3f}")
        
        # Fairness assessment
        if results['accuracy_disparity'] < 0.05:
            assessment = "✅ Excellent Fairness"
        elif results['accuracy_disparity'] < 0.1:
            assessment = "⚠️ Acceptable Fairness"
        else:
            assessment = "❌ Poor Fairness"
        
        print(f"  Assessment: {assessment}")
    
    results['group_metrics'] = group_metrics
    return results

def create_fairness_visualizations(fairness_results):
    """Create fairness analysis visualizations."""
    print("\n📊 Creating fairness visualizations...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Wearable Device Algorithmic Fairness Analysis', fontsize=16, fontweight='bold')
        
        # 1. Accuracy Disparity by Factor
        ax1 = axes[0, 0]
        factors = []
        disparities = []
        
        for factor, results in fairness_results.items():
            if 'accuracy_disparity' in results:
                factors.append(factor.replace('_', ' ').title())
                disparities.append(results['accuracy_disparity'])
        
        if factors:
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
            ax1.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='Excellent (≤0.05)')
            ax1.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Acceptable (≤0.10)')
            ax1.legend()
        
        # 2. Wear Time Category Performance
        ax2 = axes[0, 1]
        if 'wear_time' in fairness_results:
            wear_metrics = fairness_results['wear_time']['group_metrics']
            
            categories = list(wear_metrics.keys())
            accuracies = [wear_metrics[cat]['accuracy'] for cat in categories]
            sample_sizes = [wear_metrics[cat]['sample_size'] for cat in categories]
            
            bars = ax2.bar(categories, accuracies, alpha=0.7, color='skyblue')
            ax2.set_title('Model Accuracy by Wear Time Category')
            ax2.set_ylabel('Accuracy')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels with sample sizes
            for bar, acc, n in zip(bars, accuracies, sample_sizes):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}\n(N={n})', ha='center', va='bottom', fontsize=8)
        
        # 3. Data Quality Impact
        ax3 = axes[1, 0]
        if 'data_quality' in fairness_results:
            quality_metrics = fairness_results['data_quality']['group_metrics']
            
            categories = list(quality_metrics.keys())
            accuracies = [quality_metrics[cat]['accuracy'] for cat in categories]
            precisions = [quality_metrics[cat]['precision'] for cat in categories]
            
            x_pos = np.arange(len(categories))
            width = 0.35
            
            bars1 = ax3.bar(x_pos - width/2, accuracies, width, label='Accuracy', alpha=0.7)
            bars2 = ax3.bar(x_pos + width/2, precisions, width, label='Precision', alpha=0.7)
            
            ax3.set_xlabel('Data Quality Category')
            ax3.set_ylabel('Score')
            ax3.set_title('Performance by Data Quality')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(categories, rotation=45)
            ax3.legend()
        
        # 4. Ideal vs Other Users
        ax4 = axes[1, 1]
        if 'user_type' in fairness_results:
            user_metrics = fairness_results['user_type']['group_metrics']
            
            user_types = list(user_metrics.keys())
            metrics_names = ['accuracy', 'precision', 'recall', 'f1']
            
            if len(user_types) >= 2:
                x_pos = np.arange(len(metrics_names))
                width = 0.35
                
                for i, user_type in enumerate(user_types[:2]):  # Show max 2 user types
                    values = [user_metrics[user_type][metric] for metric in metrics_names]
                    label = 'Ideal Users' if str(user_type) == 'True' else 'Other Users'
                    ax4.bar(x_pos + i*width, values, width, label=label, alpha=0.7)
                
                ax4.set_xlabel('Metrics')
                ax4.set_ylabel('Score')
                ax4.set_title('Performance: Ideal vs Other Users')
                ax4.set_xticks(x_pos + width/2)
                ax4.set_xticklabels(metrics_names)
                ax4.legend()
        
        plt.tight_layout()
        plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/results/figures/wearable_fairness_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations created successfully!")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def generate_fairness_report(fairness_results, total_participants, overall_accuracy):
    """Generate a comprehensive fairness report."""
    print("\n📝 Generating fairness report...")
    
    report = []
    report.append("# WEARABLE DEVICE ALGORITHMIC FAIRNESS ANALYSIS")
    report.append("=" * 60)
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Participants: {total_participants}")
    report.append(f"Overall Model Accuracy: {overall_accuracy:.3f}")
    report.append("")
    
    # Executive summary
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
    report.append(f"• Average Fairness Disparity: {avg_disparity:.3f}")
    report.append(f"• Maximum Fairness Disparity: {max_disparity:.3f}")
    report.append(f"• Critical Issues (>0.10 disparity): {critical_issues}")
    
    if critical_issues == 0:
        report.append("• ✅ **NO CRITICAL FAIRNESS ISSUES DETECTED**")
    else:
        report.append(f"• ❌ **{critical_issues} CRITICAL FAIRNESS ISSUES REQUIRE ATTENTION**")
    
    report.append("")
    
    # Detailed analysis
    report.append("## DETAILED FAIRNESS ANALYSIS")
    report.append("-" * 40)
    
    factor_names = {
        'wear_time': 'Wear Time Categories',
        'data_quality': 'Data Quality Categories', 
        'user_type': 'User Type (Ideal vs Others)'
    }
    
    for factor, results in fairness_results.items():
        factor_name = factor_names.get(factor, factor.replace('_', ' ').title())
        report.append(f"\n### {factor_name}")
        
        if 'accuracy_disparity' in results:
            acc_disp = results['accuracy_disparity']
            prec_disp = results.get('precision_disparity', 0)
            
            report.append(f"**Accuracy Disparity:** {acc_disp:.3f}")
            report.append(f"**Precision Disparity:** {prec_disp:.3f}")
            
            # Fairness assessment
            if acc_disp < 0.05:
                assessment = "✅ **Excellent Fairness** - No action needed"
            elif acc_disp < 0.1:
                assessment = "⚠️ **Acceptable Fairness** - Monitor closely"
            else:
                assessment = "❌ **Poor Fairness** - Immediate action required"
            
            report.append(f"**Assessment:** {assessment}")
        
        # Group performance details
        report.append("\n**Group Performance:**")
        for group, metrics in results['group_metrics'].items():
            report.append(f"- **{group}:** Accuracy={metrics['accuracy']:.3f}, "
                         f"Precision={metrics['precision']:.3f}, N={metrics['sample_size']}")
    
    # Recommendations
    report.append("\n## RECOMMENDATIONS")
    report.append("-" * 30)
    
    if critical_issues > 0:
        report.append("\n### 🚨 IMMEDIATE ACTIONS REQUIRED")
        report.append("1. **Investigate High-Disparity Factors:**")
        for factor, results in fairness_results.items():
            if results.get('accuracy_disparity', 0) > 0.1:
                report.append(f"   - Address {factor_names.get(factor, factor)} bias through targeted interventions")
        
        report.append("\n2. **Model Improvements:**")
        report.append("   - Implement stratified model training for high-disparity groups")
        report.append("   - Apply fairness-aware machine learning techniques")
        report.append("   - Consider ensemble methods with fairness constraints")
    else:
        report.append("\n### ✅ MAINTENANCE RECOMMENDATIONS")
        report.append("1. **Continue Monitoring:**")
        report.append("   - Establish regular fairness auditing procedures")
        report.append("   - Monitor fairness metrics in production deployment")
    
    report.append("\n### 📊 DATA COLLECTION IMPROVEMENTS")
    report.append("1. **Ensure Balanced Representation:**")
    report.append("   - Target underrepresented wear time categories")
    report.append("   - Implement data quality thresholds for training")
    
    report.append("\n2. **Enhanced Metadata Collection:**")
    report.append("   - Collect device type and model information")
    report.append("   - Track user engagement and adherence patterns")
    report.append("   - Monitor environmental factors affecting wear patterns")
    
    report.append("\n### 🚀 DEPLOYMENT CONSIDERATIONS")
    report.append("1. **Real-time Monitoring:**")
    report.append("   - Implement fairness dashboards for production models")
    report.append("   - Set up alerts for fairness threshold violations")
    
    report.append("\n2. **User Communication:**")
    report.append("   - Provide uncertainty quantification for predictions")
    report.append("   - Communicate model limitations to users")
    report.append("   - Establish feedback mechanisms for bias reporting")
    
    # Technical appendix
    report.append("\n## TECHNICAL APPENDIX")
    report.append("-" * 30)
    
    report.append("\n### Fairness Metrics Definitions")
    report.append("- **Accuracy Disparity:** Difference in accuracy between groups")
    report.append("- **Precision Disparity:** Difference in precision between groups")
    report.append("- **Statistical Parity:** Equal positive prediction rates across groups")
    report.append("- **Equal Opportunity:** Equal true positive rates across groups")
    
    report.append("\n### Fairness Thresholds")
    report.append("- **Excellent:** Disparity < 0.05")
    report.append("- **Acceptable:** Disparity < 0.10")
    report.append("- **Poor:** Disparity ≥ 0.10")
    
    report.append("\n### Wearable Metadata Factors Analyzed")
    report.append("- **Wear Time Categories:** Low (<10h), Medium (10-15h), High (15-20h), Excellent (>20h)")
    report.append("- **Data Quality Categories:** Poor (<70%), Fair (70-85%), Good (85-95%), Excellent (>95%)")
    report.append("- **User Types:** Ideal (High wear + Good quality) vs Others")
    
    # Save report
    report_text = "\n".join(report)
    
    try:
        with open('/Users/aakashsuresh/fairness/blood_glucose_project/results/reports/wearable_algorithmic_fairness_report.md', 'w') as f:
            f.write(report_text)
        print("✅ Fairness report saved successfully!")
    except Exception as e:
        print(f"Error saving report: {e}")
    
    return report_text

if __name__ == "__main__":
    results = main()
