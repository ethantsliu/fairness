import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# Set style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')

# Color scheme for consistent visualization
COLORS = {
    'Traditional': '#2E86AB',      # Blue for traditional risk scores
    'Clinic_Free': '#A23B72',      # Purple for clinic-free models
    'User_Defined': '#F18F01',     # Orange for demographics/SDOH
    'Accel': '#27AE60',            # Green for accelerometer
    'Self_Report': '#E74C3C',      # Red for self-reported
    'Baseline': '#95A5A6'          # Gray for baseline models
}

def parse_ci_string(ci_string):
    """Parse confidence interval string like '0.64 (0.60–0.68)'"""
    if pd.isna(ci_string) or ci_string == '':
        return np.nan, np.nan, np.nan
    
    # Extract the main value and CI range
    match = re.match(r'(\d+\.\d+)\s*\((\d+\.\d+)–(\d+\.\d+)\)', str(ci_string))
    if match:
        main_val = float(match.group(1))
        ci_lower = float(match.group(2))
        ci_upper = float(match.group(3))
        return main_val, ci_lower, ci_upper
    else:
        return np.nan, np.nan, np.nan

def load_and_process_data():
    """Load and process the new data files"""
    
    # Load bootstrap results for all splits
    splits = ['A', 'C']  # Only A and C have complete bootstrap data
    all_data = []
    
    for split in splits:
        # Load bootstrap results
        bootstrap_file = f'data/bootstrap_results_summary_{split}.csv'
        bootstrap_df = pd.read_csv(bootstrap_file)
        
        # Load delta AUC results
        delta_file = f'data/delta_auc_results_{split}.csv'
        delta_df = pd.read_csv(delta_file)
        
        # Process bootstrap results
        for _, row in bootstrap_df.iterrows():
            model_name = row.iloc[0]
            auc_str = row['ROC AUC']
            f1_str = row['F1 Score']
            sens_str = row['Sensitivity']
            spec_str = row['Specificity']
            bal_acc_str = row['Balanced Accuracy']
            
            # Parse values and CIs
            auc_val, auc_lower, auc_upper = parse_ci_string(auc_str)
            f1_val, f1_lower, f1_upper = parse_ci_string(f1_str)
            sens_val, sens_lower, sens_upper = parse_ci_string(sens_str)
            spec_val, spec_lower, spec_upper = parse_ci_string(spec_str)
            bal_acc_val, bal_acc_lower, bal_acc_upper = parse_ci_string(bal_acc_str)
            
            # Determine model category
            if 'PCE' in model_name or 'SCORE' in model_name or 'LE8' in model_name:
                if 'Clinic_Free' in model_name:
                    category = 'Clinic_Free'
                elif 'Reasonable_Clinic_Free' in model_name:
                    category = 'Clinic_Free'
                else:
                    category = 'Traditional'
            elif 'Demographics' in model_name or 'SDOH' in model_name:
                category = 'User_Defined'
            elif 'only' in model_name:
                category = 'Baseline'
            else:
                category = 'Traditional'
            
            # Determine activity type
            if 'Accel Various' in model_name:
                activity_type = 'Accelerometer-derived MVPA'
            elif 'Accel' in model_name:
                activity_type = 'Accelerometer'
            elif 'Self Reported' in model_name:
                activity_type = 'Self-Reported'
            else:
                activity_type = 'None'
            
            # Find corresponding delta AUC
            delta_row = delta_df[delta_df['Comparison'].str.contains(model_name.replace(' ', ''), regex=False)]
            if not delta_row.empty:
                delta_auc = delta_row.iloc[0]['ΔAUC']
                delta_ci = delta_row.iloc[0]['ΔAUC 95% CI']
            else:
                delta_auc = np.nan
                delta_ci = np.nan
            
            all_data.append({
                'Split': f'Split{split}',
                'Model': model_name,
                'Category': category,
                'Activity_Type': activity_type,
                'AUC': auc_val,
                'AUC_CI_Lower': auc_lower,
                'AUC_CI_Upper': auc_upper,
                'F1_Score': f1_val,
                'Sensitivity': sens_val,
                'Specificity': spec_val,
                'Balanced_Accuracy': bal_acc_val,
                'Delta_AUC': delta_auc,
                'Delta_CI': delta_ci
            })
    
    return pd.DataFrame(all_data)

def plot_performance_comparison(df):
    """Plot 1: Performance comparison across model categories with clear accelerometer vs self-report contrast"""
    
    # Filter for key models to avoid clutter
    key_models = df[
        (df['Model'].str.contains('PCE|SCORE|LE8|Demographics|SDOH')) & 
        (~df['Model'].str.contains('only'))
    ].copy()
    
    # Group by model base and activity type
    key_models['Base_Model'] = key_models['Model'].str.replace(' \+ .*', '', regex=True)
    key_models['Base_Model'] = key_models['Base_Model'].str.replace('_Clinic_Free|_Reasonable_Clinic_Free', '', regex=True)
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Get unique base models and activity types
    base_models = key_models['Base_Model'].unique()
    activity_types = ['None', 'Self-Reported', 'Accelerometer', 'Accelerometer-derived MVPA']
    
    x = np.arange(len(base_models))
    width = 0.2
    
    for i, activity in enumerate(activity_types):
        values = []
        for base in base_models:
            model_data = key_models[
                (key_models['Base_Model'] == base) & 
                (key_models['Activity_Type'] == activity)
            ]
            if not model_data.empty:
                # Average across splits
                values.append(model_data['AUC'].mean())
            else:
                values.append(np.nan)
        
        # Filter out NaN values
        valid_indices = [j for j, v in enumerate(values) if not np.isnan(v)]
        valid_values = [v for v in values if not np.isnan(v)]
        
        if valid_values:
            bars = ax.bar(x[valid_indices] + i*width, valid_values, width, 
                         label=activity, alpha=0.8, 
                         color=COLORS.get(activity, '#95A5A6'))
            
            # Add value labels
            for j, (bar, value) in enumerate(zip(bars, valid_values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_xlabel('Base Model')
    ax.set_ylabel('AUC Score')
    ax.set_title('Model Performance Comparison: Accelerometer vs Self-Reported Physical Activity', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(base_models, rotation=45, ha='right')
    ax.legend(title='Physical Activity Type', loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 0.8)
    
    plt.tight_layout()
    plt.savefig('figures/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_delta_auc_analysis(df):
    """Plot 2: Delta AUC analysis showing improvement from accelerometer vs self-reported data"""
    
    # Filter for models with delta AUC data
    delta_data = df[df['Delta_AUC'].notna()].copy()
    
    # Group by model category and activity type
    delta_data = delta_data[delta_data['Activity_Type'].isin(['Self-Reported', 'Accelerometer', 'Accelerometer-derived MVPA'])]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    categories = ['Traditional', 'Clinic_Free', 'User_Defined']
    
    for i, category in enumerate(categories):
        ax = axes[i]
        cat_data = delta_data[delta_data['Category'] == category]
        
        if not cat_data.empty:
            # Create grouped bar plot
            activity_types = ['Self-Reported', 'Accelerometer', 'Accelerometer-derived MVPA']
            x = np.arange(len(cat_data['Model'].unique()) // 3)
            width = 0.25
            
            for j, activity in enumerate(activity_types):
                act_data = cat_data[cat_data['Activity_Type'] == activity]
                if not act_data.empty:
                    values = act_data['Delta_AUC'].values
                    bars = ax.bar(x + j*width, values, width, 
                                 label=activity, alpha=0.8,
                                 color=COLORS.get(activity, '#95A5A6'))
                    
                    # Add value labels
                    for k, (bar, value) in enumerate(zip(bars, values)):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('ΔAUC')
            ax.set_title(f'{category} Models: ΔAUC Improvement', fontsize=14, fontweight='bold')
            ax.set_xticks(x + width)
            ax.set_xticklabels(['PCE', 'SCORE', 'LE8'] if category == 'Traditional' else 
                              ['PCE_CF', 'SCORE_CF', 'LE8_CF'] if category == 'Clinic_Free' else
                              ['Demo', 'SDOH'], rotation=0)
            ax.legend(title='Physical Activity Type')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.set_ylim(-0.02, 0.12)
    
    plt.suptitle('ΔAUC Analysis: Accelerometer vs Self-Reported Physical Activity Impact', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/delta_auc_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_clinic_free_vs_traditional(df):
    """Plot 3: Clinic-free vs Traditional models performance comparison"""
    
    # Filter for PCE models to show the contrast
    pce_models = df[df['Model'].str.contains('PCE')].copy()
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Group models
    traditional = pce_models[pce_models['Model'].isin(['PCE', 'PCE + Accelerometer', 'PCE + Accelerometer-derived MVPA'])]
    clinic_free = pce_models[pce_models['Model'].str.contains('Clinic_Free')]
    
    # Prepare data for plotting
    model_groups = ['Traditional', 'Clinic-Free', 'Reasonable Clinic-Free']
    activity_types = ['None', 'Accelerometer', 'Accelerometer-derived MVPA']
    
    x = np.arange(len(model_groups))
    width = 0.25
    
    for i, activity in enumerate(activity_types):
        values = []
        for group in model_groups:
            if group == 'Traditional':
                models = traditional[traditional['Activity_Type'] == activity]
            elif group == 'Clinic-Free':
                models = clinic_free[clinic_free['Model'].str.contains('Clinic_Free') & 
                                   ~clinic_free['Model'].str.contains('Reasonable') & 
                                   (clinic_free['Activity_Type'] == activity)]
            else:  # Reasonable Clinic-Free
                models = clinic_free[clinic_free['Model'].str.contains('Reasonable_Clinic_Free') & 
                                   (clinic_free['Activity_Type'] == activity)]
            
            if not models.empty:
                values.append(models['AUC'].mean())
            else:
                values.append(np.nan)
        
        # Filter out NaN values
        valid_indices = [j for j, v in enumerate(values) if not np.isnan(v)]
        valid_values = [v for v in values if not np.isnan(v)]
        
        if valid_values:
            bars = ax.bar(x[valid_indices] + i*width, valid_values, width, 
                         label=activity, alpha=0.8,
                         color=COLORS.get(activity, '#95A5A6'))
            
            # Add value labels
            for j, (bar, value) in enumerate(zip(bars, valid_values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model Type')
    ax.set_ylabel('AUC Score')
    ax.set_title('PCE Model Performance: Traditional vs Clinic-Free Approaches', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_groups)
    ax.legend(title='Physical Activity Type', loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 0.75)
    
    plt.tight_layout()
    plt.savefig('figures/clinic_free_vs_traditional.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confidence_intervals(df):
    """Plot 4: Confidence intervals for key models showing statistical significance"""
    
    # Filter for key models and get confidence interval data
    key_models = ['PCE', 'PCE + Accelerometer', 'PCE + Accelerometer-derived MVPA',
                  'PCE_Clinic_Free', 'PCE_Clinic_Free + Accelerometer', 'PCE_Clinic_Free + Accelerometer-derived MVPA']
    
    ci_data = df[df['Model'].isin(key_models)].copy()
    
    # Create error bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(key_models))
    means = []
    ci_lowers = []
    ci_uppers = []
    
    for model in key_models:
        model_data = ci_data[ci_data['Model'] == model]
        if not model_data.empty:
            means.append(model_data['AUC'].mean())
            ci_lowers.append(model_data['AUC_CI_Lower'].mean())
            ci_uppers.append(model_data['AUC_CI_Upper'].mean())
        else:
            means.append(np.nan)
            ci_lowers.append(np.nan)
            ci_uppers.append(np.nan)
    
    # Calculate error bars
    yerr_lower = [means[i] - ci_lowers[i] for i in range(len(means))]
    yerr_upper = [ci_uppers[i] - means[i] for i in range(len(means))]
    yerr = [yerr_lower, yerr_upper]
    
    # Create bars with error bars
    bars = ax.bar(x, means, yerr=yerr, capsize=10, alpha=0.8, 
                  color=['#2E86AB', '#27AE60', '#27AE60', '#A23B72', '#27AE60', '#27AE60'])
    
    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('AUC Score with 95% CI')
    ax.set_title('Model Performance with Confidence Intervals', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' + ', '\n+ ') for m in key_models], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 0.8)
    
    plt.tight_layout()
    plt.savefig('figures/confidence_intervals.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_comprehensive_dashboard(df):
    """Plot 5: Comprehensive dashboard showing all key findings"""
    
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Performance comparison (top left, 2x2)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Filter for key models
    key_models = df[
        (df['Model'].str.contains('PCE|SCORE|LE8|Demographics|SDOH')) & 
        (~df['Model'].str.contains('only'))
    ].copy()
    
    # Group by model base and activity type
    key_models['Base_Model'] = key_models['Model'].str.replace(' \+ .*', '', regex=True)
    key_models['Base_Model'] = key_models['Base_Model'].str.replace('_Clinic_Free|_Reasonable_Clinic_Free', '', regex=True)
    
    base_models = key_models['Base_Model'].unique()
    activity_types = ['None', 'Self-Reported', 'Accelerometer', 'Accelerometer-derived MVPA']
    
    x = np.arange(len(base_models))
    width = 0.2
    
    for i, activity in enumerate(activity_types):
        values = []
        for base in base_models:
            model_data = key_models[
                (key_models['Base_Model'] == base) & 
                (key_models['Activity_Type'] == activity)
            ]
            if not model_data.empty:
                values.append(model_data['AUC'].mean())
            else:
                values.append(np.nan)
        
        valid_indices = [j for j, v in enumerate(values) if not np.isnan(v)]
        valid_values = [v for v in values if not np.isnan(v)]
        
        if valid_values:
            bars = ax1.bar(x[valid_indices] + i*width, valid_values, width, 
                          label=activity, alpha=0.8, 
                          color=COLORS.get(activity, '#95A5A6'))
            
            for j, (bar, value) in enumerate(zip(bars, valid_values)):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    ax1.set_xlabel('Base Model')
    ax1.set_ylabel('AUC Score')
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(base_models, rotation=45, ha='right')
    ax1.legend(title='Physical Activity Type', loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 0.8)
    
    # 2. Delta AUC summary (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Calculate average delta AUC by category and activity type
    delta_summary = df[df['Delta_AUC'].notna()].groupby(['Category', 'Activity_Type'])['Delta_AUC'].mean().reset_index()
    
    # Create heatmap-style visualization
    categories = ['Traditional', 'Clinic_Free', 'User_Defined']
    activity_types = ['Self-Reported', 'Accelerometer', 'Accelerometer-derived MVPA']
    
    heatmap_data = np.zeros((len(categories), len(activity_types)))
    for i, cat in enumerate(categories):
        for j, act in enumerate(activity_types):
            data = delta_summary[(delta_summary['Category'] == cat) & (delta_summary['Activity_Type'] == act)]
            if not data.empty:
                heatmap_data[i, j] = data.iloc[0]['Delta_AUC']
    
    im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-0.02, vmax=0.08)
    ax2.set_xticks(range(len(activity_types)))
    ax2.set_yticks(range(len(categories)))
    ax2.set_xticklabels([act.replace(' ', '\n') for act in activity_types], rotation=0, ha='center')
    ax2.set_yticklabels(categories)
    ax2.set_title('Average ΔAUC by Category', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i in range(len(categories)):
        for j in range(len(activity_types)):
            text = ax2.text(j, i, f'{heatmap_data[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax2, label='ΔAUC')
    
    # 3. Key findings summary (middle row)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    summary_text = """
    KEY FINDINGS SUMMARY:
    
    • ACCELEROMETER DATA CONSISTENTLY OUTPERFORMS SELF-REPORTED PHYSICAL ACTIVITY:
      - Traditional models: Modest improvements (ΔAUC: 0.001-0.015)
      - Clinic-free models: Substantial improvements (ΔAUC: 0.020-0.044)
      - User-defined models: Large improvements (ΔAUC: 0.071-0.082)
    
    • CLINIC-FREE MODELS BENEFIT MOST FROM ACCELEROMETER DATA:
      - PCE_Clinic_Free + Accelerometer: ΔAUC 0.0375
      - SCORE_Clinic_Free + Accelerometer: ΔAUC 0.0444
      - LE8_Clinic_Free + Accelerometer: ΔAUC 0.0302
    
    • SELF-REPORTED DATA SHOWS MINIMAL TO NEGATIVE IMPROVEMENT:
      - Traditional models: ΔAUC -0.0007 to 0.0001
      - Clinic-free models: ΔAUC -0.003 to 0.0056
    
    • ACCELEROMETER-DERIVED MVPA PROVIDES BEST PERFORMANCE:
      - Most consistent improvements across all model types
      - Particularly effective in clinic-free and user-defined models
    """
    
    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # 4. Model category performance (bottom left)
    ax4 = fig.add_subplot(gs[2, :2])
    
    # Average performance by category
    category_perf = df.groupby(['Category', 'Activity_Type'])['AUC'].mean().reset_index()
    
    categories = ['Traditional', 'Clinic_Free', 'User_Defined']
    activity_types = ['None', 'Self-Reported', 'Accelerometer', 'Accelerometer-derived MVPA']
    
    x = np.arange(len(categories))
    width = 0.2
    
    for i, activity in enumerate(activity_types):
        values = []
        for cat in categories:
            data = category_perf[(category_perf['Category'] == cat) & (category_perf['Activity_Type'] == activity)]
            if not data.empty:
                values.append(data.iloc[0]['AUC'])
            else:
                values.append(np.nan)
        
        valid_indices = [j for j, v in enumerate(values) if not np.isnan(v)]
        valid_values = [v for v in values if not np.isnan(v)]
        
        if valid_values:
            bars = ax4.bar(x[valid_indices] + i*width, valid_values, width, 
                          label=activity, alpha=0.8,
                          color=COLORS.get(activity, '#95A5A6'))
            
            for j, (bar, value) in enumerate(zip(bars, valid_values)):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_xlabel('Model Category')
    ax4.set_ylabel('Average AUC Score')
    ax4.set_title('Performance by Model Category and Physical Activity Type', fontsize=14, fontweight='bold')
    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(categories)
    ax4.legend(title='Physical Activity Type', loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.5, 0.8)
    
    # 5. Statistical significance (bottom right)
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
    
    stats_text = """
    STATISTICAL SIGNIFICANCE:
    
    • Confidence Intervals (95% CI) included for all models
    
    • Accelerometer improvements are statistically significant:
      - Clinic-free models: CI ranges mostly positive
      - User-defined models: CI ranges consistently positive
    
    • Self-reported improvements are not significant:
      - CI ranges include zero or negative values
      - High variability across splits
    
    • Sample size: 1,390 participants
      - 2011-2012: n=684
      - 2013-2014: n=706
    
    • Validation: 1,000 bootstrapped iterations
    """
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # 6. Final recommendation (bottom row)
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    recommendation_text = """
    RECOMMENDATIONS FOR CLINICAL IMPLEMENTATION:
    
    1. PRIORITIZE ACCELEROMETER-DERIVED PHYSICAL ACTIVITY MEASURES:
       - Replace self-reported physical activity in all models
       - Use "Accelerometer-derived MVPA" for best performance
    
    2. IMPLEMENT CLINIC-FREE MODELS WITH ACCELEROMETER DATA:
       - Most significant improvements in clinic-free settings
       - Enables remote monitoring and screening
    
    3. CONSIDER USER-DEFINED MODELS FOR POPULATION HEALTH:
       - Demographics + Accelerometer: AUC 0.60-0.61
       - SDOH + Accelerometer: AUC 0.63-0.64
    
    4. AVOID SELF-REPORTED PHYSICAL ACTIVITY:
       - Minimal to negative improvement in all models
       - High variability and low reliability
    """
    
    ax6.text(0.05, 0.95, recommendation_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.suptitle('NHANES Physical Activity Analysis - Comprehensive Dashboard (Updated)', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('figures/comprehensive_dashboard_updated.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run all updated visualizations"""
    print("Creating Updated NHANES Physical Activity Analysis Visualizations...")
    
    # Load and process data
    df = load_and_process_data()
    print(f"Loaded dataset with {len(df)} observations")
    
    # Create all visualizations
    print("1. Creating performance comparison plot...")
    plot_performance_comparison(df)
    
    print("2. Creating delta AUC analysis...")
    plot_delta_auc_analysis(df)
    
    print("3. Creating clinic-free vs traditional comparison...")
    plot_clinic_free_vs_traditional(df)
    
    print("4. Creating confidence intervals plot...")
    plot_confidence_intervals(df)
    
    print("5. Creating comprehensive dashboard...")
    plot_comprehensive_dashboard(df)
    
    print("\nAll updated visualizations completed and saved to 'figures/' directory!")
    print("\nKey improvements made:")
    print("- Updated terminology: 'Accel Various' → 'Accelerometer-derived MVPA'")
    print("- Clear contrast highlighting accelerometer vs self-reported performance")
    print("- Proper confidence intervals included")
    print("- Logical model grouping: Traditional, Clinic-Free, User-Defined")
    print("- Emphasis on clinic-free model benefits from accelerometer data")

if __name__ == "__main__":
    main()
