import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for better image generation
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0.5

def parse_ci_string(ci_string):
    """Parse confidence interval string like '0.69 (0.65–0.73)'"""
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

def transform_model_name(model_name):
    """Transform model names from old naming convention to new naming convention"""
    if pd.isna(model_name) or model_name == '':
        return model_name
    
    # Apply transformations in order
    transformed_name = model_name
    
    # Replace model combination names
    transformed_name = transformed_name.replace('Accel Various', 'Acc-Driven MVPA')
    transformed_name = transformed_name.replace('Accel', 'Total MIMS')
    transformed_name = transformed_name.replace('Demographics', 'DEMO')
    
    # Replace underscores with spaces for clinic-free models
    transformed_name = transformed_name.replace('PCE_Clinic_Free', 'PCE Clinic-Free')
    transformed_name = transformed_name.replace('SCORE_Clinic_Free', 'SCORE Clinic-Free')
    transformed_name = transformed_name.replace('LE8_Clinic_Free', 'LE8 Clinic-Free')
    
    # Replace underscores with spaces for reasonable clinic-free models
    transformed_name = transformed_name.replace('PCE_Reasonable_Clinic_Free', 'PCE Reasonable Clinic-Free')
    transformed_name = transformed_name.replace('SCORE_Reasonable_Clinic_Free', 'SCORE Reasonable Clinic-Free')
    transformed_name = transformed_name.replace('LE8_Reasonable_Clinic_Free', 'LE8 Reasonable Clinic-Free')
    
    return transformed_name

def load_real_data():
    """Load and process the actual data files"""
    # Load bootstrap results for all splits
    splits = ['A', 'B', 'C']
    all_data = []
    
    for split in splits:
        bootstrap_file = f'data/bootstrap_results_summary_{split}.csv'
        bootstrap_df = pd.read_csv(bootstrap_file)
        
        for _, row in bootstrap_df.iterrows():
            model_name = row.iloc[0]
            auc_str = row['ROC AUC']
            
            # Transform model names to match new naming convention
            model_name = transform_model_name(model_name)
            
            # Parse values and CIs
            auc_val, auc_lower, auc_upper = parse_ci_string(auc_str)
            
            all_data.append({
                'Split': f'Split{split}',
                'Model': model_name,
                'AUC': auc_val,
                'AUC_CI_Lower': auc_lower,
                'AUC_CI_Upper': auc_upper
            })
    
    return pd.DataFrame(all_data)

def create_figure_traditional_risk_models(df):
    """Figure for Impact of Physical Activity Measures on Clinical Risk Models"""
    
    # Filter for traditional risk models
    traditional_models = ['PCE', 'PCE + Self Reported Physical Activity', 'PCE + Total MIMS', 'PCE + Acc-Driven MVPA',
                         'SCORE', 'SCORE + Self Reported Physical Activity', 'SCORE + Total MIMS', 'SCORE + Acc-Driven MVPA',
                         'LE8', 'LE8 + Self Reported Physical Activity', 'LE8 + Total MIMS', 'LE8 + Acc-Driven MVPA']
    
    # Filter data for these models
    filtered_data = df[df['Model'].isin(traditional_models)].copy()
    
    # Create the plot with explicit white background
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Get unique models
    models = filtered_data['Model'].unique()
    x = np.arange(len(models))
    width = 0.25
    
    # Color mapping: Split A = blue, Split B = orange, Split C = green
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, split in enumerate(['SplitA', 'SplitB', 'SplitC']):
        split_data = filtered_data[filtered_data['Split'] == split]
        
        auc_values = []
        ci_lowers = []
        ci_uppers = []
        
        for model in models:
            model_data = split_data[split_data['Model'] == model]
            if not model_data.empty:
                auc_values.append(model_data['AUC'].iloc[0])
                ci_lowers.append(model_data['AUC_CI_Lower'].iloc[0])
                ci_uppers.append(model_data['AUC_CI_Upper'].iloc[0])
            else:
                auc_values.append(np.nan)
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)
        
        # Filter out NaN values
        valid_indices = [j for j, v in enumerate(auc_values) if not np.isnan(v)]
        valid_values = [v for j, v in enumerate(auc_values) if not np.isnan(v)]
        valid_ci_lowers = [ci_lowers[j] for j in valid_indices]
        valid_ci_uppers = [ci_uppers[j] for j in valid_indices]
        
        if valid_values:
            bars = ax.bar(x[valid_indices] + i*width, valid_values, width, 
                         label=f'Split {split[-1]}', alpha=0.8, color=colors[i])
            
            # Add error bars for confidence intervals
            yerr = [[valid_values[j] - valid_ci_lowers[j] for j in range(len(valid_values))],
                   [valid_ci_uppers[j] - valid_values[j] for j in range(len(valid_values))]]
            
            ax.errorbar(x[valid_indices] + i*width, valid_values, yerr=yerr, 
                       fmt='none', color='black', capsize=5, capthick=1)
            
            # Add value labels
            for j, (bar, value) in enumerate(zip(bars, valid_values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Clean up model names for display
    display_names = [m.replace('Self Reported Physical Activity', 'Self-Report MVPA').replace(' + ', '\n+ ') for m in models]
    
    ax.set_xlabel('Model')
    ax.set_ylabel('ROC AUC with 95% CI')
    ax.set_title('Performance of Traditional Risk Models', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.legend(title='Validation Split')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.40, 0.80)  # Fixed y-axis range
    
    plt.tight_layout()
    plt.savefig('figures/figure_traditional_risk_models.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def create_figure_clinic_free_models(df):
    """Figure for Clinic-Free Models"""
    
    # Filter for clinic-free models
    clinic_free_models = ['PCE Clinic-Free', 'PCE Clinic-Free + Self Reported Physical Activity', 
                         'PCE Clinic-Free + Total MIMS', 'PCE Clinic-Free + Acc-Driven MVPA',
                         'SCORE Clinic-Free', 'SCORE Clinic-Free + Self Reported Physical Activity',
                         'SCORE Clinic-Free + Total MIMS', 'SCORE Clinic-Free + Acc-Driven MVPA',
                         'LE8 Clinic-Free', 'LE8 Clinic-Free + Self Reported Physical Activity',
                         'LE8 Clinic-Free + Total MIMS', 'LE8 Clinic-Free + Acc-Driven MVPA']
    
    # Filter data for these models
    filtered_data = df[df['Model'].isin(clinic_free_models)].copy()
    
    # Create the plot with explicit white background
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Get unique models
    models = filtered_data['Model'].unique()
    x = np.arange(len(models))
    width = 0.25
    
    # Color mapping: Split A = blue, Split B = orange, Split C = green
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, split in enumerate(['SplitA', 'SplitB', 'SplitC']):
        split_data = filtered_data[filtered_data['Split'] == split]
        
        auc_values = []
        ci_lowers = []
        ci_uppers = []
        
        for model in models:
            model_data = split_data[split_data['Model'] == model]
            if not model_data.empty:
                auc_values.append(model_data['AUC'].iloc[0])
                ci_lowers.append(model_data['AUC_CI_Lower'].iloc[0])
                ci_uppers.append(model_data['AUC_CI_Upper'].iloc[0])
            else:
                auc_values.append(np.nan)
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)
        
        # Filter out NaN values
        valid_indices = [j for j, v in enumerate(auc_values) if not np.isnan(v)]
        valid_values = [v for j, v in enumerate(auc_values) if not np.isnan(v)]
        valid_ci_lowers = [ci_lowers[j] for j in valid_indices]
        valid_ci_uppers = [ci_uppers[j] for j in valid_indices]
        
        if valid_values:
            bars = ax.bar(x[valid_indices] + i*width, valid_values, width, 
                         label=f'Split {split[-1]}', alpha=0.8, color=colors[i])
            
            # Add error bars for confidence intervals
            yerr = [[valid_values[j] - valid_ci_lowers[j] for j in range(len(valid_values))],
                   [valid_ci_uppers[j] - valid_values[j] for j in range(len(valid_values))]]
            
            ax.errorbar(x[valid_indices] + i*width, valid_values, yerr=yerr, 
                       fmt='none', color='black', capsize=5, capthick=1)
            
            # Add value labels
            for j, (bar, value) in enumerate(zip(bars, valid_values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Clean up model names for display
    display_names = [m.replace('Self Reported Physical Activity', 'Self-Report MVPA').replace(' + ', '\n+ ') for m in models]
    
    ax.set_xlabel('Model')
    ax.set_ylabel('ROC AUC with 95% CI')
    ax.set_title('Performance of Clinic-Free Models', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.legend(title='Validation Split')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.40, 0.80)  # Fixed y-axis range
    
    plt.tight_layout()
    plt.savefig('figures/figure_clinic_free_models.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def create_figure_reasonable_clinic_free_models(df):
    """Figure for Reasonable Clinic-Free Models (Including Blood Pressure)"""
    
    # Filter for reasonable clinic-free models
    reasonable_models = ['PCE Reasonable Clinic-Free', 'PCE Reasonable Clinic-Free + Self Reported Physical Activity',
                        'PCE Reasonable Clinic-Free + Total MIMS', 'PCE Reasonable Clinic-Free + Acc-Driven MVPA',
                        'SCORE Reasonable Clinic-Free', 'SCORE Reasonable Clinic-Free + Self Reported Physical Activity',
                        'SCORE Reasonable Clinic-Free + Total MIMS', 'SCORE Reasonable Clinic-Free + Acc-Driven MVPA',
                        'LE8 Reasonable Clinic-Free', 'LE8 Reasonable Clinic-Free + Self Reported Physical Activity',
                        'LE8 Reasonable Clinic-Free + Total MIMS', 'LE8 Reasonable Clinic-Free + Acc-Driven MVPA']
    
    # Filter data for these models
    filtered_data = df[df['Model'].isin(reasonable_models)].copy()
    
    # Create the plot with explicit white background
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Get unique models
    models = filtered_data['Model'].unique()
    x = np.arange(len(models))
    width = 0.25
    
    # Color mapping: Split A = blue, Split B = orange, Split C = green
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, split in enumerate(['SplitA', 'SplitB', 'SplitC']):
        split_data = filtered_data[filtered_data['Split'] == split]
        
        auc_values = []
        ci_lowers = []
        ci_uppers = []
        
        for model in models:
            model_data = split_data[split_data['Model'] == model]
            if not model_data.empty:
                auc_values.append(model_data['AUC'].iloc[0])
                ci_lowers.append(model_data['AUC_CI_Lower'].iloc[0])
                ci_uppers.append(model_data['AUC_CI_Upper'].iloc[0])
            else:
                auc_values.append(np.nan)
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)
        
        # Filter out NaN values
        valid_indices = [j for j, v in enumerate(auc_values) if not np.isnan(v)]
        valid_values = [v for j, v in enumerate(auc_values) if not np.isnan(v)]
        valid_ci_lowers = [ci_lowers[j] for j in valid_indices]
        valid_ci_uppers = [ci_uppers[j] for j in valid_indices]
        
        if valid_values:
            bars = ax.bar(x[valid_indices] + i*width, valid_values, width, 
                         label=f'Split {split[-1]}', alpha=0.8, color=colors[i])
            
            # Add error bars for confidence intervals
            yerr = [[valid_values[j] - valid_ci_lowers[j] for j in range(len(valid_values))],
                   [valid_ci_uppers[j] - valid_values[j] for j in range(len(valid_values))]]
            
            ax.errorbar(x[valid_indices] + i*width, valid_values, yerr=yerr, 
                       fmt='none', color='black', capsize=5, capthick=1)
            
            # Add value labels
            for j, (bar, value) in enumerate(zip(bars, valid_values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Clean up model names for display
    display_names = [m.replace('Self Reported Physical Activity', 'Self-Report MVPA').replace(' + ', '\n+ ') for m in models]
    
    ax.set_xlabel('Model')
    ax.set_ylabel('ROC AUC with 95% CI')
    ax.set_title('Performance of Reasonable Clinic-Free Models (Including Blood Pressure)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.legend(title='Validation Split')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.40, 0.80)  # Fixed y-axis range
    
    plt.tight_layout()
    plt.savefig('figures/figure_reasonable_clinic_free_models.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def create_figure_sociodemographic_models(df):
    """Figure for Performance with Sociodemographic Variables Only"""
    
    # Filter for sociodemographic models
    socio_models = ['DEMO', 'DEMO + Self Reported Physical Activity',
                   'DEMO + Total MIMS', 'DEMO + Acc-Driven MVPA',
                   'SDOH', 'SDOH + Self Reported Physical Activity',
                   'SDOH + Total MIMS', 'SDOH + Acc-Driven MVPA']
    
    # Filter data for these models
    filtered_data = df[df['Model'].isin(socio_models)].copy()
    
    # Create the plot with explicit white background
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Get unique models
    models = filtered_data['Model'].unique()
    x = np.arange(len(models))
    width = 0.25
    
    # Color mapping: Split A = blue, Split B = orange, Split C = green
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, split in enumerate(['SplitA', 'SplitB', 'SplitC']):
        split_data = filtered_data[filtered_data['Split'] == split]
        
        auc_values = []
        ci_lowers = []
        ci_uppers = []
        
        for model in models:
            model_data = split_data[split_data['Model'] == model]
            if not model_data.empty:
                auc_values.append(model_data['AUC'].iloc[0])
                ci_lowers.append(model_data['AUC_CI_Lower'].iloc[0])
                ci_uppers.append(model_data['AUC_CI_Upper'].iloc[0])
            else:
                auc_values.append(np.nan)
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)
        
        # Filter out NaN values
        valid_indices = [j for j, v in enumerate(auc_values) if not np.isnan(v)]
        valid_values = [v for j, v in enumerate(auc_values) if not np.isnan(v)]
        valid_ci_lowers = [ci_lowers[j] for j in valid_indices]
        valid_ci_uppers = [ci_uppers[j] for j in valid_indices]
        
        if valid_values:
            bars = ax.bar(x[valid_indices] + i*width, valid_values, width, 
                         label=f'Split {split[-1]}', alpha=0.8, color=colors[i])
            
            # Add error bars for confidence intervals
            yerr = [[valid_values[j] - valid_ci_lowers[j] for j in range(len(valid_values))],
                   [valid_ci_uppers[j] - valid_values[j] for j in range(len(valid_values))]]
            
            ax.errorbar(x[valid_indices] + i*width, valid_values, yerr=yerr, 
                       fmt='none', color='black', capsize=5, capthick=1)
            
            # Add value labels
            for j, (bar, value) in enumerate(zip(bars, valid_values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Clean up model names for display
    display_names = [m.replace('Self Reported Physical Activity', 'Self-Report MVPA').replace(' + ', '\n+ ') for m in models]
    
    ax.set_xlabel('Model')
    ax.set_ylabel('ROC AUC with 95% CI')
    ax.set_title('Performance of Sociodemographic Models', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.legend(title='Validation Split')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.40, 0.80)  # Fixed y-axis range
    
    plt.tight_layout()
    plt.savefig('figures/figure_sociodemographic_models.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def main():
    """Main function to create all four manuscript figures"""
    print("Creating Updated Manuscript Figures with Real Data...")
    
    # Load real data
    df = load_real_data()
    print(f"Loaded dataset with {len(df)} observations")
    
    print("1. Creating figure for Traditional Risk Models...")
    create_figure_traditional_risk_models(df)
    
    print("2. Creating figure for Clinic-Free Models...")
    create_figure_clinic_free_models(df)
    
    print("3. Creating figure for Reasonable Clinic-Free Models...")
    create_figure_reasonable_clinic_free_models(df)
    
    print("4. Creating figure for Sociodemographic Models...")
    create_figure_sociodemographic_models(df)
    
    print("\nAll four manuscript figures completed and saved to 'figures/' directory!")
    print("Updates made:")
    print("- Y-axis range fixed to 0.40-0.80 across all figures")
    print("- 'Self-Report' changed to 'Self-Report MVPA' consistently")
    print("- Using real data from bootstrap_results_summary CSV files")
    print("- Color scheme: Split A = blue, Split B = orange, Split C = green")

if __name__ == "__main__":
    main()
