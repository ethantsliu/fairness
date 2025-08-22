import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def parse_ci_string(ci_string):
    """Parse confidence interval string like '0.68 (0.64–0.72)'"""
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

def load_data():
    """Load and process the data files"""
    # Load bootstrap results for splits A and C
    splits = ['A', 'C']
    all_data = []
    
    for split in splits:
        bootstrap_file = f'data/bootstrap_results_summary_{split}.csv'
        bootstrap_df = pd.read_csv(bootstrap_file)
        
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
            
            all_data.append({
                'Split': f'Split{split}',
                'Model': model_name,
                'AUC': auc_val,
                'AUC_CI_Lower': auc_lower,
                'AUC_CI_Upper': auc_upper,
                'F1_Score': f1_val,
                'Sensitivity': sens_val,
                'Specificity': spec_val,
                'Balanced_Accuracy': bal_acc_val
            })
    
    return pd.DataFrame(all_data)

def create_figure_3(df):
    """Figure 3: Model performance (ROC AUC) across different feature combinations"""
    
    # Filter for key models
    key_models = ['PCE', 'PCE + Accel', 'PCE + SDOH + Accel',
                  'SCORE', 'SCORE + Accel', 'SCORE + SDOH + Accel',
                  'LE8', 'LE8 + Accel', 'LE8 + DEMO + Accel', 
                  'LE8 + SDOH + Accel', 'LE8 + SDOH + DEMO + Accel',
                  'DEMO + SDOH + Accel']
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create grouped bar plot
    x = np.arange(len(key_models))
    width = 0.25
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71']  # Red, Blue, Green for splits
    
    for i, split in enumerate(['SplitA', 'SplitC']):
        split_data = df[df['Split'] == split]
        
        values = []
        ci_lowers = []
        ci_uppers = []
        
        for model in key_models:
            model_data = split_data[split_data['Model'] == model]
            if not model_data.empty:
                values.append(model_data['AUC'].iloc[0])
                ci_lowers.append(model_data['AUC_CI_Lower'].iloc[0])
                ci_uppers.append(model_data['AUC_CI_Upper'].iloc[0])
            else:
                values.append(np.nan)
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)
        
        # Filter out NaN values
        valid_indices = [j for j, v in enumerate(values) if not np.isnan(v)]
        valid_values = [v for j, v in enumerate(values) if not np.isnan(v)]
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
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_xlabel('Model Combinations')
    ax.set_ylabel('ROC AUC')
    ax.set_title('Figure 3. Model performance (ROC AUC) across different feature combinations', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([m.replace(' + ', '\n+ ') for m in key_models], rotation=45, ha='right')
    ax.legend(title='Validation Split')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig('figures/figure_3_roc_auc.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_figure_4(df):
    """Figure 4: Change in model performance (ΔAUC) with accelerometer integration"""
    
    # Calculate delta AUC for models with accelerometer
    delta_data = []
    
    baseline_models = ['PCE', 'SCORE', 'LE8', 'DEMO', 'SDOH']
    
    for split in ['SplitA', 'SplitC']:
        split_data = df[df['Split'] == split]
        
        for base in baseline_models:
            # Find baseline model
            base_data = split_data[split_data['Model'] == base]
            if not base_data.empty:
                base_auc = base_data['AUC'].iloc[0]
                
                # Find models with accelerometer
                accel_models = split_data[split_data['Model'].str.contains('Accel') & 
                                        split_data['Model'].str.contains(base)]
                
                for _, accel_row in accel_models.iterrows():
                    accel_auc = accel_row['AUC']
                    delta_auc = accel_auc - base_auc
                    
                    delta_data.append({
                        'Split': split,
                        'Baseline': base,
                        'Model': accel_row['Model'],
                        'Delta_AUC': delta_auc
                    })
    
    delta_df = pd.DataFrame(delta_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Group by baseline model
    baseline_models = ['PCE', 'SCORE', 'LE8', 'DEMO', 'SDOH']
    x_positions = np.arange(len(baseline_models))
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71']  # Red, Blue, Green for splits
    
    for i, split in enumerate(['SplitA', 'SplitC']):
        split_data = delta_df[delta_df['Split'] == split]
        
        values = []
        for base in baseline_models:
            base_data = split_data[split_data['Baseline'] == base]
            if not base_data.empty:
                # Average delta AUC for this baseline
                values.append(base_data['Delta_AUC'].mean())
            else:
                values.append(0)
        
        bars = ax.bar(x_positions + i*0.2, values, 0.2, 
                     label=f'Split {split[-1]}', alpha=0.8, color=colors[i])
        
        # Add value labels
        for j, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add dotted vertical lines to separate baseline categories
    for i in range(len(baseline_models) - 1):
        ax.axvline(x=i + 0.5, color='black', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Baseline Model')
    ax.set_ylabel('ΔAUC (Change in AUC)')
    ax.set_title('Figure 4. Change in model performance (ΔAUC) with accelerometer integration', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x_positions + 0.1)
    ax.set_xticklabels(baseline_models)
    ax.legend(title='Validation Split')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.set_ylim(-0.02, 0.12)
    
    plt.tight_layout()
    plt.savefig('figures/figure_4_delta_auc.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_table_1():
    """Table 1: Baseline characteristics of CVD cases and controls"""
    
    # Create sample data based on your example
    data = {
        'Characteristics': [
            'Age, years',
            'Female, n (%)',
            'BMI, kg/m²',
            'Total cholesterol, mg/dL',
            'HDL cholesterol, mg/dL',
            'Systolic BP, mmHg',
            'Diastolic BP, mmHg',
            'HbA1c, %',
            'Former smoker (100+ cigs, now quit), n (%)',
            'Sleep hours/night',
            'Has wage/salary income, n (%)',
            'Education <HS, n (%)',
            'Emergency food assistance, n (%)',
            'Daily MIMS, mean (SD)'
        ],
        'CVD Cases': [
            '65.83 (12.96)',
            '312 (44.9%)',
            '30.22 (7.45)',
            '178.78 (43.97)',
            '50.15 (15.62)',
            '130.85 (20.34)',
            '67.15 (14.80)',
            '6.30 (1.43)',
            '274 (39.4%)',
            '6.91 (1.62)',
            '320 (46.0%)',
            '219 (31.6%)',
            '94 (13.5%)',
            '8.41 (5.56)'
        ],
        'Controls': [
            '64.92 (3.55)',
            '374 (53.8%)',
            '29.17 (6.33)',
            '198.74 (43.54)',
            '54.79 (15.87)',
            '131.57 (18.64)',
            '71.90 (12.23)',
            '6.05 (1.17)',
            '360 (51.8%)',
            '6.93 (1.34)',
            '434 (62.4%)',
            '163 (23.5%)',
            '46 (6.6%)',
            '10.05 (5.95)'
        ],
        'P-value': [
            '0.074',
            '0.001',
            '0.005',
            '<0.001',
            '<0.001',
            '0.494',
            '<0.001',
            '<0.001',
            '<0.001',
            '0.760',
            '<0.001',
            '<0.001',
            '<0.001',
            '<0.001'
        ]
    }
    
    # Create the table
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=[[data['Characteristics'][i], data['CVD Cases'][i], 
                                data['Controls'][i], data['P-value'][i]] 
                               for i in range(len(data['Characteristics']))],
                     colLabels=['Characteristics', 'CVD Cases', 'Controls', 'P-value'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.4, 0.2, 0.2, 0.2])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style alternating rows
    for i in range(1, len(data['Characteristics']) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')
    
    # Add title and description
    ax.text(0.5, 0.95, 'Table 1. Baseline characteristics of CVD cases and controls after 1:1 undersampling', 
            transform=ax.transAxes, fontsize=14, fontweight='bold', ha='center')
    
    description = """Values are presented as mean (SD) for continuous variables and n (%) for categorical variables. 
P-values were obtained from t-tests for continuous variables and chi-square tests for categorical variables.
Abbreviations: BP, blood pressure; BMI, body mass index; HDL, high-density lipoprotein; MIMS, Monitor-Independent Movement Summary."""
    
    ax.text(0.5, 0.92, description, transform=ax.transAxes, fontsize=10, ha='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('figures/table_1_baseline_characteristics.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to create all figures and tables"""
    print("Creating NHANES CVD Analysis Figures and Tables...")
    
    # Load data
    df = load_data()
    print(f"Loaded dataset with {len(df)} observations")
    
    # Create figures and tables
    print("1. Creating Figure 3: ROC AUC performance...")
    create_figure_3(df)
    
    print("2. Creating Figure 4: Delta AUC analysis...")
    create_figure_4(df)
    
    print("3. Creating Table 1: Baseline characteristics...")
    create_table_1()
    
    print("\nAll figures and tables completed and saved to 'figures/' directory!")

if __name__ == "__main__":
    main()
