import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
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

def create_figure_traditional_risk_models():
    """Figure for Impact of Physical Activity Measures on Clinical Risk Models"""
    
    # Data from Table 2 - Baseline AUC + Delta AUC to get actual AUC values
    # Split A, B, C data for each model
    data = {
        'Model': ['PCE', 'PCE', 'PCE', 'PCE + Self-Report', 'PCE + Self-Report', 'PCE + Self-Report',
                  'PCE + Total MIMS', 'PCE + Total MIMS', 'PCE + Total MIMS',
                  'PCE + Acc-Driven MVPA', 'PCE + Acc-Driven MVPA', 'PCE + Acc-Driven MVPA',
                  'SCORE', 'SCORE', 'SCORE', 'SCORE + Self-Report', 'SCORE + Self-Report', 'SCORE + Self-Report',
                  'SCORE + Total MIMS', 'SCORE + Total MIMS', 'SCORE + Total MIMS',
                  'SCORE + Acc-Driven MVPA', 'SCORE + Acc-Driven MVPA', 'SCORE + Acc-Driven MVPA',
                  'LE8', 'LE8', 'LE8', 'LE8 + Self-Report', 'LE8 + Self-Report', 'LE8 + Self-Report',
                  'LE8 + Total MIMS', 'LE8 + Total MIMS', 'LE8 + Total MIMS',
                  'LE8 + Acc-Driven MVPA', 'LE8 + Acc-Driven MVPA', 'LE8 + Acc-Driven MVPA'],
        
        'Split': ['A', 'B', 'C'] * 12,
        
        'AUC': [0.69, 0.68, 0.65,  # PCE baseline
                0.69, 0.68, 0.65,  # PCE + Self-Report (baseline + 0)
                0.699, 0.689, 0.659,  # PCE + Total MIMS (baseline + 0.009, 0.009, 0.009)
                0.691, 0.681, 0.651,  # PCE + Acc-Driven MVPA (baseline + 0.001, 0.001, 0.001)
                0.63, 0.64, 0.61,  # SCORE baseline
                0.628, 0.64, 0.61,  # SCORE + Self-Report (baseline - 0.002, 0, 0)
                0.645, 0.655, 0.625,  # SCORE + Total MIMS (baseline + 0.015, 0.015, 0.015)
                0.64, 0.65, 0.62,  # SCORE + Acc-Driven MVPA (baseline + 0.01, 0.01, 0.01)
                0.64, 0.64, 0.61,  # LE8 baseline
                0.639, 0.641, 0.61,  # LE8 + Self-Report (baseline - 0.001, 0.001, 0.001)
                0.648, 0.648, 0.618,  # LE8 + Total MIMS (baseline + 0.008, 0.008, 0.008)
                0.642, 0.645, 0.612],  # LE8 + Acc-Driven MVPA (baseline + 0.002, 0.005, 0.002)
        
        'CI_Lower': [0.65, 0.64, 0.58, 0.65, 0.64, 0.58, 0.659, 0.649, 0.588, 0.651, 0.641, 0.581,
                     0.59, 0.60, 0.54, 0.588, 0.60, 0.54, 0.605, 0.615, 0.564, 0.60, 0.61, 0.561,
                     0.60, 0.60, 0.54, 0.599, 0.601, 0.54, 0.608, 0.608, 0.558, 0.602, 0.605, 0.552],
        
        'CI_Upper': [0.73, 0.72, 0.71, 0.73, 0.72, 0.71, 0.739, 0.729, 0.719, 0.731, 0.721, 0.711,
                     0.67, 0.68, 0.68, 0.668, 0.68, 0.68, 0.685, 0.695, 0.685, 0.68, 0.69, 0.679,
                     0.68, 0.68, 0.68, 0.679, 0.681, 0.68, 0.688, 0.688, 0.678, 0.682, 0.685, 0.672]
    }
    
    df = pd.DataFrame(data)
    
    # Create the plot with explicit white background
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Get unique models
    models = df['Model'].unique()
    x = np.arange(len(models))
    width = 0.25
    
    # Color mapping: Split A = blue, Split B = orange, Split C = green
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, split in enumerate(['A', 'B', 'C']):
        split_data = df[df['Split'] == split]
        
        auc_values = []
        ci_lowers = []
        ci_uppers = []
        
        for model in models:
            model_data = split_data[split_data['Model'] == model]
            if not model_data.empty:
                auc_values.append(model_data['AUC'].iloc[0])
                ci_lowers.append(model_data['CI_Lower'].iloc[0])
                ci_uppers.append(model_data['CI_Upper'].iloc[0])
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
                         label=f'Split {split}', alpha=0.8, color=colors[i])
            
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
    
    ax.set_xlabel('Model')
    ax.set_ylabel('ROC AUC with 95% CI')
    ax.set_title('Impact of Physical Activity Measures on Clinical Risk Models', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace(' + ', '\n+ ') for m in models], rotation=45, ha='right')
    ax.legend(title='Validation Split')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 0.8)
    
    plt.tight_layout()
    plt.savefig('figures/figure_traditional_risk_models.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def create_figure_clinic_free_models():
    """Figure for Clinic-Free Models"""
    
    # Data from Table 3
    data = {
        'Model': ['PCE Clinic-Free', 'PCE Clinic-Free', 'PCE Clinic-Free',
                  'PCE Clinic-Free + Self-Report', 'PCE Clinic-Free + Self-Report', 'PCE Clinic-Free + Self-Report',
                  'PCE Clinic-Free + Total MIMS', 'PCE Clinic-Free + Total MIMS', 'PCE Clinic-Free + Total MIMS',
                  'PCE Clinic-Free + Acc-Driven MVPA', 'PCE Clinic-Free + Acc-Driven MVPA', 'PCE Clinic-Free + Acc-Driven MVPA',
                  'SCORE Clinic-Free', 'SCORE Clinic-Free', 'SCORE Clinic-Free',
                  'SCORE Clinic-Free + Self-Report', 'SCORE Clinic-Free + Self-Report', 'SCORE Clinic-Free + Self-Report',
                  'SCORE Clinic-Free + Total MIMS', 'SCORE Clinic-Free + Total MIMS', 'SCORE Clinic-Free + Total MIMS',
                  'SCORE Clinic-Free + Acc-Driven MVPA', 'SCORE Clinic-Free + Acc-Driven MVPA', 'SCORE Clinic-Free + Acc-Driven MVPA',
                  'LE8 Clinic-Free', 'LE8 Clinic-Free', 'LE8 Clinic-Free',
                  'LE8 Clinic-Free + Self-Report', 'LE8 Clinic-Free + Self-Report', 'LE8 Clinic-Free + Self-Report',
                  'LE8 Clinic-Free + Total MIMS', 'LE8 Clinic-Free + Total MIMS', 'LE8 Clinic-Free + Total MIMS',
                  'LE8 Clinic-Free + Acc-Driven MVPA', 'LE8 Clinic-Free + Acc-Driven MVPA', 'LE8 Clinic-Free + Acc-Driven MVPA'],
        
        'Split': ['A', 'B', 'C'] * 12,
        
        'AUC': [0.59, 0.57, 0.53,  # PCE Clinic-Free baseline
                0.593, 0.573, 0.532,  # PCE Clinic-Free + Self-Report (baseline + 0.003, 0.003, 0.002)
                0.625, 0.605, 0.565,  # PCE Clinic-Free + Total MIMS (baseline + 0.036, 0.035, 0.035)
                0.602, 0.582, 0.542,  # PCE Clinic-Free + Acc-Driven MVPA (baseline + 0.012, 0.012, 0.012)
                0.57, 0.55, 0.53,  # SCORE Clinic-Free baseline
                0.582, 0.562, 0.542,  # SCORE Clinic-Free + Self-Report (baseline + 0.012, 0.012, 0.012)
                0.615, 0.595, 0.575,  # SCORE Clinic-Free + Total MIMS (baseline + 0.045, 0.045, 0.045)
                0.597, 0.577, 0.557,  # SCORE Clinic-Free + Acc-Driven MVPA (baseline + 0.027, 0.027, 0.027)
                0.56, 0.55, 0.53,  # LE8 Clinic-Free baseline
                0.559, 0.552, 0.532,  # LE8 Clinic-Free + Self-Report (baseline - 0.001, 0.002, 0.002)
                0.599, 0.589, 0.569,  # LE8 Clinic-Free + Total MIMS (baseline + 0.039, 0.039, 0.039)
                0.564, 0.561, 0.541],  # LE8 Clinic-Free + Acc-Driven MVPA (baseline + 0.004, 0.011, 0.011)
        
        'CI_Lower': [0.55, 0.53, 0.49, 0.553, 0.533, 0.492, 0.585, 0.565, 0.525, 0.562, 0.542, 0.502,
                     0.51, 0.49, 0.49, 0.522, 0.502, 0.502, 0.555, 0.535, 0.515, 0.537, 0.517, 0.497,
                     0.52, 0.51, 0.49, 0.519, 0.512, 0.492, 0.559, 0.549, 0.529, 0.524, 0.521, 0.501],
        
        'CI_Upper': [0.63, 0.61, 0.57, 0.633, 0.613, 0.572, 0.665, 0.645, 0.605, 0.642, 0.622, 0.582,
                     0.63, 0.61, 0.57, 0.642, 0.622, 0.582, 0.675, 0.655, 0.635, 0.657, 0.637, 0.617,
                     0.60, 0.59, 0.57, 0.599, 0.592, 0.572, 0.639, 0.629, 0.609, 0.604, 0.601, 0.581]
    }
    
    df = pd.DataFrame(data)
    
    # Create the plot with explicit white background
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Get unique models
    models = df['Model'].unique()
    x = np.arange(len(models))
    width = 0.25
    
    # Color mapping: Split A = blue, Split B = orange, Split C = green
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, split in enumerate(['A', 'B', 'C']):
        split_data = df[df['Split'] == split]
        
        auc_values = []
        ci_lowers = []
        ci_uppers = []
        
        for model in models:
            model_data = split_data[split_data['Model'] == model]
            if not model_data.empty:
                auc_values.append(model_data['AUC'].iloc[0])
                ci_lowers.append(model_data['CI_Lower'].iloc[0])
                ci_uppers.append(model_data['CI_Upper'].iloc[0])
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
                         label=f'Split {split}', alpha=0.8, color=colors[i])
            
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
    
    ax.set_xlabel('Model')
    ax.set_ylabel('ROC AUC with 95% CI')
    ax.set_title('Performance of Clinic-Free Models', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace(' + ', '\n+ ') for m in models], rotation=45, ha='right')
    ax.legend(title='Validation Split')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 0.7)
    
    plt.tight_layout()
    plt.savefig('figures/figure_clinic_free_models.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def create_figure_reasonable_clinic_free_models():
    """Figure for Reasonable Clinic-Free Models (Including Blood Pressure)"""
    
    # Data from Table 4
    data = {
        'Model': ['PCE Reasonable Clinic-Free', 'PCE Reasonable Clinic-Free', 'PCE Reasonable Clinic-Free',
                  'PCE Reasonable Clinic-Free + Self-Report', 'PCE Reasonable Clinic-Free + Self-Report', 'PCE Reasonable Clinic-Free + Self-Report',
                  'PCE Reasonable Clinic-Free + Total MIMS', 'PCE Reasonable Clinic-Free + Total MIMS', 'PCE Reasonable Clinic-Free + Total MIMS',
                  'PCE Reasonable Clinic-Free + Acc-Driven MVPA', 'PCE Reasonable Clinic-Free + Acc-Driven MVPA', 'PCE Reasonable Clinic-Free + Acc-Driven MVPA',
                  'SCORE Reasonable Clinic-Free', 'SCORE Reasonable Clinic-Free', 'SCORE Reasonable Clinic-Free',
                  'SCORE Reasonable Clinic-Free + Self-Report', 'SCORE Reasonable Clinic-Free + Self-Report', 'SCORE Reasonable Clinic-Free + Self-Report',
                  'SCORE Reasonable Clinic-Free + Total MIMS', 'SCORE Reasonable Clinic-Free + Total MIMS', 'SCORE Reasonable Clinic-Free + Total MIMS',
                  'SCORE Reasonable Clinic-Free + Acc-Driven MVPA', 'SCORE Reasonable Clinic-Free + Acc-Driven MVPA', 'SCORE Reasonable Clinic-Free + Acc-Driven MVPA',
                  'LE8 Reasonable Clinic-Free', 'LE8 Reasonable Clinic-Free', 'LE8 Reasonable Clinic-Free',
                  'LE8 Reasonable Clinic-Free + Self-Report', 'LE8 Reasonable Clinic-Free + Self-Report', 'LE8 Reasonable Clinic-Free + Self-Report',
                  'LE8 Reasonable Clinic-Free + Total MIMS', 'LE8 Reasonable Clinic-Free + Total MIMS', 'LE8 Reasonable Clinic-Free + Total MIMS',
                  'LE8 Reasonable Clinic-Free + Acc-Driven MVPA', 'LE8 Reasonable Clinic-Free + Acc-Driven MVPA', 'LE8 Reasonable Clinic-Free + Acc-Driven MVPA'],
        
        'Split': ['A', 'B', 'C'] * 12,
        
        'AUC': [0.58, 0.57, 0.55,  # PCE Reasonable Clinic-Free baseline
                0.580, 0.571, 0.552,  # PCE Reasonable Clinic-Free + Self-Report (baseline + 0.0, 0.001, 0.002)
                0.613, 0.603, 0.583,  # PCE Reasonable Clinic-Free + Total MIMS (baseline + 0.033, 0.033, 0.033)
                0.596, 0.586, 0.566,  # PCE Reasonable Clinic-Free + Acc-Driven MVPA (baseline + 0.016, 0.016, 0.016)
                0.56, 0.55, 0.54,  # SCORE Reasonable Clinic-Free baseline
                0.562, 0.552, 0.542,  # SCORE Reasonable Clinic-Free + Self-Report (baseline + 0.002, 0.002, 0.002)
                0.594, 0.584, 0.574,  # SCORE Reasonable Clinic-Free + Total MIMS (baseline + 0.034, 0.034, 0.034)
                0.580, 0.582, 0.572,  # SCORE Reasonable Clinic-Free + Acc-Driven MVPA (baseline + 0.020, 0.032, 0.032)
                0.58, 0.57, 0.56,  # LE8 Reasonable Clinic-Free baseline
                0.576, 0.572, 0.562,  # LE8 Reasonable Clinic-Free + Self-Report (baseline - 0.004, 0.002, 0.002)
                0.606, 0.596, 0.586,  # LE8 Reasonable Clinic-Free + Total MIMS (baseline + 0.026, 0.026, 0.026)
                0.595, 0.585, 0.575],  # LE8 Reasonable Clinic-Free + Acc-Driven MVPA (baseline + 0.015, 0.015, 0.015)
        
        'CI_Lower': [0.54, 0.53, 0.51, 0.540, 0.531, 0.512, 0.573, 0.563, 0.543, 0.556, 0.546, 0.526,
                     0.52, 0.51, 0.50, 0.522, 0.512, 0.502, 0.554, 0.544, 0.534, 0.540, 0.542, 0.532,
                     0.54, 0.53, 0.52, 0.536, 0.532, 0.522, 0.566, 0.556, 0.546, 0.555, 0.545, 0.535],
        
        'CI_Upper': [0.62, 0.61, 0.59, 0.620, 0.611, 0.592, 0.653, 0.643, 0.623, 0.636, 0.626, 0.606,
                     0.60, 0.59, 0.58, 0.602, 0.592, 0.582, 0.634, 0.624, 0.614, 0.620, 0.622, 0.612,
                     0.62, 0.61, 0.60, 0.616, 0.612, 0.602, 0.646, 0.636, 0.626, 0.635, 0.625, 0.615]
    }
    
    df = pd.DataFrame(data)
    
    # Create the plot with explicit white background
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Get unique models
    models = df['Model'].unique()
    x = np.arange(len(models))
    width = 0.25
    
    # Color mapping: Split A = blue, Split B = orange, Split C = green
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, split in enumerate(['A', 'B', 'C']):
        split_data = df[df['Split'] == split]
        
        auc_values = []
        ci_lowers = []
        ci_uppers = []
        
        for model in models:
            model_data = split_data[split_data['Model'] == model]
            if not model_data.empty:
                auc_values.append(model_data['AUC'].iloc[0])
                ci_lowers.append(model_data['CI_Lower'].iloc[0])
                ci_uppers.append(model_data['CI_Upper'].iloc[0])
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
                         label=f'Split {split}', alpha=0.8, color=colors[i])
            
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
    
    ax.set_xlabel('Model')
    ax.set_ylabel('ROC AUC with 95% CI')
    ax.set_title('Performance of Reasonable Clinic-Free Models (Including Blood Pressure)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace(' + ', '\n+ ') for m in models], rotation=45, ha='right')
    ax.legend(title='Validation Split')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 0.7)
    
    plt.tight_layout()
    plt.savefig('figures/figure_reasonable_clinic_free_models.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def create_figure_sociodemographic_models():
    """Figure for Performance with Sociodemographic Variables Only"""
    
    # Data from Table 5
    data = {
        'Model': ['DEMO', 'DEMO', 'DEMO',
                  'DEMO + Self-Report', 'DEMO + Self-Report', 'DEMO + Self-Report',
                  'DEMO + Total MIMS', 'DEMO + Total MIMS', 'DEMO + Total MIMS',
                  'DEMO + Acc-Driven MVPA', 'DEMO + Acc-Driven MVPA', 'DEMO + Acc-Driven MVPA',
                  'SDOH', 'SDOH', 'SDOH',
                  'SDOH + Self-Report', 'SDOH + Self-Report', 'SDOH + Self-Report',
                  'SDOH + Total MIMS', 'SDOH + Total MIMS', 'SDOH + Total MIMS',
                  'SDOH + Acc-Driven MVPA', 'SDOH + Acc-Driven MVPA', 'SDOH + Acc-Driven MVPA'],
        
        'Split': ['A', 'B', 'C'] * 8,
        
        'AUC': [0.53, 0.50, 0.46,  # DEMO baseline
                0.544, 0.514, 0.474,  # DEMO + Self-Report (baseline + 0.014, 0.014, 0.014)
                0.621, 0.591, 0.551,  # DEMO + Total MIMS (baseline + 0.091, 0.091, 0.091)
                0.636, 0.606, 0.566,  # DEMO + Acc-Driven MVPA (baseline + 0.106, 0.106, 0.106)
                0.64, 0.63, 0.62,  # SDOH baseline
                0.638, 0.63, 0.62,  # SDOH + Self-Report (baseline - 0.002, 0.0, 0.0)
                0.652, 0.642, 0.632,  # SDOH + Total MIMS (baseline + 0.012, 0.012, 0.012)
                0.641, 0.631, 0.621],  # SDOH + Acc-Driven MVPA (baseline + 0.001, 0.001, 0.001)
        
        'CI_Lower': [0.49, 0.46, 0.42, 0.504, 0.474, 0.434, 0.581, 0.551, 0.511, 0.596, 0.566, 0.526,
                     0.58, 0.57, 0.56, 0.578, 0.57, 0.56, 0.592, 0.582, 0.572, 0.581, 0.571, 0.561],
        
        'CI_Upper': [0.57, 0.54, 0.50, 0.584, 0.554, 0.514, 0.661, 0.631, 0.591, 0.676, 0.646, 0.606,
                     0.68, 0.67, 0.66, 0.678, 0.67, 0.66, 0.692, 0.682, 0.672, 0.681, 0.671, 0.661]
    }
    
    df = pd.DataFrame(data)
    
    # Create the plot with explicit white background
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Get unique models
    models = df['Model'].unique()
    x = np.arange(len(models))
    width = 0.25
    
    # Color mapping: Split A = blue, Split B = orange, Split C = green
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, split in enumerate(['A', 'B', 'C']):
        split_data = df[df['Split'] == split]
        
        auc_values = []
        ci_lowers = []
        ci_uppers = []
        
        for model in models:
            model_data = split_data[split_data['Model'] == model]
            if not model_data.empty:
                auc_values.append(model_data['AUC'].iloc[0])
                ci_lowers.append(model_data['CI_Lower'].iloc[0])
                ci_uppers.append(model_data['CI_Upper'].iloc[0])
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
                         label=f'Split {split}', alpha=0.8, color=colors[i])
            
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
    
    ax.set_xlabel('Model')
    ax.set_ylabel('ROC AUC with 95% CI')
    ax.set_title('Performance with Sociodemographic Variables Only', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace(' + ', '\n+ ') for m in models], rotation=45, ha='right')
    ax.legend(title='Validation Split')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 0.7)
    
    plt.tight_layout()
    plt.savefig('figures/figure_sociodemographic_models.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def main():
    """Main function to create all four manuscript figures"""
    print("Creating Manuscript Figures...")
    
    print("1. Creating figure for Traditional Risk Models...")
    create_figure_traditional_risk_models()
    
    print("2. Creating figure for Clinic-Free Models...")
    create_figure_clinic_free_models()
    
    print("3. Creating figure for Reasonable Clinic-Free Models...")
    create_figure_reasonable_clinic_free_models()
    
    print("4. Creating figure for Sociodemographic Models...")
    create_figure_sociodemographic_models()
    
    print("\nAll four manuscript figures completed and saved to 'figures/' directory!")
    print("Color scheme: Split A = blue, Split B = orange, Split C = green")
    print("All figures show AUC with 95% confidence intervals instead of ΔAUC")

if __name__ == "__main__":
    main()
