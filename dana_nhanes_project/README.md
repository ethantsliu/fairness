# NHANES Physical Activity Analysis - Dana Project

This project contains comprehensive visualizations and analysis of NHANES physical activity measures based on the research findings shared by the team.

## Project Overview

The analysis compares three different physical activity measurement approaches:
1. **Accelerometer daily averages** - Device-based total activity measures
2. **Accelerometer-derived activity intensity** - Device-based moderate+ intensity activity
3. **Self-reported physical activity** - Questionnaire-based activity measures

## Key Findings

### Performance Stability
- **Accelerometer data** demonstrates the most stable predictive performance across all data splits
- **Self-reported data** shows high sensitivity but suffers from low specificity and balanced accuracy
- **Accelerometer-derived activity intensity** maintains ~0.7 sensitivity while keeping specificity around 0.4

### Model Improvement Effects
- **Self-reported Physical Activity**: Little to no improvement, sometimes negative effects
- **Accelerometer daily averages**: Modest improvements in clinical models, clearer gains in clinic-free models
- **Accelerometer-derived activity intensity**: Most stable and substantial improvements across all cases

## Data Structure

The analysis includes three data splits (SplitA, SplitB, SplitC) with the following metrics:
- AUC scores with confidence intervals
- Sensitivity, Specificity, and Balanced Accuracy
- Overall performance measures

## Files

- `visualize_results.py` - Main visualization script
- `requirements.txt` - Python dependencies
- `figures/` - Directory for generated visualizations
- `README.md` - This file

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the visualization script:
```bash
python visualize_results.py
```

3. Generated visualizations will be saved in the `figures/` directory:
   - `auc_comparison.png` - AUC comparison across splits and metrics
   - `performance_metrics.png` - Sensitivity, specificity, and balanced accuracy comparison
   - `auc_heatmap.png` - AUC performance heatmap
   - `stability_analysis.png` - Performance stability analysis
   - `comprehensive_dashboard.png` - All-in-one dashboard

## Visualizations Included

1. **AUC Comparison Plot**: Grouped bar chart showing AUC scores across all splits and metrics
2. **Performance Metrics**: Side-by-side comparison of sensitivity, specificity, and balanced accuracy
3. **AUC Heatmap**: Color-coded matrix showing performance across splits and metrics
4. **Stability Analysis**: Error bar plot showing mean ± SD performance across splits
5. **Comprehensive Dashboard**: Multi-panel summary with radar charts and key findings

## Sample Characteristics

- **Age**: CVD cases 65.81±13.02 years, controls 65.57±13.08 years (p=0.733)
- **Gender**: Cases 45.8% female, controls 45.2% female (p=0.864)
- **Race**: Cases 54.8% Non-Hispanic White, controls 57.0% (p=0.443)

## Physical Activity Measures

### Self-reported Activity
- Calculated total weekly minutes of moderate+ activity
- Converted to average daily values
- Includes work, transportation, and leisure domains

### Accelerometer-derived Activity
- Minute-level data classification (≥30 counts as moderate intensity)
- Averaged across valid wear days (≥4 days, ≥600 minutes/day)
- Represents actual device-measured activity

## Clinic-Free Model Definitions

- **Strict clinic-free**: Blood pressure excluded
- **Reasonable clinic-free**: Blood pressure included (home measurement capability)
- **Nicotine exposure**: Self-reported included, cotinine (lab-based) excluded

## Contact

For questions about this analysis, please contact the research team.
