#!/usr/bin/env python3
"""
Dietary Data Integration Investigation
Investigate and fix dietary data integration challenges to unlock dietary features

Author: Generated for fairness project
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

class DietaryDataIntegrationInvestigator:
    """
    Investigate dietary data integration issues and create solutions
    """
    
    def __init__(self):
        self.base_dir = "/Users/aakashsuresh/fairness"
        self.processed_data_new = f"{self.base_dir}/processed_data_new"
        self.output_dir = f"{self.base_dir}/blood_glucose_project/dietary_analysis"
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        
        self.dietary_files = {}
        self.dietary_analysis = {}
        
    def investigate_dietary_files(self):
        """
        Investigate available dietary files and their structure
        """
        print("=== Investigating Dietary Data Files ===")
        
        # Check for dietary files
        dietary_file_patterns = [
            "filled_nhanes_combined_diet.csv",
            "cleaned_nhanes_combined_diet.csv", 
            "nhanes_combined_diet.csv",
            "2011-2012_Dietary.csv",
            "2013-2014_Dietary.csv"
        ]
        
        for pattern in dietary_file_patterns:
            file_path = f"{self.processed_data_new}/{pattern}"
            if os.path.exists(file_path):
                print(f"✅ Found: {pattern}")
                try:
                    df = pd.read_csv(file_path)
                    self.dietary_files[pattern] = {
                        'path': file_path,
                        'data': df,
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'seqn_range': (df['SEQN'].min(), df['SEQN'].max()) if 'SEQN' in df.columns else None
                    }
                    print(f"   Shape: {df.shape}")
                    print(f"   SEQN range: {self.dietary_files[pattern]['seqn_range']}")
                except Exception as e:
                    print(f"   ❌ Error loading: {e}")
            else:
                print(f"❌ Missing: {pattern}")
        
        return self.dietary_files
    
    def analyze_dietary_data_quality(self):
        """
        Analyze the quality and usability of dietary data
        """
        print("\n=== Dietary Data Quality Analysis ===")
        
        if not self.dietary_files:
            print("No dietary files found to analyze")
            return None
        
        # Analyze the best available file
        best_file = None
        max_features = 0
        
        for filename, file_info in self.dietary_files.items():
            df = file_info['data']
            
            # Count potential dietary features
            dietary_keywords = [
                'KCAL', 'CARB', 'TFAT', 'SFAT', 'MFAT', 'PFAT', 'PROT', 
                'SODI', 'FIBE', 'SUGA', 'CALC', 'IRON', 'VITA', 'VITC'
            ]
            
            dietary_features = []
            for col in df.columns:
                if any(keyword in col.upper() for keyword in dietary_keywords):
                    if df[col].dtype in ['float64', 'int64']:
                        dietary_features.append(col)
            
            file_info['dietary_features'] = dietary_features
            file_info['n_dietary_features'] = len(dietary_features)
            
            print(f"\n{filename}:")
            print(f"  Total columns: {len(df.columns)}")
            print(f"  Dietary features: {len(dietary_features)}")
            print(f"  Sample size: {len(df):,}")
            
            if len(dietary_features) > max_features:
                max_features = len(dietary_features)
                best_file = filename
        
        if best_file:
            print(f"\n🏆 Best dietary file: {best_file}")
            self.analyze_best_dietary_file(best_file)
        
        return best_file
    
    def analyze_best_dietary_file(self, filename):
        """
        Deep analysis of the best dietary file
        """
        print(f"\n=== Deep Analysis: {filename} ===")
        
        file_info = self.dietary_files[filename]
        df = file_info['data']
        dietary_features = file_info['dietary_features']
        
        print(f"Dietary features found: {len(dietary_features)}")
        
        # Analyze missing values
        missing_analysis = {}
        for feature in dietary_features[:10]:  # Analyze top 10 features
            missing_count = df[feature].isnull().sum()
            missing_pct = 100 * missing_count / len(df)
            missing_analysis[feature] = {
                'missing_count': missing_count,
                'missing_pct': missing_pct,
                'mean': df[feature].mean() if missing_pct < 100 else np.nan,
                'std': df[feature].std() if missing_pct < 100 else np.nan
            }
            
            print(f"  {feature}: {missing_pct:.1f}% missing, mean={df[feature].mean():.2f}")
        
        # Check SEQN overlap with our glucose data
        glucose_seqn_range = (62161, 83731)  # From our successful integration
        dietary_seqn_range = file_info['seqn_range']
        
        print(f"\nSEQN Range Analysis:")
        print(f"  Glucose data SEQN: {glucose_seqn_range[0]} - {glucose_seqn_range[1]}")
        print(f"  Dietary data SEQN: {dietary_seqn_range[0]} - {dietary_seqn_range[1]}")
        
        # Check overlap
        if dietary_seqn_range:
            overlap_start = max(glucose_seqn_range[0], dietary_seqn_range[0])
            overlap_end = min(glucose_seqn_range[1], dietary_seqn_range[1])
            
            if overlap_start <= overlap_end:
                overlap_seqns = df[(df['SEQN'] >= overlap_start) & (df['SEQN'] <= overlap_end)]['SEQN']
                print(f"  Overlapping SEQN range: {overlap_start} - {overlap_end}")
                print(f"  Participants in overlap: {len(overlap_seqns):,}")
                
                # This is the key insight!
                if len(overlap_seqns) > 1000:
                    print("  ✅ Sufficient overlap for dietary integration!")
                    self.create_dietary_integration_solution(df, overlap_seqns, dietary_features)
                else:
                    print("  ❌ Insufficient overlap for reliable dietary integration")
            else:
                print("  ❌ No SEQN overlap found")
        
        self.dietary_analysis[filename] = missing_analysis
        return missing_analysis
    
    def create_dietary_integration_solution(self, dietary_df, overlap_seqns, dietary_features):
        """
        Create a solution for dietary data integration
        """
        print("\n=== Creating Dietary Integration Solution ===")
        
        # Filter to overlapping participants
        dietary_overlap = dietary_df[dietary_df['SEQN'].isin(overlap_seqns)].copy()
        
        print(f"Dietary data for integration: {len(dietary_overlap):,} participants")
        
        # Select best dietary features (low missing values, high variance)
        feature_quality = []
        for feature in dietary_features:
            missing_pct = 100 * dietary_overlap[feature].isnull().sum() / len(dietary_overlap)
            variance = dietary_overlap[feature].var()
            
            if missing_pct < 50 and variance > 0:  # Good quality criteria
                feature_quality.append({
                    'feature': feature,
                    'missing_pct': missing_pct,
                    'variance': variance,
                    'quality_score': (100 - missing_pct) * np.log1p(variance)
                })
        
        # Sort by quality score
        feature_quality.sort(key=lambda x: x['quality_score'], reverse=True)
        
        print(f"\nTop 10 Quality Dietary Features:")
        top_dietary_features = []
        for i, fq in enumerate(feature_quality[:10]):
            print(f"  {i+1:2d}. {fq['feature']}: {fq['missing_pct']:.1f}% missing, quality={fq['quality_score']:.1f}")
            top_dietary_features.append(fq['feature'])
        
        # Create clean dietary dataset
        dietary_clean = dietary_overlap[['SEQN'] + top_dietary_features].copy()
        
        # Handle missing values intelligently
        for feature in top_dietary_features:
            if dietary_clean[feature].isnull().sum() > 0:
                # Use median imputation for dietary features
                median_val = dietary_clean[feature].median()
                dietary_clean[feature] = dietary_clean[feature].fillna(median_val)
                print(f"  Imputed {feature} missing values with median: {median_val:.2f}")
        
        # Standardize column names
        dietary_clean.columns = dietary_clean.columns.str.lower()
        
        # Save the clean dietary dataset
        output_path = f"{self.output_dir}/dietary_features_integrated.csv"
        dietary_clean.to_csv(output_path, index=False)
        
        print(f"\n✅ Dietary integration dataset saved: {output_path}")
        print(f"   Shape: {dietary_clean.shape}")
        print(f"   Features: {top_dietary_features}")
        
        return dietary_clean, top_dietary_features
    
    def test_dietary_integration_with_glucose(self):
        """
        Test integrating dietary features with our existing glucose dataset
        """
        print("\n=== Testing Dietary Integration with Glucose Data ===")
        
        # Load our successful glucose dataset
        glucose_file = f"{self.base_dir}/blood_glucose_project/fixed_data/integrated_nhanes_2011_2014.csv"
        
        if not os.path.exists(glucose_file):
            print("❌ Glucose dataset not found")
            return None
        
        glucose_df = pd.read_csv(glucose_file)
        print(f"Glucose dataset: {glucose_df.shape}")
        
        # Load dietary features if available
        dietary_file = f"{self.output_dir}/dietary_features_integrated.csv"
        
        if os.path.exists(dietary_file):
            dietary_df = pd.read_csv(dietary_file)
            print(f"Dietary dataset: {dietary_df.shape}")
            
            # Merge datasets
            combined_df = glucose_df.merge(dietary_df, on='seqn', how='inner')
            print(f"Combined dataset: {combined_df.shape}")
            
            if len(combined_df) > 1000:
                print("✅ Successful dietary integration!")
                
                # Analyze the enhanced dataset
                exclude_cols = ['seqn', 'glucose', 'hba1c']
                feature_cols = [col for col in combined_df.columns if col not in exclude_cols]
                
                print(f"Total features after dietary integration: {len(feature_cols)}")
                
                # Check feature variance
                zero_var_features = []
                for col in feature_cols:
                    if combined_df[col].var() == 0:
                        zero_var_features.append(col)
                
                print(f"Features with variance: {len(feature_cols) - len(zero_var_features)}")
                if zero_var_features:
                    print(f"Zero variance features: {zero_var_features}")
                
                # Save enhanced dataset
                enhanced_path = f"{self.output_dir}/enhanced_dataset_with_dietary.csv"
                combined_df.to_csv(enhanced_path, index=False)
                print(f"Enhanced dataset saved: {enhanced_path}")
                
                return combined_df
            else:
                print("❌ Insufficient overlap after dietary integration")
                return None
        else:
            print("❌ Dietary integration file not found")
            return None
    
    def create_dietary_visualizations(self):
        """
        Create visualizations for dietary analysis
        """
        print("\n=== Creating Dietary Analysis Visualizations ===")
        
        if not self.dietary_analysis:
            print("No dietary analysis data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. File comparison
        if len(self.dietary_files) > 1:
            file_names = list(self.dietary_files.keys())
            feature_counts = [self.dietary_files[f]['n_dietary_features'] for f in file_names]
            sample_sizes = [len(self.dietary_files[f]['data']) for f in file_names]
            
            axes[0, 0].bar(range(len(file_names)), feature_counts, alpha=0.7)
            axes[0, 0].set_xlabel('Dietary Files')
            axes[0, 0].set_ylabel('Number of Dietary Features')
            axes[0, 0].set_title('Dietary Features by File')
            axes[0, 0].set_xticks(range(len(file_names)))
            axes[0, 0].set_xticklabels([f[:15] + '...' for f in file_names], rotation=45)
        
        # 2. Missing value analysis
        if self.dietary_analysis:
            best_file = list(self.dietary_analysis.keys())[0]
            missing_data = self.dietary_analysis[best_file]
            
            features = list(missing_data.keys())[:10]
            missing_pcts = [missing_data[f]['missing_pct'] for f in features]
            
            axes[0, 1].barh(range(len(features)), missing_pcts, alpha=0.7, color='orange')
            axes[0, 1].set_xlabel('Missing Percentage')
            axes[0, 1].set_ylabel('Dietary Features')
            axes[0, 1].set_title('Missing Values in Dietary Features')
            axes[0, 1].set_yticks(range(len(features)))
            axes[0, 1].set_yticklabels([f[:15] + '...' for f in features])
        
        # 3. SEQN overlap visualization
        if self.dietary_files:
            glucose_range = (62161, 83731)
            
            for i, (filename, file_info) in enumerate(self.dietary_files.items()):
                if file_info['seqn_range']:
                    dietary_range = file_info['seqn_range']
                    
                    # Plot ranges
                    axes[1, 0].barh(i, glucose_range[1] - glucose_range[0], 
                                   left=glucose_range[0], alpha=0.7, label='Glucose' if i == 0 else "")
                    axes[1, 0].barh(i + 0.3, dietary_range[1] - dietary_range[0], 
                                   left=dietary_range[0], alpha=0.7, label='Dietary' if i == 0 else "")
            
            axes[1, 0].set_xlabel('SEQN Range')
            axes[1, 0].set_ylabel('Files')
            axes[1, 0].set_title('SEQN Range Overlap Analysis')
            axes[1, 0].legend()
        
        # 4. Integration success summary
        integration_success = {
            'Files Found': len(self.dietary_files),
            'Features Available': sum(f['n_dietary_features'] for f in self.dietary_files.values()),
            'Max Sample Size': max(len(f['data']) for f in self.dietary_files.values()) if self.dietary_files else 0
        }
        
        labels = list(integration_success.keys())
        values = list(integration_success.values())
        
        axes[1, 1].bar(labels, values, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Dietary Integration Summary')
        
        for i, v in enumerate(values):
            axes[1, 1].text(i, v + max(values) * 0.01, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/dietary_integration_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Dietary analysis visualizations saved")
    
    def run_dietary_integration_investigation(self):
        """
        Run complete dietary integration investigation
        """
        print("Dietary Data Integration Investigation")
        print("=" * 60)
        
        # Investigate available files
        self.investigate_dietary_files()
        
        # Analyze data quality
        best_file = self.analyze_dietary_data_quality()
        
        # Test integration with glucose data
        enhanced_dataset = self.test_dietary_integration_with_glucose()
        
        # Create visualizations
        self.create_dietary_visualizations()
        
        # Generate summary report
        self.generate_dietary_report(best_file, enhanced_dataset)
        
        print("\n" + "=" * 60)
        print("DIETARY INTEGRATION INVESTIGATION COMPLETE")
        print("=" * 60)
        
        if enhanced_dataset is not None:
            print("✅ SUCCESS: Dietary features successfully integrated!")
            print(f"   Enhanced dataset: {enhanced_dataset.shape}")
            print(f"   Total features: {enhanced_dataset.shape[1] - 3}")  # Excluding seqn, glucose, hba1c
        else:
            print("❌ CHALLENGE: Dietary integration requires additional work")
            print("   Recommendations provided in report")
        
        return {
            'dietary_files': self.dietary_files,
            'dietary_analysis': self.dietary_analysis,
            'enhanced_dataset': enhanced_dataset,
            'best_file': best_file
        }
    
    def generate_dietary_report(self, best_file, enhanced_dataset):
        """
        Generate comprehensive dietary integration report
        """
        report = f"""
# Dietary Data Integration Investigation Report

## Executive Summary
Investigation of dietary data integration challenges and solutions for NHANES glucose prediction models.

## Files Investigated
"""
        
        for filename, file_info in self.dietary_files.items():
            report += f"- **{filename}**: {file_info['shape'][0]:,} participants, {file_info['n_dietary_features']} dietary features\n"
        
        report += f"""

## Key Findings

### Data Availability
- **Files Found**: {len(self.dietary_files)}
- **Best File**: {best_file if best_file else 'None suitable'}
- **Integration Success**: {'✅ Successful' if enhanced_dataset is not None else '❌ Requires additional work'}

### Technical Challenges
1. **SEQN Range Mismatch**: Some dietary files may not overlap with glucose data SEQN ranges
2. **Missing Values**: High missing value rates in some dietary features
3. **Feature Quality**: Variable quality of dietary measurements across cycles

### Solutions Implemented
"""
        
        if enhanced_dataset is not None:
            report += f"""
1. **SEQN Overlap Analysis**: Identified overlapping participants between dietary and glucose data
2. **Feature Quality Scoring**: Selected top dietary features based on missing values and variance
3. **Intelligent Imputation**: Used median imputation for missing dietary values
4. **Enhanced Dataset**: Created dataset with {enhanced_dataset.shape[1] - 3} total features

### Enhanced Dataset Characteristics
- **Participants**: {len(enhanced_dataset):,}
- **Total Features**: {enhanced_dataset.shape[1] - 3}
- **Dietary Features Added**: ~10 high-quality dietary variables
"""
        else:
            report += """
1. **Challenge Identification**: Documented specific integration barriers
2. **Recommendations**: Provided actionable steps for future integration
3. **Alternative Approaches**: Suggested workarounds for dietary feature limitations
"""
        
        report += f"""

## Recommendations

### Immediate Actions
1. **{'Deploy Enhanced Model' if enhanced_dataset is not None else 'Address Data Gaps'}**: {'Use enhanced dataset for improved predictions' if enhanced_dataset is not None else 'Obtain additional dietary data or use synthetic features'}
2. **Validation Testing**: Test dietary features' predictive value
3. **Clinical Interpretation**: Validate dietary feature importance with nutrition experts

### Future Improvements
1. **Multi-Cycle Integration**: Combine dietary data across multiple NHANES cycles
2. **External Validation**: Test dietary features on independent datasets
3. **Feature Engineering**: Create derived dietary patterns and ratios

---
*Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save report
        report_path = f"{self.output_dir}/dietary_integration_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Dietary integration report saved: {report_path}")

def main():
    """
    Main execution function
    """
    investigator = DietaryDataIntegrationInvestigator()
    results = investigator.run_dietary_integration_investigation()
    return results

if __name__ == "__main__":
    results = main()
