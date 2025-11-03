#!/usr/bin/env python3
"""
Optimized Classification Model with Complete Lifestyle Features
Uses the enhanced 20-feature dataset for diabetes risk classification

Author: Generated for fairness project
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class OptimizedClassificationModel:
    """
    Optimized classification model for diabetes risk prediction
    """
    
    def __init__(self, dataset_path="/Users/aakashsuresh/fairness/blood_glucose_project/fixed_data/integrated_nhanes_2011_2014.csv"):
        self.dataset_path = dataset_path
        self.df = None
        self.X = None
        self.y_binary = None
        self.y_ada = None
        self.y_strict = None
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """
        Load data and create classification targets
        """
        print("=== Loading and Preparing Enhanced Dataset ===")
        
        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Participants: {len(self.df):,}")
        
        # Prepare features
        exclude_cols = ['seqn', 'glucose', 'hba1c']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        self.X = self.df[feature_cols]
        
        print(f"Features: {len(feature_cols)}")
        
        # Create classification targets
        self.create_classification_targets()
        
        return self.X, self.y_binary, self.y_ada, self.y_strict
    
    def create_classification_targets(self):
        """
        Create multiple classification targets for diabetes risk
        """
        print("\n=== Creating Classification Targets ===")
        
        glucose = self.df['glucose']
        hba1c = self.df['hba1c']
        
        # 1. Binary Risk (Prediabetes/Diabetes vs Normal)
        # Glucose ≥100 mg/dL OR HbA1c ≥5.7%
        self.y_binary = ((glucose >= 100) | (hba1c >= 5.7)).astype(int)
        
        # 2. Strict Diabetes (Binary)
        # Only definitive diabetes: Glucose ≥126 OR HbA1c ≥6.5
        self.y_strict = ((glucose >= 126) | (hba1c >= 6.5)).astype(int)
        
        # Print target distributions
        print("Target Distributions:")
        print(f"Binary Risk (≥100 mg/dL or ≥5.7%): {self.y_binary.sum():,} / {len(self.y_binary):,} ({100*self.y_binary.mean():.1f}%)")
        print(f"Strict Diabetes (≥126 mg/dL or ≥6.5%): {self.y_strict.sum():,} / {len(self.y_strict):,} ({100*self.y_strict.mean():.1f}%)")
    
    def prepare_features_for_classification(self):
        """
        Prepare features specifically for classification
        """
        print("\n=== Preparing Features for Classification ===")
        
        X_processed = self.X.copy()
        
        # Encode categorical variables
        categorical_cols = ['gender', 'race_ethnicity', 'education_level', 'activity_level']
        
        for col in categorical_cols:
            if col in X_processed.columns:
                if X_processed[col].dtype == 'object' or X_processed[col].nunique() < 10:
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        # Handle any remaining missing values
        X_processed = X_processed.fillna(X_processed.median())
        
        print(f"Processed features shape: {X_processed.shape}")
        print(f"Feature types: {X_processed.dtypes.value_counts().to_dict()}")
        
        return X_processed
    
    def train_optimized_models(self, target_name, y_target):
        """
        Train optimized models for a specific target
        """
        print(f"\n=== Training Optimized Models for {target_name} ===")
        
        # Prepare data
        X_processed = self.prepare_features_for_classification()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_target, test_size=0.2, random_state=42, stratify=y_target
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models with optimized hyperparameters
        models_config = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                }
            }
        }
        
        target_results = {}
        
        for model_name, config in models_config.items():
            print(f"\nOptimizing {model_name}...")
            
            # Grid search for hyperparameter optimization
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=3,  # Reduced for speed
                scoring='roc_auc' if len(np.unique(y_target)) == 2 else 'f1_macro',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(X_test_scaled)
            y_pred_proba = best_model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC (for binary classification)
            if len(np.unique(y_target)) == 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            
            print(f"  Best params: {grid_search.best_params_}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-score: {f1:.4f}")
            print(f"  ROC AUC: {roc_auc:.4f}")
            
            # Feature importance (for tree-based models)
            feature_importance = None
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = best_model.feature_importances_
            
            target_results[model_name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'feature_names': X_processed.columns.tolist(),
                'scaler': scaler,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc
                },
                'feature_importance': feature_importance
            }
        
        return target_results
    
    def evaluate_fairness_enhanced(self, target_name, target_results):
        """
        Evaluate fairness across demographic subgroups with enhanced features
        """
        print(f"\n=== Fairness Evaluation for {target_name} ===")
        
        fairness_results = {}
        
        # Define demographic subgroups
        subgroups = {
            'Gender': {
                'Male': self.df['gender'] == 1,
                'Female': self.df['gender'] == 2
            },
            'Age Groups': {
                'Young (18-40)': self.df['age'] < 40,
                'Middle (40-60)': (self.df['age'] >= 40) & (self.df['age'] < 60),
                'Older (60+)': self.df['age'] >= 60
            }
        }
        
        for model_name, model_data in target_results.items():
            print(f"\nFairness for {model_name}:")
            
            model_fairness = {}
            
            for group_name, group_dict in subgroups.items():
                group_fairness = {}
                
                for subgroup_name, mask in group_dict.items():
                    # Get test indices for this subgroup
                    test_indices = model_data['y_test'].index
                    subgroup_test_mask = mask.loc[test_indices]
                    
                    if subgroup_test_mask.sum() > 10:  # Minimum sample size
                        subgroup_y_test = model_data['y_test'][subgroup_test_mask]
                        subgroup_y_pred = model_data['y_pred'][subgroup_test_mask]
                        
                        subgroup_accuracy = accuracy_score(subgroup_y_test, subgroup_y_pred)
                        subgroup_f1 = f1_score(subgroup_y_test, subgroup_y_pred, average='weighted', zero_division=0)
                        
                        group_fairness[subgroup_name] = {
                            'n_samples': subgroup_test_mask.sum(),
                            'accuracy': subgroup_accuracy,
                            'f1_score': subgroup_f1
                        }
                        
                        print(f"  {group_name} - {subgroup_name}: Acc={subgroup_accuracy:.3f}, F1={subgroup_f1:.3f} (n={subgroup_test_mask.sum()})")
                
                model_fairness[group_name] = group_fairness
            
            fairness_results[model_name] = model_fairness
        
        return fairness_results
    
    def create_enhanced_classification_visualizations(self, target_name, target_results, fairness_results):
        """
        Create comprehensive visualizations for enhanced classification results
        """
        print(f"\n=== Creating Visualizations for {target_name} ===")
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Performance metrics
        models = list(target_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        performance_data = []
        for model_name in models:
            for metric in metrics:
                performance_data.append({
                    'Model': model_name,
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': target_results[model_name]['metrics'][metric]
                })
        
        perf_df = pd.DataFrame(performance_data)
        
        # Bar plot of performance
        pivot_perf = perf_df.pivot(index='Model', columns='Metric', values='Score')
        pivot_perf.plot(kind='bar', ax=axes[0, 0], rot=45)
        axes[0, 0].set_title(f'Model Performance Comparison - {target_name}')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Feature Importance (Random Forest)
        if 'Random Forest' in target_results:
            rf_data = target_results['Random Forest']
            if rf_data['feature_importance'] is not None:
                feature_names = rf_data['feature_names']
                importance_scores = rf_data['feature_importance']
                
                # Top 10 features
                top_indices = np.argsort(importance_scores)[-10:]
                top_features = [feature_names[i] for i in top_indices]
                top_scores = importance_scores[top_indices]
                
                axes[0, 1].barh(range(len(top_features)), top_scores, color='lightgreen')
                axes[0, 1].set_yticks(range(len(top_features)))
                axes[0, 1].set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in top_features])
                axes[0, 1].set_title(f'Top 10 Feature Importance - {target_name}')
                axes[0, 1].set_xlabel('Importance Score')
        
        # 3. ROC Curve (for binary classification)
        if len(np.unique(list(target_results.values())[0]['y_test'])) == 2:
            for model_name, model_data in target_results.items():
                fpr, tpr, _ = roc_curve(model_data['y_test'], model_data['y_pred_proba'][:, 1])
                auc_score = model_data['metrics']['roc_auc']
                axes[1, 0].plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.3f})')
            
            axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title(f'ROC Curves - {target_name}')
            axes[1, 0].legend()
        
        # 4. Fairness Comparison
        if fairness_results:
            fairness_data = []
            for model_name, model_fairness in fairness_results.items():
                for group_name, group_data in model_fairness.items():
                    for subgroup_name, metrics in group_data.items():
                        fairness_data.append({
                            'Model': model_name,
                            'Group': f"{group_name} - {subgroup_name}",
                            'Accuracy': metrics['accuracy'],
                            'F1_Score': metrics['f1_score']
                        })
            
            if fairness_data:
                fairness_df = pd.DataFrame(fairness_data)
                
                # Plot accuracy by subgroup
                pivot_fairness = fairness_df.pivot(index='Group', columns='Model', values='Accuracy')
                pivot_fairness.plot(kind='bar', ax=axes[1, 1], rot=45)
                axes[1, 1].set_title(f'Fairness: Accuracy by Subgroup - {target_name}')
                axes[1, 1].set_ylabel('Accuracy')
                axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'/Users/aakashsuresh/fairness/blood_glucose_project/enhanced_classification_{target_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved for {target_name}")
    
    def run_comprehensive_classification_analysis(self):
        """
        Run comprehensive classification analysis with enhanced features
        """
        print("Enhanced Classification Analysis with Complete Lifestyle Features")
        print("=" * 80)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Define classification tasks
        classification_tasks = {
            'Binary Risk': self.y_binary,
            'Strict Diabetes': self.y_strict
        }
        
        all_results = {}
        
        for target_name, y_target in classification_tasks.items():
            print(f"\n{'='*20} {target_name} {'='*20}")
            
            # Train models
            target_results = self.train_optimized_models(target_name, y_target)
            
            # Evaluate fairness
            fairness_results = self.evaluate_fairness_enhanced(target_name, target_results)
            
            # Create visualizations
            self.create_enhanced_classification_visualizations(target_name, target_results, fairness_results)
            
            all_results[target_name] = {
                'model_results': target_results,
                'fairness_results': fairness_results
            }
        
        # Summary
        self.create_classification_summary(all_results)
        
        return all_results
    
    def create_classification_summary(self, all_results):
        """
        Create summary of classification results
        """
        print("\n" + "="*80)
        print("ENHANCED CLASSIFICATION ANALYSIS SUMMARY")
        print("="*80)
        
        summary_data = []
        
        for target_name, results in all_results.items():
            model_results = results['model_results']
            
            print(f"\n{target_name}:")
            
            best_model = None
            best_score = 0
            
            for model_name, model_data in model_results.items():
                metrics = model_data['metrics']
                
                print(f"  {model_name}:")
                print(f"    Accuracy: {metrics['accuracy']:.4f}")
                print(f"    F1-Score: {metrics['f1_score']:.4f}")
                print(f"    ROC AUC: {metrics['roc_auc']:.4f}")
                
                if metrics['f1_score'] > best_score:
                    best_score = metrics['f1_score']
                    best_model = model_name
                
                summary_data.append({
                    'Target': target_name,
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'F1_Score': metrics['f1_score'],
                    'ROC_AUC': metrics['roc_auc']
                })
            
            print(f"  Best Model: {best_model} (F1={best_score:.4f})")
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('/Users/aakashsuresh/fairness/blood_glucose_project/enhanced_classification_summary.csv', index=False)
        print(f"\nSummary saved to: enhanced_classification_summary.csv")
        
        return summary_df

def main():
    """
    Main execution function
    """
    classifier = OptimizedClassificationModel()
    results = classifier.run_comprehensive_classification_analysis()
    return results

if __name__ == "__main__":
    results = main()
