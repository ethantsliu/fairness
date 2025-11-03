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
        
        # 2. ADA Standard (3 categories)
        # Normal: Glucose <100 AND HbA1c <5.7
        # Prediabetes: Glucose 100-125 OR HbA1c 5.7-6.4
        # Diabetes: Glucose ≥126 OR HbA1c ≥6.5
        y_ada = np.zeros(len(self.df))
        
        # Prediabetes
        prediabetes_mask = (
            ((glucose >= 100) & (glucose < 126)) | 
            ((hba1c >= 5.7) & (hba1c < 6.5))
        ) & ~((glucose >= 126) | (hba1c >= 6.5))
        
        # Diabetes
        diabetes_mask = (glucose >= 126) | (hba1c >= 6.5)
        
        y_ada[prediabetes_mask] = 1
        y_ada[diabetes_mask] = 2
        
        self.y_ada = y_ada.astype(int)
        
        # 3. Strict Diabetes (Binary)
        # Only definitive diabetes: Glucose ≥126 OR HbA1c ≥6.5
        self.y_strict = ((glucose >= 126) | (hba1c >= 6.5)).astype(int)
        
        # Print target distributions
        print("Target Distributions:")
        print(f"Binary Risk (≥100 mg/dL or ≥5.7%): {self.y_binary.sum():,} / {len(self.y_binary):,} ({100*self.y_binary.mean():.1f}%)")
        
        ada_counts = pd.Series(self.y_ada).value_counts().sort_index()
        print(f"ADA Standard - Normal: {ada_counts.get(0, 0):,} ({100*ada_counts.get(0, 0)/len(self.y_ada):.1f}%)")
        print(f"ADA Standard - Prediabetes: {ada_counts.get(1, 0):,} ({100*ada_counts.get(1, 0)/len(self.y_ada):.1f}%)")
        print(f"ADA Standard - Diabetes: {ada_counts.get(2, 0):,} ({100*ada_counts.get(2, 0)/len(self.y_ada):.1f}%)")
        
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
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
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
                cv=5, 
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
        
        return target_results\n    \n    def evaluate_fairness_enhanced(self, target_name, target_results):\n        \"\"\"\n        Evaluate fairness across demographic subgroups with enhanced features\n        \"\"\"\n        print(f\"\\n=== Fairness Evaluation for {target_name} ===\")\n        \n        fairness_results = {}\n        \n        # Define demographic subgroups\n        subgroups = {\n            'Gender': {\n                'Male': self.df['gender'] == 1,\n                'Female': self.df['gender'] == 2\n            },\n            'Age Groups': {\n                'Young (18-40)': self.df['age'] < 40,\n                'Middle (40-60)': (self.df['age'] >= 40) & (self.df['age'] < 60),\n                'Older (60+)': self.df['age'] >= 60\n            }\n        }\n        \n        for model_name, model_data in target_results.items():\n            print(f\"\\nFairness for {model_name}:\")\n            \n            model_fairness = {}\n            \n            for group_name, group_dict in subgroups.items():\n                group_fairness = {}\n                \n                for subgroup_name, mask in group_dict.items():\n                    # Get test indices for this subgroup\n                    test_indices = model_data['y_test'].index\n                    subgroup_test_mask = mask.loc[test_indices]\n                    \n                    if subgroup_test_mask.sum() > 10:  # Minimum sample size\n                        subgroup_y_test = model_data['y_test'][subgroup_test_mask]\n                        subgroup_y_pred = model_data['y_pred'][subgroup_test_mask]\n                        \n                        subgroup_accuracy = accuracy_score(subgroup_y_test, subgroup_y_pred)\n                        subgroup_f1 = f1_score(subgroup_y_test, subgroup_y_pred, average='weighted', zero_division=0)\n                        \n                        group_fairness[subgroup_name] = {\n                            'n_samples': subgroup_test_mask.sum(),\n                            'accuracy': subgroup_accuracy,\n                            'f1_score': subgroup_f1\n                        }\n                        \n                        print(f\"  {group_name} - {subgroup_name}: Acc={subgroup_accuracy:.3f}, F1={subgroup_f1:.3f} (n={subgroup_test_mask.sum()})\")\n                \n                model_fairness[group_name] = group_fairness\n            \n            fairness_results[model_name] = model_fairness\n        \n        return fairness_results\n    \n    def create_enhanced_classification_visualizations(self, target_name, target_results, fairness_results):\n        \"\"\"\n        Create comprehensive visualizations for enhanced classification results\n        \"\"\"\n        print(f\"\\n=== Creating Visualizations for {target_name} ===\")\n        \n        # 1. Model Performance Comparison\n        fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n        \n        # Performance metrics\n        models = list(target_results.keys())\n        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']\n        \n        performance_data = []\n        for model_name in models:\n            for metric in metrics:\n                performance_data.append({\n                    'Model': model_name,\n                    'Metric': metric.replace('_', ' ').title(),\n                    'Score': target_results[model_name]['metrics'][metric]\n                })\n        \n        perf_df = pd.DataFrame(performance_data)\n        \n        # Bar plot of performance\n        pivot_perf = perf_df.pivot(index='Model', columns='Metric', values='Score')\n        pivot_perf.plot(kind='bar', ax=axes[0, 0], rot=45)\n        axes[0, 0].set_title(f'Model Performance Comparison - {target_name}')\n        axes[0, 0].set_ylabel('Score')\n        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n        \n        # 2. Feature Importance (Random Forest)\n        if 'Random Forest' in target_results:\n            rf_data = target_results['Random Forest']\n            if rf_data['feature_importance'] is not None:\n                feature_names = rf_data['feature_names']\n                importance_scores = rf_data['feature_importance']\n                \n                # Top 10 features\n                top_indices = np.argsort(importance_scores)[-10:]\n                top_features = [feature_names[i] for i in top_indices]\n                top_scores = importance_scores[top_indices]\n                \n                axes[0, 1].barh(range(len(top_features)), top_scores, color='lightgreen')\n                axes[0, 1].set_yticks(range(len(top_features)))\n                axes[0, 1].set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in top_features])\n                axes[0, 1].set_title(f'Top 10 Feature Importance - {target_name}')\n                axes[0, 1].set_xlabel('Importance Score')\n        \n        # 3. ROC Curve (for binary classification)\n        if len(np.unique(list(target_results.values())[0]['y_test'])) == 2:\n            for model_name, model_data in target_results.items():\n                fpr, tpr, _ = roc_curve(model_data['y_test'], model_data['y_pred_proba'][:, 1])\n                auc_score = model_data['metrics']['roc_auc']\n                axes[1, 0].plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.3f})')\n            \n            axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)\n            axes[1, 0].set_xlabel('False Positive Rate')\n            axes[1, 0].set_ylabel('True Positive Rate')\n            axes[1, 0].set_title(f'ROC Curves - {target_name}')\n            axes[1, 0].legend()\n        \n        # 4. Fairness Comparison\n        if fairness_results:\n            fairness_data = []\n            for model_name, model_fairness in fairness_results.items():\n                for group_name, group_data in model_fairness.items():\n                    for subgroup_name, metrics in group_data.items():\n                        fairness_data.append({\n                            'Model': model_name,\n                            'Group': f\"{group_name} - {subgroup_name}\",\n                            'Accuracy': metrics['accuracy'],\n                            'F1_Score': metrics['f1_score']\n                        })\n            \n            if fairness_data:\n                fairness_df = pd.DataFrame(fairness_data)\n                \n                # Plot accuracy by subgroup\n                pivot_fairness = fairness_df.pivot(index='Group', columns='Model', values='Accuracy')\n                pivot_fairness.plot(kind='bar', ax=axes[1, 1], rot=45)\n                axes[1, 1].set_title(f'Fairness: Accuracy by Subgroup - {target_name}')\n                axes[1, 1].set_ylabel('Accuracy')\n                axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n        \n        plt.tight_layout()\n        plt.savefig(f'/Users/aakashsuresh/fairness/blood_glucose_project/enhanced_classification_{target_name.lower().replace(\" \", \"_\")}.png', \n                   dpi=300, bbox_inches='tight')\n        plt.show()\n        \n        print(f\"Visualization saved for {target_name}\")\n    \n    def run_comprehensive_classification_analysis(self):\n        \"\"\"\n        Run comprehensive classification analysis with enhanced features\n        \"\"\"\n        print(\"Enhanced Classification Analysis with Complete Lifestyle Features\")\n        print(\"=\" * 80)\n        \n        # Load and prepare data\n        self.load_and_prepare_data()\n        \n        # Define classification tasks\n        classification_tasks = {\n            'Binary Risk': self.y_binary,\n            'Strict Diabetes': self.y_strict\n        }\n        \n        all_results = {}\n        \n        for target_name, y_target in classification_tasks.items():\n            print(f\"\\n{'='*20} {target_name} {'='*20}\")\n            \n            # Train models\n            target_results = self.train_optimized_models(target_name, y_target)\n            \n            # Evaluate fairness\n            fairness_results = self.evaluate_fairness_enhanced(target_name, target_results)\n            \n            # Create visualizations\n            self.create_enhanced_classification_visualizations(target_name, target_results, fairness_results)\n            \n            all_results[target_name] = {\n                'model_results': target_results,\n                'fairness_results': fairness_results\n            }\n        \n        # Summary\n        self.create_classification_summary(all_results)\n        \n        return all_results\n    \n    def create_classification_summary(self, all_results):\n        \"\"\"\n        Create summary of classification results\n        \"\"\"\n        print(\"\\n\" + \"=\"*80)\n        print(\"ENHANCED CLASSIFICATION ANALYSIS SUMMARY\")\n        print(\"=\"*80)\n        \n        summary_data = []\n        \n        for target_name, results in all_results.items():\n            model_results = results['model_results']\n            \n            print(f\"\\n{target_name}:\")\n            \n            best_model = None\n            best_score = 0\n            \n            for model_name, model_data in model_results.items():\n                metrics = model_data['metrics']\n                \n                print(f\"  {model_name}:\")\n                print(f\"    Accuracy: {metrics['accuracy']:.4f}\")\n                print(f\"    F1-Score: {metrics['f1_score']:.4f}\")\n                print(f\"    ROC AUC: {metrics['roc_auc']:.4f}\")\n                \n                if metrics['f1_score'] > best_score:\n                    best_score = metrics['f1_score']\n                    best_model = model_name\n                \n                summary_data.append({\n                    'Target': target_name,\n                    'Model': model_name,\n                    'Accuracy': metrics['accuracy'],\n                    'F1_Score': metrics['f1_score'],\n                    'ROC_AUC': metrics['roc_auc']\n                })\n            \n            print(f\"  Best Model: {best_model} (F1={best_score:.4f})\")\n        \n        # Save summary\n        summary_df = pd.DataFrame(summary_data)\n        summary_df.to_csv('/Users/aakashsuresh/fairness/blood_glucose_project/enhanced_classification_summary.csv', index=False)\n        print(f\"\\nSummary saved to: enhanced_classification_summary.csv\")\n        \n        return summary_df\n\ndef main():\n    \"\"\"\n    Main execution function\n    \"\"\"\n    classifier = OptimizedClassificationModel()\n    results = classifier.run_comprehensive_classification_analysis()\n    return results\n\nif __name__ == \"__main__\":\n    results = main()
