#!/usr/bin/env python3
"""
Classification-Focused Approach for Diabetes Risk Prediction
Focus on clinically meaningful classification rather than precise regression

Author: Generated for fairness project
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class DiabetesRiskClassifier:
    """
    Classification-focused approach for diabetes risk prediction
    """
    
    def __init__(self, dataset_path="/Users/aakashsuresh/fairness/blood_glucose_project/finetuning/improved_dataset.csv"):
        self.dataset_path = dataset_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """
        Load data and create meaningful classification targets
        """
        print("=== Loading Data for Classification ===")
        
        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset shape: {self.df.shape}")
        
        # Prepare features (only valid ones)
        exclude_cols = ['seqn', 'glucose', 'hba1c']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        valid_features = []
        for col in feature_cols:
            if self.df[col].notna().sum() > 0 and self.df[col].var() > 0:
                valid_features.append(col)
        
        self.X = self.df[valid_features]
        print(f"Valid features: {valid_features}")
        
        return self.X
    
    def create_clinical_classification_targets(self):
        """
        Create clinically meaningful classification targets
        """
        print("\n=== Creating Clinical Classification Targets ===")
        
        # Multiple classification schemes
        classification_schemes = {}
        
        # 1. Standard ADA diabetes classification
        def ada_classification(row):
            glucose = row['glucose']
            hba1c = row['hba1c']
            
            if glucose >= 126 or hba1c >= 6.5:
                return 'Diabetes'
            elif glucose >= 100 or hba1c >= 5.7:
                return 'Pre-diabetes'
            else:
                return 'Normal'
        
        self.df['ada_class'] = self.df.apply(ada_classification, axis=1)
        classification_schemes['ADA Standard'] = self.df['ada_class']
        
        # 2. Binary high-risk classification
        def high_risk_classification(row):
            glucose = row['glucose']
            hba1c = row['hba1c']
            
            if glucose >= 100 or hba1c >= 5.7:
                return 'High Risk'
            else:
                return 'Low Risk'
        
        self.df['high_risk_class'] = self.df.apply(high_risk_classification, axis=1)
        classification_schemes['Binary Risk'] = self.df['high_risk_class']
        
        # 3. Strict diabetes classification
        def strict_diabetes_classification(row):
            glucose = row['glucose']
            hba1c = row['hba1c']
            
            if glucose >= 126 or hba1c >= 6.5:
                return 'Diabetes'
            else:
                return 'No Diabetes'
        
        self.df['strict_diabetes_class'] = self.df.apply(strict_diabetes_classification, axis=1)
        classification_schemes['Strict Diabetes'] = self.df['strict_diabetes_class']
        
        # Print distributions
        for scheme_name, classes in classification_schemes.items():
            print(f"\n{scheme_name} distribution:")
            class_counts = classes.value_counts()
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count} ({count/len(classes)*100:.1f}%)")
        
        return classification_schemes
    
    def train_classification_models(self, target_scheme='ADA Standard'):
        """
        Train classification models for specified target scheme
        """
        print(f"\n=== Training Classification Models for {target_scheme} ===")
        
        # Select target
        if target_scheme == 'ADA Standard':
            y = self.df['ada_class']
        elif target_scheme == 'Binary Risk':
            y = self.df['high_risk_class']
        elif target_scheme == 'Strict Diabetes':
            y = self.df['strict_diabetes_class']
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Calculate class weights for imbalanced data
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        print(f"Class weights: {class_weight_dict}")
        
        # Define models with class weights
        models = {
            'Logistic Regression': LogisticRegression(
                class_weight='balanced', random_state=42, max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, class_weight='balanced', random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, random_state=42, eval_metric='mlogloss'
            ),
            'SVM': SVC(
                class_weight='balanced', random_state=42, probability=True
            )
        }
        
        results = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                # ROC AUC (for binary classification)
                if len(classes) == 2 and y_pred_proba is not None:
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    roc_auc = None
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'y_pred': y_pred,
                    'y_test': y_test,
                    'y_pred_proba': y_pred_proba
                }
                
                print(f"  Accuracy: {accuracy:.3f}")
                print(f"  Precision: {precision:.3f}")
                print(f"  Recall: {recall:.3f}")
                print(f"  F1-Score: {f1:.3f}")
                if roc_auc:
                    print(f"  ROC AUC: {roc_auc:.3f}")
                
                # Detailed classification report
                print(f"\nDetailed Classification Report for {name}:")
                print(classification_report(y_test, y_pred))
                
            except Exception as e:
                print(f"  {name} failed: {e}")
        
        return results, X_train, X_test, y_train, y_test, scaler
    
    def hyperparameter_tuning_classification(self, results, X_train_scaled, y_train):
        """
        Hyperparameter tuning for best classification models
        """
        print("\n=== Hyperparameter Tuning for Classification ===")
        
        # Find best models to tune
        best_models = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)[:3]
        
        tuned_results = {}
        
        for model_name, model_data in best_models:
            print(f"\nTuning {model_name}...")
            
            if model_name == 'Random Forest':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': ['balanced']
                }
                base_model = RandomForestClassifier(random_state=42)
                
            elif model_name == 'XGBoost':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'eval_metric': ['mlogloss']
                }
                base_model = xgb.XGBClassifier(random_state=42)
                
            elif model_name == 'Logistic Regression':
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear'],
                    'class_weight': ['balanced']
                }
                base_model = LogisticRegression(random_state=42, max_iter=1000)
            
            else:
                continue
            
            try:
                # Grid search with stratified cross-validation
                grid_search = GridSearchCV(
                    base_model, param_grid,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='f1_weighted',
                    n_jobs=-1, verbose=0
                )
                
                grid_search.fit(X_train_scaled, y_train)
                
                tuned_results[f'{model_name}_Tuned'] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_
                }
                
                print(f"  Best parameters: {grid_search.best_params_}")
                print(f"  Best CV F1-score: {grid_search.best_score_:.3f}")
                
            except Exception as e:
                print(f"  Tuning failed for {model_name}: {e}")
        
        return tuned_results
    
    def evaluate_fairness_classification(self, results, X_test, y_test):
        """
        Evaluate fairness for classification models
        """
        print("\n=== Classification Fairness Evaluation ===")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
        best_model = results[best_model_name]['model']
        
        print(f"Analyzing fairness for best model: {best_model_name}")
        print(f"Best F1-score: {results[best_model_name]['f1']:.3f}")
        
        # Create test dataframe
        test_df = X_test.copy()
        test_df['y_true'] = y_test
        test_df['y_pred'] = results[best_model_name]['y_pred']
        
        fairness_results = {}
        
        # Gender-based fairness
        if 'gender' in test_df.columns:
            gender_fairness = {}
            for gender in test_df['gender'].unique():
                gender_data = test_df[test_df['gender'] == gender]
                if len(gender_data) > 10:
                    accuracy = accuracy_score(gender_data['y_true'], gender_data['y_pred'])
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        gender_data['y_true'], gender_data['y_pred'], average='weighted'
                    )
                    
                    gender_fairness[f'Gender_{gender}'] = {
                        'n': len(gender_data),
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
            
            fairness_results['gender'] = gender_fairness
        
        # Age-based fairness
        if 'age' in test_df.columns:
            test_df['age_group'] = pd.cut(test_df['age'], bins=[18, 40, 60, 100], labels=['<40', '40-60', '>60'])
            age_fairness = {}
            
            for age_group in test_df['age_group'].unique():
                if pd.notna(age_group):
                    age_data = test_df[test_df['age_group'] == age_group]
                    if len(age_data) > 10:
                        accuracy = accuracy_score(age_data['y_true'], age_data['y_pred'])
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            age_data['y_true'], age_data['y_pred'], average='weighted'
                        )
                        
                        age_fairness[str(age_group)] = {
                            'n': len(age_data),
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1
                        }
            
            fairness_results['age'] = age_fairness
        
        # Print fairness results
        for group_type, group_results in fairness_results.items():
            print(f"\nFairness by {group_type}:")
            for group_name, metrics in group_results.items():
                print(f"  {group_name} (n={metrics['n']}):")
                print(f"    Accuracy: {metrics['accuracy']:.3f}")
                print(f"    F1-Score: {metrics['f1']:.3f}")
        
        return fairness_results
    
    def create_classification_visualizations(self, results, scheme_name):
        """
        Create visualizations for classification results
        """
        print(f"\n=== Creating Visualizations for {scheme_name} ===")
        
        # Model performance comparison
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        f1_scores = [results[name]['f1'] for name in model_names]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        axes[0].barh(model_names, accuracies, color='skyblue', alpha=0.7)
        axes[0].set_xlabel('Accuracy')
        axes[0].set_title(f'Model Accuracy Comparison\n{scheme_name}')
        axes[0].set_xlim(0, 1)
        
        # F1-score comparison
        axes[1].barh(model_names, f1_scores, color='lightcoral', alpha=0.7)
        axes[1].set_xlabel('F1-Score')
        axes[1].set_title(f'Model F1-Score Comparison\n{scheme_name}')
        axes[1].set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'/Users/aakashsuresh/fairness/blood_glucose_project/finetuning/classification_performance_{scheme_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Confusion matrix for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
        y_test = results[best_model_name]['y_test']
        y_pred = results[best_model_name]['y_pred']
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.title(f'Confusion Matrix - {best_model_name}\n{scheme_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'/Users/aakashsuresh/fairness/blood_glucose_project/finetuning/confusion_matrix_{scheme_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_classification_analysis(self):
        """
        Run complete classification-focused analysis
        """
        print("Classification-Focused Diabetes Risk Prediction")
        print("=" * 60)
        
        # Load and prepare data
        self.load_and_prepare_data()
        classification_schemes = self.create_clinical_classification_targets()
        
        all_results = {}
        
        # Analyze each classification scheme
        for scheme_name in ['Binary Risk', 'ADA Standard', 'Strict Diabetes']:
            print(f"\n{'='*20} {scheme_name} {'='*20}")
            
            # Train models
            results, X_train, X_test, y_train, y_test, scaler = self.train_classification_models(scheme_name)
            
            # Hyperparameter tuning
            X_train_scaled = scaler.transform(X_train)
            tuned_results = self.hyperparameter_tuning_classification(results, X_train_scaled, y_train)
            
            # Fairness evaluation
            fairness_results = self.evaluate_fairness_classification(results, X_test, y_test)
            
            # Create visualizations
            self.create_classification_visualizations(results, scheme_name)
            
            all_results[scheme_name] = {
                'results': results,
                'tuned_results': tuned_results,
                'fairness': fairness_results
            }
        
        print("\n" + "=" * 60)
        print("CLASSIFICATION ANALYSIS COMPLETE")
        print("=" * 60)
        
        # Summary of best approaches
        print("\nSUMMARY OF BEST CLASSIFICATION APPROACHES:")
        for scheme_name, scheme_results in all_results.items():
            best_model = max(scheme_results['results'].keys(), 
                           key=lambda x: scheme_results['results'][x]['f1'])
            best_f1 = scheme_results['results'][best_model]['f1']
            print(f"{scheme_name}: {best_model} (F1={best_f1:.3f})")
        
        return all_results

def main():
    """
    Main execution function
    """
    classifier = DiabetesRiskClassifier()
    results = classifier.run_complete_classification_analysis()
    return results

if __name__ == "__main__":
    results = main()
