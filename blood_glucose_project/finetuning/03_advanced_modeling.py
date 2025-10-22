#!/usr/bin/env python3
"""
Advanced Modeling Approaches for Blood Glucose Prediction
Implements ensemble methods, deep learning, and classification approaches

Author: Generated for fairness project
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class AdvancedGlucoseModeling:
    """
    Advanced modeling approaches for glucose prediction
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
        
    def load_improved_dataset(self):
        """
        Load the improved dataset created by feature engineering
        """
        print("=== Loading Improved Dataset ===")
        
        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Features: {list(self.df.columns)}")
        
        # Prepare features and targets
        exclude_cols = ['seqn', 'glucose', 'hba1c']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        # Remove features with no variance or all NaN
        valid_features = []
        for col in feature_cols:
            if self.df[col].notna().sum() > 0 and self.df[col].var() > 0:
                valid_features.append(col)
        
        self.X = self.df[valid_features]
        self.y = self.df[['glucose', 'hba1c']]
        
        print(f"Valid features: {len(valid_features)}")
        print(f"Features used: {valid_features}")
        
        return self.X, self.y
    
    def prepare_data_for_modeling(self):
        """
        Prepare data for advanced modeling
        """
        print("\n=== Preparing Data for Advanced Modeling ===")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.X['gender'] if 'gender' in self.X.columns else None
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_ensemble_models(self):
        """
        Train ensemble methods for regression
        """
        print("\n=== Training Ensemble Models ===")
        
        # Define base models
        base_models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=1.0, random_state=42),
            'svr': SVR(kernel='rbf', C=1.0),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        # Train individual models
        print("Training individual models...")
        for name, model in base_models.items():
            print(f"Training {name}...")
            try:
                multi_model = MultiOutputRegressor(model)
                multi_model.fit(self.X_train_scaled, self.y_train)
                self.models[name] = multi_model
                
                # Evaluate
                y_pred = multi_model.predict(self.X_test_scaled)
                mae = mean_absolute_error(self.y_test, y_pred)
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                
                self.results[name] = {'mae': mae, 'mse': mse, 'r2': r2}
                print(f"  {name}: MAE={mae:.3f}, MSE={mse:.3f}, R²={r2:.3f}")
                
            except Exception as e:
                print(f"  {name} failed: {e}")
        
        # Create voting ensemble
        print("\nTraining voting ensemble...")
        try:
            # Select best performing models for ensemble
            best_models = []
            for name, model in self.models.items():
                if name in ['random_forest', 'gradient_boosting', 'xgboost']:
                    best_models.append((name, model.estimators_[0]))  # Get first estimator for glucose
            
            if len(best_models) >= 2:
                voting_regressor = VotingRegressor(best_models)
                voting_multi = MultiOutputRegressor(voting_regressor)
                voting_multi.fit(self.X_train_scaled, self.y_train)
                
                y_pred = voting_multi.predict(self.X_test_scaled)
                mae = mean_absolute_error(self.y_test, y_pred)
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                
                self.models['voting_ensemble'] = voting_multi
                self.results['voting_ensemble'] = {'mae': mae, 'mse': mse, 'r2': r2}
                print(f"  Voting Ensemble: MAE={mae:.3f}, MSE={mse:.3f}, R²={r2:.3f}")
        
        except Exception as e:
            print(f"Voting ensemble failed: {e}")
        
        return self.models, self.results
    
    def hyperparameter_tuning(self):
        """
        Perform hyperparameter tuning on best models
        """
        print("\n=== Hyperparameter Tuning ===")
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'estimator__n_estimators': [100, 200, 300],
                'estimator__max_depth': [10, 20, None],
                'estimator__min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'estimator__n_estimators': [100, 200],
                'estimator__max_depth': [3, 6, 9],
                'estimator__learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        tuned_models = {}
        
        for model_name, param_grid in param_grids.items():
            if model_name == 'random_forest':
                base_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
            elif model_name == 'xgboost':
                base_model = MultiOutputRegressor(xgb.XGBRegressor(random_state=42))
            
            print(f"Tuning {model_name}...")
            try:
                grid_search = GridSearchCV(
                    base_model, param_grid, 
                    cv=5, scoring='neg_mean_squared_error',
                    n_jobs=-1, verbose=0
                )
                
                grid_search.fit(self.X_train_scaled, self.y_train)
                
                # Evaluate tuned model
                y_pred = grid_search.predict(self.X_test_scaled)
                mae = mean_absolute_error(self.y_test, y_pred)
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                
                tuned_models[f'{model_name}_tuned'] = grid_search
                self.results[f'{model_name}_tuned'] = {'mae': mae, 'mse': mse, 'r2': r2}
                
                print(f"  {model_name} tuned: MAE={mae:.3f}, MSE={mse:.3f}, R²={r2:.3f}")
                print(f"  Best params: {grid_search.best_params_}")
                
            except Exception as e:
                print(f"  {model_name} tuning failed: {e}")
        
        self.models.update(tuned_models)
        return tuned_models
    
    def create_classification_targets(self):
        """
        Create classification targets for diabetes risk categories
        """
        print("\n=== Creating Classification Targets ===")
        
        # Create diabetes risk categories based on glucose and HbA1c
        def classify_diabetes_risk(row):
            glucose = row['glucose']
            hba1c = row['hba1c']
            
            # ADA criteria for diabetes classification
            if glucose >= 126 or hba1c >= 6.5:
                return 2  # Diabetes
            elif glucose >= 100 or hba1c >= 5.7:
                return 1  # Pre-diabetes
            else:
                return 0  # Normal
        
        self.df['diabetes_risk'] = self.df.apply(classify_diabetes_risk, axis=1)
        
        # Distribution of classes
        class_dist = self.df['diabetes_risk'].value_counts().sort_index()
        print("Diabetes risk distribution:")
        print(f"  Normal (0): {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(self.df)*100:.1f}%)")
        print(f"  Pre-diabetes (1): {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(self.df)*100:.1f}%)")
        print(f"  Diabetes (2): {class_dist.get(2, 0)} ({class_dist.get(2, 0)/len(self.df)*100:.1f}%)")
        
        return self.df['diabetes_risk']
    
    def train_classification_models(self):
        """
        Train classification models for diabetes risk prediction
        """
        print("\n=== Training Classification Models ===")
        
        # Create classification targets
        y_class = self.create_classification_targets()
        
        # Split data for classification
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
            self.X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        
        # Scale features
        X_train_class_scaled = self.scaler.transform(X_train_class)
        X_test_class_scaled = self.scaler.transform(X_test_class)
        
        # Define classification models
        class_models = {
            'rf_classifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb_classifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'xgb_classifier': xgb.XGBClassifier(n_estimators=100, random_state=42)
        }
        
        classification_results = {}
        
        for name, model in class_models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train_class_scaled, y_train_class)
                y_pred_class = model.predict(X_test_class_scaled)
                
                accuracy = accuracy_score(y_test_class, y_pred_class)
                classification_results[name] = {
                    'accuracy': accuracy,
                    'model': model,
                    'y_pred': y_pred_class,
                    'y_test': y_test_class
                }
                
                print(f"  {name}: Accuracy={accuracy:.3f}")
                
                # Print classification report
                print(f"  Classification Report for {name}:")
                print(classification_report(y_test_class, y_pred_class, 
                                          target_names=['Normal', 'Pre-diabetes', 'Diabetes']))
                
            except Exception as e:
                print(f"  {name} failed: {e}")
        
        return classification_results
    
    def compare_model_performance(self):
        """
        Compare performance of all models
        """
        print("\n=== Model Performance Comparison ===")
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'MAE': metrics['mae'],
                'MSE': metrics['mse'],
                'R²': metrics['r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('MAE')
        
        print("Regression Model Performance (sorted by MAE):")
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # Visualize performance
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # MAE comparison
        axes[0].barh(comparison_df['Model'], comparison_df['MAE'])
        axes[0].set_xlabel('MAE (mg/dL)')
        axes[0].set_title('Mean Absolute Error Comparison')
        
        # R² comparison
        axes[1].barh(comparison_df['Model'], comparison_df['R²'])
        axes[1].set_xlabel('R² Score')
        axes[1].set_title('R² Score Comparison')
        
        # MSE comparison
        axes[2].barh(comparison_df['Model'], comparison_df['MSE'])
        axes[2].set_xlabel('MSE')
        axes[2].set_title('Mean Squared Error Comparison')
        
        plt.tight_layout()
        plt.savefig('/Users/aakashsuresh/fairness/blood_glucose_project/finetuning/advanced_model_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def analyze_best_model_fairness(self):
        """
        Analyze fairness of the best performing model
        """
        print("\n=== Best Model Fairness Analysis ===")
        
        # Find best model
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]['mae'])
        best_model = self.models[best_model_name]
        
        print(f"Best model: {best_model_name}")
        print(f"Best MAE: {self.results[best_model_name]['mae']:.3f}")
        
        # Get predictions
        y_pred = best_model.predict(self.X_test_scaled)
        
        # Create test dataframe with predictions
        test_df = self.X_test.copy()
        test_df['glucose_true'] = self.y_test.iloc[:, 0].values
        test_df['hba1c_true'] = self.y_test.iloc[:, 1].values
        test_df['glucose_pred'] = y_pred[:, 0]
        test_df['hba1c_pred'] = y_pred[:, 1]
        
        # Analyze fairness by demographic groups
        fairness_results = {}
        
        # Gender-based fairness
        if 'gender' in test_df.columns:
            gender_results = {}
            for gender in test_df['gender'].unique():
                gender_data = test_df[test_df['gender'] == gender]
                glucose_mae = mean_absolute_error(gender_data['glucose_true'], gender_data['glucose_pred'])
                gender_results[f'Gender_{gender}'] = {
                    'n': len(gender_data),
                    'glucose_mae': glucose_mae
                }
            fairness_results['gender'] = gender_results
        
        # Age-based fairness
        if 'age' in test_df.columns:
            test_df['age_group'] = pd.cut(test_df['age'], bins=[18, 40, 60, 100], labels=['<40', '40-60', '>60'])
            age_results = {}
            for age_group in test_df['age_group'].unique():
                if pd.notna(age_group):
                    age_data = test_df[test_df['age_group'] == age_group]
                    glucose_mae = mean_absolute_error(age_data['glucose_true'], age_data['glucose_pred'])
                    age_results[str(age_group)] = {
                        'n': len(age_data),
                        'glucose_mae': glucose_mae
                    }
            fairness_results['age'] = age_results
        
        # Print fairness results
        for group_type, results in fairness_results.items():
            print(f"\nFairness by {group_type}:")
            for group_name, metrics in results.items():
                print(f"  {group_name}: MAE={metrics['glucose_mae']:.3f} (n={metrics['n']})")
        
        return fairness_results, best_model_name
    
    def run_complete_advanced_analysis(self):
        """
        Run complete advanced modeling analysis
        """
        print("Advanced Blood Glucose Modeling Analysis")
        print("=" * 60)
        
        # Load data and prepare
        self.load_improved_dataset()
        self.prepare_data_for_modeling()
        
        # Train ensemble models
        self.train_ensemble_models()
        
        # Hyperparameter tuning
        self.hyperparameter_tuning()
        
        # Classification approach
        classification_results = self.train_classification_models()
        
        # Compare performance
        comparison_df = self.compare_model_performance()
        
        # Fairness analysis
        fairness_results, best_model = self.analyze_best_model_fairness()
        
        print("\n" + "=" * 60)
        print("ADVANCED MODELING COMPLETE")
        print("=" * 60)
        
        return {
            'regression_results': self.results,
            'classification_results': classification_results,
            'comparison': comparison_df,
            'fairness': fairness_results,
            'best_model': best_model
        }

def main():
    """
    Main execution function
    """
    analyzer = AdvancedGlucoseModeling()
    results = analyzer.run_complete_advanced_analysis()
    return results

if __name__ == "__main__":
    results = main()
