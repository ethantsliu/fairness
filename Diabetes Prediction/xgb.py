import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
import torch

# Check for Metal Performance Shaders on mac
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Read the accelerometry and glucose data
acc_data = pd.read_csv("processed_data_new/nhanes_combined_acc.csv")  # Accelerometry data
glu_data = pd.read_csv("processed_data_new/nhanes_combined_glu.csv")  # Plasma fasting glucose

# Drop unnecessary columns from glu_data (including diabetes_status if present)
glu_data = glu_data.drop(columns=['diabetes_status'], axis=1, errors='ignore')

# Merge the accelerometry data with glucose data on 'SEQN' (subject identifier)
merged_data = pd.merge(acc_data, glu_data, on='SEQN', how='inner')

# Handle missing values in the LBXGLU column for the target variable (diabetes status)
merged_data['LBXGLU'] = merged_data['LBXGLU'].fillna(0)  # Or choose another strategy like mean or median

# Create a target variable 'diabetes_status' based on GLU (LBXGLU)
merged_data['diabetes_status'] = pd.cut(merged_data['LBXGLU'], 
                                         bins=[0, 99, 125, float('inf')], 
                                         labels=[0, 1, 2])  # 0 = No diabetes, 1 = Pre-diabetes, 2 = Diabetes

# Handle missing values in the 'diabetes_status' column (in case LBXGLU had NaNs)
merged_data['diabetes_status'] = merged_data['diabetes_status'].fillna(0)  # Default to 0 (no diabetes)

# Ensure the 'diabetes_status' column is of integer type
merged_data['diabetes_status'] = merged_data['diabetes_status'].astype(int)

# Define input features (only accelerometry columns, no LBXGLU)
# Drop non-numeric columns like 'SEQN', 'LBXGLU', and 'diabetes_status'
features = merged_data.drop(columns=['SEQN', 'LBXGLU', 'diabetes_status'])

# Convert features to numeric and handle any missing values (e.g., fill with mean or median)
features = features.apply(pd.to_numeric, errors='coerce')
# Check for columns with all NaN values
all_nan_columns = features.columns[features.isna().all()].tolist()

# If any column is fully NaN, we can either drop it or fill with a constant
if all_nan_columns:
    print(f"Columns with all NaN values: {all_nan_columns}")
    features = features.drop(columns=all_nan_columns)  # Drop columns with all NaN values
    # Columns related to glucose
    features = features.drop(columns=['LBDGLUSI', 'WTSAF2YR', 'PHAFSTHR', 'PHAFSTMN'])

# Handle NaN values in accelerometry columns by filling with column mean
features.fillna(features.mean(), inplace=True)  # Fill NaNs with mean of each column
print(features.head())

print("Features used to predict diabetes: " + features.columns)

# Check again for NaN values after filling
nan_counts = features.isna().sum()
print("Columns with NaN values after filling:")
print(nan_counts[nan_counts > 0])

# Target variable (diabetes status based on LBXGLU)
y = merged_data['diabetes_status'].values  # Convert to integer labels

# Standardize the features (important for many ML algorithms, including XGBoost)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Cross-validation procedure
def cross_validate_model(X, y, n_splits=5):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    # XGBoost parameters (feel free to tweak these)
    params = {
        'objective': 'multi:softmax',  # multi-class classification
        'num_class': 3,  # Number of classes (0, 1, 2)
        'eval_metric': 'mlogloss',  # Log loss as the evaluation metric
        'max_depth': 6,  # Maximum depth of the trees
        'learning_rate': 0.1,  # Learning rate
        'subsample': 0.8,  # Subsample ratio of the training instances
        'colsample_bytree': 0.8  # Subsample ratio of features
    }

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"Fold {fold+1}/{n_splits}")
        
        # Split data into train and validation sets for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert data to DMatrix (XGBoost's internal data structure)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train the XGBoost model
        model = xgb.train(params, dtrain, num_boost_round=100)

        # Validate the model
        y_pred = model.predict(dval)
        y_pred = np.round(y_pred)  # Round predictions to nearest integer (class)

        # Calculate accuracy for this fold
        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)
        print(f"Validation Accuracy for Fold {fold+1}: {accuracy:.4f}")

        # Display confusion matrix for fold to understand predictions better
        print("Confusion Matrix for Fold", fold+1)
        print(confusion_matrix(y_val, y_pred))

    # Print final average accuracy across all folds
    print(f"\nAverage Cross-Validation Accuracy: {np.mean(accuracies):.4f}")
    return accuracies

# Call the function to perform cross-validation
accuracies = cross_validate_model(features_scaled, y, n_splits=5)