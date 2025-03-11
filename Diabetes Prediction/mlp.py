import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Check for Metal Performance Shaders on mac
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Read the accelerometry, glucose, and diet data
acc_data = pd.read_csv("processed_data_new/nhanes_combined_acc.csv")  # Accelerometry data
glu_data = pd.read_csv("processed_data_new/nhanes_combined_glu.csv")  # Plasma fasting glucose
diet_data = pd.read_csv("processed_data_new/filled_nhanes_combined_diet.csv")  # Diet data

# Drop unnecessary columns from glu_data (including diabetes_status if present)
glu_data = glu_data.drop(columns=['diabetes_status'], axis=1, errors='ignore')

# Merge accelerometry, glucose, and diet data on 'SEQN' (subject identifier)
merged_data = pd.merge(acc_data, glu_data, on='SEQN', how='inner')
merged_data = pd.merge(merged_data, diet_data, on='SEQN', how='inner')

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

# Define input features (include both accelerometry and diet columns)
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

# Handle NaN values in the columns by filling with column mean
features.fillna(features.mean(), inplace=True)  # Fill NaNs with mean of each column
print(features.head())

# Check again for NaN values after filling
nan_counts = features.isna().sum()
print("Columns with NaN values after filling:")
print(nan_counts[nan_counts > 0])

# Target variable (diabetes status based on LBXGLU)
y = merged_data['diabetes_status'].values  # Convert to integer labels

# Standardize the features (important for neural network training)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Define the MLP model with Dropout for regularization
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=1024, output_size=3):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, hidden_size)  # Additional hidden layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))  # Additional hidden layer
        x = self.output_layer(x)
        return self.softmax(x)

# Cross-validation procedure
def cross_validate_model(X, y, n_splits=5, epochs=100, batch_size=32):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"Fold {fold+1}/{n_splits}")
        
        # Split data into train and validation sets for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # for classification (long type)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        # Instantiate the model
        input_size = X_train.shape[1]  # Number of features
        model = MLP(input_size).to(device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

        # Training loop for this fold
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor.to(device))
            loss = criterion(outputs, y_train_tensor.to(device))
            loss.backward()
            optimizer.step()

        # Validation loop for this fold
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.to(device))
            _, predicted = torch.max(val_outputs, 1)
            
            # Move the predictions to the CPU before using them in sklearn functions
            predicted = predicted.cpu().numpy()
            
            accuracy = accuracy_score(y_val, predicted)
            f1 = f1_score(y_val, predicted, average='weighted')
            auc = roc_auc_score(y_val, val_outputs.cpu().numpy(), multi_class='ovr', average='weighted')

            accuracies.append(accuracy)
            f1_scores.append(f1)
            auc_scores.append(auc)

            print(f"Validation Accuracy for Fold {fold+1}: {accuracy:.4f}")
            print(f"Validation F1 Score for Fold {fold+1}: {f1:.4f}")
            print(f"Validation AUC Score for Fold {fold+1}: {auc:.4f}")

    # Print final average metrics across all folds
    print(f"\nAverage Cross-Validation Accuracy: {np.mean(accuracies):.4f}")
    print(f"Average Cross-Validation F1 Score: {np.mean(f1_scores):.4f}")
    print(f"Average Cross-Validation AUC Score: {np.mean(auc_scores):.4f}")
    return accuracies, f1_scores, auc_scores

# Call the function to perform cross-validation
accuracies, f1_scores, auc_scores = cross_validate_model(features_scaled, y, n_splits=5, epochs=100)