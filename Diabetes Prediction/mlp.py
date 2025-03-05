import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Standardize the features (important for neural network training)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # for classification (long type)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

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
    
# Instantiate the model
input_size = X_train.shape[1]  # Number of features
model = MLP(input_size)

# Define the loss function (cross-entropy loss for classification) and optimizer (Adam)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)  # Reduced learning rate


# Training loop with early stopping (if needed)
num_epochs = 100
early_stopping_patience = 20
best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    _, predicted = torch.max(y_pred, 1)
    accuracy = accuracy_score(y_test_tensor, predicted)
    print(f'Test Accuracy: {accuracy:.4f}')