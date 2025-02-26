import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load NHANES datasets
nhanes_2011_2012 = pd.read_csv('processed_data/2011-2012_NHANES.csv')
nhanes_2013_2014 = pd.read_csv('processed_data/2013-2014_NHANES.csv')

# Merge both datasets for combined analysis
nhanes = pd.concat([nhanes_2011_2012, nhanes_2013_2014], ignore_index=True)

# Print column names
print("\nColumn Names:\n", ", ".join(nhanes.columns))

# Print first row values in a single line
print("\nFirst Row:\n", " | ".join(nhanes.iloc[0].astype(str)))

# Display basic information
print("\nDataset Info:\n")
print(nhanes.info())

# Check for missing values
missing_values = nhanes.isnull().sum()
print("\nMissing Values:\n", missing_values[missing_values > 0])

# Check for duplicates
print("\nNumber of Duplicate Rows:", nhanes.duplicated().sum())

# Summary statistics for numeric columns
print("\nDescriptive Statistics:\n")
print(nhanes.describe())

# Visualizations
plt.figure(figsize=(10, 6))
sns.heatmap(nhanes.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()

# Select categorical columns for bar plots
categorical_cols = ['RIAGENDR', 'RIDSTATR', 'DMDHRBR4', 'DMDHREDU', 'DMDHRMAR']
for col in categorical_cols:
    if col in nhanes.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(y=nhanes[col], order=nhanes[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.show()

# Correlation heatmap for numeric variables
plt.figure(figsize=(12, 8))
sns.heatmap(nhanes.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()