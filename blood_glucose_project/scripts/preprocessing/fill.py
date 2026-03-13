import pandas as pd

# Load dataset
diet_data = pd.read_csv("/Users/aakashsuresh/fairness/blood_glucose_project/data/processed/nhanes_combined/nhanes_combined_diet.csv")

# Define threshold for dropping columns 
threshold = 0.3 * len(diet_data)

# Drop columns with >= threshold missing values
diet_data_cleaned = diet_data.dropna(axis=1, thresh=threshold)

# Fill remaining NaN values with column mean
diet_data_cleaned = diet_data_cleaned.apply(lambda col: col.fillna(col.mean()) if col.dtype in ['float64', 'int64'] else col)

# Save cleaned dataset
diet_data_cleaned.to_csv("/Users/aakashsuresh/fairness/blood_glucose_project/data/processed/nhanes_combined/filled_nhanes_combined_diet.csv", index=False)

print("Cleaned dataset saved as 'filled_nhanes_combined_diet.csv'")
