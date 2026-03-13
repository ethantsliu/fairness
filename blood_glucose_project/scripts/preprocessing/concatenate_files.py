import pandas as pd

# Read the CSV files
nhanes_2011_2012 = pd.read_csv('/Users/aakashsuresh/fairness/blood_glucose_project/data/processed/nhanes_combined/2011-2012_GLU_G.csv')
nhanes_2013_2014 = pd.read_csv('/Users/aakashsuresh/fairness/blood_glucose_project/data/processed/nhanes_combined/2013-2014_GLU_H.csv')
nhanes_2011_2012 = nhanes_2011_2012.drop(columns=['LBXIN', 'LBDINSI'])

# Concatenate the DataFrames
nhanes_combined = pd.concat([nhanes_2011_2012, nhanes_2013_2014], axis=0, ignore_index=True)

# Save to a new CSV file
nhanes_combined.to_csv('/Users/aakashsuresh/fairness/blood_glucose_project/data/processed/nhanes_combined/nhanes_combined_glu.csv', index=False)

print("Files combined successfully!")