import pandas as pd
import pyreadstat

# Load the .xpt file
path = "/Users/aakashsuresh/fairness/blood_glucose_project/data/processed/nhanes_combined/2013-2014_GHB_H" 
df, meta = pyreadstat.read_xport(path + ".xpt")

# Save to CSV
df.to_csv(path + ".csv", index=False)