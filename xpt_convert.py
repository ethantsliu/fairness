import pandas as pd
import pyreadstat

# Load the .xpt file
path = "processed_data_new/2013-2014_GHB_H" 
df, meta = pyreadstat.read_xport(path + ".xpt")

# Save to CSV
df.to_csv(path + ".csv", index=False)