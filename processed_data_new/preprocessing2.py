import os
import pandas as pd
import zipfile
from pathlib import Path

# Define NEW folder for processed data
output_dir = Path("/Users/aakashsuresh/fairness/processed_data_new")
output_dir.mkdir(exist_ok=True)  # Ensure the new folder exists

# Define input file paths
input_paths = {
    "2011-2012_Dietary": "/Users/aakashsuresh/fairness/NHANES Dietary Data 2011-2012.xpt",
    "2013-2014_Dietary": "/Users/aakashsuresh/fairness/NHANES Dietary Data 2013-2014.xpt",
    "2011-2012_Accelerometry": "/Users/aakashsuresh/fairness/NHANES Acceleromatry Data 2011-2012.xpt",
    "2013-2014_Accelerometry": "/Users/aakashsuresh/fairness/NHANES Acceleromatry Data 2013-2014.xpt"
}

def convert_xpt_to_csv(input_xpt, output_csv):
    """Converts NHANES .XPT file to CSV."""
    df = pd.read_sas(input_xpt, format='xport')
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved CSV: {output_csv}")

def zip_csv_file(csv_path, zip_path):
    """Zips the CSV file."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, csv_path.name)
    print(f"✅ Zipped file: {zip_path}")

# Process each dataset and save to processed_data_new/
for dataset, xpt_path in input_paths.items():
    csv_path = output_dir / f"{dataset}.csv"
    zip_path = output_dir / f"{dataset}.zip"

    convert_xpt_to_csv(xpt_path, csv_path)
    zip_csv_file(csv_path, zip_path)

