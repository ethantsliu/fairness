import os
import pandas as pd
import zipfile
from pathlib import Path

# Define file paths
input_paths = {
    "2011-2012": "/Users/aakashsuresh/fairness/2011-2012 NHANES Data.xpt",
    "2013-2014": "/Users/aakashsuresh/fairness/2013-2014 NHANES Data.xpt"
}

output_dir = Path("/Users/aakashsuresh/fairness/processed_data")
output_dir.mkdir(exist_ok=True)

def convert_xpt_to_csv(input_xpt, output_csv):
    """Converts an NHANES .XPT file to a CSV file."""
    df = pd.read_sas(input_xpt, format='xport')
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV: {output_csv}")

def zip_csv_file(csv_path, zip_path):
    """Zips the CSV file."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, csv_path.name)
    print(f"Zipped file: {zip_path}")

# Process each dataset
for year, xpt_path in input_paths.items():
    csv_path = output_dir / f"{year}_NHANES.csv"
    zip_path = output_dir / f"{year}_NHANES.zip"

    convert_xpt_to_csv(xpt_path, csv_path)
    zip_csv_file(csv_path, zip_path)


