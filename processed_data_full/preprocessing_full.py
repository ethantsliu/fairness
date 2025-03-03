import os
import pandas as pd
import zipfile
from pathlib import Path

# Define NHANES survey cycles
survey_years = [
    "2011-2012", "2013-2014", "2015-2016", "2017-2018", "2019-2020", "2021-2022", "2023-2024"
]

# Base directory
base_dir = Path("/Users/aakashsuresh/fairness")

# Output directory for full dataset processing
output_dir = base_dir / "processed_data_full"
output_dir.mkdir(exist_ok=True)  # Ensure the new folder exists

# Expected file name patterns
file_templates = {
    "Dietary": "NHANES Dietary Data {}.xpt",
    "Accelerometry": "NHANES Acceleromatry Data {}.xpt"
}

def standardize_columns(df):
    """ Standardizes column names to lowercase and removes spaces for merging datasets. """
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df

def convert_xpt_to_csv(input_xpt, output_csv):
    """ Converts NHANES .XPT file to CSV with standardized columns. """
    if not input_xpt.exists():
        print(f"Skipping missing file: {input_xpt}")
        return False
    df = pd.read_sas(input_xpt, format='xport')
    df = standardize_columns(df)  # Standardize column names
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV: {output_csv}")
    return True

def zip_csv_file(csv_path, zip_path):
    """ Zips the CSV file. """
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, csv_path.name)
    print(f"Zipped file: {zip_path}")

# Process each dataset for all survey cycles
for year in survey_years:
    for dataset, template in file_templates.items():
        xpt_path = base_dir / template.format(year)
        csv_path = output_dir / f"{year}_{dataset}.csv"
        zip_path = output_dir / f"{year}_{dataset}.zip"

        if convert_xpt_to_csv(xpt_path, csv_path):
            zip_csv_file(csv_path, zip_path)
