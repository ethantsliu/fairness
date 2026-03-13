import os
import requests
from tqdm import tqdm

ftp_url = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/nhanes/2011-2012/PAX80_G/"
output_dir = "NHANES_PAX80_G"

os.makedirs(output_dir, exist_ok=True)

# Example: Download first 10 participant files (Adjust range as needed)
for seqn in range(100001, 100011):  # Adjust SEQN numbers as needed
    filename = f"{seqn}.tar.bz2"
    file_url = f"{ftp_url}{filename}"
    output_path = os.path.join(output_dir, filename)

    response = requests.get(file_url, stream=True)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in tqdm(response.iter_content(1024), desc=filename, unit="KB"):
                f.write(chunk)
    else:
        print(f"Failed to download {filename}")