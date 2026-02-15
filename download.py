from huggingface_hub import snapshot_download

"""snapshot_download(
    repo_id="Enxin/MovieChat-1K_train",
    repo_type="dataset",
    allow_patterns="jsons/*",   # only json folder
    local_dir="MovieChat_jsons",   # folder inside your project
    local_dir_use_symlinks=False
)"""

"""snapshot_download(
    repo_id="TIGER-Lab/VISTA-400K",
    repo_type="dataset",
    allow_patterns="jsons/*",   # only json folder
    local_dir="VISTA-400K_jsons",   # folder inside your project
)"""

"""snapshot_download(
    repo_id="MLVU/MVLU",
    repo_type="dataset",
    allow_patterns="MLVU/json/*",   # correct path inside repo
    local_dir="MVLU_jsons",         # will contain the folder structure
)"""

import os
import pandas as pd
from huggingface_hub import snapshot_download

# ===============================
# Step 1: Download dataset
# ===============================

local_dir = "ActivityNetQA_data"

snapshot_download(
    repo_id="lmms-lab/ActivityNetQA",
    repo_type="dataset",
    allow_patterns="data/*",
    local_dir=local_dir,
    resume_download=True
)

print("Download completed!")

# ===============================
# Step 2: Find parquet file
# ===============================

parquet_path = None

for root, dirs, files in os.walk(local_dir):
    for file in files:
        if file.endswith(".parquet"):
            parquet_path = os.path.join(root, file)
            break

if parquet_path is None:
    raise FileNotFoundError("Parquet file not found!")

print("Found parquet at:", parquet_path)

# ===============================
# Step 3: Convert and save JSON in SAME folder
# ===============================

df = pd.read_parquet(parquet_path)

# Save JSON in same directory as parquet
json_output_path = os.path.join(
    os.path.dirname(parquet_path),
    "ActivityNetQA_test.json"
)

df.to_json(json_output_path, orient="records", indent=4)

print("JSON saved at:", json_output_path)


print("Download completed inside project folder.")
