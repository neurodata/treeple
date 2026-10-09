%pip install -q ydf numpy pandas openpyxl matplotlib scikit-learn
print("Environment setup complete.")

import os
import subprocess
import sys
import pandas as pd

dataset = "/content/Sam's length_filtered.reconstructed.20pc.xlsx"
runner = "/content/ydf_standalone_runner_20261009.py"
trees = 1000

subprocess.run(
    [sys.executable, runner, "--dataset", dataset, "--trees", str(trees)],
    check=True
)

if os.path.exists("Algorithm_Summary_Metrics.csv"):
    display(pd.read_csv("Algorithm_Summary_Metrics.csv"))
