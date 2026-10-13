%pip install -q torch numpy pandas openpyxl scikit-learn

import subprocess
import sys
import pandas as pd
from IPython.display import display

# Only these two inputs need editing.
dataset = "/content/dummy_dataset.xlsx"
networks = 100

subprocess.run(
    [sys.executable, "/content/mighty_deep_standalone_runner_20261013.py",
     "--dataset", dataset, "--networks", str(networks)],
    check=True,
    cwd="/content",
)
display(pd.read_csv("/content/Algorithm_Summary_Metrics.csv"))