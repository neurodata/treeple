
import argparse
import os
import numpy as np
import pandas as pd
import ydf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--trees", type=int, default=1000)
args = parser.parse_args()

df = pd.read_excel(args.dataset)
targets = {"target", "cancer status", "cancer_status", "status", "label"}
target = next(
    (c for c in df.columns if str(c).strip().lower() in targets),
    df.columns[-1]
)
df = df.dropna(subset=[target]).copy()

# Convert cancer status to binary (healthy=0, cancer=1)
negative = {
    "false", "0", "healthy", "normal", "control",
    "negative", "no", "benign", "non-cancer", "noncancer"
}
positive = {
    "true", "1", "cancer", "tumor", "tumour",
    "positive", "yes", "malignant", "disease",
    "colon", "prostate", "breast", "lung", "ovarian",
    "pancreatic", "liver", "kidney", "bladder",
    "colorectal", "melanoma", "leukemia", "lymphoma"
}

labels = df[target].astype(str).str.strip().str.lower()
unknown = set(labels.unique()) - negative - positive
if unknown:
    raise ValueError(f"Unrecognized target labels: {unknown}")

df[target] = labels.isin(positive).astype(int).astype(str)

if df[target].nunique() != 2:
    raise ValueError("Both healthy and cancer samples are required.")

train, test = train_test_split(
    df, test_size=0.2, stratify=df[target], random_state=42
)

model = ydf.RandomForestLearner(
    label=target,
    num_trees=args.trees,
    split_axis="SPARSE_OBLIQUE",
    random_seed=23,
    winner_take_all=False,
).train(train)

y_true = test[target].astype(int).to_numpy()
y_prob = model.predict(test)
fpr, tpr, _ = roc_curve(y_true, y_prob)

sens = {
    s: float(tpr[fpr <= 1 - s / 100 + 1e-12].max())
    for s in [92, 94, 95, 96, 98, 100]
}

result = {
    "Classification Algorithm": "YDF-SPORF",
    "Dataset": os.path.basename(args.dataset),
    "# of Trees": args.trees,
    "# of Features": df.shape[1] - 1,
    "AUC": roc_auc_score(y_true, y_prob),
    **{
        f"Sensitivity at {s}% Specificity": sens[s]
        for s in sens
    },
    "Average of S@94, S@95, S@96": np.mean(
        [sens[94], sens[95], sens[96]]
    ),
}

output = "Algorithm_Summary_Metrics.csv"
pd.DataFrame([result]).round(6).to_csv(output, index=False)
print(f"Saved {output}")
