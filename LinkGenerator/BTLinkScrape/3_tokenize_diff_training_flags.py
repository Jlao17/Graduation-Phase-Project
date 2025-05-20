import pandas as pd
import sys
import numpy as np
from LinkGenerator import preprocessor

maxInt = sys.maxsize
repo = "GIRAPH"
# Load your data
df = pd.read_csv(f"../../data/Processed{repo.title()}/2.5_{repo.lower()}_link_false_rn.csv")
print(len(df))
# if release_notes = "nan", then target_rn = 0, else 1
df["target_rn"] = df["release_notes"].notna().astype(int)
# delete all rows with target_rn = 0
# df = df[df["target_rn"] == 1]
for index, row in df.iterrows():
    df.at[index, "Diff_processed"] = preprocessor.processCode(str(row["Diff_processed"]).strip("[]"))

df['train_flag'] = 1

# Ensure reproducibility
np.random.seed(42)

# Assign train_flag: 80% for training and 20% for testing
df['train_flag'] = np.random.choice([1, 0], size=len(df), p=[0.8, 0.2])
print(df['train_flag'].value_counts())
print(len(df))
df.to_csv(f"../../data/Processed{repo.title()}/3_{repo.lower()}_link_final.csv", index=False)