import pandas as pd
import random

repo = "GIRAPH"
df = pd.read_csv(f"../data/Processed{repo.title()}/2_{repo.lower()}_link_merged.csv")

# Drop rows with NaN in either 'release_notes' or 'tracking_id' and get the valid pairs
valid_pairs = df[["release_notes", "tracking_id"]].dropna().values.tolist()

# Replace release_notes and tracking_id when target_rn == 0
def replace_with_random_pair(row):
    if row["target_rn"] == 0:
        rn, tid = random.choice(valid_pairs)
        row["release_notes"] = rn
        row["tracking_id"] = tid
    return row

df = df.apply(replace_with_random_pair, axis=1)

df.to_csv(f"../data/Processed{repo.title()}/2.5_{repo.lower()}_link_false_rn.csv", index=False)
