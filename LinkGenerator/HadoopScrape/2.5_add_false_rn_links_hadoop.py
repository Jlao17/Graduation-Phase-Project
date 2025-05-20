import pandas as pd
import random

import pandas as pd
import random

repo = "HADOOP"
df = pd.read_csv(f"../../data/Processed{repo.title()}/2.4_{repo.lower()}_link_merge_false_rn.csv")

# Filter out rows with valid release_notes and tracking_id to sample from
valid_pairs = df[["release_notes", "tracking_id"]].dropna().values.tolist()

# Get indices of half the dataset randomly
num_false_links = len(df) // 2
false_indices = random.sample(range(len(df)), num_false_links)

# Replace release_notes and tracking_id for selected indices
for idx in false_indices:
    rn, tid = random.choice(valid_pairs)
    df.loc[idx, "release_notes"] = rn
    df.loc[idx, "tracking_id"] = tid
    df.loc[idx, "target_rn"] = 0

# Ensure target_rn column is filled and properly typed
df["target_rn"] = df["target_rn"].fillna(0).astype(int)

# Save the modified dataset
df.to_csv(f"../../data/Processed{repo.title()}/2.5_{repo.lower()}_link_false_rn.csv", index=False)

print("False link generation complete! 🚀")
