import pandas as pd
import random

repo = "HADOOP"
df = pd.read_csv(f"../data/Processed{repo.title()}/2_{repo.lower()}_link_merged.csv")

df["target_rn"] = 1

false_rows = []
# Precompute the set of all unique release notes
release_notes_list = df["release_notes"].tolist()

for index, row in df.iterrows():
    if row["target"] == 0:
        continue
    new_row = row.copy()
    current_release_note = row["release_notes"]

    # Create a list of release notes that are NOT the current release note
    available_release_notes = [rn for rn in release_notes_list if rn != current_release_note]

    new_release_note = random.choice(available_release_notes)
    new_row["release_notes"] = new_release_note
    new_row["target_rn"] = 0

    false_rows.append(new_row)

false_df = pd.DataFrame(false_rows)

# Concatenate the original DataFrame with the false-link DataFrame
final_df = pd.concat([df, false_df], ignore_index=True)
final_df = final_df.sample(frac=1).reset_index(drop=True)

final_df.to_csv(f"../data/Processed{repo.title()}/2.5_{repo.lower()}_link_false_rn.csv", index=False)
