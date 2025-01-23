import ast

import pandas as pd
from LinkGenerator import preprocessor
import re

# Load data
repo = "ISIS"
df = pd.read_csv(f"../data/Processed{repo.title()}/1_{repo.lower()}_process.csv")
isis_release_notes = pd.read_csv("../../data/ReleaseNotes/Isis/release_notes_isis_merged.csv")

df["tracking_id"] = df["tracking_id"].astype(str)

def get_release_notes_isis(fix_versions, isis_df):
    release_notes = []
    fix_versions_list = fix_versions.strip("[]").replace("'", "").split(",")
    fix_versions_list = [v.strip() for v in fix_versions_list]
    for version in fix_versions_list:
        for index, row in isis_df.iterrows():
            try:
                components_list = ast.literal_eval(row["component"])
            except:
                components_list = []
            if version in components_list:
                isis_note = row["content"]
                release_notes.append(isis_note)
                break
            else:
                continue

    return release_notes if release_notes else "nan"


if __name__ == "__main__":
    expanded_rows = []

    for index, row in df.iterrows():
        # Split `fix_version` if there are multiple versions
        print("Processing row: ", index + 1, "/", len(df))
        fix_versions_list = row["fix_version"].strip("[]").replace("'", "").split(",")
        fix_versions_list = [v.strip() for v in fix_versions_list]
        for version in fix_versions_list:
            # Get the release note for the current fix_version
            release_note = get_release_notes_isis(version, isis_release_notes)

            try:
                release_note_list = ast.literal_eval(release_note[0])
            except:
                # print("No release note found for row: ", index)
                release_note_list = []

            # Copy the original row and modify the 'fix_version' and 'release_notes'
            new_row = row.copy()
            new_row["fix_version"] = version
            # new_row["tracking_id"] = new_row["tracking_id"].strip("[]")
            for note in release_note_list:
                if new_row["tracking_id"] in note:
                    pattern = rf"\b{repo}-\d+\b"
                    cleaned_release_notes = re.sub(pattern, "", note)
                    release_notes_processed_cleaned = preprocessor.preprocessNoCamel(str(cleaned_release_notes).strip("[]"))
                    new_row["release_notes_original"] = note
                    new_row["release_notes"] = release_notes_processed_cleaned
                    break

            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)

    # Save the expanded DataFrame
    expanded_df.to_csv(f"../data/Processed{repo.title()}/1.5_{repo.lower()}_process_notes_cleaned.csv", index=False)
