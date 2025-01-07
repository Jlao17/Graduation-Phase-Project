import ast

import pandas as pd
from LinkGenerator import preprocessor
import re

# Load data
repo = "CALCITE"
df = pd.read_csv(f"../data/Processed{repo.title()}/1_{repo.lower()}_process.csv")
calcite_release_notes = pd.read_csv("../data/ReleaseNotes/Calcite/release_notes_calcite.csv")
avatica_release_notes = pd.read_csv("../data/ReleaseNotes/Calcite/release_notes_calcite_avatica.csv")
avatica_go_release_notes = pd.read_csv("../data/ReleaseNotes/Calcite/release_notes_calcite_avatica_go.csv")

df["tracking_id"] = df["tracking_id"].astype(str)


def get_release_notes_calcite(fix_versions, calcite_df, avatica_df, avatica_go_df):
    release_notes = []

    # Ensure fix_versions is a list
    fix_versions_list = fix_versions.strip("[]").replace("'", "").split(",")
    fix_versions_list = [v.strip() for v in fix_versions_list]

    for version in fix_versions_list:
        if "avatica-go" in version:
            version_number = version.replace("avatica-go-", "").strip()
            avatica_go_note = avatica_go_df[avatica_go_df["version"] == version_number]["content"]
            if len(avatica_go_note) > 0:
                release_notes.extend(avatica_go_note)
        elif "avatica-" in version and "avatica-go" not in version:
            version_number = version.replace("avatica-", "").strip()
            avatica_note = avatica_df[avatica_df["version"] == version_number]["content"]
            if len(avatica_note) > 0:
                release_notes.extend(avatica_note)
        else:
            calcite_note = calcite_df[calcite_df["version"] == version]["content"]
            if len(calcite_note) > 0:
                release_notes.extend(calcite_note)
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
            release_note = get_release_notes_calcite(version, calcite_release_notes, avatica_release_notes,
                                                     avatica_go_release_notes)

            try:
                release_note_list = ast.literal_eval(release_note[0])
            except:
                # print("No release note found for row: ", index)
                release_note_list = []

            # Copy the original row and modify the 'fix_version' and 'release_notes'
            new_row = row.copy()
            new_row["fix_version"] = version
            new_row["tracking_id"] = new_row["tracking_id"].strip("[]")
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
