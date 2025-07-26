import csv
import re
import pandas as pd
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
from LinkGenerator import preprocessor
from extract_commit_version import get_fix_versions_from_jira
import ast
import requests
from tqdm import tqdm

# Increase CSV field size limit to handle large data
csv.field_size_limit(500 * 1024 * 1024)

repo = "GIRAPH"

# Initialize an empty list to store processed data
process = []
# Load the CSV file containing the dataset
dummy_link = pd.read_csv(f"../../data/OriginalData/{repo.title()}/{repo.lower()}_link_raw_merged_test.csv")
release_notes = pd.read_csv(f"../../data/ReleaseNotes/{repo.title()}/release_notes_{repo.lower()}.csv")

print(f"Processing information for {repo} issues")
# Rename the column "commitid" to "hash"
tqdm.pandas()
bar = tqdm(total=len(dummy_link))

# # Iterate through each row in the dataset
for index, row in dummy_link.iterrows():
    bar.update(1)
    # Preprocess the summary, description, commit message, and diff
    if isinstance(row["message"], str):
        clean_message = re.sub(rf"{repo}-\d+\s*", "", row["message"])
        clean_message = re.sub(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', '', clean_message)
    else:
        clean_message = "nan"

    summary_processed = preprocessor.preprocessNoCamel(str(row["summary"]).strip("[]"))
    description_processed = preprocessor.preprocessNoCamel(str(row["description"]).strip("[]"))
    message_processed = preprocessor.preprocessNoCamel(str(clean_message).strip("[]"))

    # Fetch the fix version for the current commit hash using the `get_fix_versions_from_jira` function
    # Combine summary, description, and comment to process issue code
    print("random")
    for i, r in release_notes.iterrows():
        if row["tracking_id"] == r["tracking_id"]:
            pattern = rf"\b({repo}-\d+)\b"
            cleaned_release_notes = re.sub(pattern, "", r["content"])
            release_notes_processed_cleaned = preprocessor.preprocessNoCamel(str(cleaned_release_notes).strip("[]"))
            release_notes_content = release_notes_processed_cleaned
            target_rn = 1

            break
        else:
            release_notes_content = "nan"
            target_rn = 0
    # Prepare the list for the current row, including the fix_version
    list1 = [
        row["issue_id"], summary_processed,
        description_processed,  row["hash"], row["tracking_id"],
        message_processed, row["Diff"], release_notes_content, row["target"], target_rn
    ]

    # Append the processed row to the process list
    process.append(list1)

bar.close()
columns = [
    "issue_id", "summary_processed", "description_processed", "hash", "tracking_id", "message_processed", "Diff_processed","release_notes", "target", "target_rn"
]

# Create a DataFrame from the processed data
df = pd.DataFrame(process, columns=columns)
# # Remove malformed rows (Tika)
# df = df[df["target"].apply(lambda x: x == "0" or x == "1")]
# Write the DataFrame to a CSV file
df.to_csv(f"../../data/Processed{repo.title()}/2_{repo.lower()}_link_merged.csv", index=False)
