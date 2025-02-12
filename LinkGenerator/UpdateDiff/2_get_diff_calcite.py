import numpy as np
import random
from pydriller import Repository
import pandas as pd
from tqdm import tqdm

# Define the repo and the specific commit hash you want
repo_path = "https://github.com/apache/calcite"  # Change to your repo path or  local directory
repo_path2 = "https://github.com/apache/calcite-avatica-go"
repo_path3 = "https://github.com/apache/calcite-avatica"

df = pd.read_csv("../../data/ProcessedCalcite/1.5_calcite_process_notes_cleaned.csv")
# add commit hases with 'avatica' in `fix_version` to the list
commit_hashes_avatica = df[df["fix_version"].notna() & df["fix_version"].str.contains("avatica")]["hash"].tolist()
commit_hashes_avatica_go = df[df["fix_version"].notna() & df["fix_version"].str.contains("avatica-go")]["hash"].tolist()
commit_hashes = df["hash"].tolist()
# Subtract hashes in avatica and avatica_go lists
commit_hashes_filtered = [hash for hash in commit_hashes if hash not in commit_hashes_avatica and hash not in commit_hashes_avatica_go]

structured_diffs = {}

for commit in tqdm(Repository(repo_path, only_commits=commit_hashes).traverse_commits(), total=len(commit_hashes), desc="Processing Calcite commits - Total commits is not accurate, no fast way to get all specific commits from pydriller"):
    structured_diff = []

    for file in commit.modified_files:

        change_type = file.change_type.name  # "ADD", "DELETE", "MODIFY"
        # Remove < and > from the method names
        changed_methods = [method.name.split("::")[-1] for method in file.methods]


        # Format structured diff
        diff_entry = f"{change_type} {file.filename} " + " ".join(f"<{m}>" for m in changed_methods) if changed_methods else f"{change_type} {file.filename}"
        structured_diff.append(diff_entry)

    # Store the structured diff in the dictionary
    structured_diffs[commit.hash] = " ".join(structured_diff)


for commit in tqdm(Repository(repo_path, only_commits=commit_hashes).traverse_commits(), total=len(commit_hashes), desc="Processing Avatica Go commits - Total commits is not accurate, no fast way to get all specific commits from pydriller"):
    structured_diff = []

    for file in commit.modified_files:

        change_type = file.change_type.name  # "ADD", "DELETE", "MODIFY"
        # Remove < and > from the method names
        changed_methods = [method.name.split("::")[-1] for method in file.methods]


        # Format structured diff
        diff_entry = f"{change_type} {file.filename} " + " ".join(f"<{m}>" for m in changed_methods) if changed_methods else f"{change_type} {file.filename}"
        structured_diff.append(diff_entry)

    # Store the structured diff in the dictionary
    structured_diffs[commit.hash] = " ".join(structured_diff)
# Update the DataFrame by mapping structured_diffs to the correct rows

for commit in tqdm(Repository(repo_path, only_commits=commit_hashes).traverse_commits(), total=len(commit_hashes), desc="Processing Avatica commits - Total commits is not accurate, no fast way to get all specific commits from pydriller"):
    structured_diff = []

    for file in commit.modified_files:

        change_type = file.change_type.name  # "ADD", "DELETE", "MODIFY"
        # Remove < and > from the method names
        changed_methods = [method.name.split("::")[-1] for method in file.methods]


        # Format structured diff
        diff_entry = f"{change_type} {file.filename} " + " ".join(f"<{m}>" for m in changed_methods) if changed_methods else f"{change_type} {file.filename}"
        structured_diff.append(diff_entry)

    # Store the structured diff in the dictionary
    structured_diffs[commit.hash] = " ".join(structured_diff)
# Update the DataFrame by mapping structured_diffs to the correct rows


df["Diff_processed"] = df["hash"].map(structured_diffs)
df.rename(columns={"label": "target"}, inplace=True)
df['Diff_processed'] = df['Diff_processed'].apply(lambda x: str(x).replace("<", "").replace(">", ""))
df = df[['issue_id', 'summary_processed', 'description_processed', 'issuecode', 'hash', 'fix_version', 'tracking_id',
 'message_processed', 'changed_files', 'codelist_processed', 'release_notes', 'Diff_processed', 'train_flag',
 'target']]
# Save the updated DataFrame back to CSV
df.to_csv("../../data/ProcessedCalcite/2_calcite_link_merged.csv", index=False)
