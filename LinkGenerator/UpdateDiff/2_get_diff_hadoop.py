import numpy as np
import random
from pydriller import Repository
import pandas as pd
from tqdm import tqdm

# Define the repo and the specific commit hash you want
repo_path = "https://github.com/apache/hadoop"  # Change to your repo path or  local directory

df = pd.read_csv("../../data/ProcessedHadoop/1.5_hadoop_process.csv")

commit_hashes = df["hash"].tolist()

structured_diffs = {}

for commit in tqdm(Repository(repo_path, only_commits=commit_hashes).traverse_commits(), total=len(commit_hashes), desc="Processing commits - Total commits is not accurate, no fast way to get all specific commits from pydriller"):
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
df = df[['issue_id', 'summary_processed', 'description_processed', 'hash', 'fix_version', 'tracking_id',
 'message_processed', 'release_notes', 'Diff_processed', 'train_flag', 'target', 'target_rn']]
df.to_csv("../../data/ProcessedHadoop/2_hadoop_link_merged.csv", index=False)
