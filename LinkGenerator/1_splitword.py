import csv
import re
import pandas as pd
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import preprocessor
from extract_commit_version import get_fix_versions_from_jira
import ast
import requests
from tqdm import tqdm

# Increase CSV field size limit to handle large data
csv.field_size_limit(500 * 1024 * 1024)

# Set language and parser
lang = "java"
LANGUAGE = Language(tsjava.language())
parser = Parser(LANGUAGE)

repo = "NETBEANS"
# Initialize an empty list to store processed data
process = []
# Cloned repo path for fetching commit messages
cloned_repo_path = f"C:/Users/Jason/Desktop/{repo.lower()} repo/{repo.lower()}"
# Load the CSV file containing the dataset
dummy_link = pd.read_csv(f"../data/OriginalData/{repo.lower()}_link_raw.csv")
JIRA_API = "https://issues.apache.org/jira/rest/api/2/issue/"

# Rename the column "commitid" to "hash"
dummy_link.rename(columns={"commitid": "hash"}, inplace=True)
tqdm.pandas()

def extract_tracking_id(issue_id, message):
    # Fetch the tracking ID from the issue ID
    url = f"{JIRA_API}/{issue_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        tracking_id = data["key"]
        clean_message = re.sub(rf"\[?{repo}-\d+\]?:?\s*", "", message)
        return pd.Series([str(tracking_id), clean_message, message])
    else:
        return pd.Series([None, message, message])

# if tracking_id column exists, skip this step

dummy_link[["tracking_id", "message", "old_message"]] = dummy_link.progress_apply(
    lambda row: extract_tracking_id(row["issue_id"], row["message"]),
    axis=1
)


# # Iterate through each row in the dataset
for index, row in dummy_link.iterrows():
    # Preprocess the summary, description, and commit message
    summary_processed = preprocessor.preprocessNoCamel(str(row["summary"]).strip("[]"))
    description_processed = preprocessor.preprocessNoCamel(str(row["description"]).strip("[]"))
    message_processed = preprocessor.preprocessNoCamel(str(row["message"]).strip("[]"))

    # Fetch the fix version for the current commit hash using the `get_fix_versions_from_jira` function
    fix_version = get_fix_versions_from_jira(cloned_repo_path, row["hash"], row["old_message"], repo)
    # Combine summary, description, and comment to process issue code

    # Prepare the list for the current row, including the fix_version
    list1 = [
        row["source"], row["product"], row["issue_id"], row["component"], summary_processed,
        description_processed, row["repo"], row["hash"], fix_version, row["tracking_id"],
        message_processed, None, row["label"], row["train_flag"]
    ]

    # Append the processed row to the process list
    process.append(list1)

    # Print progress
    print(f"Processed row {index + 1}/{len(dummy_link)}")
columns = [
    "source", "product", "issue_id", "component", "summary_processed", "description_processed", "repo", "hash", "fix_version",
    "tracking_id", "message_processed", "Diff_processed", "label", "train_flag"
]

# Create a DataFrame from the processed data
df = pd.DataFrame(process, columns=columns)

# Write the DataFrame to a CSV file
df.to_csv(f"../data/Processed{repo.title()}/1_{repo.lower()}_process.csv", index=False)
