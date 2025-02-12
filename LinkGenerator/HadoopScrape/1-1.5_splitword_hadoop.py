import csv
import pandas as pd
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import LinkGenerator.preprocessor as preprocessor
import ast
import re

# Increase CSV field size limit to handle large data
csv.field_size_limit(500 * 1024 * 1024)

# Set language and parser
lang = "java"
LANGUAGE = Language(tsjava.language())
parser = Parser(LANGUAGE)

repo = "HADOOP"
# Initialize an empty list to store processed data
process = []
# Load the CSV file containing the dataset
dummy_link = pd.read_csv(f"../../data/OriginalData/Hadoop/{repo.lower()}_link_raw_merged.csv")

# Rename the column "commitid" to "hash"
dummy_link.rename(columns={"commitid": "hash"}, inplace=True)

def remove_tracking_id(row):
    tracking_id = row['tracking_id']
    text = row['message']
    # Remove tracking_id from text if it exists
    return re.sub(rf'\b{re.escape(tracking_id)}\b', '', text).strip()
# Apply the function to remove tracking_id from text
dummy_link['message'] = dummy_link.apply(remove_tracking_id, axis=1)
# Alternative regular function to remove labels
dummy_link["message"] = dummy_link["message"].apply(lambda x: re.sub(r'^\[?\s*\S+-\d+\s*]?\s*', '', str(x)))
# Remove issue code from commit messages
dummy_link["message"] = dummy_link["message"].apply(lambda x: re.sub(r'\(#\d+\)', '', str(x)).strip())
# Remove Reviewed-by, Signed-off-by, and Contributed by from commit messages
dummy_link["message"] = dummy_link["message"].apply(lambda x: re.sub(r'(Reviewed-by:.*|Signed-off-by:.*|Contributed by.*)', '', str(x), flags=re.IGNORECASE)).str.strip()
# # Iterate through each row in the dataset
for index, row in dummy_link.iterrows():
    # Preprocess the summary, description, and commit message

    summary_processed = preprocessor.preprocessNoCamel(str(row["summary"]).strip("[]"))
    description_processed = preprocessor.preprocessNoCamel(str(row["description"]).strip("[]"))
    message_processed = preprocessor.preprocessNoCamel(str(row["message"]).strip("[]"))
    changed_files = []
    codelist_processed = []
    # Combine summary, description, and comment to process issue code
    issue_text = str(row["summary"]) + str(row["description"]) + str(row["comment"])
    issuecode = preprocessor.getIssueCode(issue_text)

    # Prepare the list for the current row, including the fix_version
    list1 = [
        row["source"], row["product"], row["issue_id"], row["component"],
        row["create_date"], row["update_date"], row["last_resolved_date"], summary_processed,
        description_processed, issuecode, row["issue_type"], row["status"], row["repo"], row["hash"], row["fix_version"],
        row["parents"], row["author"], row["committer"], row["author_time_date"], row["commit_time_date"], row["tracking_id"],
        message_processed, row["commit_issue_id"], changed_files, None, codelist_processed, row["label"],
        row["train_flag"], row["release_notes_original"], row["release_notes"]
    ]

    # Append the processed row to the process list
    process.append(list1)

    # Print progress
    print(f"Processed row {index + 1}/{len(dummy_link)}")

columns = [
    "source", "product", "issue_id", "component", "create_date", "update_date",
    "last_resolved_date", "summary_processed", "description_processed", "issuecode", "issue_type",
    "status", "repo", "hash", "fix_version", "parents", "author", "committer", "author_time_date", "commit_time_date",
    "tracking_id", "message_processed", "commit_issue_id", "changed_files", "Diff_processed", "codelist_processed",
    "label", "train_flag", "release_notes_original", "release_notes"
]

# Create a DataFrame from the processed data
df = pd.DataFrame(process, columns=columns)

# Release notes already added during scraping, so output is saved to 1.5_ instead of 1_ file
df.to_csv(f"../../data/Processed{repo.title()}/1.5_{repo.lower()}_process.csv", index=False)
