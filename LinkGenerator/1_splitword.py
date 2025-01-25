import csv
import re
import pandas as pd
from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
import preprocessor
from extract_commit_version import get_fix_versions_from_jira
import ast
import requests
# Increase CSV field size limit to handle large data
csv.field_size_limit(500 * 1024 * 1024)

# Set language and parser
lang = "java"
LANGUAGE = Language(tsjava.language())
parser = Parser(LANGUAGE)

repo = "CALCITE"
# Initialize an empty list to store processed data
process = []
# Cloned repo path for fetching commit messages
cloned_repo_path = f"C:/Users/Jason/Desktop/{repo.lower()} repo/{repo.lower()}"
# Load the CSV file containing the dataset
dummy_link = pd.read_csv(f"../data/OriginalData/{repo.lower()}_link_raw.csv")
JIRA_API = "https://issues.apache.org/jira/rest/api/2/issue/"

# Rename the column "commitid" to "hash"
dummy_link.rename(columns={"commitid": "hash"}, inplace=True)

def extract_tracking_id(issue_id, message):
    # Fetch the tracking ID from the issue ID
    url = f"{JIRA_API}/{issue_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        tracking_id = data["key"]
        clean_message = re.sub(rf"\[{repo}-\d+]\s*", "", message)
        return pd.Series([str(tracking_id), clean_message, message])
    else:
        return pd.Series([None, message, message])

dummy_link[["tracking_id", "message", "old_message"]] = dummy_link.apply(
    lambda row: extract_tracking_id(row["issue_id"], row["message"]),
    axis=1
)

# # Iterate through each row in the dataset
for index, row in dummy_link.iterrows():
    # Preprocess the summary, description, and commit message
    summary_processed = preprocessor.preprocessNoCamel(str(row["summary"]).strip("[]"))
    description_processed = preprocessor.preprocessNoCamel(str(row["description"]).strip("[]"))
    message_processed = preprocessor.preprocessNoCamel(str(row["message"]).strip("[]"))

    # Process the changed files
    changed_files = []
    cf = ast.literal_eval(row["changed_files"])  # Convert string representation to list safely
    for f in cf:
        f_name = f.split("/")[-1]  # Correctly split by a forward slash
        changed_files.append(f_name)

    # Process the codelist using the parser
    clist = eval(row["codelist"])
    codelist_processed = []
    for code in clist:
        codelist_processed.append(preprocessor.extract_codetoken(code, parser, lang))

    # Fetch the fix version for the current commit hash using the `get_fix_versions_from_jira` function
    fix_version = get_fix_versions_from_jira(cloned_repo_path, row["hash"], row["old_message"], repo)
    # Combine summary, description, and comment to process issue code
    issue_text = str(row["summary"]) + str(row["description"]) + str(row["comment"])
    issuecode = preprocessor.getIssueCode(issue_text)

    # Prepare the list for the current row, including the fix_version
    list1 = [
        row["source"], row["product"], row["issue_id"], row["component"],
        row["create_date"], row["update_date"], row["last_resolved_date"], summary_processed,
        description_processed, issuecode, row["issue_type"], row["status"], row["repo"], row["hash"], fix_version,
        row["parents"], row["author"], row["committer"], row["author_time_date"], row["commit_time_date"], row["tracking_id"],
        message_processed, row["commit_issue_id"], changed_files, None, codelist_processed, row["label"],
        row["train_flag"]
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
    "label", "train_flag"
]

# Create a DataFrame from the processed data
df = pd.DataFrame(process, columns=columns)

# Write the DataFrame to a CSV file
df.to_csv(f"../data/Processed{repo.title()}/1_{repo.lower()}_process.csv", index=False)
