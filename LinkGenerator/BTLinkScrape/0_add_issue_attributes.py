import pandas as pd
import requests
from tqdm import tqdm
import time

repo = "TIKA"

df = pd.read_csv(f"../../data/OriginalData/{repo.title()}/{repo.title()}_DEV.csv")
df2 = pd.read_csv(f"../../data/OriginalData/{repo.title()}/{repo.title()}_TRAIN.csv")
df3 = pd.read_csv(f"../../data/OriginalData/{repo.title()}/{repo.title()}_TEST.csv")


combined_df = pd.concat([df, df2, df3], ignore_index=True)
combined_df = combined_df.rename(columns={"label": "target", "Issue_KEY": "tracking_id", "Commit_SHA": "hash", "Issue_Text": "summary", "Commit_Text": "message", "Commit_Code": "Diff"})

JIRA_BASE_URL = "https://issues.apache.org/jira/rest/api/2/issue"

tqdm.pandas()
def extract_issue_description(issue_id):
    time.sleep(0.05)
    url = f"{JIRA_BASE_URL}/{issue_id}"
    response = requests.get(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
    })
    if response.status_code == 200:
        data = response.json()
        description = data["fields"]["description"]
        issueid = data["id"]
        return description, issueid
    else:
        return None

# Create new column "description" and "issueid" and add for each row the description and issueid respectively of the issue
combined_df[["description", "issue_id"]] = combined_df.progress_apply(
    lambda row: extract_issue_description(row["tracking_id"]),
    axis=1,
    result_type="expand"
)
# Save the updated dataframe to a new CSV file
combined_df.to_csv(f"../../data/OriginalData/{repo.title()}/{repo.lower()}_link_raw_merged.csv", index=False)

