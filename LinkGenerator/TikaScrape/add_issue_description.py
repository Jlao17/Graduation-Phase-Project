import pandas as pd
import requests
from tqdm import tqdm

df = pd.read_csv("../../data/OriginalData/Tika/tika_link_raw_merged.csv")

JIRA_BASE_URL = "https://issues.apache.org/jira/rest/api/2/issue"

tqdm.pandas()
def extract_issue_description(issue_id):
    url = f"{JIRA_BASE_URL}/{issue_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        description = data["fields"]["description"]
        return description
    else:
        return None

# Create new column "description" and add for each row the description of the issue
df["description"] = df["tracking_id"].progress_apply(extract_issue_description)

# Save the updated dataframe to a new CSV file
df.to_csv("../../data/OriginalData/Tika/tika_link_raw_merged.csv", index=False)

