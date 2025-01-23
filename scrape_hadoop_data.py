import pandas as pd
import sys
import requests
from bs4 import BeautifulSoup
from constants import GITHUB_TOKEN
from LinkGenerator import preprocessor
import time

maxInt = sys.maxsize

# "3.4.1", "3.4.0", "3.3.6", "3.3.5", "3.3.4", "3.2.4", "2.10.2", "3.3.3", "3.2.3", "3.3.2", "3.3.1","3.2.2", "3.3.0", "2.8.4", "2.9.1", "2.9.2" "2.10.1", "3.2.0", "3.2.1", "3.1.2", "3.2.0", "2.9.2", "3.1.1", "3.0.3", "3.1.0", "3.0.1", "2.9.0",
RELEASE_NOTE_URLS = ["3.4.1", "3.4.0", "3.3.6", "3.3.5", "3.3.4", "3.2.4", "2.10.2", "3.3.3", "3.2.3", "3.3.2", "3.3.1","3.2.2", "3.3.0", "2.8.4", "2.9.1", "2.9.2" "2.10.1", "3.2.0", "3.2.1", "3.1.2", "3.2.0", "2.9.2", "3.1.1", "3.0.3", "3.1.0", "3.0.1", "2.9.0", "3.0.0", "2.8.3", "2.8.2"]
for url in RELEASE_NOTE_URLS:
    JIRA_API = "https://issues.apache.org/jira/rest/api/2/issue"
    RELEASE_NOTE_URL = f"https://hadoop.apache.org/docs/r{url}/hadoop-project-dist/hadoop-common/release/{url}/CHANGELOG.{url}.html"
    RELEASE_NOTE_URL_2 = f"https://hadoop.apache.org/docs/r{url}/hadoop-project-dist/hadoop-common/release/{url}/CHANGES.{url}.html"
    GITHUB_RELEASE_NOTE_API = "https://api.github.com/repos/apache/hadoop/pulls"
    GITHUB_COMMITS_DIFF_API = "https://api.github.com/repos/apache/hadoop/commits"

    data = []
    response = requests.get(RELEASE_NOTE_URL)
    if response.status_code != 200:
        response = requests.get(RELEASE_NOTE_URL_2)

    html_data = response.text
    soup_data = BeautifulSoup(html_data, "html.parser")
    rows = soup_data.find_all('tr', class_=True)  # Find all <tr> tags with a class attribute

    # Process each row
    for index, row in enumerate(rows):
        time.sleep(2.0)
        print(f"Processing row {index}/{len(rows)}")


        # Extract all <td> content
        columns = row.find_all("td")
        # Get each column text
        release_note_data = [column.get_text() for column in columns]
        try:
            release_note = release_note_data[1]
            release_notes_processed = preprocessor.preprocessNoCamel(str(release_note))
        except IndexError:
            print("No release note found")
            continue

        a = row.find("a", href=True)
        try:
            jira_request_url = a["href"]
        except TypeError:
            print("No JIRA link found")
            continue

        jira_id = jira_request_url.split("/")[-1]
        jira_json_response = requests.get(JIRA_API + "/" + jira_request_url.split("/")[-1])
        jira_json_data = jira_json_response.json()

        # Columns of the dataset
        source = "apache"
        product = "HADOOP"
        jira_issue_id = jira_json_data["id"]

        jira_fields = jira_json_data["fields"]

        if len(jira_fields["components"]) > 0:
            # Handle the case where there are multiple commits (or at least one)
            jira_component = jira_fields["components"][0]["name"]  # First commit in the list
        else:
            # Handle the case where there is only one commit (not a list, or directly the commit object)
            jira_component = []

        jira_creator_key = jira_fields["creator"]["key"]

        jira_create_date = jira_fields["created"]

        jira_update_date = jira_fields["updated"]

        jira_last_resolved_date = jira_fields["resolutiondate"]

        jira_comments = jira_fields["comment"]["comments"]

        jira_comments_content = []
        for comment in jira_comments:
            jira_comments_content.append(comment["body"])

        jira_summary = jira_fields["summary"]

        jira_description = jira_fields["description"]

        jira_issue_type = jira_fields["issuetype"]["name"]

        jira_status = jira_fields["status"]["name"]

        jira_fix_versions = []
        for fix_version in jira_fields["fixVersions"]:
            jira_fix_versions.append(fix_version["name"])

        repo = "hadoop"

        # Get the github data from the JIRA issue
        jira_response = requests.get(jira_request_url)
        jira_html_data = jira_response.text
        jira_soup_data = BeautifulSoup(jira_html_data, "html.parser")

        # Find issue details
        issue_details = jira_soup_data.find("ul", id="issuedetails")

        # Get links to pull request from JIRA issue
        issue_links_div = jira_soup_data.find("div", id="linkingmodule")

        # Find the <dt> with title="links to"
        dt_tag = jira_soup_data.find("dt", title="links to")

        # Find the parent <dl> of the <dt> tag
        if dt_tag:
            dl_tag = dt_tag.find_parent("dl")
        else:
            print("No pull request found")
            continue

        # Find pull request id in the JIRA issue
        a_tag = dl_tag.find_all("a")[-1]
        pull_request_url = a_tag["href"]
        if "github" not in pull_request_url:
            print("Different source than github found")
            continue
        pull_request_id = pull_request_url.split("/")[-1]


        github_commits_url = f"{GITHUB_RELEASE_NOTE_API}/{pull_request_id}/commits"

        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}"
        }

        github_commits_response = requests.get(github_commits_url, headers=headers)
        if github_commits_response.status_code == 200:
            github_commits_json_data = github_commits_response.json()
        else:
            continue

        if len(github_commits_json_data) > 0:
            commits = github_commits_json_data
        else:
            # Handle the case where there is only one commit (not a list, or directly the commit object)
            print(github_commits_json_data)
            continue

        for commit in commits:
            commit_hash = commit["sha"]
            github_commit_parent = commit["parents"][0]["sha"]

            github_commit_author = commit["commit"]["author"]["name"]
            github_commit_committers = commit["commit"]["committer"]["name"]
            github_commit_author_date = commit["commit"]["author"]["date"]
            github_commit_committer_date = commit["commit"]["committer"]["date"]
            github_commit_message = commit["commit"]["message"]

            github_changed_files = []
            github_codelist = []
            label = 1
            train_flag = 1

            # Get the diff data
            github_diff_url = f"{GITHUB_COMMITS_DIFF_API}/{commit_hash}"
            headers_diff = {
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github.diff",
            }
            github_diff_response = requests.get(github_diff_url, headers=headers_diff)
            github_diff_data = [github_diff_response.text]

            data.append({"source": source, "product": product, "issue_id": jira_issue_id, "component": jira_component,
                         "creator_key": jira_creator_key, "create_date": jira_create_date,
                         "update_date": jira_update_date, "last_resolved_date": jira_last_resolved_date,
                         "comment": jira_comments_content,
                         "summary": jira_summary, "description": jira_description, "issue_type": jira_issue_type,
                         "status": jira_status, "repo": repo, "commitid": commit_hash, "fix_version": jira_fix_versions,
                         "parents": github_commit_parent,
                         "author": github_commit_author, "committer": github_commit_committers,
                         "author_time_date": github_commit_author_date,
                         "commit_time_date": github_commit_committer_date, "tracking_id": jira_id,
                         "message": github_commit_message, "commit_issue_id": None, "changed_files": github_changed_files,
                         "Diff": github_diff_data, "codelist": github_codelist, "label": label,
                         "train_flag": train_flag, "release_notes_original": release_note, "release_notes": release_notes_processed})

    df = pd.DataFrame(data)
    df.to_csv(f"data/OriginalData/Hadoop/hadoop_link_raw_{url}.csv")
    print(f"Data saved to 'hadoop_link_raw_{url}.csv'")
    # Create a DataFrame from the list of rows with column names: ,Unnamed: 0,source,product,issue_id,component,creator_key,create_date,update_date,last_resolved_date,comment,summary,description,issue_type,status,repo,commitid,parents,author,committer,author_time_date,commit_time_date,message,commit_issue_id,changed_files,Diff,codelist,nosource,label,train_flag
    # Unnamed: 0,source,product,issue_id,component,
    # creator_key,create_date,update_date,last_resolved_date,
    # comment,summary,description,issue_type,status,repo,commitid,
    # parents,author,committer,author_time_date,commit_time_date,
    # message,commit_issue_id,changed_files,Diff,codelist,nosource,label,train_flag
