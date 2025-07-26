import requests
import re
from constants import GITHUB_TOKEN
import time
import subprocess

# GitHub API settings
GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_REPO_OWNER = "apache"

# JIRA API settings
JIRA_BASE_URL = "https://issues.apache.org/jira/rest/api/2/issue"


def get_title_from_github(repo_name, commit_hash):
    time.sleep(1)
    url = f"{GITHUB_API_BASE_URL}/repos/{GITHUB_REPO_OWNER}/{repo_name}/commits/{commit_hash}"

    headers = {}
    if GITHUB_TOKEN:
        headers['Authorization'] = f"token {GITHUB_TOKEN}"

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        commit_data = response.json()
        commit_message = commit_data['commit']['message']
        return commit_message
    else:
        raise Exception(f"Error fetching commit from GitHub: {response.status_code}, {response.text}")


def extract_jira_issue_key(commit_message, repo):
    jira_issue_pattern = re.compile(rf"{repo}-\d+")
    match = jira_issue_pattern.search(commit_message)
    if match:
        return match.group(0)
    else:
        raise ValueError("No JIRA issue key found in the commit message")


def get_fix_versions_from_jira(commit_message, repo):
    # if cloned:
    #     commit_message = get_title_from_cloned_repo(repo_name, hash)
    # else:
    #     commit_message = get_title_from_github(repo_name, hash)
    #
    # if commit_message == "nan":
    #     return []
    try:
        issue_key = extract_jira_issue_key(commit_message, repo)
    except ValueError as e:
        print(f"Error: {e}")
        issue_key = "nan"

    url = f"{JIRA_BASE_URL}/{issue_key}"

    response = requests.get(url)

    if response.status_code == 200:
        issue_data = response.json()
        fix_versions = issue_data['fields']['fixVersions']
        return [version['name'] for version in fix_versions]
    else:
        print(f"Error fetching issue from JIRA: {response.status_code}, {response.text}")
        return []


def get_title_from_cloned_repo(repo_path, commit_hash):
    """
    Get the commit message (title) for a given commit hash from a cloned repository.

    :param repo_path: Path to the local cloned repository
    :param commit_hash: The hash of the commit
    :return: Commit message (title), or None if the commit is not found
    """
    try:
        # Run the git show command to extract the commit message
        result = subprocess.run(
            ["git", "-C", repo_path, "show", "--quiet", "--pretty=format:%B", commit_hash],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()  # Return the commit message
        else:
            # Log an error message for debugging
            error_message = f"Error fetching commit {commit_hash}: {result.stderr.strip()}"
            print(error_message)
            return "nan"
    except Exception as e:
        # Handle unexpected exceptions gracefully
        error_message = f"An unexpected error occurred: {e}"
        print(error_message)
        return "nan"


def main():
    # Test the function
    commit_ids = ["c016e6440bed51057e83d5178c769f582d685e11", "92978d981e6ca2eb352c1654d14c7b64fb8710fd",
                  "406372f1274a6a0c9fe2b471ce6d65669e798633", "81960613e7750a9191280719352ae941a7d6a22d",
                  "7273a972e4cc514702f4797bd57ab8af929bc726"]
    for commit_id in commit_ids:
        try:
            fix_versions = get_fix_versions_from_jira("calcite", commit_id)
            print(fix_versions)
        except Exception as e:
            continue


if __name__ == "__main__":
    main()
