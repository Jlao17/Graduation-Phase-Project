import ast
from typing import Any
from transformers import AutoModelForMaskedLM, BertForMaskedLM, RobertaModel, RobertaConfig, AutoModel
import pickle
import torch
import pandas as pd
import sys
import csv
import pandas as pd
import numpy as np
maxInt = sys.maxsize

import subprocess

def get_title_from_cloned_repo(repo_path, commit_hash):
    """
    Get the commit message (title) for a given commit hash from a cloned repository.

    :param repo_path: Path to the local cloned repository
    :param commit_hash: The hash of the commit
    :return: Commit message (title)
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
            return result.stdout  # The commit message
        else:
            raise Exception(f"Error: {result.stderr}")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")


# function to find row with hash
def find_row_with_hash(df, hash):
    return df.loc[df['commitid'] == hash]

# Example usage
# repo_path = "C:/Users/Jason/Desktop/isis repo/causeway"  # Update with the path to your cloned repo
# commit_hash = "dc1bb3d74ff3f769ff5b00c3a47c1d74e2bdba3f"  # Replace with your desired commit hash
# subprocess.run(["git", "-C", repo_path, "fetch", "--all"])
# try:
#     title = get_title_from_cloned_repo(repo_path, commit_hash)
#     print(f"Commit Title: {title}")
# except Exception as e:
#     print(f"Failed to get commit title: {e}")

# row = find_row_with_hash(pd.read_csv("./data/OriginalData/isis_link_raw.csv"), "c9e5343516f4dd96e5487b3aa2f92c8f3e654edd")
# print(row["message"])
df = pd.read_csv("./data/ProcessedIsis/1_isis_process.csv")

# print all unique values in df["fix_version"]
print(df["fix_version"].unique())