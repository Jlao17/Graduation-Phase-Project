import git
import pandas as pd
from tqdm import tqdm  # Progress bar

# Initialize the repository object
repo_path = "C:/Users/Jason/Desktop/hadoop/hadoop"
repo = git.Repo(repo_path)

# Load Data
df = pd.read_csv("../../data/ProcessedHadoop/1.5_hadoop_process.csv")

# Step 1: Pre-fetch all commits into a DataFrame
commit_data = [{"message": commit.message.strip(), "hash": commit.hexsha} for commit in repo.iter_commits()]
commit_df = pd.DataFrame(commit_data)

# Step 2: Define function to search for a matching commit hash
def find_commit_hash(query):
    result = commit_df[commit_df["message"].str.contains(query, case=False, na=False)]
    hash = result.iloc[0]["hash"] if not result.empty else None  # Return first matching commit hash
    message = result.iloc[0]["message"] if not result.empty else None  # Return first matching commit message
    return hash, message


# Step 3: Apply the search function with a progress bar
tqdm.pandas()  # Enable progress tracking
df[["hash", "message_processed"]] = df["tracking_id"].progress_apply(find_commit_hash).tolist()

# Save the updated file
df.to_csv("../../data/ProcessedHadoop/1.5_hadoop_process.csv", index=False)

print("Processing complete! ✅")