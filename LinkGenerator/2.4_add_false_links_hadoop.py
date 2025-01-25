import pandas as pd
import sys
import numpy as np
import random

maxInt = sys.maxsize

df = pd.read_csv("../data/ProcessedHadoop/2_hadoop_link_merged.csv")

# Extract issue_ids and commit_hashes
issue_ids = df['issue_id'].tolist()
commit_hashes = df['hash'].tolist()

# Generate existing links
existing_links = set(zip(df['issue_id'], df['hash']))

# Generate all possible pairs (Cartesian product of issue_ids and commit_hashes)
all_possible_pairs = set((issue_id, commit_hash) for issue_id in issue_ids for commit_hash in commit_hashes)

# Filter out existing links to get non-link candidates
non_link_candidates = all_possible_pairs - existing_links

# Randomly sample non-links
num_non_links = len(existing_links)  # Match the number of existing links
for i in range(len(df)):
    print(f"Warning: Only {len(non_link_candidates)} unique non-links can be generated.")
non_links = random.sample(non_link_candidates, min(num_non_links, len(non_link_candidates)))
print(non_links)
# Convert non-links to DataFrame
non_linked_data = pd.DataFrame(non_links, columns=['issue_id', 'hash'])
non_linked_data['target'] = 0

# Merge non-links with issue and commit details
non_linked_data_df = non_linked_data.merge(
    df[["issue_id", "summary_processed", "description_processed", "issuecode"]].drop_duplicates(),
    on='issue_id',
    how='left'
).merge(
    df[["hash", "fix_version", "tracking_id", "message_processed", "changed_files", "codelist_processed", "release_notes", "Diff_processed", "train_flag"]].drop_duplicates(),
    on='hash',
    how='left'
)

# Remove duplicates after merging (if any)
non_linked_data_df = non_linked_data_df.drop_duplicates(subset=['issue_id', 'hash'])
# Reorder columns
non_linked_data_df = non_linked_data_df[['issue_id', 'summary_processed', 'description_processed', 'issuecode', 'hash', 'fix_version', 'tracking_id', 'message_processed', 'changed_files', 'codelist_processed', 'release_notes', 'Diff_processed', 'train_flag', 'target']]
# Save non-links to CSV
# non_linked_data_df.to_csv("data/ProcessedHadoop/hadoop_link_false_rn.csv", index=False)

# Combine non-links with original data
final_data = pd.concat([df, non_linked_data_df], ignore_index=True)
final_data.to_csv("../data/ProcessedHadoop/2.4_hadoop_link_false.csv", index=False)