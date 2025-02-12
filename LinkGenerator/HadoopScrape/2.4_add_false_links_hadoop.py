import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("../../data/ProcessedHadoop/2_hadoop_link_merged.csv")

# Extract necessary columns
issue_texts = df[['issue_id', 'summary_processed', 'description_processed']].drop_duplicates()
issue_texts['combined_text'] = issue_texts['summary_processed'].fillna('') + " " + issue_texts[
    'description_processed'].fillna('')

# Convert issue texts into a dictionary for easy lookup
issue_text_dict = dict(zip(issue_texts['issue_id'], issue_texts['combined_text']))

# Vectorize issue descriptions using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(issue_texts['combined_text'])

# Compute pairwise cosine similarities between all issues
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Convert cosine similarity matrix into a DataFrame
issue_ids = issue_texts['issue_id'].tolist()
sim_df = pd.DataFrame(cosine_sim_matrix, index=issue_ids, columns=issue_ids)
print(sim_df)
# Set self-similarity to a high value (to ignore itself)
np.fill_diagonal(sim_df.values, 1.0)

# Extract true links from the dataset
true_links = df[['issue_id', 'hash']].drop_duplicates()

# Find the least similar issue for each true link
false_links = []
for issue_id, commit_hash in true_links.itertuples(index=False):
    # Find the issue with the lowest cosine similarity (i.e., least similar issue)
    least_similar_issue = sim_df.loc[issue_id].idxmin()

    # Generate a false link by linking the commit to the least similar issue
    false_links.append((least_similar_issue, commit_hash))

print(false_links)
# Convert false links to DataFrame
false_links_df = pd.DataFrame(false_links, columns=['issue_id', 'hash'])
false_links_df['target'] = 0  # Label as false link

# Merge false links with issue and commit details
false_links_df = false_links_df.merge(
    df[['issue_id', 'summary_processed', 'description_processed', 'issuecode', 'release_notes']].drop_duplicates(),
    on='issue_id',
    how='left'
).merge(
    df[['hash', 'fix_version', 'tracking_id', 'message_processed', 'changed_files', 'codelist_processed', 'Diff_processed', 'train_flag']].drop_duplicates(),
    on='hash',
    how='left'
)

# Remove duplicates after merging (if any)
false_links_df = false_links_df.drop_duplicates(subset=['issue_id', 'hash'])

# Reorder columns
false_links_df = false_links_df[
    ['issue_id', 'summary_processed', 'description_processed', 'issuecode', 'hash', 'fix_version', 'tracking_id',
     'message_processed', 'changed_files', 'codelist_processed', 'release_notes', 'Diff_processed', 'train_flag',
     'target']]


# Combine false links with original data
final_data = pd.concat([df, false_links_df], ignore_index=True)
final_data.to_csv("../../data/ProcessedHadoop/2_hadoop_link_merged.csv", index=False)

print("False link generation complete! 🚀")