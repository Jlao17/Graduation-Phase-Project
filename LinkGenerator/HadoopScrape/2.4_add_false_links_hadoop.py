import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("../../data/ProcessedHadoop/2.1_hadoop_link_merged_drop_dup.csv")

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

# Set self-similarity to a high value (to ignore itself)
np.fill_diagonal(sim_df.values, 1.0)

# Extract true links from the dataset
true_links = df[['issue_id', 'hash']]
print(len(true_links))
# Find the least similar issue for each true link
false_links = []
num_false_links = len(true_links)  # 14041 in your case

for issue_id, commit_hash in true_links.itertuples(index=False):
    # Get least similar issues (excluding itself)
    sorted_issues = sim_df.loc[issue_id].sort_values()

    # Exclude the most similar ones and pick a random one from the least similar ones
    least_similar_issues = sorted_issues.iloc[:20].index.tolist()  # Take the least 20 similar issues
    false_issue = random.choice(least_similar_issues)  # Pick one randomly

    false_links.append((false_issue, commit_hash))

# Convert to DataFrame
false_links_df = pd.DataFrame(false_links, columns=['issue_id', 'hash'])
false_links_df['target'] = 0  # Label as false link

print(len(false_links))
# Convert false links to DataFrame
false_links_df = pd.DataFrame(false_links, columns=['issue_id', 'hash'])
false_links_df['target'] = 0  # Label as false link

# Merge false links with issue and commit details
false_links_df = false_links_df.merge(
    df[['issue_id', 'summary_processed', 'description_processed', 'release_notes']].drop_duplicates(),
    on='issue_id',
    how='left'
).merge(
    df[['hash', 'fix_version', 'tracking_id', 'message_processed', 'Diff_processed', 'train_flag']].drop_duplicates(),
    on='hash',
    how='left'
)

# # Remove duplicates after merging (if any)
false_links_df = false_links_df.drop_duplicates()

# Reorder columns
false_links_df = false_links_df[
    ['issue_id', 'summary_processed', 'description_processed', 'hash', 'fix_version', 'tracking_id',
     'message_processed', 'release_notes', 'Diff_processed', 'train_flag','target']]


# Combine false links with original data
final_data = pd.concat([df, false_links_df], ignore_index=True)
final_data.to_csv("../../data/ProcessedHadoop/2.4_hadoop_link_merge_false_rn.csv", index=False)

print("False link generation complete! 🚀")