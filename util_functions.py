import ast
import random
import os
import re

import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch
from click.formatting import iter_rows
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import DataLoader
from transformers import AutoModel, RobertaConfig, AutoTokenizer

from LinkGenerator import preprocessor
from LinkGenerator.preprocessor import preprocessNoCamel
from models.model import Multi_Model
import seaborn as sns
import numpy as np
import transformers

from models.train_test_utils import IssueCommitReleaseDataset

"""
Combine datasets functions
"""
#
# df = pd.read_csv("data/OriginalData/Giraph/Giraph_DEV.csv")
# for index, row in df.iterrows():
#     if pd.isna(row["description_processed"]):  # or pd.isnull(row["description_processed"])
#         print(row)
# release_notes_list = df["release_notes"].tolist()
# # remove 'nan' from the list
# release_notes_list = [x for x in release_notes_list if str(x) != 'nan']
# # add random release notes to df['release_notes'] where df["release_notes"] == 'nan'
# df['release_notes'] = df['release_notes'].apply(lambda x: random.choice(release_notes_list) if str(x) == 'nan' else x)
# df.to_csv("data/ProcessedGiraph/3_giraph_link_final_false_rn.csv", index=False)
# df = pd.read_csv("data/ProcessedGiraph/3_giraph_link_final.csv") # G
# df6 = pd.read_csv("data/ProcessedHadoop/3_hadoop_link_final.csv") # H
# df3 = pd.read_csv("data/ProcessedIsis/3_isis_link_final.csv") # I
# df4 = pd.read_csv("data/ProcessedTika/3_tika_link_final.csv") # T
# df5 = pd.read_csv("data/ProcessedNetbeans/3_netbeans_link_final.csv") # N
# df2 = pd.read_csv("data/ProcessedCalcite/3_calcite_link_final.csv") # C
#
#
# df_combined = pd.concat([df6, df3, df4, df5, df2], ignore_index=True)
# # print(df_combined)
# df_combined.to_csv("data/CombinedProjects/[HITNC]hadoop_isis_tika_netbeans_calcite.csv", index=False)

# df_pre = pd.read_csv("data/ProcessedGiraph/3_giraph_link_final.csv")
# df6_pre = pd.read_csv("data/ProcessedHadoop/2_hadoop_link_merged.csv")
# df3_pre = pd.read_csv("data/ProcessedIsis/2_isis_link_merged.csv")
# df4_pre = pd.read_csv("data/ProcessedTika/3_tika_link_final.csv")
# df5_pre = pd.read_csv("data/ProcessedNetbeans/2_netbeans_link_merged.csv")
# df2_pre = pd.read_csv("data/ProcessedCalcite/2_calcite_link_merged.csv")

# df_test = pd.read_csv("data/ProcessedHadoop/2.4_hadoop_link_merge_false_rn.csv")
# df123 = df6_pre.drop_duplicates()
# print(df6_pre["target"].value_counts())
# print(df_test["target"].value_counts())



# #
# df2_pre["target_rn"] = df2_pre["release_notes"].apply(lambda x: 0 if pd.isna(x) else 1).astype(int)
# df2_pre.to_csv("data/ProcessedCalcite/2_calcite_link_merge_false_rn.csv", index=False)
#
# df3_pre["target_rn"] = df3_pre["release_notes"].apply(lambda x: 0 if pd.isna(x) else 1).astype(int)
# df3_pre.to_csv("data/ProcessedIsis/2_isis_link_merge_false_rn.csv", index=False)
#
# df5_pre["target_rn"] = df5_pre["release_notes"].apply(lambda x: 0 if pd.isna(x) else 1).astype(int)
# df5_pre.to_csv("data/ProcessedNetbeans/2_netbeans_link_merge_false_rn.csv", index=False)



"""
Find value counts of `target_rn` for each dataset
"""

# df_hadoop_fix = df6_pre[df6_pre["target"] == 1]
# df_hadoop_fix.to_csv("data/ProcessedHadoop/2_hadoop_link_merged.csv")
# print(df6["target_rn"].value_counts())
# print(df3["target_rn"].value_counts())
# print(df4["target_rn"].value_counts())
# print(df5["target_rn"].value_counts())
# print(df2["target_rn"].value_counts())


"""
Read release notes and count the number of release notes
"""
# rl = pd.read_csv("data/ReleaseNotes/Calcite/release_notes_calcite.csv")
# rl2 = pd.read_csv("data/ReleaseNotes/Calcite/release_notes_calcite_avatica.csv")
# rl3 = pd.read_csv("data/ReleaseNotes/Calcite/release_notes_calcite_avatica_go.csv")
# rl_giraph = pd.read_csv("data/ReleaseNotes/Giraph/release_notes_giraph.csv")
# rl_isis = pd.read_csv("data/ReleaseNotes/Isis/release_notes_isis_merged.csv")
# rl_netbeans = pd.read_csv("data/ReleaseNotes/Netbeans/release_notes_netbeans.csv")
# rl_tika = pd.read_csv("data/ReleaseNotes/Tika/release_notes_tika.csv")


# total_rl = 0
# for index, row in rl.iterrows():
#     count = len(ast.literal_eval(row["content"]))
#     total_rl += count
# print(total_rl)
# for index, row in rl2.iterrows():
#     count = len(ast.literal_eval(row["content"]))
#     total_rl += count
# print(total_rl)
# for index, row in rl3.iterrows():
#     count = len(ast.literal_eval(row["content"]))
#     total_rl += count

"""
Print number of unique entries for each dataset
"""
# df = pd.read_csv("data/OriginalData/calcite_link_raw.csv")
# # df_isis = pd.read_csv("data/OriginalData/isis_link_raw.csv")
# df_hadoop = pd.read_csv("data/OriginalData/Hadoop/hadoop_link_raw_merged.csv")
# # df_false = pd.read_csv("data/ProcessedHadoop/2_hadoop_link_merged.csv")
# df_netbeans = pd.read_csv("data/OriginalData/netbeans_link_raw.csv")
# df_tika = pd.read_csv("data/ProcessedTika/3_tika_link_final.csv")
# print(df_tika["target_rn"].value_counts())
#Print unique amount of issue_id and commitid
# print(df_hadoop["issue_id"].nunique())
# print(df_hadoop["commitid"].nunique())
# print(df_hadoop["release_notes_original"].nunique())


"""
Find model with best average f1 score
"""
#
# directory = 'Epochs'  # set directory path
#
# for entry in os.scandir(directory):
#     if entry.is_file():  # check if it's a file
#         df = pd.read_csv(entry.path)
#         f1_df = df[df["Metrics"] == "average_f1"]
#         best_epoch = f1_df.loc[f1_df["Score"].idxmax()]
#
#         print(f"Best Epoch for {entry.name}: {best_epoch['Epoch']}, Best F1 Score: {best_epoch['Score']}")

"""
Print the scores for overleaf table format
"""
# result_directory = "Results_HPC/Metrics_Hard"
# for entry in os.scandir(result_directory):
#     if entry.is_file():  # check if it's a file
#         df = pd.read_csv(entry.path)
#         f1_ic = df[df["Metrics"] == "eval_f1_release_issue"]["Score"].iloc[0]
#         recall_ic = df[df["Metrics"] == "eval_recall_release_issue"]["Score"].iloc[0]
#         precision_ic = df[df["Metrics"] == "eval_precision_release_issue"]["Score"].iloc[0]
#         mcc_ic = df[df["Metrics"] == "eval_mcc_release_issue"]["Score"].iloc[0]
#         auc_ic = df[df["Metrics"] == "eval_auc_release_issue"]["Score"].iloc[0] if "eval_auc_release_issue" in df["Metrics"].values else 0
#         acc_ic = df[df["Metrics"] == "eval_acc_release_issue"]["Score"].iloc[0]
#         pf_ic = df[df["Metrics"] == "eval_pf_release_issue"]["Score"].iloc[0]
#
#         print(
#             f"{entry.name} & {float(f1_ic) * 100:.2f} & {float(recall_ic) * 100:.2f} & {float(precision_ic) * 100:.2f} & {float(mcc_ic) * 100:.2f} & {float(auc_ic) * 100:.2f} & {float(acc_ic) * 100:.2f} & {float(pf_ic) * 100:.2f}")


"""
Plot confusion matrices
"""

# result_directory = "Results_HPC/Metrics_Soft"
# for entry in os.scandir(result_directory):
#     if entry.is_file():  # check if it's a file
#         group_names = ["True Negative", "False Positive", "False Negative", "True Positive"]
#         df = pd.read_csv(entry.path)
#
#         matrix_ic = df[df["Metrics"] == "confusion_matrix_release_issue"]["Score"].iloc[0]
#         arr = ast.literal_eval(matrix_ic)
#         reshaped_matrix = np.array(arr).reshape(2, 2)  # Reshape to 2x2
#         group_counts = ["{0:0.0f}".format(value) for value in reshaped_matrix.flatten()]
#         group_percentages = ["{0:.2%}".format(value) for value in reshaped_matrix.flatten() / np.sum(reshaped_matrix)]
#         labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
#         labels = np.asarray(labels).reshape(2, 2)
#         data_name = entry.name.split('_')[0]
#         sns.heatmap(reshaped_matrix, annot=labels, fmt="", cmap='Oranges')
#         # plt.show()
#         plt.savefig(f"Confusion Matrices/Release-Issue Soft/{data_name}_Release_Issue_Matrix_Soft", bbox_inches='tight')
#         plt.clf()


"""
Find False Positives of datasets
"""
# import pandas as pd
#
# # Load the original issue commit data
# df_issues_commits = pd.read_csv("data/ProcessedNetbeans/3_netbeans_link_final.csv")
#
# # Load the prediction results
# df_predictions = pd.read_csv("Results_HPC/Predictions_Hard/Netbeans_Test_Prediction_Hard.csv")
#
# # Step 1: Filter df_predictions to find false positives (preds issue commit = 1 and labels issue commit = 0)
# false_positives = df_predictions[(df_predictions['preds issue commit'] == 1) & (df_predictions['labels issue commit'] == 0)]
# # Keep Issue_Key and Commit_SHa only
# false_positives = false_positives[['Issue_Key', 'Commit_SHA']]
# false_positives['Issue_Key'] = false_positives['Issue_Key'].str.replace('tensor(', '').str.replace(')', '')
#
# # Find the corresponding issue commit pairs in the original data Issue_Key = issue_id, Commit_SHA = hash
# # Assuming 'df' is the dataframe where you want to find the matching row
# # Assuming 'df' is the dataframe where you want to find the matching row
# false_positives['Issue_Key'] = false_positives['Issue_Key'].astype(str)
# false_positives['Commit_SHA'] = false_positives['Commit_SHA'].astype(str)
#
# df_issues_commits['issue_id'] = df_issues_commits['issue_id'].astype(str)
# df_issues_commits['hash'] = df_issues_commits['hash'].astype(str)
#
# # Now perform the merge
# result = pd.merge(false_positives, df_issues_commits, left_on=['Issue_Key', 'Commit_SHA'], right_on=['issue_id', 'hash'], how='inner')
# # remove duplicate rows in result
# result = result.drop_duplicates()
# result.to_csv("Netbeans_False_Positives_Hard.csv", index=False)
# print(result)


"""
Find False Negatives of datasets
"""
# import pandas as pd
#
# # Load the original issue commit data
# df_issues_commits = pd.read_csv("data/ProcessedIsis/3_isis_link_final.csv")
#
# # Load the prediction results
# df_predictions = pd.read_csv("Results_HPC/Predictions_Hard/Isis_Test_Prediction_Hard.csv")
#
# # Step 1: Filter df_predictions to find false positives (preds issue commit = 1 and labels issue commit = 0)
# false_positives = df_predictions[(df_predictions['preds issue commit'] == 0) & (df_predictions['labels issue commit'] == 1)]
# # Keep Issue_Key and Commit_SHa only
# false_positives = false_positives[['Issue_Key', 'Commit_SHA']]
# false_positives['Issue_Key'] = false_positives['Issue_Key'].str.replace('tensor(', '').str.replace(')', '')
#
# # Find the corresponding issue commit pairs in the original data Issue_Key = issue_id, Commit_SHA = hash
# # Assuming 'df' is the dataframe where you want to find the matching row
# # Assuming 'df' is the dataframe where you want to find the matching row
# false_positives['Issue_Key'] = false_positives['Issue_Key'].astype(str)
# false_positives['Commit_SHA'] = false_positives['Commit_SHA'].astype(str)
#
# df_issues_commits['issue_id'] = df_issues_commits['issue_id'].astype(str)
# df_issues_commits['hash'] = df_issues_commits['hash'].astype(str)
#
# # Now perform the merge
# result = pd.merge(false_positives, df_issues_commits, left_on=['Issue_Key', 'Commit_SHA'], right_on=['issue_id', 'hash'], how='inner')
# # remove duplicate rows in result
# result = result.drop_duplicates()
# result.to_csv("Isis_False_Negatives_Hard.csv", index=False)
# print(result)

"""
Calculate the similarity between issue and commit messages using FastText
"""
# import fasttext.util
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from tqdm import tqdm
#
# tqdm.pandas()
# # Load Data
# df_giraph_fp = pd.read_csv("Hadoop_False_Positives_Soft.csv")
#
# # Fill missing values to avoid AttributeError
# df_giraph_fp[['description_processed', 'message_processed']] = df_giraph_fp[['description_processed', 'message_processed']].fillna("")
#
# # Download and Load FastText Model
# fasttext.util.download_model('en', if_exists='ignore')
# ft = fasttext.load_model('cc.en.300.bin')
#
# # Function to get sentence embedding
# def fasttext_sentence_embedding(text):
#     if not isinstance(text, str):  # Ensure text is a string
#         text = ""
#
#     words = text.split()  # Tokenize into words
#     word_vectors = [ft.get_word_vector(word) for word in words if word in ft]
#
#     if len(word_vectors) == 0:  # Handle empty cases
#         return np.zeros(ft.get_dimension())
#
#     return np.mean(word_vectors, axis=0)  # Average word vectors
#
# # Function to compute FastText similarity
# def fasttext_text_similarity(text1, text2):
#     vec1, vec2 = fasttext_sentence_embedding(text1), fasttext_sentence_embedding(text2)
#     return cosine_similarity([vec1], [vec2])[0][0]  # Cosine similariy
#
# df_giraph_fp["issue_commit_sim"] = df_giraph_fp.progress_apply(
#     lambda x: fasttext_text_similarity(
#         str(x["summary_processed"]) + " " + str(x["description_processed"]),  # Concatenate summary + description
#         str(x["message_processed"]) + " " + str(x["Diff_processed"])  # Concatenate message + diff
#     ),
#     axis=1
# )

##Display updated DataFrame
# df_giraph_fp.to_csv("Hadoop_False_Positives_Soft_Sim.csv", index=False)

"""
Plot the distribution of similarity scores
"""
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Load data
# df_fp_sim = pd.read_csv("Tika_False_Positives_Soft_Sim.csv")
#
# # Define bin edges (10 bins from 0.0 to 1.0)
# bins = np.linspace(0, 1, 11)  # 11 edges for 10 bins
#
# # Plot histogram
# plt.hist(df_fp_sim['issue_commit_sim'], bins=bins, edgecolor='black', alpha=0.7)
#
# # Set xticks at the bin edges
# plt.xticks(bins)
#
# # Labels and title
# plt.xlabel('Scores')
# plt.ylabel('Frequency')
# plt.savefig("Tika_Similarity_Scores_Histogram.png", bbox_inches='tight')
# # Show plot
# plt.show()

# df = pd.read_csv("Tika_False_Positives_Soft_Sim.csv")
# df = df[df["issue_commit_sim"] > 0.5]
# df.to_csv("Tika_False_Positives_Soft_Sim_0.5.csv", index=False)

# df = pd.read_csv("Giraph_False_Positives_Hard_Sim_0.5.csv").sample()

"""
Captum
"""

# import torch
# from captum.attr import IntegratedGradients
# from captum.attr import visualization as viz
# import matplotlib.pyplot as plt
# from models.utils import IssueCommitReleaseDataset  # Replace with your actual import
# from torch.utils.data import DataLoader
# from tqdm import tqdm
#
#
# class ExplanationWrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#
#     def forward(self, issue_inputs, code_inputs):
#         # Convert to long inside the forward pass
#         issue_long = issue_inputs.long()
#         code_long = code_inputs.long()
#
#         # Create dummy inputs for the unused task
#         batch_size = issue_long.size(0)
#         device = issue_long.device
#         dummy_release = torch.zeros_like(issue_long)
#         dummy_target = torch.zeros(batch_size, dtype=torch.long).to(device)
#
#         # Forward pass - get only the task we want to explain
#         total_loss, issue_logits, _, _, _ = self.model(
#             issue_commit_inputs=issue_long,
#             issue_code_inputs=code_long,
#             release_issue_inputs=dummy_release,
#             release_description_inputs=dummy_release,
#             issue_commit_target=dummy_target,
#             release_issue_target=dummy_target,
#             mode='eval'
#         )
#
#         # Return only the positive class probability (scalar output)
#         return issue_logits[:, 1]  # Positive class logits only
#
#
# def explain_issue_commit_link(model_path, tokenizer, data_csv_path, device="cuda", num_samples=3):
#     # Load model
#     original_model = torch.load(model_path, map_location=device)
#     model = ExplanationWrapper(original_model).to(device)
#     model.eval()
#
#     # Initialize Integrated Gradients
#     ig = IntegratedGradients(model)
#
#     # Load data
#     df = pd.read_csv(data_csv_path)
#     dataset = IssueCommitReleaseDataset(df, tokenizer=tokenizer)
#     loader = DataLoader(dataset, batch_size=1, shuffle=False)
#
#     results = []
#     for i, batch in enumerate(tqdm(loader, desc="Explaining samples")):
#         if i >= num_samples:
#             break
#
#         issue_id, commit_hash, _, issue_input, code_input, _, _, target, _ = batch
#
#         # Prepare inputs
#         issue_input = issue_input.to(device).float().requires_grad_(True)
#         code_input = code_input.to(device).float().requires_grad_(True)
#
#         # Compute attributions
#         attrs_issue, attrs_code = ig.attribute(
#             inputs=(issue_input, code_input),
#             baselines=(
#                 torch.full_like(issue_input, tokenizer.pad_token_id),
#                 torch.full_like(code_input, tokenizer.pad_token_id)
#             ),
#             n_steps=1,
#             internal_batch_size=1
#         )
#
#         # Get prediction
#         with torch.no_grad():
#             pred_prob = torch.sigmoid(model(issue_input, code_input)).item()
#             prediction = 1 if pred_prob > 0.5 else 0
#
#         results.append({
#             'issue_id': issue_id.item(),
#             'commit_hash': commit_hash[0],
#             'issue_attrs': attrs_issue.cpu().detach().numpy(),
#             'code_attrs': attrs_code.cpu().detach().numpy(),
#             'prediction': prediction,
#             'label': target.item()
#         })
#
#     return results
#
# codeBert_config = RobertaConfig.from_pretrained("microsoft/codebert-base")
# codeBert_config.num_labels = 2
# codeBert = AutoModel.from_pretrained("microsoft/codebert-base", config=codeBert_config)
# roberta_config = RobertaConfig.from_pretrained("distilbert/distilroberta-base")
# roberta_config.num_labels = 2
# tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
# roberta = AutoModel.from_pretrained("distilbert/distilroberta-base", config=roberta_config)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# output_dir = "model_epoch-6.bin"
#
# results = explain_issue_commit_link(
#     model_path=output_dir,
#     tokenizer=tokenizer,
#     data_csv_path="data/ProcessedTika/3_tika_link_final.csv",
#     num_samples=1,
#     device=device
# )
#
# # Print results
# for result in results:
#     print(f"\nIssue {result['issue_id']} - Commit {result['commit_hash']}")
#     print(f"Predicted: {result['prediction']}, Actual: {result['label']}")
#
#     # Visualize important tokens
#     print("\nImportant Issue Tokens:")
#     for token, attr in zip(tokenizer.convert_ids_to_tokens(issue_input[0]), result['issue_attrs']):
#         if token not in ['[PAD]', '[CLS]', '[SEP]']:
#             print(f"{token:20s}: {attr:+.4f}")

"""
Release Note Labels Fix
"""


from LinkGenerator import preprocessor
from rapidfuzz import process, fuzz
from tqdm import tqdm
import pandas as pd

# # Enable progress bar
# tqdm.pandas()
#
# # Load data
# df = pd.read_csv("data/ProcessedTika/3_tika_link_final.csv")
# rl = pd.read_csv("tika_release_notes_extracted.csv")
#
# # Matching pool
# rl_notes = rl["release_note"].tolist()
# rl_label_map = dict(zip(rl["release_note"], rl["label"]))
#
# # Only false links
# false_links = df[df["target_rn"] == 0].copy()
#
# # Matching function with RapidFuzz
# def get_best_label(note, choices, label_map):
#     match = process.extractOne(note, choices, scorer=fuzz.ratio)
#     best_match, score, _ = match if match else (None, 0, None)
#     return label_map[best_match] if best_match else None
#
# # Apply with progress bar
# false_links["tracking_id"] = false_links["release_notes"].progress_apply(
#     lambda x: get_best_label(x, rl_notes, rl_label_map)
# )
#
# # Update and save
# df.update(false_links)
# df.to_csv("data/ProcessedTika/3.5_tika_link_fuzzy_matched.csv", index=False)



# Extract release notes seperately from the combined release notes
# df = pd.read_csv("data/ReleaseNotes/Netbeans/release_notes_netbeans.csv")
# rows = []
#
# for rn_list in df["content"].dropna():
#     # Convert from string to list if necessary
#     if isinstance(rn_list, str):
#         try:
#             rn_items = ast.literal_eval(rn_list)
#         except:
#             rn_items = [rn_list]  # Fallback if it's just a string
#     else:
#         rn_items = rn_list
#
#     for note in rn_items:
#         match = re.search(r"(NETBEANS-\d+)\s*(.*)", note)
#         if match:
#             label = match.group(1)
#             text = match.group(2)
#             processed_text = preprocessor.preprocessNoCamel(str(text))
#             rows.append({"label": label, "release_note": processed_text})
#
# # Create new DataFrame
# rn_df = pd.DataFrame(rows)
# rn_df.drop_duplicates(inplace=True)
# # Save to CSV
# rn_df.to_csv("netbeans_release_notes_extracted.csv", index=False)

# df = pd.read_csv("data/ReleaseNotes/Tika/release_notes_tika.csv")
#
# for index, row in df.iterrows():
#     row["content"] = preprocessNoCamel(row["content"])
#
# df.rename(columns={"content": "release_note", "tracking_id": "label"}, inplace=True)
#
# df.to_csv("tika_release_notes_extracted.csv", index=False)

# df = pd.read_csv("data/ProcessedHadoop/3.5_hadoop_link_final_diff_processed.csv")
# df = df[df["hash"].notna()]
# df["message_processed"] = df["message_processed"].apply(lambda x: re.sub(r'^\[?\s*\S+-\d+\s*]?\s*', '', str(x)))
# df["message_processed"] = df["message_processed"].apply(lambda x: preprocessor.preprocessNoCamel(str(x)))
# # Remove all rows where hash = nan/null
#
#
# df.to_csv("3.5_hadoop_link_final_remove_nan.csv", index=False)

# df = pd.read_csv("3.5_hadoop_link_final_remove_nan.csv")
#
# print(df["target"].value_counts())
#
# print(df["target_rn"].value_counts())

"""
Create visualizations for predictions (venn)
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from matplotlib.patches import Patch

# df = pd.read_csv("Results_HPC/Predictions_Soft/Calcite_Test_Prediction_Soft.csv")
#
# issue_unique = df["Issue_Key"].dropna().unique()
# commit_unique = df["Commit_SHA"].dropna().unique()
# release_unique = df["Release Note Label"].dropna().unique()
#
# issue_commit_pairs = df[df["labels issue commit"] == 1][["Issue_Key", "Commit_SHA"]].dropna().drop_duplicates() # A intersect B
# issue_rn_pairs = df[df["labels release issue"] == 1][["Issue_Key", "Release Note Label"]].dropna().drop_duplicates() # A intersect C
# commit_rn_pairs = pd.merge(issue_commit_pairs, issue_rn_pairs, on="Issue_Key")[["Commit_SHA", "Release Note Label"]].drop_duplicates() # B intersect C
#
# # get the issues not in the issue_commit_pairs
# issue_not_in_commit_rn = set(issue_unique) - set(issue_commit_pairs["Issue_Key"].unique()) - set(issue_rn_pairs["Issue_Key"].unique()) # A
# commit_not_in_issue_rn = set(commit_unique) - set(issue_commit_pairs["Commit_SHA"].unique()) - set(commit_rn_pairs["Commit_SHA"].unique()) # B
# release_not_in_issue_commit = set(release_unique) - set(issue_rn_pairs["Release Note Label"].unique()) - set(commit_rn_pairs["Release Note Label"].unique()) # C
#
#
# issue_commit_rn_triples = (
#     pd.merge(issue_commit_pairs, issue_rn_pairs, on="Issue_Key")
#     .drop_duplicates()
# ) # A intersect B intersect C
#
# venn_sizes = (
#     len(issue_not_in_commit_rn),
#     len(commit_not_in_issue_rn),
#     len(issue_commit_pairs),
#     len(release_not_in_issue_commit),
#     len(issue_rn_pairs),
#     len(commit_rn_pairs),
#     len(issue_commit_rn_triples)
# )
#
# print(venn_sizes)
#
# v = venn3(
#     subsets=venn_sizes,
#     set_labels=("Issues", "Commits", "Release Notes"),
#     set_colors=("blue", "green", "red"),
#     alpha=0.7
# )
#
# # Manually set the correct intersection colors
# v.get_patch_by_id("100").set_color("blue")    # Issues only
# v.get_patch_by_id("010").set_color("green")   # Commits only
# v.get_patch_by_id("001").set_color("red")     # Release Notes only
# v.get_patch_by_id("110").set_color("#0b7bd0") # Blue+Green (darker cyan)
# v.get_patch_by_id("101").set_color("#a000c8") # Blue+Red (purple)
# v.get_patch_by_id("011").set_color("#c88000") # Green+Red (orange-brown)
# v.get_patch_by_id("111").set_color("#804040") # All three (dark brown)
#
# # Create accurate legend
# legend_patches = [
#     Patch(facecolor="blue", label="Issues only"),
#     Patch(facecolor="green", label="Commits only"),
#     Patch(facecolor="red", label="Release Notes only"),
#     Patch(facecolor="#0b7bd0", label="Issues ∩ Commits"),
#     Patch(facecolor="#a000c8", label="Issues ∩ Release Notes"),
#     Patch(facecolor="#c88000", label="Commits ∩ Release Notes (inferred)"),
#     Patch(facecolor="#804040", label="All three linked")
# ]
#
# plt.legend(
#     handles=legend_patches,
#     title="Link Types:",
#     bbox_to_anchor=(1.05, 1),
#     loc="upper left"
# )
#
# plt.title("Artifact Links Ground Truth", pad=20)
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib_venn import venn3
# from matplotlib.patches import Patch
#
# df = pd.read_csv("Results_HPC/Predictions_Soft/Calcite_Test_Prediction_Soft.csv")
#
# issue_unique = df["Issue_Key"].dropna().unique()
# commit_unique = df["Commit_SHA"].dropna().unique()
# release_unique = df["Release Note Label"].dropna().unique()
#
# issue_commit_pairs = df[df["preds issue commit"] == 1][["Issue_Key", "Commit_SHA"]].dropna().drop_duplicates() # A intersect B
# issue_rn_pairs = df[df["preds release issue"] == 1][["Issue_Key", "Release Note Label"]].dropna().drop_duplicates() # A intersect C
# commit_rn_pairs = pd.merge(issue_commit_pairs, issue_rn_pairs, on="Issue_Key")[["Commit_SHA", "Release Note Label"]].drop_duplicates() # B intersect C
#
# issue_not_in_commit_rn = set(issue_unique) - set(issue_commit_pairs["Issue_Key"].unique()) - set(issue_rn_pairs["Issue_Key"].unique()) # A
# commit_not_in_issue_rn = set(commit_unique) - set(issue_commit_pairs["Commit_SHA"].unique()) - set(commit_rn_pairs["Commit_SHA"].unique()) # B
# release_not_in_issue_commit = set(release_unique) - set(issue_rn_pairs["Release Note Label"].unique()) - set(commit_rn_pairs["Release Note Label"].unique()) # C
#
# # triplets are same as commit-rn intersection at the moment
# issue_commit_rn_triples = (
#     pd.merge(issue_commit_pairs, issue_rn_pairs, on="Issue_Key")
#     .drop_duplicates()
# ) # A intersect B intersect C
#
# venn_sizes = (
#     len(issue_not_in_commit_rn),
#     len(commit_not_in_issue_rn),
#     len(issue_commit_pairs),
#     len(release_not_in_issue_commit),
#     len(issue_rn_pairs),
#     len(commit_rn_pairs),
#     len(issue_commit_rn_triples)
# )
#
# print(venn_sizes)
#
# v = venn3(
#     subsets=venn_sizes,
#     set_labels=("Issues", "Commits", "Release Notes"),
#     set_colors=("blue", "green", "red"),
#     alpha=0.7
# )
#
# # Manually set the correct intersection colors
# v.get_patch_by_id("100").set_color("blue")    # Issues only
# v.get_patch_by_id("010").set_color("green")   # Commits only
# v.get_patch_by_id("001").set_color("red")     # Release Notes only
# v.get_patch_by_id("110").set_color("#0b7bd0") # Blue+Green (darker cyan)
# v.get_patch_by_id("101").set_color("#a000c8") # Blue+Red (purple)
# v.get_patch_by_id("011").set_color("#c88000") # Green+Red (orange-brown)
# v.get_patch_by_id("111").set_color("#804040") # All three (dark brown)
#
# # Create accurate legend
# legend_patches = [
#     Patch(facecolor="blue", label="Issues only"),
#     Patch(facecolor="green", label="Commits only"),
#     Patch(facecolor="red", label="Release Notes only"),
#     Patch(facecolor="#0b7bd0", label="Issues ∩ Commits"),
#     Patch(facecolor="#a000c8", label="Issues ∩ Release Notes"),
#     Patch(facecolor="#c88000", label="Commits ∩ Release Notes (inferred)"),
#     Patch(facecolor="#804040", label="All three linked")
# ]
#
# plt.legend(
#     handles=legend_patches,
#     title="Link Types:",
#     bbox_to_anchor=(1.05, 1),
#     loc="upper left"
# )
#
# plt.title("Artifact Links Predictions", pad=20)
# plt.tight_layout()
# plt.show()

"""
Process data for cosmograph
"""
import pandas as pd

df = pd.read_csv("models/Results/Test/Hitnc_Soft_Share_4_8Model/Giraph_Test_Prediction.csv")

# Clean possible tensor strings
def clean(val):
    if isinstance(val, str) and val.startswith("tensor("):
        return val.replace("tensor(", "").replace(")", "")
    return val

for col in ['Release Note Label', 'Issue_Key', 'Commit_SHA']:
    df[col] = df[col].apply(clean)

edges = []

for _, row in df.iterrows():
    release = row['Release Note Label']
    issue = row['Issue_Key']
    commit = row['Commit_SHA']

    if row['preds release issue'] == 1 and pd.notna(release) and pd.notna(issue):
        edges.append({'source': release, 'target': issue, 'type': 'release_issue'})
    if row['preds issue commit'] == 1 and pd.notna(issue) and pd.notna(commit):
        edges.append({'source': issue, 'target': commit, 'type': 'issue_commit'})

# Get all nodes from each column
release_nodes = pd.Series(df['Release Note Label'].dropna().unique(), name='id')
issue_nodes = pd.Series(df['Issue_Key'].dropna().unique(), name='id')
commit_nodes = pd.Series(df['Commit_SHA'].dropna().unique(), name='id')

all_nodes = pd.concat([release_nodes, issue_nodes, commit_nodes]).drop_duplicates()

# Get all linked node IDs
linked_ids = pd.Series([e['source'] for e in edges] + [e['target'] for e in edges]).dropna().unique()

# Find unlinked nodes and add them with type-specific targets
for node in all_nodes[~all_nodes.isin(linked_ids)]:
    if node in release_nodes.values:
        edges.append({'source': node, 'target': 'unlinked_release_notes', 'type': 'unconnected'})
    elif node in issue_nodes.values:
        edges.append({'source': node, 'target': 'unlinked_issues', 'type': 'unconnected'})
    elif node in commit_nodes.values:
        edges.append({'source': node, 'target': 'unlinked_commits', 'type': 'unconnected'})

cosmo_df = pd.DataFrame(edges)

cosmo_df.to_csv("cosmograph_edges_and_nodes_giraph.csv", index=False)

"""
Create meta data for cosmograph
"""

df = pd.read_csv("models/Results/Test/Hitnc_Soft_Share_4_8Model/Giraph_Test_Prediction.csv")

# Clean possible tensor strings
def clean(val):
    if isinstance(val, str) and val.startswith("tensor("):
        return val.replace("tensor(", "").replace(")", "")
    return val

for col in ['Release Note Label', 'Issue_Key', 'Commit_SHA']:
    df[col] = df[col].apply(clean)

node_metadata = []

# Colors for each type of artifact
type_colors = {
    'Issue': '#4c8ccf',
    'Commit': '#4e9f70',
    'Release Note': '#e67e22'
}


# Loop through the nodes and assign type, color, and size
for issue in df['Issue_Key'].dropna().unique():
    node_metadata.append({'id': issue, 'color': type_colors['Issue']})

for commit in df['Commit_SHA'].dropna().unique():
    node_metadata.append({'id': commit, 'color': type_colors['Commit']})

for release in df['Release Note Label'].dropna().unique():
    node_metadata.append({'id': release, 'color': type_colors['Release Note']})

metadata_df = pd.DataFrame(node_metadata)

metadata_df.to_csv("cosmograph_metadata_giraph.csv", index=False)

print("Metadata file created successfully!")

"""
Create interactive node diagram
"""
# from pyvis.network import Network
# import pandas as pd
#
# # Load your data
# df = pd.read_csv('Results_HPC/Predictions_Soft/Calcite_Test_Prediction_Soft.csv')
#
# # subset random
# df = df.sample(n=1000, random_state=42)
#
#
# # Create a network
# net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', directed=True)
#
# # Add nodes and edges
# for _, row in df.iterrows():
#     net.add_node(row['Issue_Key'], title=row['Issue_Key'], color='lightblue', shape='dot', group='issue')
#
#     if row['preds issue commit'] == 1:
#         net.add_node(row['Commit_SHA'], title=row['Commit_SHA'], color='lightgreen', shape='square', group='commit')
#         net.add_edge(row['Issue_Key'], row['Commit_SHA'])
#
#     if row['preds release issue'] == 1:
#         net.add_node(row['Release Note Label'], title=row['Release Note Label'], color='pink', shape='diamond',
#                      group='release_note')
#         net.add_edge(row['Release Note Label'], row['Issue_Key'])
#
# # Configure physics for better layout
# net.toggle_physics(True)
# net.show_buttons(filter_=['physics'])
#
# # Save and show
# net.show('artifact_network.html', notebook=False)
