import argparse
import math

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_score, recall_score, f1_score, \
    roc_auc_score, matthews_corrcoef, brier_score_loss, confusion_matrix


def truncate_tokens(first_token, second_token, max_len):
    if len(first_token) + len(second_token) > max_len - 3:
        if len(first_token) > (max_len - 3) / 2 and len(second_token) > (max_len - 3) / 2:
            first_token = first_token[:int((max_len - 3) / 2)]
            second_token = second_token[:max_len - 3 - len(first_token)]
        elif len(first_token) > (max_len - 3) / 2:
            first_token = first_token[:max_len - 3 - len(second_token)]
        elif len(second_token) > (max_len - 3) / 2:
            second_token = second_token[:max_len - 3 - len(first_token)]
    return first_token, second_token

def combine_tokens(tokenizer, first_token, second_token):
    combined_token = [tokenizer.cls_token] + first_token + [tokenizer.sep_token] + second_token + [tokenizer.sep_token]
    return combined_token


class IssueCommitReleaseDataset(Data.Dataset):
    def __init__(self, df, tokenizer=None, max_len=512):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the dataset.
            tokenizer (callable, optional): Tokenizer function for text processing.
            max_len (int, optional): Maximum length for tokenized sequences. Defaults to 512.
            label_pad_value (int, optional): Padding value for labels. Defaults to 3.
        """
        # df_link = MySubSampler(df, 42)
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Extract row
        row = self.df.iloc[idx]

        # Extract fields
        issue_id = int(row['issue_id']) if pd.notnull(row['issue_id']) else 0
        summary = row['summary_processed']
        description = row['description_processed']
        commit_hash = row['hash']
        tracking_id = row['tracking_id']
        message = row['message_processed']
        release_notes = row['release_notes']
        diff = str(row["Diff_processed"]) if pd.notnull(row["Diff_processed"]) else ""
        # num = float(row['num'])
        target = int(row['target'])
        target_rn = int(row['target_rn'])


        if isinstance(release_notes, list):
            release_notes = " ".join(release_notes)  # Join the list if it's a list of strings
        elif not isinstance(release_notes, str):
            release_notes = str(release_notes)  # Convert to string if not already a string

        if isinstance(commit_hash, float) and math.isnan(commit_hash):
            commit_hash = "NaN"

        # Tokenize individual fields
        issue_token = self.tokenizer.tokenize(str(summary))
        issue_description_token = self.tokenizer.tokenize(str(description))
        commit_token = self.tokenizer.tokenize(str(message))
        release_note_token = self.tokenizer.tokenize(str(release_notes))
        code_token = self.tokenizer.tokenize(str(diff))

        # These tokens will be combined later on in the model
        if pd.isna(description):
            issue_token_for_commit, commit_token_for_issue = truncate_tokens(issue_token, commit_token, self.max_len)
            issue_token_for_code, code_token_for_issue = truncate_tokens(issue_token, code_token, self.max_len)
        else:
            issue_token_for_commit, commit_token_for_issue = truncate_tokens(issue_description_token, commit_token, self.max_len)
            issue_token_for_code, code_token_for_issue = truncate_tokens(issue_description_token, code_token, self.max_len)

        issue_token_for_release, release_token_for_issue = truncate_tokens(issue_token, release_note_token, self.max_len)
        description_token_for_release, release_token_for_description = truncate_tokens(issue_description_token, release_note_token, self.max_len)

        issue_commit_input_ids = combine_tokens(self.tokenizer, issue_token_for_commit, commit_token_for_issue)
        issue_commit_combined_ids = self.tokenizer.convert_tokens_to_ids(issue_commit_input_ids)

        if len(issue_commit_combined_ids) < self.max_len:
            padding_length = self.max_len - len(issue_commit_combined_ids)
            issue_commit_combined_ids += [self.tokenizer.pad_token_id] * padding_length

        issue_code_input_ids = combine_tokens(self.tokenizer, issue_token_for_code, code_token_for_issue)
        issue_code_combined_ids = self.tokenizer.convert_tokens_to_ids(issue_code_input_ids)
        if len(issue_code_combined_ids) < self.max_len:
            padding_length = self.max_len - len(issue_code_combined_ids)
            issue_code_combined_ids += [self.tokenizer.pad_token_id] * padding_length

        release_issue_input_ids = combine_tokens(self.tokenizer, issue_token_for_release, release_token_for_issue)
        release_issue_combined_ids = self.tokenizer.convert_tokens_to_ids(release_issue_input_ids)
        if len(release_issue_combined_ids) < self.max_len:
            padding_length = self.max_len - len(release_issue_combined_ids)
            release_issue_combined_ids += [self.tokenizer.pad_token_id] * padding_length

        release_issue_description_input_ids = combine_tokens(self.tokenizer, description_token_for_release, release_token_for_description)
        release_issue_description_combined_ids = self.tokenizer.convert_tokens_to_ids(release_issue_description_input_ids)
        if len(release_issue_description_combined_ids) < self.max_len:
            padding_length = self.max_len - len(release_issue_description_combined_ids)
            release_issue_description_combined_ids += [self.tokenizer.pad_token_id] * padding_length

        # print("issue_commit_combined_ids", len(issue_commit_combined_ids))
        # print("issue_code_combined_ids", len(issue_code_combined_ids))
        # print("release_issue_combined_ids", len(release_issue_combined_ids))
        # print("*" * 100)
        issue_commit_combined_ids = torch.tensor(issue_commit_combined_ids, dtype=torch.long)
        issue_code_combined_ids = torch.tensor(issue_code_combined_ids, dtype=torch.long)
        release_issue_combined_ids = torch.tensor(release_issue_combined_ids, dtype=torch.long)
        release_issue_description_combined_ids = torch.tensor(release_issue_description_combined_ids, dtype=torch.long)

        issue_id = torch.tensor(issue_id, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)
        target_rn = torch.tensor(target_rn, dtype=torch.long)


        # Return all processed fields
        return issue_id, commit_hash, tracking_id, issue_commit_combined_ids, issue_code_combined_ids, release_issue_combined_ids, release_issue_description_combined_ids, target, target_rn

def calculate_metrics(logits_issue_commit, logits_release_issue, target_issue_commit, target_release_issue):
    # Convert logits to predicted class labels for both tasks
    preds_issue_commit = logits_issue_commit.argmax(-1)
    preds_release_issue = logits_release_issue.argmax(-1)

    # Calculate accuracy for both tasks
    eval_acc_issue_commit = np.mean(target_issue_commit == preds_issue_commit)
    eval_acc_release_issue = np.mean(target_release_issue == preds_release_issue)

    # Precision, Recall, F1 score for issue-commit task
    eval_precision_issue_commit = precision_score(target_issue_commit, preds_issue_commit)
    eval_recall_issue_commit = recall_score(target_issue_commit, preds_issue_commit)
    eval_f1_issue_commit = f1_score(target_issue_commit, preds_issue_commit)

    # Precision, Recall, F1 score for release-issue task
    eval_precision_release_issue = precision_score(target_release_issue, preds_release_issue)
    eval_recall_release_issue = recall_score(target_release_issue, preds_release_issue)
    eval_f1_release_issue = f1_score(target_release_issue, preds_release_issue)

    # Matthews Correlation Coefficient (MCC) for both tasks
    eval_mcc_issue_commit = matthews_corrcoef(target_issue_commit, preds_issue_commit)
    eval_mcc_release_issue = matthews_corrcoef(target_release_issue, preds_release_issue)

    # Confusion matrix for both tasks
    print("Issue - commit", confusion_matrix(target_issue_commit, preds_issue_commit).ravel())
    tn_issue_commit, fp_issue_commit, fn_issue_commit, tp_issue_commit = confusion_matrix(target_issue_commit,
                                                                                          preds_issue_commit).ravel()
    print("Release - issue ", confusion_matrix(target_release_issue, preds_release_issue).ravel())
    tn_release_issue, fp_release_issue, fn_release_issue, tp_release_issue = confusion_matrix(target_release_issue,
                                                                                              preds_release_issue).ravel()

    # Calculate Precision Fallout metrics (PF for both tasks)
    eval_pf_issue_commit = fp_issue_commit / (fp_issue_commit + tn_issue_commit) if (fp_issue_commit + tn_issue_commit) != 0 else 0
    eval_pf_release_issue = fp_release_issue / (fp_release_issue + tn_release_issue) if (fp_release_issue + tn_release_issue) != 0 else 0

    # Brier score loss for both tasks
    eval_brier_issue_commit = brier_score_loss(target_issue_commit, preds_issue_commit)
    eval_brier_release_issue = brier_score_loss(target_release_issue, preds_release_issue)

    # # Average loss for both tasks
    # eval_loss_issue_commit = np.mean(logits_issue_commit)  # If you need to compute per-task loss
    # eval_loss_release_issue = np.mean(logits_release_issue)

    tp_total = tp_issue_commit + tp_release_issue
    fp_total = fp_issue_commit + fp_release_issue
    fn_total = fn_issue_commit + fn_release_issue

    precision_micro = tp_total / (tp_total + fp_total)
    recall_micro = tp_total / (tp_total + fn_total)

    f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)

    # Return all the metrics
    return {
        "eval_acc_issue_commit": round(eval_acc_issue_commit, 4),
        "eval_acc_release_issue": round(eval_acc_release_issue, 4),
        "eval_precision_issue_commit": round(eval_precision_issue_commit, 4),
        "eval_recall_issue_commit": round(eval_recall_issue_commit, 4),
        "eval_f1_issue_commit": round(eval_f1_issue_commit, 4),
        "eval_precision_release_issue": round(eval_precision_release_issue, 4),
        "eval_recall_release_issue": round(eval_recall_release_issue, 4),
        "eval_f1_release_issue": round(eval_f1_release_issue, 4),
        "eval_mcc_issue_commit": round(eval_mcc_issue_commit, 4),
        "eval_mcc_release_issue": round(eval_mcc_release_issue, 4),
        "eval_pf_issue_commit": round(eval_pf_issue_commit, 4),
        "eval_pf_release_issue": round(eval_pf_release_issue, 4),
        "eval_brier_issue_commit": round(eval_brier_issue_commit, 4),
        "eval_brier_release_issue": round(eval_brier_release_issue, 4),
        "average_f1": round(f1_micro, 4),
        "confusion_matrix_issue_commit": [tn_issue_commit, fp_issue_commit, fn_issue_commit, tp_issue_commit],
        "confusion_matrix_release_issue": [tn_release_issue, fp_release_issue, fn_release_issue, tp_release_issue]
    }

def MySubSampler(df, x):
    X = df[['summary_processed', 'description_processed', 'message_processed', 'Diff_processed']]
    y1 = df['target']  # Target for issue-commit links
    y2 = df['target_rn']  # Target for issue-release links

    # Step 1: Balance y1 (target) while keeping features (X) intact
    rus1 = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled1, y1_resampled = rus1.fit_resample(X, y1)

    # Create a new DataFrame with the first resampling result
    df_resampled1 = pd.DataFrame(X_resampled1, columns=X.columns)
    df_resampled1['target'] = y1_resampled

    # Merge `y2` back from the original dataset (keeping alignment)
    df_resampled1['target_rn'] = df.loc[df_resampled1.index, 'target_rn']

    # Step 2: Balance y2 (target_rn) while keeping the already balanced y1
    rus2 = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled2, y2_resampled = rus2.fit_resample(df_resampled1.drop(columns=['target_rn']),
                                                   df_resampled1['target_rn'])

    # Create the final balanced dataset
    df_final = pd.DataFrame(X_resampled2, columns=X.columns)
    df_final['target'] = df_resampled1.loc[df_final.index, 'target']
    df_final['target_rn'] = y2_resampled

    # Save the final balanced dataset
    df_final.to_csv("balanced_dataset.csv", index=False)
    return df.sample(frac=1)