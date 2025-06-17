from __future__ import absolute_import
from __future__ import print_function

import argparse
import math
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import (RobertaConfig, RobertaModel,
                          get_linear_schedule_with_warmup)

from train_test_utils import IssueCommitReleaseDataset, calculate_metrics
from model import Multi_Model

warnings.simplefilter(action='ignore', category=FutureWarning)

train_dataset_project = "HITNC_COMBINED" # Project name of dataset to train the model
test_dataset_project = "GIRAPH" # Project name of dataset to test the model
model_project = "HITNC_COMBINED"         # Project name of dataset of the model on which the model was trained

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_args():
    train_result_dir = "../models/Results/Train"
    test_result_dir = f"../models/Results/Test/{model_project.title()}Model"
    output_dir = "../models/"
    

    parser = argparse.ArgumentParser(description="EALink.py")
    parser.add_argument("--end_epoch", type=int, default=10,
                        help="Epoch to stop training.")

    parser.add_argument("--tra_batch_size", type=int, default=2,
                        help="Batch size set during training")

    parser.add_argument("--val_batch_size", type=int, default=16,
                        help="Batch size set during predicting")

    parser.add_argument("--tes_batch_size", type=int, default=8,
                        help="Batch size set during predicting")
    parser.add_argument("--output_model", type=str, default='',
                        help="The path to save model")
    parser.add_argument("--train_result_dir", type=str, default=train_result_dir,
                        help="The path to save training results")
    parser.add_argument("--test_result_dir", type=str, default=test_result_dir,
                        help="The path to save testing results")
    parser.add_argument("--output_dir", type=str, default=output_dir,
                        help="The path to save model")
    parser.add_argument("--pro", type=str, default=f"{train_dataset_project}",
                        help="The path to save model")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    return args


def custom_collate_fn(batch):
    batch_dict = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            # Stack tensors
            batch_dict[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            # Keep non-tensor fields as a list
            batch_dict[key] = [item[key] for item in batch]
    return batch_dict


def train_model(args, model, tokenizer):
    # df = pd.read_csv(f'../data/Processed{train_dataset_project.title()}/3_{train_dataset_project.lower()}_link_final.csv')
    df = pd.read_csv("../data/CombinedProjects/hadoop_tika_netbeans_calcite_df.csv")
    torch.set_grad_enabled(True)
    train_df_sum = df.loc[df['train_flag'] == 1]
    cnt = len(train_df_sum)/4
    train_df = train_df_sum.loc[:cnt*3-1]
    valid_df = train_df_sum.loc[cnt*3:]

    train_dataset = IssueCommitReleaseDataset(train_df, tokenizer=tokenizer)
    valid_dataset = IssueCommitReleaseDataset(valid_df, tokenizer=tokenizer)
    dfScores = pd.DataFrame(columns=['Epoch', 'Metrics', 'Score'])

    # Training settings
    train_sampler = RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.tra_batch_size, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.val_batch_size, num_workers=4)

    no_decay = ['bias', 'LayerNorm.weight']

    # Define parameter groups with their respective learning rates
    param_groups = {
        'issue_commit': ([], 1e-5),
        'release_issue': ([], 1e-6),
        'code_encoder': ([], 5e-7),
        'text_encoder': ([], 5e-7),
        'other': ([], 1e-4)
    }

    # Categorize parameters based on their name
    for name, param in model.named_parameters():
        assigned = False
        for key in param_groups.keys():
            if key in name:
                param_groups[key][0].append((name, param))
                assigned = True
                break
        if not assigned:
            param_groups['other'][0].append((name, param))  # For any additional layers

    # Construct optimizer parameter groups
    optimizer_grouped_parameters = []

    for key, (params, lr) in param_groups.items():
        optimizer_grouped_parameters.append(
            {
                'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                'lr': lr,
                'weight_decay': 1e-3
            }
        )
        optimizer_grouped_parameters.append(
            {
                'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                'lr': lr,
                'weight_decay': 0.0
            }
        )


    # Initialize optimizer
    optimizer = optim.AdamW(optimizer_grouped_parameters)
    max_steps = len(train_dataloader) * args.end_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1, num_training_steps=max_steps)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5, total_steps=max_steps)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-6)
    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    print("********** Running training **********")
    print("  Num examples = {}".format(len(train_dataset)))
    print("  Num Epochs = {}".format(args.end_epoch))
    print("  batch size = {}".format(args.tra_batch_size))
    print("  Total optimization steps = {}".format(max_steps))
    best_f = 0.0

    bar = tqdm(train_dataloader, total=len(train_dataloader))
    model.train()
    for idx in range(args.end_epoch):
        losses = []
        for step, batch in enumerate(bar):
            issue_id, commit_hash, tracking_id, issue_commit_ids, issue_code_ids, release_issue_ids, release_desc_ids, target, target_rn = batch

            issue_commit_inputs = issue_commit_ids.to(args.device)
            issue_code_inputs = issue_code_ids.to(args.device)
            release_issue_inputs = release_issue_ids.to(args.device)
            release_description_inputs = release_desc_ids.to(args.device)
            issue_commit_target = target.to(args.device)
            release_issue_target = target_rn.to(args.device)

            # Autocast and scaler are used to speed up training process and reduce memory usage
            # Use autocast to enable mixed precision during the forward pass
            with autocast(device_type='cuda', dtype=torch.float16):  # Use FP16 for mixed precision
                loss = model(issue_commit_inputs, issue_code_inputs, release_issue_inputs, release_description_inputs,
                             issue_commit_target, release_issue_target)

            # Scale the loss to prevent underflow in mixed precision
            scaler.scale(loss).backward()

            # Clip gradients to prevent exploding gradients (optional)
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)

            # Update the model parameters with scaler.step() to handle mixed precision
            scaler.step(optimizer)

            # Update the scaler for the next iteration
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()  # Adjust learning rate

            losses.append(loss.item())
            bar.set_description("epoch {} loss {}".format(idx, round(float(np.mean(losses)), 3)))


        results = evaluate_model(args, model, valid_dataloader)
        results.update({"train_loss": round(float(np.mean(losses)), 4)})
        for key, value in results.items():
            if isinstance(value, list):  # Leave lists unchanged
                print('-' * 10 + f"  {key} = {value}")
            elif isinstance(value, np.ndarray):  # Leave numpy arrays unchanged
                print('-' * 10 + f"  {key} = {value}")
            else:
                print('-' * 10 + f"  {key} = {round(value, 4)}")

        for key in sorted(results.keys()):
            if isinstance(results[key], list):  # Leave lists unchanged
                print('-' * 10 + f"  {key} = {results[key]}")
                dfScores.loc[len(dfScores)] = [idx, key, results[key]]  # Store array as is
            elif isinstance(results[key], np.ndarray):  # Leave numpy arrays unchanged
                print('-' * 10 + f"  {key} = {results[key]}")
                dfScores.loc[len(dfScores)] = [idx, key, results[key]]  # Store array as is
            else:
                rounded_value = round(results[key], 4)
                print('-' * 10 + f"  {key} = {rounded_value}")
                dfScores.loc[len(dfScores)] = [idx, key, str(rounded_value)]

        eval_f1 = results["average_f1"]
        best_f = eval_f1
        print("  " + "*" * 20)
        print("  Best f1: {}".format(round(best_f, 4)))
        print("  " + "*" * 20)
        checkpoint_prefix = args.pro + '_checkpoint_epochs'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir = os.path.join(output_dir, '{}'.format(f'model_epoch-{str(idx)}.bin'))
        torch.save(model, output_dir)
        print("Saving model checkpoint to {}".format(output_dir))
        dfScores.loc[len(dfScores)] = [idx, '___best___', '___best___']
        if not os.path.exists(args.train_result_dir):
            os.makedirs(args.train_result_dir)
        dfScores.to_csv(os.path.join(args.train_result_dir, args.pro + "_Epoch_Metrics.csv"), index=False)



def evaluate_model(args, model, eval_dataset):
    stime = datetime.now()

    # Evaluation settings
    print("***** Running evaluation *****")
    print("  Num examples = {}".format(len(eval_dataset)))
    print("  Batch size = {}".format(args.val_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    bar = tqdm(eval_dataset, total=len(eval_dataset))

    logits_issue_commit_list = []  # Store logits for issue-commit task
    logits_release_issue_list = []  # Store logits for release-issue task
    target_issue_commit_list = []  # Store true labels for issue-commit
    target_release_issue_list = []  # Store true labels for release-issue

    for step, batch in enumerate(bar):
        # Unpack batch
        with torch.no_grad():
            issue_id, commit_hash, tracking_id, issue_commit_ids, issue_code_ids, release_issue_ids, release_desc_ids, target, target_rn = batch

            issue_commit_inputs = issue_commit_ids.to(args.device)
            issue_code_inputs = issue_code_ids.to(args.device)
            release_issue_inputs = release_issue_ids.to(args.device)
            release_description_inputs = release_desc_ids.to(args.device)
            issue_commit_target = target.to(args.device)
            release_issue_target = target_rn.to(args.device)

            loss, logits_issue_commit, logits_release_issue = model(issue_commit_inputs, issue_code_inputs, release_issue_inputs,
                                                                    release_description_inputs, issue_commit_target,release_issue_target, mode='eval')

            eval_loss += loss.mean().item()
            logits_issue_commit_list.append(logits_issue_commit.cpu().numpy())
            logits_release_issue_list.append(logits_release_issue.cpu().numpy())
            target_issue_commit_list.append(issue_commit_target.cpu().numpy())
            target_release_issue_list.append(release_issue_target.cpu().numpy())
        nb_eval_steps += 1

    # Calculate metrics
    logits_issue_commit_list = np.concatenate(logits_issue_commit_list, axis=0)
    logits_release_issue_list = np.concatenate(logits_release_issue_list, axis=0)
    target_issue_commit_list = np.concatenate(target_issue_commit_list, axis=0)
    target_release_issue_list = np.concatenate(target_release_issue_list, axis=0)
    # print("Predictions (Issue Commit):", logits_issue_commit_list[:10])  # First 10 predictions
    # print("Targets (Issue Commit):", target_issue_commit_list[:10])
    # print("Predictions (Release - Issue):", logits_release_issue_list[:10])  # First 10 predictions
    # print("Targets (Release - Issue):", target_release_issue_list[:10])
    etime = datetime.now()
    eval_time = (etime - stime).seconds
    metrics = calculate_metrics(logits_issue_commit_list, logits_release_issue_list, target_issue_commit_list,
                                target_release_issue_list)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = math.exp(eval_loss)

    result = {
        "eval_loss": round(eval_loss, 4),
        "perplexity": round(perplexity, 4),
        "eval_time": float(eval_time),
        "predictions_issue_commit": logits_issue_commit_list[:25],
        "target_issue_commit": target_issue_commit_list[:25],
        "predictions_release_issue": logits_release_issue_list[:25],
        "target_release_issue": target_release_issue_list[:25]

    }

    result.update(metrics)  # Add task-specific metrics to the result dictionary

    return result


def test_model(args, model, tokenizer, df, test_dataset_project, model_project):
    print(f"Testing model: {model_project} on dataset: {test_dataset_project}")
    stime = datetime.now()
    # Replace NaN values with an empty string for all columns except the excluded ones
    test_result_dir = f"../models/Results/Test/{model_project.title()}Model"

    test_dataset = IssueCommitReleaseDataset(df, tokenizer=tokenizer)

    test_dataloader = DataLoader(test_dataset, batch_size=args.tes_batch_size,
                                 num_workers=4, shuffle=True)

    # Evaluation settings
    print("***** Running testing*****")
    print("  Num examples = {}".format(len(test_dataset)))
    print("  Batch size = {}".format(args.tes_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    bar = tqdm(test_dataloader, total=len(test_dataloader))

    logits_issue_commit_list = []  # Store logits for issue-commit task
    logits_release_issue_list = []  # Store logits for release-issue task
    target_issue_commit_list = []  # Store true labels for issue-commit
    target_release_issue_list = []  # Store true labels for release-issue
    issue_ids = []
    commit_hashes = []
    release_note_ids = []

    for step, batch in enumerate(bar):
        # Unpack batch
        with torch.no_grad():
            issue_id, commit_hash, tracking_id, issue_commit_ids, issue_code_ids, release_issue_ids, release_desc_ids, target, target_rn = batch

            issue_commit_inputs = issue_commit_ids.to(args.device)
            issue_code_inputs = issue_code_ids.to(args.device)
            release_issue_inputs = release_issue_ids.to(args.device)
            release_description_inputs = release_desc_ids.to(args.device)
            issue_commit_target = target.to(args.device)
            release_issue_target = target_rn.to(args.device)
            issue_ids.extend(issue_id)
            commit_hashes.extend(commit_hash)
            release_note_ids.extend(tracking_id)

            loss, logits_issue_commit, logits_release_issue = model(issue_commit_inputs, issue_code_inputs, release_issue_inputs,
                                                                    release_description_inputs, issue_commit_target, release_issue_target, mode='eval')


            eval_loss += loss.mean().item()
            logits_issue_commit_list.append(logits_issue_commit.cpu().numpy())
            logits_release_issue_list.append(logits_release_issue.cpu().numpy())
            target_issue_commit_list.append(issue_commit_target.cpu().numpy())
            target_release_issue_list.append(release_issue_target.cpu().numpy())
        nb_eval_steps += 1

    # Calculate metrics
    logits_issue_commit_list = np.concatenate(logits_issue_commit_list, axis=0)
    logits_release_issue_list = np.concatenate(logits_release_issue_list, axis=0)
    target_issue_commit_list = np.concatenate(target_issue_commit_list, axis=0)
    target_release_issue_list = np.concatenate(target_release_issue_list, axis=0)
    preds_issue_commit = logits_issue_commit_list.argmax(-1)
    preds_release_issue = logits_release_issue_list.argmax(-1)

    etime = datetime.now()
    eval_time = (etime - stime).seconds
    metrics = calculate_metrics(logits_issue_commit_list, logits_release_issue_list, target_issue_commit_list,
                                target_release_issue_list)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = math.exp(eval_loss)

    result = {
        "eval_loss": round(eval_loss, 4),
        "perplexity": round(perplexity, 4),
        "eval_time": float(eval_time),

    }

    result.update(metrics)  # Add task-specific metrics to the result dictionary
    print("********** Prediction - Target Issue Commit **********")
    print(preds_issue_commit[:25], target_issue_commit_list[:25])
    print("********** Prediction - Target Release Issue **********")
    print(preds_release_issue[:25], target_release_issue_list[:25])
    print("********** Test results **********")
    dfScores = pd.DataFrame(columns=['Metrics', 'Score'])

    for key in sorted(result.keys()):
        if isinstance(result[key], list):
            dfScores.loc[len(dfScores)] = [key, result[key]]
        else:
            print('-' * 10 + "  {} = {}".format(key, str(round(result[key], 4))))
            dfScores.loc[len(dfScores)] = [key, str(round(result[key], 4))]

    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)

    dfScores.to_csv(os.path.join(test_result_dir, f"{test_dataset_project.title()}_Test_Metrics.csv"), index=False)

    assert len(logits_issue_commit_list) == len(target_issue_commit_list) and len(logits_issue_commit_list) == len(
        preds_issue_commit), 'error'
    assert len(logits_release_issue_list) == len(target_release_issue_list) and len(logits_release_issue_list) == len(
        preds_release_issue), 'error'

    logits4class0_issue_commit, logits4class1_issue_commit, logits4class0_relnote_issue, logits4class1_relnote_issue = \
        [logits_issue_commit_list[iclass][0] for iclass in range(len(logits_issue_commit_list))], \
        [logits_issue_commit_list[iclass][1] for iclass in range(len(logits_issue_commit_list))], \
        [logits_release_issue_list[iclass][0] for iclass in range(len(logits_release_issue_list))], \
        [logits_release_issue_list[iclass][1] for iclass in range(len(logits_release_issue_list))]

    df = pd.DataFrame(np.transpose([issue_ids, commit_hashes, release_note_ids, logits4class0_issue_commit,
                                    logits4class1_issue_commit, preds_issue_commit, target_issue_commit_list,
                                    logits4class0_relnote_issue, logits4class1_relnote_issue, preds_release_issue,
                                    target_release_issue_list]),
                      columns=["Issue_Key", "Commit_SHA", "Release Note Label", "0_logit issue commit",
                               "1_logit issue commit", "preds issue commit", "labels issue commit",
                               "0_logit release issue", "1_logit release issue", "preds release issue",
                               "labels release issue"])
    df.to_csv(os.path.join(test_result_dir, f"{test_dataset_project.title()}_Test_Prediction.csv"), index=False)

    return result



def main():
    args = get_args()
    print(torch.cuda.is_available())
    set_seed(args.seed)

    codeBert_config = RobertaConfig.from_pretrained("microsoft/codebert-base")
    codeBert_config.num_labels = 2
    codeBert = AutoModel.from_pretrained("microsoft/codebert-base", config=codeBert_config)

    roberta_config = RobertaConfig.from_pretrained("distilbert/distilroberta-base")
    roberta_config.num_labels = 2
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
    roberta = AutoModel.from_pretrained("distilbert/distilroberta-base", config=roberta_config)

    # Continue training from a checkpoint
    # output_dir = "../models/HTNC_COMBINED_checkpoint_epochs/model_epoch-0.bin"
    # model = torch.load(output_dir)

    # Train model
    # model = Multi_Model(roberta, codeBert, roberta_config.hidden_size, codeBert_config.hidden_size)
    # train_model(args, model, tokenizer)
    # args.seed += 3

    # output_dir = f"../models/{model_project}_checkpoint_epochs/model_epoch-0.bin"
    # model = torch.load(output_dir)
    # model.to(args.device)

    # Load and test trained model
    giraph_df = pd.read_csv(f"../data/ProcessedGiraph/3_giraph_link_final.csv")

    output_dir = f"../models/Old_Models/model_epoch-9HITNC.bin"
    model = torch.load(output_dir)
    model.to(args.device)
    model.eval()

    test_model(args, model, tokenizer, giraph_df, "GIRAPH", "HITNC_SOFT_SHARE_4_8")

if __name__ == '__main__':
    main()