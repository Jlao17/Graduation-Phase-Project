import ast
from typing import Any

from sklearn.metrics import precision_score, recall_score, f1_score, \
    roc_auc_score, matthews_corrcoef, brier_score_loss, confusion_matrix
from transformers import AutoModelForMaskedLM, BertForMaskedLM, RobertaModel, RobertaConfig, AutoModel
import pickle
import torch
import pandas as pd
import sys
import requests
from bs4 import BeautifulSoup
import regex as re
from constants import GITHUB_TOKEN
import time
import glob
import os
maxInt = sys.maxsize
import subprocess
import numpy as np
import random

# # combine all csv files in the folder
# input_folder = "data/OriginalData/Hadoop"
# input_files_pattern = "hadoop_link_raw*.csv"  # This matches all CSV files with the specified pattern
# output_file = "data/OriginalData/Hadoop/hadoop_link_raw_merged.csv"  # Output file where the merged data will be saved
# def append_csv_files(input_folder, input_files_pattern, output_file):
#     # Create the full path pattern for CSV files
#     input_pattern = os.path.join(input_folder, input_files_pattern)
#
#     # Get a list of all CSV files matching the pattern
#     csv_files = glob.glob(input_pattern)
#
#     # Read and append each file to a list
#     df_list = []
#     for file in csv_files:
#         df = pd.read_csv(file)
#         df_list.append(df)
#
#     # Concatenate all dataframes into one
#     final_df = pd.concat(df_list, ignore_index=True)
#
#     # Save the final concatenated dataframe to the output file
#     final_df.to_csv(output_file, index=False)
#     print(f"All files appended successfully into {output_file}")

# Load your data
df = pd.read_csv("data/ProcessedHadoop/2.5_hadoop_link_false_rn.csv")
print(df['train_flag'].value_counts())
print(df["target"].value_counts())
