import ast

import pandas as pd
import sys
import numpy as np

maxInt = sys.maxsize
# Load your data
df = pd.read_csv("../data/ProcessedHadoop/2.5_hadoop_link_false_rn.csv")
df['train_flag'] = 1

# Ensure reproducibility
np.random.seed(42)

# Assign train_flag: 80% for training and 20% for testing
df['train_flag'] = np.random.choice([1, 0], size=len(df), p=[0.8, 0.2])
print(df['train_flag'].value_counts())
print(len(df))
# remove empty lists inside a list message_rpocessed
df['message_processed'] = df['message_processed'].apply(lambda x: ast.literal_eval(x))
df['message_processed'] = df['message_processed'].apply(lambda x: [y for y in x if y])

df.to_csv("../data/ProcessedHadoop/2.5_hadoop_link_false_rn.csv", index=False)