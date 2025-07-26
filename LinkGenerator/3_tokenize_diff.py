import pandas as pd
from nltk import word_tokenize
import nltk.data
import re
import LinkGenerator.preprocessor as preprocessor
from tqdm import tqdm

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = nltk.stem.SnowballStemmer('english')
repo = "HADOOP"

df = pd.read_csv(f"../data/Processed{repo.title()}/2.5_{repo.lower()}_link_false_rn.csv")
tqdm.pandas()

df['Diff_processed'] = df['Diff_processed'].progress_apply(preprocessor.processCode)
df.to_csv(f"../data/Processed{repo.title()}/3_{repo.lower()}_link_final_diff_processed.csv", index=False)
