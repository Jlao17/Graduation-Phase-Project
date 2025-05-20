import pandas as pd
from nltk import word_tokenize
import nltk.data
import re
import LinkGenerator.preprocessor as preprocessor
from tqdm import tqdm

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = nltk.stem.SnowballStemmer('english')
repo = "HADOOP"
# Define the repo and the specific commit hash you want
# repo_path = "https://github.com/apache/calcite"  # Change to your repo path or  local directory
df = pd.read_csv(f"../data/Processed{repo.title()}/2.5_{repo.lower()}_link_false_rn.csv")
tqdm.pandas()
# def codeMatch(word):
#     identifier_pattern = r'''[A-Zz-z]+[0-9]*_.*
#                             |[A-Za-z]+[0-9]*[\.].+
#                             |[A-Za-z]+.*[A-Z]+.*
#                             |[A-Z0-9]+
#                             |_ +[A-Za-z0-9]+.+
#                             |[a-zA-Z]+[:]{2,}.+
#                             '''
#     identifier_pattern = re.compile(identifier_pattern)
#     if identifier_pattern.match(word):
#         return True
#     else:
#         return False
#
# def diffProcess(text):
#     if not isinstance(text, str) or pd.isna(text):
#         return ' '  # Return empty space or other default value
#     sentences = tokenizer.tokenize(text)
#     for sentence in sentences:
#         word_tokens = word_tokenize(sentence)
#         # if a token is 2 characters or less, remove it
#         word_tokens = [word for word in word_tokens if len(word) > 2]
#         print(word_tokens)
#         return word_tokens


df['Diff_processed'] = df['Diff_processed'].progress_apply(preprocessor.processCode)
df.to_csv(f"../data/Processed{repo.title()}/3_{repo.lower()}_link_final_diff_processed.csv", index=False)