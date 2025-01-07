#分开运行 处理分词
import csv
# from keras.preprocessing.text import Tokenizer
import re

import pandas as pd
from tree_sitter import Language, Parser
import preprocessor
import tree_sitter_python as tspython
import tree_sitter_java as tsjava

repo = "NETBEANS"
# from tkinter import _flatten
csv.field_size_limit(500 * 1024 * 1024)
lang = "java"

LANGUAGE = Language(tsjava.language())
parser = Parser(LANGUAGE)

newlist = []
dummy_link = pd.read_csv(f"../data/OriginalData/{repo.lower()}_link_raw.csv")
for index, row in dummy_link.iterrows():
    labelist = []
    Diff_processed = []
    difflist = eval(row["Diff"])
    tg = row["label"]
    num = len(difflist)
    if tg == 0:
        labelist = [0]*num
    elif tg==1:
        text = str(row["comment"]) + str(row["summary"]) + str(row["description"])
        text = text.lower()
        cf = eval(row["changed_files"])
        len1 = len(cf)
        if len1 == num:
            for i in range(0,len1):
                func_name = cf[i].split(".")[0].split("/")[-1].lower()
                if text.find(func_name) != -1:
                    labelist.append(1)
                else:
                    labelist.append(0)
        else:
            labelist = [1]*num

    # Preprocess Diff code
    for d in difflist:
        diff = re.sub(r"\<ROW.[0-9]*\>", "", str(d))
        diff = re.sub(r"\<CODE.[0-9]*\>", "", diff)
        dl = preprocessor.processDiffCode(diff)
        # Filter out tokens <= 2 characters to reduce noise
        dl = [token for token in dl if len(token) > 2]
        Diff_processed.append(dl)
    list1 = [Diff_processed,labelist,num]
    newlist.append(list1)
    print(index)
pd.DataFrame(newlist,columns=["Diff_processed","labelist","num"]).to_csv(
    f"../data/Processed{repo.title()}/0_{repo.lower()}_sub.csv")
