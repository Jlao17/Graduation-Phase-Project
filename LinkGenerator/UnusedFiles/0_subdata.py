
import csv
import re

import pandas as pd
import preprocessor


repo = "HADOOP"
# from tkinter import _flatten
csv.field_size_limit(500 * 1024 * 1024)

newlist = []
dummy_link = pd.read_csv(f"../data/OriginalData/Hadoop/{repo.lower()}_link_raw_merged.csv")
for index, row in dummy_link.iterrows():
    Diff_processed = []
    difflist = eval(row["Diff"])


    # Preprocess Diff code
    for d in difflist:
        diff = re.sub(r"\<ROW.[0-9]*\>", "", str(d))
        diff = re.sub(r"\<CODE.[0-9]*\>", "", diff)
        dl = preprocessor.processDiffCode(diff)
        # Filter out tokens <= 2 characters to reduce noise
        dl = [token for token in dl if len(token) > 2]
        Diff_processed.append(dl)
    list1 = [Diff_processed]
    newlist.append(list1)
    print(index)
pd.DataFrame(newlist,columns=["Diff_processed"]).to_csv(
    f"../data/Processed{repo.title()}/0_{repo.lower()}_test_remove_after.csv")
