import pandas as pd
from numpy import nan as NaN

repo = "ISIS"
df1 = pd.read_csv(f"../data/Processed{repo.title()}/1.5_{repo.lower()}_process_notes_cleaned.csv")
df2 = pd.read_csv(f"../data/Processed{repo.title()}/0_{repo.lower()}_sub.csv")
list1 = ["issue_id", "summary_processed", "description_processed", "issuecode", "hash", "fix_version", "tracking_id",
         "message_processed", "changed_files", "codelist_processed", "label", "train_flag", "release_notes"]

df = df1[list1]
df.insert(len(df.columns), "Diff_processed", value=NaN)
df.insert(len(df.columns), "labelist", value=NaN)
df.insert(len(df.columns), "num", value=NaN)
tg = df.label

res = df.drop("label", axis=1)
res.insert(len(res.columns), "target", tg)
flag = res.train_flag
res = res.drop("train_flag", axis=1)
res.insert(len(res.columns), "train_flag", flag)

# Apply filtering to Diff_processed to remove tokens <= 2 characters to reduce noise
res["Diff_processed"] = df2["Diff_processed"].apply(
    lambda x: [token for token in eval(x) if len(token) > 2] if pd.notna(x) else [])
res["labelist"] = df2.labelist
res["num"] = df2.num

res = res.loc[:, ~res.columns.str.contains("Unnamed")]
res.to_csv(f"../data/Processed{repo.title()}/2_{repo.lower()}_link_merged.csv")
print(res.columns)
