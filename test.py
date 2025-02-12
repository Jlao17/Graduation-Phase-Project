import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
import nltk.data
import re
from nltk.stem import WordNetLemmatizer,PorterStemmer
from torch.utils.data import DataLoader
from transformers import AutoModel, RobertaConfig, AutoConfig, AutoTokenizer

import LinkGenerator.preprocessor as preprocessor
from models.utils import IssueCommitReleaseDataset

lemmatizer = WordNetLemmatizer()
from tqdm import tqdm

tqdm.pandas()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stemmer = nltk.stem.SnowballStemmer('english')


def RemoveHttp(str):
    httpPattern = '[a-zA-z]+://[^\s]*'
    return re.sub(httpPattern, ' ', str)

def RemoveGit(str):
    gitPattern = '[Gg]it-svn-id'
    return re.sub(gitPattern, ' ', str)

def clean_en_text(text):
    # keep English, digital and space
    comp = re.compile('[^A-Z^a-z^0-9^ ]')
    return comp.sub(' ', text)

def preprocessNoCamel(paragraph):
    if not isinstance(paragraph, str):  # Ensure input is a string
        return ""  # Or use str(paragraph) if you want to keep the data
    result = []

    # Remove text inside brackets
    paragraph = re.sub(r'(\[[\s\S]*?\])', '', paragraph, 0, re.I)
    # print("1.", paragraph)
    # Remove URLs
    paragraph = RemoveHttp(paragraph)
    # print("2.", paragraph)
    # Remove git-svn-id
    paragraph = RemoveGit(paragraph)
    # print("3.", paragraph)
    # Replace inline code with a placeholder (CODE_TAG needs to be defined)
    paragraph = re.sub(r'`[\s\S]*?`', '', paragraph, 0, re.I)
    # print("4.", paragraph)
    # Convert to lowercase
    paragraph = paragraph.lower()
    # print("5.", paragraph)
    # Tokenize sentences
    sentences = tokenizer.tokenize(paragraph)

    for sentence in sentences:
        sentence = clean_en_text(sentence)
        word_tokens = word_tokenize(sentence)
        word_tokens = [word for word in word_tokens if word.lower() not in stopwords.words('english')]
        # print("5.", word_tokens)
        for word in word_tokens:
            if word in stopwords.words('english'):
                continue
            else:
                processed = lemmatizer.lemmatize(word) if lemmatizer.lemmatize(word).endswith('e') else stemmer.stem(word)
                print("6.", processed)
                result.append(str(processed))
    if len(result) == 0:
        text = ' '
    else:
        text = ' '.join(result)

    return text

# test = preprocessNoCamel("We deployed hadoop 3.1.1 on centos 7.2 servers whose timezone is configured as GMT+8,  the web browser time zone is GMT+8 too. yarn ui page loaded failed due to js error: !image-2018-09-05-18-54-03-991.png! The moment-timezone js component raised that error. This has been fixed in moment-timezone v0.5.1([see|[https://github.com/moment/moment-timezone/issues/294]).] We need to update moment-timezone version accordingly")
#
# print(stemmer.stem("timezone"))
# df = pd.read_csv("data/ProcessedHadoop/1.5_hadoop_process.csv")
# df["release_notes"] = df["release_notes_original"].progress_apply(preprocessor.preprocessNoCamel)

word = "MODIFY ModelHandler.java ModelHandler create visit visit stringListList stringList populateSchema visit operandMap visit visit visit populateLattice visit visit currentSchemaPath currentSchema currentSchemaName currentMutableSchema visit visit visit ExtraOperand MODIFY CsvSchemaFactory.java CsvSchemaFactory create MODIFY CsvTableFactory.java CsvTableFactory create MODIFY CsvTest.java close toLinux testVanityDriver testVanityDriverArgsInUrl testBadDirectory testSelect testSelectSingleProjectGz testSelectSingleProject testCustomTable testPushDownProjectDumb testPushDownProject testPushDownProject2 testFilterableSelect testFilterableSelectStar testFilterableWhere testFilterableWhere2 testJson checkSql output checkSql expect checkSql jsonPath collect output testJoinOnString testWackyColumns testBoolean testDateType MODIFY bug.json MODIFY model-with-custom-table.json MODIFY model.json MODIFY smart.json"
word2 = "modify"
# processed_word = lemmatizer.lemmatize(word2) if lemmatizer.lemmatize(word2).endswith('e') else stemmer.stem(word2)
#
# df = pd.read_csv("data/ProcessedHadoop/3_hadoop_link_final.csv")
# print(df.duplicated(subset=df.columns.difference(['target', 'target_rn'])).sum())
# df.drop_duplicates(inplace=True)
# print(df.duplicated(subset=df.columns.difference(['target', 'target_rn'])).sum())
# df.to_csv("data/ProcessedHadoop/3_hadoop_link_final_drop_duplicate.csv", index=False)

df1 = pd.read_csv("data/OriginalData/Tika/Tika_DEV.csv")
df2 = pd.read_csv("data/OriginalData/Tika/Tika_TEST.csv")
df3 = pd.read_csv("data/OriginalData/Tika/Tika_TRAIN.csv")

combined_df = pd.concat([df1, df2, df3], ignore_index=True)

#rename columns
combined_df.rename(columns={"label": "target", "Issue_KEY": "tracking_id", "Commit_SHA": "commitid", "Issue_Text": "summary", "Commit_Text": "message", "Commit_Code": "Diff"}, inplace=True)
combined_df.to_csv("data/OriginalData/Tika/tika_link_raw_merged.csv", index=False)






