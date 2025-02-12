import re
# import dgl
# from tkinter import _flatten
import sys
from queue import Queue

import nltk
import nltk.data
import nltk.stem
from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import WordPunctTokenizer, word_tokenize
# import chardet   #需要导入这个模块，检测编码格式
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
# from keras.preprocessing.text import Tokenizer
from LinkGenerator.parser_lang import (tree_to_token_index,
                         index_to_code_token)
import tree_sitter_java as tsjava
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download('wordnet')
sys.setrecursionlimit(16385)
stop_word = ["i",
            "me",
            "my",
            "myself",
            "we",
            "us",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "will",
            "would",
            "shall",
            "should",
            "can",
            "could",
            "may",
            "might",
            "must",
            "ought",
            "i'm",
            "you're",
            "he's",
            "she's",
            "it's",
            "we're",
            "they're",
            "i've",
            "you've",
            "we've",
            "they've",
            "i'd",
            "you'd",
            "he'd",
            "she'd",
            "we'd",
            "they'd",
            "i'll",
            "you'll",
            "he'll",
            "she'll",
            "we'll",
            "they'll",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "hasn't",
            "haven't",
            "hadn't",
            "doesn't",
            "don't",
            "didn't",
            "won't",
            "wouldn't",
            "shan't",
            "shouldn't",
            "can't",
            "cannot",
            "couldn't",
            "mustn't",
            "let's",
            "that's",
            "who's",
            "what's",
            "here's",
            "there's",
            "when's",
            "where's",
            "why's",
            "how's",
            "daren't",
            "needn't",
            "oughtn't ",
            "mightn't",
            "a",
            "an",
            "the",
            "and",
            "but",
            "if",
            "or",
            "because",
            "as",
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "to",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "ever",
            "also",
            "just",
            "whether",
            "like",
            "even",
            "still",
            "since",
            "another",
            "however",
            "please",
            "much",
            "many"]

pattern = r"""(?x)                   
              (?:[a-zA-Z]\.)+           
              |\d+(?:\.\d+)?%?       
              |\w+(?:[-']\w+)*       
              |\.\.\.               
              |(?:[.,;"'?():-_`])    
            """
lang = 'python'

LANGUAGE = Language(tspython.language())
parser = Parser(LANGUAGE)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle') 
# wordtokenizer = RegexpTokenizer(pattern)
lemmatizer = WordNetLemmatizer()
stemmer = nltk.stem.SnowballStemmer('english')

codePattern = re.compile(r'<code>([\s\S]*?)</code>', re.I)
singleCodePattern = re.compile(r'`([\s\S]*?)`', re.I)
multiCodePattern = re.compile(r'```([\s\S]*?)```', re.I)
termPattern = re.compile(r'[A-Za-z]+.*[A-Z]+.*')
camelCase1 = re.compile(r'^[A-Z]+[a-z]+.*[A-Z]+.*$')
camelCase2 = re.compile(r'^[a-z]+.*[A-Z]+.*$')
upperCase = re.compile(r'^[A-Z]+[0-9]*$')
upperExtCase = re.compile(r'^[A-Z]*(_+[A-Z]*)+[0-9]*$')
methodInvocationCase = re.compile(r'([A-Za-z0-9_]+\.[A-Za-z0-9_]+)')

LINK_TAG = 'rhlinkrh'
CODE_TAG = 'rhcoderh'

stop_words = set(stopwords.words('english'))

def split_camel(camel_str,test=''):
    try:
        split_str = re.sub(
            r'(?<=[a-z]|[0-9])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\s+',
            '_',
            camel_str)
    except TypeError:
        return ['']
    try:
        if split_str[0] == '_':
            return [camel_str]
    except IndexError:
        return []
    return [i for i in split_str.lower().split('_') if i != '']


def index_to_code_token(index, code):
    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s = ""
        s += code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1, end_point[0]):
            s += code[i]
        s += code[end_point[0]][:end_point[1]]
    return s

def not_empty(s):
    return s and s.strip()

# 获取code token
def extract_codetoken(code, parser,lang):  
    #obtain codetoken   
    tree = parser.parse(bytes(code,'utf8'))    
    root_node = tree.root_node  
    tokens_index=tree_to_token_index(root_node)  
    code_tokens = []  
    code=code.split('\n')
    tokens = [index_to_code_token(x,code) for x in tokens_index]  
    code_tokens = [tokens[i] for i in range(len(tokens)) if (tokens[i].isalpha() or (tokens[i].isalnum() and not tokens[i].isdigit()))]
    code_tokens = list(filter(not_empty,code_tokens))
    return code_tokens

# 获取函数名
def get_func_name(code):        
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    node_queue = Queue()
    node_queue.put(root_node)
    func_name = ''
    while not node_queue.empty():
        cur_node = node_queue.get()
        if not cur_node.children is None:
            childs = cur_node.children
        if 'identifier' in cur_node.type:
            func_name = index_to_code_token([cur_node.start_point, cur_node.end_point], code.split('\n'))
            break
        for child in childs:
            node_queue.put(child)
    return split_camel(func_name)

def isDelete(word):
    if word in stop_word:
        return True
    if not re.match('.*[a-zA-Z]+.*', word, re.I):
        return True
    return False        


def preprocess(paragraph):
    result = []
    sentences = tokenizer.tokenize(paragraph)
    for sentence in sentences:
        words = nltk.regexp_tokenize(sentence, pattern) 
        temp = []
        for word in words:
            toDeal = []
            if camelCase1.match(word) or camelCase2.match(word):
                toDeal = splitCode(word)
            elif upperExtCase.match(word):
                toDeal = splitFinalExt(word)
            else:
                toDeal.append(word)
            for deal in toDeal:
                    if not isDelete(deal.lower()):
                        temp.append(stemmer.stem(deal))
        result.append(temp)
    return result


def preprocessBad(paragraph):
    result = []
    sentences = tokenizer.tokenize(paragraph)
    for sentence in sentences:
        words = WordPunctTokenizer().tokenize(sentence)
        temp = []
        for word in words:
            if not isDelete(word.lower()):
                temp.append(stemmer.stem(word.lower()))
        result.append(temp)
    return result

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

    # Remove URLs
    paragraph = RemoveHttp(paragraph)

    # Remove git-svn-id
    paragraph = RemoveGit(paragraph)

    # Replace inline code with a placeholder (CODE_TAG needs to be defined)
    paragraph = re.sub(r'`[\s\S]*?`', CODE_TAG, paragraph, 0, re.I)

    # Tokenize sentences
    sentences = tokenizer.tokenize(paragraph)

    for sentence in sentences:
        sentence = clean_en_text(sentence)

        word_tokens = word_tokenize(sentence)
        word_tokens = [word for word in word_tokens if word.lower() not in stopwords.words('english')]
        for word in word_tokens:
            if word in stopwords.words('english'):
                continue
            else:
                processed_word = lemmatizer.lemmatize(word) if lemmatizer.lemmatize(word).endswith('e') else stemmer.stem(word)
                result.append(str(processed_word))
    if len(result) == 0:
        text = ' '
    else:
        text = ' '.join(result)

    return text


def processHTMLNoCamel(html):
    texts = re.sub(r'(```[\s\S]*?```)', '', html, 0, re.I)
    texts = re.sub(r'(<.*?>)', '', texts, 0, re.I)
    texts = re.sub(r'(</.*?>)', '', texts, 0, re.I)
    return preprocessNoCamel(texts)


def preprocessToWord(paragraph):
    result = []
    sentences = tokenizer.tokenize(paragraph)
    for sentence in sentences:
        words = nltk.regexp_tokenize(sentence, pattern)
        for word in words:
            toDeal = []
            if camelCase1.match(word) or camelCase2.match(word):
                toDeal = splitCode(word)
            elif upperExtCase.match(word):
                toDeal = splitFinalExt(word)
            else:
                toDeal.append(word)
            for deal in toDeal:
                    if not isDelete(deal.lower()):
                        result.append(stemmer.stem(deal))
    return result


def processDiffCode(code):
    code = re.sub(r'(\"[\s\S]*?\")', '', code, 0, re.I)
    code = re.sub(r'(@@[\s\S]*?\n)', '', code, 0, re.I)
    code = re.sub(r'(-[\s\S]*?\n)', '', code, 0, re.I)
    result = []
    mis = methodInvocationCase.findall(code)
    for mi in mis:
        miWords = mi.split('\.')
        for miWord in miWords:
            toDeal = []
            if camelCase1.match(miWord) or camelCase2.match(miWord):
                toDeal = splitCode(miWord)
            elif upperExtCase.match(miWord):
                toDeal = splitFinalExt(miWord)
            elif upperCase.match(miWord):
                toDeal.append(miWord)
            for deal in toDeal:
                if not isDelete(deal.lower()):
                    result.append(lemmatizer.lemmatize(deal.lower()))

    code = re.sub(r'([A-Za-z0-9_]+\.[A-Za-z0-9_]+)', '', code, 0, re.I)
    sentences = tokenizer.tokenize(code)
    for sentence in sentences:
        words = nltk.regexp_tokenize(sentence, pattern)
        for word in words:
            toDeal = []
            if camelCase1.match(word) or camelCase2.match(word):
                toDeal = splitCode(word)
            elif upperExtCase.match(word):
                toDeal = splitFinalExt(word)
            elif upperCase.match(word):
                toDeal.append(word)
            for deal in toDeal:
                if not isDelete(deal.lower()):
                    result.append(lemmatizer.lemmatize(deal.lower()))
    return result

def processCode(code):
    code = eval(code)
    result = []
    for c in code:
        c = re.sub(r'(\"[\s\S]*?\")', '', c, 0, re.I)
        c = re.sub(r'(@@[\s\S]*?\n)', '', c, 0, re.I)
        c = re.sub(r'(-[\s\S]*?\n)', '', c, 0, re.I)   
        mis = methodInvocationCase.findall(c)
        for mi in mis:
            miWords = mi.split('\.')
            for miWord in miWords:
                toDeal = []
                if camelCase1.match(miWord) or camelCase2.match(miWord):
                    toDeal = splitCode(miWord)
                elif upperExtCase.match(miWord):
                    toDeal = splitFinalExt(miWord)
                elif upperCase.match(miWord):
                    toDeal.append(miWord)
                for deal in toDeal:
                    if not isDelete(deal.lower()):
                        result.append(lemmatizer.lemmatize(deal.lower()))

        code = re.sub(r'([A-Za-z0-9_]+\.[A-Za-z0-9_]+)', '', code, 0, re.I)
        sentences = tokenizer.tokenize(code)
        for sentence in sentences:
            words = nltk.regexp_tokenize(sentence, pattern)
            for word in words:
                toDeal = []
                if camelCase1.match(word) or camelCase2.match(word):
                    toDeal = splitCode(word)
                elif upperExtCase.match(word):
                    toDeal = splitFinalExt(word)
                elif upperCase.match(word):
                    toDeal.append(word)
                for deal in toDeal:
                    if not isDelete(deal.lower()):
                        result.append(lemmatizer.lemmatize(deal.lower()))
    return result

def processPreDiffCode(code):
    code = re.sub(r'(\"[\s\S]*?\")', '', code, 0, re.I)
    code = re.sub(r'(@@[\s\S]*?\n)', '', code, 0, re.I)
    code = re.sub(r'(\+[\s\S]*?\n)', '', code, 0, re.I)
    result = []
    mis = methodInvocationCase.findall(code)
    for mi in mis:
        miWords = mi.split('\.')
        for miWord in miWords:
            toDeal = []
            if camelCase1.match(miWord) or camelCase2.match(miWord):
                toDeal = splitCode(miWord)
            elif upperExtCase.match(miWord):
                toDeal = splitFinalExt(miWord)
            elif upperCase.match(miWord):
                toDeal.append(miWord)
            for deal in toDeal:
                if not isDelete(deal.lower()):
                    result.append(lemmatizer.lemmatize(deal.lower()))

    code = re.sub(r'([A-Za-z0-9_]+\.[A-Za-z0-9_]+)', '', code, 0, re.I)
    sentences = tokenizer.tokenize(code)
    for sentence in sentences:
        words = nltk.regexp_tokenize(sentence, pattern)
        for word in words:
            toDeal = []
            if camelCase1.match(word) or camelCase2.match(word):
                toDeal = splitCode(word)
            elif upperExtCase.match(word):
                toDeal = splitFinalExt(word)
            elif upperCase.match(word):
                toDeal.append(word)
            for deal in toDeal:
                if not isDelete(deal.lower()):
                    result.append(lemmatizer.lemmatize(deal.lower()))
    return result


def processHTML(html):
      multiCodes = multiCodePattern.findall(html)
      texts = re.sub(r'(```[\s\S]*?```)', '', html, 0, re.I)
      singleCodes = singleCodePattern.findall(html)
      texts = re.sub(r'(<.*?>)', '', texts, 0, re.I)
      texts = re.sub(r'(</.*?>)', '', texts, 0, re.I)
      texts = re.sub(r'(`)', '', texts, 0, re.I)
      preText = preprocessToWord(texts)
      result = []
      codes = multiCodes + singleCodes
      for code in codes:
          code = re.sub(r'(\"[\s\S]*?\")', '', code, 0, re.I)
          mis = methodInvocationCase.findall(code)
          for mi in mis:
              miWords = mi.split('.')
              for miWord in miWords:
                  toDeal = []
                  if camelCase1.match(miWord) or camelCase2.match(miWord):
                      toDeal = splitCode(miWord)
                  elif upperExtCase.match(miWord):
                      toDeal = splitFinalExt(miWord)
                  elif upperCase.match(miWord):
                      toDeal.append(miWord)
                  else:
                      toDeal.append(miWord)
                  for deal in toDeal:
                      if not isDelete(deal.lower()):
                          result.append(stemmer.stem(deal))
          code = re.sub(r'([A-Za-z0-9_]+\.[A-Za-z0-9_]+)', '', code, 0, re.I)
          sentences = tokenizer.tokenize(code)
          for sentence in sentences:
              words = nltk.regexp_tokenize(sentence, pattern)
              for word in words:
                  toDeal = []
                  if camelCase1.match(word) or camelCase2.match(word):
                      toDeal = splitCode(word)
                  elif upperExtCase.match(word):
                      toDeal = splitFinalExt(word)
                  elif upperCase.match(word):
                      toDeal.append(word)
                  for deal in toDeal:
                      if not isDelete(deal.lower()):
                          result.append(stemmer.stem(deal))
      return result, preText


def getIssueCode(html):
    multiCodes = multiCodePattern.findall(html)
    texts = re.sub(r'(```[\s\S]*?```)', '', html, 0, re.I)
    singleCodes = singleCodePattern.findall(texts)
    result = []
    codes = multiCodes + singleCodes
    for code in codes:
        code = re.sub(r'(\"[\s\S]*?\")', '', code, 0, re.I)
        mis = methodInvocationCase.findall(code)
        for mi in mis:
            miWords = mi.split('.')
            for miWord in miWords:
                toDeal = []
                if camelCase1.match(miWord) or camelCase2.match(miWord):
                    toDeal = splitCode(miWord)
                elif upperExtCase.match(miWord):
                    toDeal = splitFinalExt(miWord)
                elif upperCase.match(miWord):
                    toDeal.append(miWord)
                else:
                    toDeal.append(miWord)
                for deal in toDeal:
                    if not isDelete(deal.lower()):
                        result.append(lemmatizer.lemmatize(deal.lower()))
        code = re.sub(r'([A-Za-z0-9_]+\.[A-Za-z0-9_]+)', '', code, 0, re.I)
        sentences = tokenizer.tokenize(code)
        for sentence in sentences:
            words = nltk.regexp_tokenize(sentence, pattern)
            for word in words:
                toDeal = []
                if camelCase1.match(word) or camelCase2.match(word):
                    toDeal = splitCode(word)
                elif upperExtCase.match(word):
                    toDeal = splitFinalExt(word)
                elif upperCase.match(word):
                    toDeal.append(word)
                for deal in toDeal:
                    if not isDelete(deal.lower()):
                        result.append(lemmatizer.lemmatize(deal.lower()))
    return result


def processHTMLByTag(html):
      codes = codePattern.findall(html)
      texts = re.sub(r'(<pre>\s*?<code>[\s\S]*?</code>\s*?</pre>)', '', html, 0, re.I)
      texts = re.sub(r'(<.*?>)', '', texts, 0, re.I)
      texts = re.sub(r'(</.*?>)', '', texts, 0, re.I)
      preText = preprocessToWord(texts)
      result = []
      for code in codes:
          code = re.sub(r'(\"[\s\S]*?\")', '', code, 0, re.I)
          mis = methodInvocationCase.findall(code)
          for mi in mis:
              miWords = mi.split('.')
              for miWord in miWords:
                  toDeal = []
                  if camelCase1.match(miWord) or camelCase2.match(miWord):
                      toDeal = splitCode(miWord)
                  elif upperExtCase.match(miWord):
                      toDeal = splitFinalExt(miWord)
                  elif upperCase.match(miWord):
                      toDeal.append(miWord)
                  for deal in toDeal:
                      if not isDelete(deal.lower()):
                          result.append(stemmer.stem(deal))
          code = re.sub(r'([A-Za-z0-9_]+\.[A-Za-z0-9_]+)', '', code, 0, re.I)
          sentences = tokenizer.tokenize(code)
          for sentence in sentences:
              words = nltk.regexp_tokenize(sentence, pattern)
              for word in words:
                  toDeal = []
                  if camelCase1.match(word) or camelCase2.match(word):
                      toDeal = splitCode(word)
                  elif upperExtCase.match(word):
                      toDeal = splitFinalExt(word)
                  elif upperCase.match(word):
                      toDeal.append(word)
                  for deal in toDeal:
                      if not isDelete(deal.lower()):
                          result.append(stemmer.stem(deal))
      return result, preText


def splitCode(code):
    res = []
    words = re.split(r"([A-Z]+[a-z]*)", code)
    for word in words:
        if word:
            res.append(word)
    return res


def splitFinalExt(ext):
    res = []
    words = ext.split('_')
    for word in words:
        if word:
            res.append(word)
    return res
import pandas as pd
def test():
    data  = pd.read_csv('/data/zcy/LinkR/data/train_flag/ignite_link_flag_comment_wash.csv')
    cl = eval(data.loc[2138,'codelist'])
    print(cl[0][:400])
    # print(data.loc[79,'description'])
    # print(extract_codetoken(cl[0], parser,lang))


if __name__ == "__main__":
    test()
