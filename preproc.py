import pandas as pd
import re
import os.path
import jieba
import jieba.analyse
from nltk import tokenize
from nltk.corpus import wordnet as wn

def cleanTXT(text):
    #all regex are tested using regex tester at https://regex101.com/
    text = re.sub(r'@[\w.]+', '',text) #remove @mentions + '_' + '.' : @yew_lee
    text = re.sub(r'#', '', text) #remove '#'
    text = re.sub(r'\\n', '', text) #remove '\n'
    text = re.sub(r'http\S+', '', text) #links
    text = re.sub(r'\W', ' ', text) #replace every symbol+emoji with empty space
    text = re.sub(r'_', ' ', text) #replace '_' since '\W' fail to catch underscore

    return re.compile(r'\s{2,}').sub(' ', text) #remove multiple whitespace to single whitespace

def check_lang_eng(token_list): #retrieve english token by comparing to synset
    eng_list=[]
    for token in token_list:
        if wn.synsets(token, lang='eng'):
            eng_list.append(token)
    return eng_list

def check_lang_eng_perc(token_list): # retrieve english token percentage in the sentence
    counts_1 = 0
    for token in token_list:
        if wn.synsets(token, lang='eng'):
            counts_1+=1
    if len(token_list)>0:
        percentage = counts_1/(len(token_list))
        return percentage
    else:
        return 0

def check_lang_zsm(token_list): #retrieve malay token by comparing to synset zsm
    zsm_list=[]
    for token in token_list:
        if wn.synsets(token, lang='zsm'):  #ind
            #no need to load .tab since Open Multilingual Wordnet has integrated it into nltk wn
            zsm_list.append(token)
    return zsm_list

def check_lang_zsm_perc(token_list): #retrieve malay token percentage in sentence
    counts_1 = 0
    for token in token_list:
        if wn.synsets(token, lang='zsm'): #ind
            counts_1+=1
    if len(token_list)>0:
        percentage = counts_1/(len(token_list))
        return percentage
    else:
        return 0
    
def check_lang_zho(token_list): #retrieve malay token
    token_list = re.sub(r'\W', '', token_list)
    token_list = list(jieba.cut(token_list, cut_all=False)) #Tokenize Chinese Word using jieba
    cn_list = []
    #check if is chinese
    for word in token_list:
        if re.search("[\u4e00-\u9FFF]", word): #search all chinese words using utf code
            cn_list.append(word)
    return cn_list

def check_lang_zho_perc(token_list):
    count = 0
    token_list = re.sub(r'\W', '', token_list)
    token_list = list(jieba.cut(token_list, cut_all=False)) #Tokenize Chinese Word
    #check if is chinese
    for word in token_list:
        if re.search("[\u4e00-\u9FFF]", word):
            count +=1
    if len(token_list)>0:
        percentage = count/(len(token_list))
        return percentage
    else:
        return 0

def whitespaceTokenizer(data): #whitespace tokenizer function
    token_list=[]
    token_list = tokenize.WhitespaceTokenizer().tokenize(data)
    return token_list
    
################ MAIN ###########################################

csv_dir = os.path.join("in_csv") #directory for CSV
csv_path = os.path.join(csv_dir, 'tester.csv') #testing CSV

#read csv
df = pd.read_csv(csv_path, encoding='utf-8')

#clean text
df['comment'] = df['comment'].apply(cleanTXT)

#tokenize word
df['comments_tokens'] = df['comment'].apply(whitespaceTokenizer) 

#calculating percentage
df['eng_tokens'] = df['comments_tokens'].apply(check_lang_eng)
df['eng_percentage'] = df['comments_tokens'].apply(check_lang_eng_perc)
df['zsm_tokens'] = df['comments_tokens'].apply(check_lang_zsm)
df['zsm_percentage'] = df['comments_tokens'].apply(check_lang_zsm_perc)
df['zho_tokens'] = df['comment'].apply(check_lang_zho)
df['zho_perc'] = df['comment'].apply(check_lang_zho_perc)