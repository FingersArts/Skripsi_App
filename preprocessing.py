# Import Library
import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
import matplotlib.pyplot as plt
import emoji

nltk.download(['stopwords', 'punkt'])

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Cleaning
def cleaning_text(text):
    # Hapus URL
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)

    # Hapus mention
    text = re.sub(r'@[\w]*', ' ', text)

    # Hapus tag HTML seperti <a href=...>
    text = re.sub(r'<a\s+href.*?>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'a\s+href', '', text, flags=re.IGNORECASE)

    # Hapus emoji
    text = emoji.replace_emoji(text, replace='')

    # Hapus tanda baca
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^+&*_~'''
    for x in text.lower():
        if x in punctuations:
            text = text.replace(x, " ")

    # Bersihkan karakter khusus
    text = text.replace('\\t', " ")
    text = text.replace('\\n', " ")
    text = text.replace('\\u', " ")
    text = text.replace('\\', "")
    text = text.replace('&quot;', "")
    text = text.replace('quot', "")
    text = text.replace('=', "")
    
    # Hapus pola seperti "(a)" atau "(A)"
    text = re.sub(r'\(a\)', '', text, flags=re.IGNORECASE)

    # Hapus angka
    text = re.sub(r"\d+", "", text)

    # Hapus spasi ekstra
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Muat kamus slangword dari file CSV
kamus_slang = pd.read_csv("kamus-slang.csv", header=None, names=['slang', 'formal'])
lookup_dict = dict(zip(kamus_slang['slang'], kamus_slang['formal']))
# Mengganti kata slang dengan kata asli
def slangremove(text, lookup_dict=lookup_dict):
    words = text.split()
    new_words = [lookup_dict.get(word, word) for word in words]
    return ' '.join(new_words)

# Define stopwords
sastrawi_stopword = "https://raw.githubusercontent.com/onlyphantom/elangdev/master/elang/word2vec/utils/stopwords-list/sastrawi-stopwords.txt"
stopwords_l = stopwords.words('indonesian')
response = requests.get(sastrawi_stopword)
stopwords_l += response.text.split('\n')

custom_st = '''
yg yang dgn ane smpai bgt gua gwa si tu ama utk udh btw
ntar lol ttg emg aj aja tll sy sih kalo nya trsa mnrt nih
'''
st_words = set(stopwords_l)
custom_stopword = set(custom_st.split())
stop_words = st_words | custom_stopword

def stopword(text):
    word_tokens = text.split()
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)

# Stemming
def stemming(text_cleaning):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  do = []
  for w in text_cleaning:
    dt = stemmer.stem(w)
    do.append(dt)
  d_clean = []
  d_clean = " ".join(do)
  print(d_clean)
  return d_clean

# Main function to process the dataframe
def preproces(df, progress_callback=None):
    if progress_callback:
        progress_callback(10, "Melakukan case folding...")
    df['casefolding'] = df['full_text'].str.lower()

    if progress_callback:
        progress_callback(25, "Membersihkan teks (cleaning)...")
    df['cleanedtext'] = df['casefolding'].apply(cleaning_text)

    df = df.dropna(subset=['cleanedtext'])
    df = df[df['cleanedtext'].str.strip() != ""]

    if progress_callback:
        progress_callback(40, "Menghapus kata tidak baku (slang removal)...")
    df['slangremoved'] = df['cleanedtext'].apply(lambda x: slangremove(x, lookup_dict))

    if progress_callback:
        progress_callback(60, "Menghapus stopword...")
    df['stopwordremoved'] = df['slangremoved'].apply(stopword)

    if progress_callback:
        progress_callback(75, "Tokenisasi...")
    df['tokenized'] = df['stopwordremoved'].apply(lambda x: x.split())

    if progress_callback:
        progress_callback(90, "Stemming...")
    df['stemming'] = df['tokenized'].apply(stemming)

    if progress_callback:
        progress_callback(100, "Preprocessing selesai!")

    return df
