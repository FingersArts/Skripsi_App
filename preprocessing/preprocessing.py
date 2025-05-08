import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from functools import lru_cache
import requests
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
import emoji

# Download resource NLTK
nltk.download(['stopwords', 'punkt'])

# Inisialisasi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# --- CLEANING FUNCTION ---
def cleaning_text(text):
    if not isinstance(text, str):
        return ""

    # Hapus URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Hapus mention
    text = re.sub(r'@[\w]*', ' ', text)
    # Hapus HTML tag <a href=...>
    text = re.sub(r'<a\s+href.*?>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'a\s+href', '', text, flags=re.IGNORECASE)
    # Hapus emoji
    text = emoji.replace_emoji(text, replace='')
    # Hapus tanda baca
    text = re.sub(r'[^\w\s]', ' ', text)
    # Bersihkan karakter khusus
    text = re.sub(r'(\\t|\\n|\\u|\\|&quot;|quot|=)', ' ', text)
    # Hapus angka
    text = re.sub(r"\d+", "", text)
    # Hapus spasi ekstra
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- SLANG REMOVER ---
kamus_slang = pd.read_csv("preprocessing/kamus-slang.csv", header=None, names=['slang', 'formal'])
lookup_dict = dict(zip(kamus_slang['slang'], kamus_slang['formal']))

def slangremove(text, lookup_dict=lookup_dict):
    words = text.split()
    new_words = [lookup_dict.get(word, word) for word in words]
    return ' '.join(new_words)

# --- STOPWORDS ---
sastrawi_stopword = "https://raw.githubusercontent.com/onlyphantom/elangdev/master/elang/word2vec/utils/stopwords-list/sastrawi-stopwords.txt"
stopwords_l = stopwords.words('indonesian')
response = requests.get(sastrawi_stopword)
stopwords_l += response.text.split('\n')

with open('preprocessing/stopwords.txt', 'r', encoding='utf-8') as f:
    custom_st = f.read()

st_words = set(stopwords_l)
custom_stopword = set(custom_st.split())
stop_words = st_words | custom_stopword

def remove_stopwords(text):
    word_tokens = text.split()
    filtered_sentence = [w for w in word_tokens if w not in stop_words]
    return ' '.join(filtered_sentence)

# --- STEMMING ---
@lru_cache(maxsize=None)
def cached_stem(word):
    return stemmer.stem(word)

def stemming(word_list):
    return [cached_stem(word) for word in word_list]

# --- MAIN PREPROCESSING FUNCTION ---
def preproces(df, progress_callback=None):
    if progress_callback:
        progress_callback(10, "Melakukan case folding...")
    df['casefolding'] = df['full_text'].str.lower()

    if progress_callback:
        progress_callback(25, "Membersihkan teks (cleaning)...")
    df['cleanedtext'] = df['casefolding'].apply(cleaning_text)
    df = df.drop_duplicates(subset='cleanedtext').reset_index(drop=True)
    df = df.dropna(subset=['cleanedtext'])
    df = df[df['cleanedtext'].str.strip() != ""]

    if progress_callback:
        progress_callback(40, "Menghapus kata tidak baku (slang removal)...")
    df['slangremoved'] = df['cleanedtext'].apply(slangremove)

    if progress_callback:
        progress_callback(60, "Menghapus stopword...")
    df['stopwordremoved'] = df['slangremoved'].apply(remove_stopwords)
    df = df.dropna(subset=['stopwordremoved'])
    df = df[df['stopwordremoved'].str.strip() != ""]

    if progress_callback:
        progress_callback(75, "Tokenisasi...")
    df['tokenized'] = df['stopwordremoved'].apply(lambda x: x.split())

    if progress_callback:
        progress_callback(90, "Stemming...")
    df['stemming'] = df['tokenized'].apply(stemming)

    # (Opsional) Gabungkan kembali hasil stemming menjadi string
    df['stemming_str'] = df['stemming'].apply(lambda x: ' '.join(x))

    if progress_callback:
        progress_callback(100, "Preprocessing selesai!")

    return df
