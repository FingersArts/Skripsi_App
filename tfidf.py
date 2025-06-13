import pandas as pd
import numpy as np
import ast
from imblearn.over_sampling import SMOTE

# Fungsi konversi teks list
def convert_text_list(texts):
    if isinstance(texts, list):
        return texts
    return ast.literal_eval(texts)

# Fungsi membuat n-gram fleksibel
def generate_ngrams(tokens, ngram_range=(1, 3)):
    ngrams = []
    min_n, max_n = ngram_range
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngram = "_".join(tokens[i:i+n])
            ngrams.append(ngram)
    return ngrams

# Hitung TF
def calc_TF(document, use_sublinear=True):
    TF_dict = {}
    for term in document:
        TF_dict[term] = TF_dict.get(term, 0) + 1
    
    doc_len = len(document) if len(document) != 0 else 1
    for term in TF_dict:
        if use_sublinear and TF_dict[term] > 0:
            TF_dict[term] = 1 + np.log(TF_dict[term])
        else:
            TF_dict[term] /= doc_len
    return TF_dict

# Hitung DF
def calc_DF(tfDict, min_df=2, max_df_ratio=0.9):
    count_DF = {}
    n_docs = len(tfDict)
    max_df = int(n_docs * max_df_ratio)
    
    # Hitung kemunculan term di seluruh dokumen
    for document in tfDict:
        for term in document:
            count_DF[term] = count_DF.get(term, 0) + 1
    
    # Filter terms berdasarkan frekuensi
    filtered_DF = {
        term: freq for term, freq in count_DF.items() 
        if freq >= min_df and freq <= max_df
    }
    
    return filtered_DF

# Hitung IDF
def calc_IDF(n_document, DF, smooth=True):
    IDF_Dict = {}
    for term in DF:
        if smooth:
            IDF_Dict[term] = np.log((n_document + 1) / (DF[term] + 1)) + 1
        else:
            IDF_Dict[term] = np.log(n_document / DF[term])
    return IDF_Dict

# Hitung TF-IDF
def calc_TF_IDF(TF, IDF):
    TF_IDF_Dict = {}
    for key in TF:
        if key in IDF:
            TF_IDF_Dict[key] = TF[key] * IDF[key]
    return TF_IDF_Dict

# Hitung TF-IDF Vector
def calc_TF_IDF_Vec(TF_IDF_Dict, selected_terms):
    TF_IDF_vector = [0.0] * len(selected_terms)
    for i, term in enumerate(selected_terms.keys()):  # Hanya ambil kunci (kata)
        if term in TF_IDF_Dict:
            TF_IDF_vector[i] = TF_IDF_Dict[term]
    return TF_IDF_vector

# Fungsi utama menghitung TF-IDF
def calculate_tfidf(df, top_k=500, ngram_range=(1, 3), label_column='sentiment', apply_smote=True, smote_random_state=42):
    # Konversi teks dan buat n-gram
    df["text_list"] = df["stemming"].apply(convert_text_list)
    df["ngrams"] = df["text_list"].apply(lambda x: generate_ngrams(x, ngram_range))
    df["TF_dict"] = df["ngrams"].apply(lambda x: calc_TF(x, use_sublinear=True))
    
    # Hitung DF dan IDF
    DF = calc_DF(df["TF_dict"], min_df=3, max_df_ratio=0.8)
    n_document = len(df)
    IDF = calc_IDF(n_document, DF, smooth=True)
    df["TF-IDF_dict"] = df["TF_dict"].apply(lambda tf: calc_TF_IDF(tf, IDF))

    # Buat ranking berdasarkan hasil TF-IDF
    TF_IDF_Sum = {}
    TF_Sum = {}
    TF_Count = {}

    for tfidf_dict, tf_dict in zip(df["TF-IDF_dict"], df["TF_dict"]):
        for term in tfidf_dict:
            TF_IDF_Sum[term] = TF_IDF_Sum.get(term, 0) + tfidf_dict[term]
        for term in tf_dict:
            TF_Sum[term] = TF_Sum.get(term, 0) + tf_dict[term]
            TF_Count[term] = TF_Count.get(term, 0) + 1

    data = []
    for term in TF_IDF_Sum:
        tf_avg = TF_Sum[term] / TF_Count[term]
        df_val = DF.get(term, 0)
        idf_val = IDF.get(term, 0)
        data.append((term, TF_IDF_Sum[term], tf_avg, df_val, idf_val))

    ranking = pd.DataFrame(data, columns=['term', 'TF-IDF', 'tf', 'df', 'idf'])
    ranking = ranking.sort_values('TF-IDF', ascending=False)

    # Seleksi fitur berdasarkan TF-IDF tertinggi
    ranking_top = ranking.head(top_k)
    selected_terms = {term: 1 for term in ranking_top['term']}  # gunakan hanya key-nya

    # Buat TF-IDF Vector berdasarkan selected_terms yang sudah dibuat
    df["TF_IDF_Vec"] = df["TF-IDF_dict"].apply(lambda tfidf: calc_TF_IDF_Vec(tfidf, selected_terms))


    # Terapkan SMOTE jika diaktifkan
    if apply_smote:
        X = np.array(df["TF_IDF_Vec"].to_list())
        y = df[label_column].to_numpy()
        
        smote = SMOTE(random_state=smote_random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Buat DataFrame baru untuk data yang sudah di-resample
        df_resampled = pd.DataFrame({
            'TF_IDF_Vec': list(X_resampled),
            label_column: y_resampled
        })
        return ranking, df, df_resampled, selected_terms, IDF
    else:
        return ranking, df, None, selected_terms, IDF