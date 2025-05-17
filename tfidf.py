import pandas as pd
import numpy as np
import ast
from collections import Counter

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
    
    # Normalisasi
    doc_len = len(document) if len(document) != 0 else 1
    
    for term in TF_dict:
        # Sublinear TF scaling (1 + log(tf))
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
        # Smooth IDF: ln((N+1)/(df+1)) + 1
        if smooth:
            IDF_Dict[term] = np.log((n_document + 1) / (DF[term] + 1)) + 1
        else:
            # Classic IDF: ln(N/df)
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
    for i, term in enumerate(selected_terms):
        if term in TF_IDF_Dict:
            TF_IDF_vector[i] = TF_IDF_Dict[term]
    return TF_IDF_vector

# Fungsi untuk term selection dengan metode chi-square
def select_features_chi2(df, DF, label_column='sentiment', top_k=500):
    n_docs = len(df)
    class_counts = Counter(df[label_column])
    classes = sorted(class_counts.keys())
    
    # Hitung frekuensi term per kelas
    term_class_freq = {}
    for term in DF:
        term_class_freq[term] = {c: 0 for c in classes}
    
    for idx, row in df.iterrows():
        class_label = row[label_column]
        for term in set(row['ngrams']):  # menggunakan set untuk menghindari double counting
            if term in term_class_freq:
                term_class_freq[term][class_label] += 1
    
    # Hitung chi-square score untuk setiap term
    chi2_scores = {}
    for term, class_freq in term_class_freq.items():
        chi2 = 0
        term_freq = DF[term]  # Frekuensi dokumen yang mengandung term
        
        for c in classes:
            # Frekuensi kelas
            class_freq_c = class_counts[c]
            
            # Observed values
            O_tc = class_freq[c]  # Dokumen kelas c dengan term
            O_tnc = term_freq - O_tc  # Dokumen dengan term tapi bukan kelas c
            O_ntc = class_freq_c - O_tc  # Dokumen kelas c tanpa term
            O_ntnc = n_docs - O_tc - O_tnc - O_ntc  # Dokumen bukan kelas c tanpa term
            
            # Expected values
            E_tc = (term_freq * class_freq_c) / n_docs
            if E_tc == 0:
                continue
                
            E_tnc = (term_freq * (n_docs - class_freq_c)) / n_docs
            if E_tnc == 0:
                continue
                
            E_ntc = ((n_docs - term_freq) * class_freq_c) / n_docs
            if E_ntc == 0:
                continue
                
            E_ntnc = ((n_docs - term_freq) * (n_docs - class_freq_c)) / n_docs
            if E_ntnc == 0:
                continue
            
            # Chi-square calculation
            chi2 += ((O_tc - E_tc) ** 2) / E_tc
            chi2 += ((O_tnc - E_tnc) ** 2) / E_tnc
            chi2 += ((O_ntc - E_ntc) ** 2) / E_ntc
            chi2 += ((O_ntnc - E_ntnc) ** 2) / E_ntnc
            
        chi2_scores[term] = chi2
    
    # Seleksi top-k term dengan chi-square tertinggi
    sorted_terms = sorted(chi2_scores.items(), key=lambda x: x[1], reverse=True)
    selected_terms = [term for term, score in sorted_terms[:top_k]]
    
    return selected_terms

# Fungsi utama menghitung TF-IDF
def calculate_tfidf(df, top_k=500, ngram_range=(1, 3), use_chi2=True, label_column='sentiment'):
    # Ubah string ke list
    df["text_list"] = df["stemming"].apply(convert_text_list)

    # Buat n-gram sesuai range
    df["ngrams"] = df["text_list"].apply(lambda x: generate_ngrams(x, ngram_range))

    # Hitung TF
    df["TF_dict"] = df["ngrams"].apply(lambda x: calc_TF(x, use_sublinear=True))

    # Hitung DF dan IDF
    DF = calc_DF(df["TF_dict"], min_df=3, max_df_ratio=0.8)
    n_document = len(df)
    IDF = calc_IDF(n_document, DF, smooth=True)

    # Hitung TF-IDF Dictionary
    df["TF-IDF_dict"] = df["TF_dict"].apply(lambda tf: calc_TF_IDF(tf, IDF))

    # Seleksi fitur
    if use_chi2:
        # Gunakan chi-square untuk seleksi fitur
        selected_terms = select_features_chi2(df, DF, label_column=label_column, top_k=top_k)
    else:
        # Seleksi berdasarkan DF (cara lama)
        sorted_DF = sorted(DF.items(), key=lambda kv: kv[1], reverse=True)
        selected_terms = [item[0] for item in sorted_DF[:top_k]]

    # Hitung vektor TF-IDF
    df["TF_IDF_Vec"] = df["TF-IDF_dict"].apply(lambda tfidf: calc_TF_IDF_Vec(tfidf, selected_terms))

    # Ranking berdasarkan bobot total
    TF_IDF_Vec_List = np.array(df["TF_IDF_Vec"].to_list())
    sums = TF_IDF_Vec_List.sum(axis=0)

    data = []
    for col, term in enumerate(selected_terms):
        data.append((term, sums[col]))

    ranking = pd.DataFrame(data, columns=['term', 'rank'])
    ranking = ranking.sort_values('rank', ascending=False)

    return ranking
