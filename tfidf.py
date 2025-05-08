import pandas as pd
import numpy as np
import ast

# Define functions for TF-IDF calculation
def convert_text_list(texts):
    if isinstance(texts, list):
        return texts
    return ast.literal_eval(texts)

def calc_TF(document):
    TF_dict = {}
    for term in document:
        TF_dict[term] = TF_dict.get(term, 0) + 1
    for term in TF_dict:
        TF_dict[term] /= len(document)
    return TF_dict

def calc_DF(tfDict):
    count_DF = {}
    for document in tfDict:
        for term in document:
            count_DF[term] = count_DF.get(term, 0) + 1
    return count_DF

def calc_IDF(__n_document, __DF):
    IDF_Dict = {}
    for term in __DF:
        IDF_Dict[term] = np.log(__n_document / (__DF[term] + 1))
    return IDF_Dict

def calc_TF_IDF(TF, IDF):
    TF_IDF_Dict = {}
    for key in TF:
        TF_IDF_Dict[key] = TF[key] * IDF[key]
    return TF_IDF_Dict

def calc_TF_IDF_Vec(__TF_IDF_Dict, unique_term):
    TF_IDF_vector = [0.0] * len(unique_term)
    for i, term in enumerate(unique_term):
        if term in __TF_IDF_Dict:
            TF_IDF_vector[i] = __TF_IDF_Dict[term]
    return TF_IDF_vector

def calculate_tfidf(df):
    df["text_list"] = df["tokenized"].apply(convert_text_list)
    df["TF_dict"] = df["text_list"].apply(calc_TF)
    DF = calc_DF(df["TF_dict"])
    n_document = len(df)
    IDF = calc_IDF(n_document, DF)
    df["TF-IDF_dict"] = df["TF_dict"].apply(lambda tf: calc_TF_IDF(tf, IDF))
    
    sorted_DF = sorted(DF.items(), key=lambda kv: kv[1], reverse=True)
    unique_term = [item[0] for item in sorted_DF]
    
    df["TF_IDF_Vec"] = df["TF-IDF_dict"].apply(lambda tfidf: calc_TF_IDF_Vec(tfidf, unique_term))
    
    TF_IDF_Vec_List = np.array(df["TF_IDF_Vec"].to_list())
    sums = TF_IDF_Vec_List.sum(axis=0)
    
    data = []
    for col, term in enumerate(unique_term):
        data.append((term, sums[col]))
    
    ranking = pd.DataFrame(data, columns=['term', 'rank'])
    ranking = ranking.sort_values('rank', ascending=False)
    
    return ranking