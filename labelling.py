import pandas as pd

def load_lexicon(file_path="lexicon_sentimen.csv"):
    try:
        lexicon_df = pd.read_csv(file_path)
        positive_lexicon = lexicon_df[lexicon_df['sentiment'] == 'positive']['word'].tolist()
        negative_lexicon = lexicon_df[lexicon_df['sentiment'] == 'negative']['word'].tolist()
        return positive_lexicon, negative_lexicon
    except Exception as e:
        raise IOError(f"Gagal membaca lexicon: {e}")

def lexicon_label(text, positive_lexicon, negative_lexicon):
    pos_count = sum(1 for word in text.split() if word in positive_lexicon)
    neg_count = sum(1 for word in text.split() if word in negative_lexicon)
    if pos_count > neg_count:
        return "positif"
    elif neg_count > pos_count:
        return "negatif"
    else:
        return "netral"

def apply_lexicon_labeling(df, text_column='stemming', lexicon_path="lexicon_sentimen.csv"):
    positive_lexicon, negative_lexicon = load_lexicon(lexicon_path)
    df['sentiment'] = df[text_column].apply(lambda text: lexicon_label(text, positive_lexicon, negative_lexicon))
    return df
