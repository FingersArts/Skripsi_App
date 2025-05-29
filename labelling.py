import pandas as pd

def load_scored_lexicon():
    url_positive = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv"
    url_negative = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"

    try:
        pos_df = pd.read_csv(url_positive, sep="\t", header=None, names=["word", "score"])
        neg_df = pd.read_csv(url_negative, sep="\t", header=None, names=["word", "score"])

        # Menangani nilai non-numeric di kolom 'score'
        pos_df = pos_df[pd.to_numeric(pos_df['score'], errors='coerce').notnull()]
        neg_df = neg_df[pd.to_numeric(neg_df['score'], errors='coerce').notnull()]

        # Konversi kolom 'score' ke integer
        pos_df['score'] = pos_df['score'].astype(int)
        neg_df['score'] = neg_df['score'].astype(int)

        # Gabungkan data dan buat kamus
        lexicon_df = pd.concat([pos_df, neg_df], ignore_index=True)
        lexicon_dict = dict(zip(lexicon_df['word'], lexicon_df['score']))

        return lexicon_dict
    except Exception as e:
        raise IOError(f"Gagal memuat lexicon dari GitHub: {e}")


def label_by_total_score(text_tokens, lexicon_dict):
    if not isinstance(text_tokens, list):
        print("Input bukan list kata.")
        return "netral"

    total_score = 0
    log_lines = ["Perhitungan Skor Kata:"]

    for word in text_tokens:
        score = lexicon_dict.get(word, 0)
        try:
            score = int(score)
        except:
            score = 0
        total_score += score
        log_lines.append(f"  - {word:<15}: {score}")

    log_lines.append(f"Total Skor Akhir    : {total_score}")
    log_lines.append(f"Label Sentimen      : {'positif' if total_score > 0 else 'negatif' if total_score < 0 else 'netral'}")
    
    print("\n".join(log_lines))

    if total_score > 0:
        return "positif"
    elif total_score < 0:
        return "negatif"
    else:
        return "netral"


def apply_score_based_labeling(df, text_column="stemming"):
    lexicon_dict = load_scored_lexicon()
    df['sentiment'] = df[text_column].apply(lambda tokens: label_by_total_score(tokens, lexicon_dict))
    return df