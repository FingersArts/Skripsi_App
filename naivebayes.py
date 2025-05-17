import math
import random
from collections import defaultdict, Counter
import seaborn as sns
import matplotlib.pyplot as plt

def train_naive_bayes(tfidf_dicts, labels, alpha=0.01):
    class_docs = defaultdict(list)
    for tfidf, label in zip(tfidf_dicts, labels):
        class_docs[label].append(tfidf)

    classes = set(labels)
    class_priors = {cls: len(class_docs[cls]) / len(labels) for cls in classes}

    # Bangun global vocabulary
    vocab = set()
    for doc in tfidf_dicts:
        vocab.update(doc.keys())

    class_term_probs = {}
    for cls in classes:
        term_counts = defaultdict(float)

        for doc in class_docs[cls]:
            for term, tfidf in doc.items():
                term_counts[term] += tfidf

        # Total bobot + smoothing denominator
        total_sum = sum(term_counts.values()) + alpha * len(vocab)

        # Smoothing eksplisit: semua term dalam vocab harus muncul
        probs = {}
        for term in vocab:
            probs[term] = (term_counts.get(term, 0.0) + alpha) / total_sum

        class_term_probs[cls] = probs

    return {
        "priors": class_priors,
        "term_probs": class_term_probs,
        "vocab": vocab
    }


def predict(tfidf_dict, model):
    scores = {}
    for cls in model['priors']:
        log_prob = math.log(model['priors'][cls])
        for term, tfidf in tfidf_dict.items():
            prob = model['term_probs'][cls].get(term, 0.01)  # fallback kalau tidak ada (shouldn’t happen)
            log_prob += tfidf * math.log(prob)
        scores[cls] = log_prob
    return max(scores, key=scores.get)


def stratified_split(df, label_col='sentiment', test_ratio=0.2, seed=42):
    random.seed(seed)
    grouped = defaultdict(list)

    # Kelompokkan data berdasarkan label
    for i, row in df.iterrows():
        grouped[row[label_col]].append(i)

    train_indices = []
    test_indices = []

    for label, indices in grouped.items():
        random.shuffle(indices)
        split_point = int(len(indices) * (1 - test_ratio))
        train_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)

    return train_df, test_df

def evaluate_model(y_true, y_pred):
    classes = sorted(set(y_true + y_pred))
    label_to_idx = {label: i for i, label in enumerate(classes)}
    matrix = [[0]*3 for _ in range(3)]
    report = {}

    correct = 0  # Untuk akurasi

    for true, pred in zip(y_true, y_pred):
        i, j = label_to_idx[true], label_to_idx[pred]
        matrix[i][j] += 1
        if true == pred:
            correct += 1

    for i, label in enumerate(classes):
        TP = matrix[i][i]
        FP = sum(matrix[j][i] for j in range(3)) - TP
        FN = sum(matrix[i][j] for j in range(3)) - TP
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        report[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1-score": round(f1, 3),
            "support": sum(matrix[i])
        }

    accuracy = correct / len(y_true)
    return report, matrix, round(accuracy, 4)

# Fungsi visualisasi confusion matrix
def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    return fig