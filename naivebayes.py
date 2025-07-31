from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union

class NaiveBayes:
    
    def __init__(self, alpha: float = 1.0):

        # Inisialisasi model Naive Bayes
        
        self.alpha = alpha  # Laplace smoothing
        self.class_priors = {}  # Prior probability untuk setiap kelas
        self.feature_probs = {}  # Conditional probability untuk setiap fitur
        self.classes = None  # Daftar kelas unik
        self.feature_count = None  # Jumlah fitur/dimensi
        self.feature_sums = None  # Jumlah nilai fitur per kelas
        self.class_counts = None  # Jumlah sampel per kelas
        self.selected_terms = None  # Menyimpan fitur (term) yang digunakan
        
    def fit(self, X: List[List[float]], y: List[Union[int, str]], selected_terms: Dict = None):

        # Melatih model Naive Bayes menggunakan data pelatihan

        X_array = np.array(X)
        y_array = np.array(y)
        
        # Simpan selected_terms jika diberikan
        self.selected_terms = selected_terms
        
        # Identifikasi kelas unik
        self.classes = np.unique(y_array)
        self.feature_count = X_array.shape[1]
        
        # Hitung prior probability untuk setiap kelas
        self.class_counts = {}
        total_samples = len(y_array)
        
        for c in self.classes:
            class_samples = np.sum(y_array == c)
            self.class_counts[c] = class_samples
            self.class_priors[c] = class_samples / total_samples
            
        # Hitung likelihood untuk setiap fitur di setiap kelas
        self.feature_sums = {}
        self.feature_probs = {}
        
        for c in self.classes:
            # Filter data untuk kelas tertentu
            X_class = X_array[y_array == c]
            
            # Jumlahkan nilai untuk setiap fitur dalam kelas ini
            feature_sum = np.sum(X_class, axis=0)
            self.feature_sums[c] = feature_sum
            
            # Total nilai fitur untuk kelas ini
            total_sum = np.sum(feature_sum)
            
            # Hitung probabilitas kondisional dengan Laplace smoothing
            smoothed_probs = (feature_sum + self.alpha) / (total_sum + self.alpha * self.feature_count)
            self.feature_probs[c] = smoothed_probs
        
        return self
    
    def predict_proba(self, X: List[List[float]]) -> np.ndarray:
        
        # Prediksi probabilitas untuk setiap kelas

        X_array = np.array(X)
        n_samples = X_array.shape[0]
        n_classes = len(self.classes)
        
        # Inisialisasi array untuk probabilitas (log probability)
        log_probs = np.zeros((n_samples, n_classes))
        
        # Hitung log probability untuk setiap sampel dan kelas
        for i, c_idx in enumerate(range(len(self.classes))):
            c = self.classes[c_idx]
            
            # Log prior
            log_prior = np.log(self.class_priors[c])
            
            # Log likelihood untuk semua fitur
            # Gunakan perkalian dot untuk efisiensi (dalam log, ini menjadi penjumlahan)
            log_likelihood = X_array @ np.log(self.feature_probs[c])
            
            # Log posterior (tidak dinormalisasi)
            log_probs[:, i] = log_prior + log_likelihood
            
        # Konversi dari log space ke probabilitas dan normalisasi
        # Gunakan trick untuk mencegah underflow numerik
        log_prob_max = np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs - log_prob_max)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X: List[List[float]]) -> np.ndarray:
        
        # Prediksi kelas untuk sampel
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]


def prepare_naive_bayes_model(X_train, y_train, selected_terms=None, alpha=1.0):
    
    # Fungsi untuk mempersiapkan model Naive Bayes
    nb_model = NaiveBayes(alpha=alpha)
    nb_model.fit(X_train, y_train, selected_terms=selected_terms)
    return nb_model


def evaluate_model(model, X_test, y_test):
    
    # Evaluasi model dan tampilkan metrik
    # Dapatkan prediksi
    y_pred = model.predict(X_test)
    y_test_array = np.array(y_test)
    
    # Hitung akurasi
    accuracy = np.mean(y_pred == y_test_array)
    
    # Hitung confusion matrix
    classes = np.unique(y_test_array)
    n_classes = len(classes)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i in range(len(y_test_array)):
        true_idx = np.where(classes == y_test_array[i])[0][0]
        pred_idx = np.where(classes == y_pred[i])[0][0]
        conf_matrix[true_idx, pred_idx] += 1
    
    # Hitung presisi, recall, dan f1-score untuk setiap kelas
    precision = {}
    recall = {}
    f1_score = {}
    
    for i, c in enumerate(classes):
        true_positive = conf_matrix[i, i]
        false_positive = np.sum(conf_matrix[:, i]) - true_positive
        false_negative = np.sum(conf_matrix[i, :]) - true_positive
        
        precision[c] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall[c] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0
    
    # Hitung rata-rata makro
    macro_precision = np.mean(list(precision.values()))
    macro_recall = np.mean(list(recall.values()))
    macro_f1 = np.mean(list(f1_score.values()))
    
    # Kemas hasil evaluasi dalam dictionary
    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'classes': classes
    }
    
    return results

def stratified_split(df, label_col='sentiment', test_ratio=0.2):
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=test_ratio, stratify=df[label_col], random_state=42)
    return train_df, test_df

def stratified_split_scratch(df, label_col='sentiment', test_ratio=0.2):
    # Set random seed
    np.random.seed(42)
    
    # Ambil kelas unik dari kolom label
    classes = df[label_col].unique()
    
    # Inisialisasi dataframe kosong untuk train dan test
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    # Untuk setiap kelas, bagi data dengan mempertahankan proporsi kelas
    for cls in classes:
        # Dapatkan semua sampel untuk kelas ini
        class_data = df[df[label_col] == cls]

        # Hitung jumlah sampel untuk set test
        n_test = int(len(class_data) * test_ratio)
        
        # acak indeks untuk membagi data
        indices = np.random.permutation(len(class_data))
        
        # split indeks menjadi train dan test
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        # Ambil data untuk train dan test
        train_df = pd.concat([train_df, class_data.iloc[train_indices]])
        test_df = pd.concat([test_df, class_data.iloc[test_indices]])
    
    # Reset index untuk train dan test dataframe
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    return train_df, test_df

# Fungsi untuk visualisasi confusion matrix (tetap sama)
def plot_confusion_matrix_streamlit(conf_matrix, labels):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    return fig

def k_fold_cross_validation_nb(X, y, selected_terms=None, alpha=1.0, k=5):
    from naivebayes import NaiveBayes
    from naivebayes import evaluate_model
    
    X = np.array(X)
    y = np.array(y)
    n_samples = len(y)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1
    current = 0
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    all_class_accuracies = []
    total_cm = None

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = NaiveBayes(alpha=alpha)
        model.fit(X_train, y_train, selected_terms=selected_terms)
        results = evaluate_model(model, X_test, y_test)
        accuracy_scores.append(results['accuracy'])
        precision_scores.append(list(results['precision'].values()))
        recall_scores.append(list(results['recall'].values()))
        all_class_accuracies.append(list(results['f1_score'].values()))

        if total_cm is None:
            total_cm = results['confusion_matrix'].astype(float)
        else:
            total_cm += results['confusion_matrix']
        current = stop
    
    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores, axis=0)
    avg_recall = np.mean(recall_scores, axis=0)
    avg_cm = total_cm / k
    avg_class_accuracies = np.mean(all_class_accuracies, axis=0)
    return avg_accuracy, avg_precision, avg_recall, avg_cm, avg_class_accuracies

