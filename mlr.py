import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Logistic Regression Model
class LogisticRegression:
    def __init__(self, num_iter, learning_rate):
        self.num_iter = num_iter
        self.learning_rate = learning_rate

    def compute_cost(self, X, y):
        num_samples = X.shape[0]
        scores = np.dot(X, self.weights) + self.bias
        probs = softmax(scores)
        correct_logprobs = -np.log(probs[range(num_samples), y])
        cost = np.sum(correct_logprobs) / num_samples
        return cost, probs

    def compute_gradients(self, X, y, probs):
        num_samples = X.shape[0]
        dscores = probs
        dscores[range(num_samples), y] -= 1
        dscores /= num_samples
        dweights = np.dot(X.T, dscores)
        dbias = np.sum(dscores, axis=0)
        return dweights, dbias

    def train(self, X, y):
        num_features = X.shape[1]
        num_classes = np.max(y) + 1
        self.weights = np.random.randn(num_features, num_classes) * 0.01
        self.bias = np.zeros(num_classes)

        for i in range(self.num_iter):
            cost, probs = self.compute_cost(X, y)
            if i % 199 == 0:
                print('Iteration: %d, Cost: %f' % (i, cost))
            dweights, dbias = self.compute_gradients(X, y, probs)
            self.weights -= self.learning_rate * dweights
            self.bias -= self.learning_rate * dbias

    def predict(self, X):
        scores = np.dot(X, self.weights) + self.bias
        return np.argmax(scores, axis=1)

# Function to perform stratified split (80:20) of a DataFrame
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

# Function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

# Function to calculate precision
def calculate_precision(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    precision = np.zeros(num_classes)
    for class_ in range(num_classes):
        true_positives = np.sum((y_true == class_) & (y_pred == class_))
        false_positives = np.sum((y_true != class_) & (y_pred == class_))
        precision[class_] = true_positives / (true_positives + false_positives + 1e-10)
    return precision

# Function to calculate recall
def calculate_recall(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    recall = np.zeros(num_classes)
    for class_ in range(num_classes):
        true_positives = np.sum((y_true == class_) & (y_pred == class_))
        false_negatives = np.sum((y_true == class_) & (y_pred != class_))
        recall[class_] = true_positives / (true_positives + false_negatives + 1e-10)
    return recall

def calculate_f1_score(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    f1_scores = np.zeros(num_classes)
    for class_ in range(num_classes):
        true_positives = np.sum((y_true == class_) & (y_pred == class_))
        false_positives = np.sum((y_true != class_) & (y_pred == class_))
        false_negatives = np.sum((y_true == class_) & (y_pred != class_))
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1_scores[class_] = 2 * precision * recall / (precision + recall + 1e-10)
    return f1_scores

# Function to calculate the confusion matrix
def calculate_confusion_matrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        true_index = y_true[i]
        pred_index = y_pred[i]
        confusion_matrix[true_index, pred_index] += 1
    return confusion_matrix

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, labels):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    return fig

def k_fold_cross_validation(X, y, model, k=10):
    num_samples = X.shape[0]
    fold_size = num_samples // k
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    all_class_f1_scores = []
    all_precision = []
    all_recall = []

    # Get the number of classes from the entire dataset
    num_classes = len(np.unique(y))
    total_cm = np.zeros((num_classes, num_classes))

    for i in range(k):
        test_indices = indices[i*fold_size:(i+1)*fold_size]
        train_indices = np.concatenate((indices[:i*fold_size], indices[(i+1)*fold_size:]))
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = calculate_accuracy(y_test, y_pred)
        precision = calculate_precision(y_test, y_pred)
        recall = calculate_recall(y_test, y_pred)
        f1_score = calculate_f1_score(y_test, y_pred)
        confusion = calculate_confusion_matrix(y_test, y_pred, num_classes)

        total_cm += confusion
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        all_class_f1_scores.append(f1_score)
        all_precision.append(precision)
        all_recall.append(recall)

    # Convert all_class_f1_scores to numpy array for easier averaging
    all_class_f1_scores = np.array(all_class_f1_scores)
    
    # Calculate average F1-Score per class
    avg_class_f1_scores = np.nanmean(all_class_f1_scores, axis=0)

    # Make sure all_precision and all_recall are arrays of the same length
    all_precision = np.array([np.pad(p, (0, num_classes - len(p)), 'constant', constant_values=np.nan) for p in all_precision])
    all_recall = np.array([np.pad(r, (0, num_classes - len(r)), 'constant', constant_values=np.nan) for r in all_recall])

    avg_precision_per_class = np.nanmean(all_precision, axis=0)
    avg_recall_per_class = np.nanmean(all_recall, axis=0)

    return (np.mean(accuracy_scores), avg_precision_per_class, avg_recall_per_class,
            total_cm / k, avg_class_f1_scores, avg_precision_per_class, avg_recall_per_class)