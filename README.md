# Naive Bayes vs Multinomial Logistic Regression for Sentiment Analysis

This repository contains a sentiment analysis application uses **Multinomial Logistic Regression** and **Naive Bayes** as the core machine learning model, implemented in Python, to analyze which one is better in your data.

## 📦 Application Features

The app is divided into 6 main tabs:

1. **Upload Data**  
   - Upload CSV file  
   - Remove duplicates and NaN values  
   - Display raw and cleaned data  

2. **Preprocessing**  
   - Case folding  
   - Text cleaning  
   - Slangword replacement  
   - Stopword removal  
   - Tokenization  
   - Stemming  
   - Top 10 word frequency visualization (bar chart)  
   - Custom slangword/stopword database integration  

3. **Sentiment Labeling**  
   - Automatic lexicon-based labeling using **InSet**  
   - Manual label editing  
   - Sentiment distribution visualization  
   - Save and download labeled data  

4. **TF-IDF Feature Extraction**  
   - Configure number of top features (top-k) and N-Gram range  
   - Optional **SMOTE** balancing  
   - Wordcloud and top term visualizations  
   - Download TF-IDF ranking and matrix  

5. **Model Testing**  
   - Implemented models: Naive Bayes and Logistic Regression  
   - Options: Stratified 80:20 split or K-Fold Cross Validation  
   - Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
   - View predicted vs actual samples  

6. **Model Comparison**  
   - Bar chart and pie chart visualizations of performance  
   - Visualized confusion matrices  
   - Compare models side by side  

---

## 🛠️ Technologies & Libraries

- **Python**, **Streamlit**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, **Plotly**, **WordCloud**
- **SMOTE** via `imbalanced-learn`
- **Sastrawi** for stemming
- **SQLite** for managing stopwords/slangwords

---

## 🔍 Methodology

1. **Web Scraping**: Harvest tweets using API tokens, scrape Detik.com via HTML
2. **Text Preprocessing**: Cleaning, case folding, stemming, etc.
3. **Sentiment Labeling**: Lexicon-based labeling using InSet
4. **TF-IDF**: Keyword weighting & feature extraction
5. **SMOTE**: Balance the dataset by generating synthetic minority samples
6. **Modeling & Evaluation**:
   - **Multinomial Naive Bayes**
   - **Multinomial Logistic Regression**
   - Evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## 🧪 Best Evaluation Results

| Model                | Without SMOTE | With SMOTE |
|----------------------|---------------|------------|
| Naive Bayes          | 68%           | 71%        |
| Logistic Regression  | 74%           | 75%        |

---

## ✅ Why TF-IDF?

- Reduces weight of common terms (e.g., “and”, “that”)
- More accurate than CountVectorizer
- Simple, fast, and effective for text classification

---

## ✅ Why SMOTE?

- Reduces class imbalance bias
- Improves recall for minority classes (positive/negative)
- Generates synthetic samples instead of duplicating existing ones

---

## ✅ Why These Models?

### Naive Bayes:
- Assumes word independence  
- Fast and scalable  
- Works well with simple text

### Logistic Regression:
- Uses softmax function for multiclass classification  
- No independence assumption  
- More accurate, stable, and interpretable

---

## ❌ Why Not Other Methods?

- **SVM**: Requires complex hyperparameter tuning
- **Random Forest**: Not optimal for high-dimensional text data
- **Neural Network**: Needs GPU & large datasets; overkill for this project

---

## ▶️ How to Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure (Example)

```
Skripsi_App/
├── app.py
├── preprocessing.py
├── naive_bayes.py
├── mlr.py
├── tfidf.py
├── db.py
├── data/
│   └── dataset.csv
├── assets/
│   └── wordclouds/
├── templates/
├── README.md
└── requirements.txt
```

Developed by FingersArts 🚗
