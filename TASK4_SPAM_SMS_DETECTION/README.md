# SPAM SMS Detection Using Machine Learning

## Project Overview

This project is focused on detecting spam SMS messages using various machine learning algorithms. The dataset is preprocessed, and text data is transformed into numeric representations to enable classification into `Spam` or `Ham` (non-spam) messages.

---

## Dataset

- **Source:** SMS Spam Collection dataset (`spam.csv`).
- **Target Variable:** `target` (1 for spam messages, 0 for ham messages).
- **Key Features (before preprocessing):**
  - `text`: The content of the SMS message.

---

## Features Used

After preprocessing, the main features include:

- **Textual Features:**
  - `num_characters`: Number of characters in the message.
  - `num_words`: Number of words in the message.
  - `num_sentences`: Number of sentences in the message.
- **Transformed Text:** Preprocessed version of the original text, including stemming and removal of stopwords and punctuation.
- **Vectorized Features:** Text data transformed using `TfidfVectorizer`.

---

## Libraries Used

- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `wordcloud`
- **Text Preprocessing:** `nltk`
- **Machine Learning:**
  - Models: `MultinomialNB`, `SVC`, `RandomForestClassifier`, `XGBClassifier`, and others.
  - Preprocessing: `CountVectorizer`, `TfidfVectorizer`
- **Evaluation Metrics:** `accuracy_score`, `precision_score`, `confusion_matrix`

---

## Workflow

1. **Data Loading:** Load the dataset.
2. **Data Cleaning:**
   - Remove unnecessary columns.
   - Rename columns for clarity.
   - Handle duplicates and missing values.
3. **Exploratory Data Analysis (EDA):**
   - Visualize message lengths and word distributions.
   - Generate word clouds for spam and ham messages.
4. **Text Preprocessing:**
   - Tokenize text.
   - Remove stopwords and punctuation.
   - Apply stemming to reduce words to their root form.
5. **Feature Engineering:**
   - Extract `num_characters`, `num_words`, and `num_sentences`.
   - Vectorize text using `TfidfVectorizer`.
6. **Model Training:**
   - Train models: `MultinomialNB`, `SVC`, `RandomForestClassifier`, and others.
   - Compare performance based on accuracy and precision.
7. **Model Evaluation:**
   - Evaluate models using accuracy, precision, and confusion matrices.
   - Select the best-performing model.
8. **Model Deployment:**
   - Save the vectorizer and model using `pickle` for deployment.

---

## Results

- **Performance Comparison:**

| Algorithm     | Accuracy | Precision |
| ------------- | -------- | --------- |
| MultinomialNB | 96.73%   | 95.88%    |
| SVC           | 97.11%   | 96.00%    |
| RandomForest  | 97.58%   | 96.50%    |
| XGBClassifier | 97.21%   | 96.35%    |

- **Best Model:** Random Forest achieved the highest performance.

---

## System Requirements

- Python 3.7+
- Install required libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn nltk scikit-learn xgboost wordcloud
  ```
