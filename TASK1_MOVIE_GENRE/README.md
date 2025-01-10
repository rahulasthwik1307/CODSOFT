# Movie Genre Classification Using Machine Learning

## Overview

This project focuses on **Movie Genre Classification** using machine learning techniques. It processes textual data related to movies (e.g., plot summaries) and predicts the genre.

## Dataset

The dataset includes movie-related data such as:

- Movie titles
- Plot summaries or descriptions
- Genres (target labels)

## Features

- **Textual Data**: Movie descriptions or plot summaries.
- **TF-IDF Features**: Generated from the text data to capture important words and phrases.
- **N-grams**: Unigrams and bigrams are used to enrich feature representation.

## Libraries Used

- **NumPy** (`numpy`)
- **Pandas** (`pandas`)
- **Scikit-learn** (`sklearn`)
  - `TfidfVectorizer` for text feature extraction
  - `MultinomialNB`, `LogisticRegression`, `LinearSVC` for classification
  - `train_test_split`, `cross_val_score` for model evaluation
  - `classification_report`, `confusion_matrix`, `accuracy_score` for performance metrics
- **Matplotlib** (`matplotlib`) and **Seaborn** (`seaborn`) for visualization
- **NLTK** (`nltk`) for text preprocessing (tokenization, stopword removal, lemmatization)
- **Warnings** for suppressing unnecessary warnings

## Workflow

1. **Data Loading**: Read the dataset into a Pandas DataFrame.
2. **Preprocessing**:
   - Clean text (remove special characters, convert to lowercase).
   - Tokenize, remove stopwords, and lemmatize.
3. **Feature Extraction**: Apply **TF-IDF Vectorization** with n-grams (1,2).
4. **Model Training**:
   - Train classifiers: **Naive Bayes**, **Logistic Regression**, **Linear SVM**.
   - Use **train-test split** for validation.
5. **Evaluation**:
   - Compute **accuracy**, **confusion matrix**, and **classification report**.
   - Plot results for comparison.

## Results

Models are evaluated using metrics like accuracy, precision, recall, and F1-score. The confusion matrix and plots visualize the model's performance. The best-performing model is highlighted.

## Requirements

```bash
pip install numpy pandas scikit-learn matplotlib seaborn nltk
```

## Example Input and Output

**Input:**\
A movie plot summary, e.g.,\
"A young boy discovers he has magical powers and attends a school for wizards."

**Output:**\
Predicted Genre: **Fantasy**


