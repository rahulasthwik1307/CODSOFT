# Credit Card Fraud Detection

## Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning algorithms. It involves data preprocessing, feature scaling, model training, and evaluation to classify transactions as fraudulent or legitimate.

---

## Dataset

- **Source:** [Kaggle - Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- **Download Instructions:**
  1. Visit the [dataset link](https://www.kaggle.com/datasets/kartik2112/fraud-detection).
  2. Sign in with your Kaggle account.
  3. Click the **Download** button to get the `fraudTrain.csv` and `fraudTest.csv` files.
- **Dataset Shape:** (1,852,394 rows, 23 columns)
- **Missing Values:** None in key features (`category`, `amt`, `gender`, `city_pop`, `is_fraud`)
- **Target Variable:** `is_fraud` (1 for fraud, 0 for legitimate)
- **Key Features (before preprocessing):**
  - `amt`: Transaction amount
  - `category`: Transaction category
  - `gender`: Customer's gender
  - `city_pop`: City population
  - `merchant`: Merchant name
  - Unnecessary columns like `trans_date_trans_time`, `cc_num` were dropped during preprocessing

---

## Features Used

After preprocessing, important features include:

- `amt`: Transaction amount
- `category`: Transaction type
- `gender`: Gender of the customer
- `city_pop`: Population of the customer's city

---

## Libraries Used

- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn` (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`)
- **Evaluation Metrics:** `accuracy_score`, `classification_report`, `confusion_matrix`, `roc_auc_score`

---

## Workflow

1. **Data Download:** Download the dataset from Kaggle.
2. **Data Loading:** Read `fraudTrain.csv` and `fraudTest.csv`.
3. **Data Cleaning:** Dropped irrelevant columns.
4. **Data Preprocessing:** Applied feature scaling using `StandardScaler`.
5. **Model Training:** Trained Logistic Regression, Decision Tree, and Random Forest classifiers.
6. **Model Evaluation:** Evaluated models using accuracy, confusion matrix, and ROC-AUC score.
7. **Results Visualization:** Plotted ROC curves and confusion matrices.

---

## Results

- **Logistic Regression:**
  - **Accuracy:** 99.43%
  - **Fraud Detection:** Low recall for fraud detection (model struggles with minority class)

- **Decision Tree:**
  - **Accuracy:** 99.62%
  - **Fraud Detection:** Moderate detection with 65% precision and 64% recall

- **Random Forest:**
  - **Accuracy:** 99.73%
  - **Fraud Detection:** Best performance with 80% precision and 65% recall

**Best Model:** Random Forest achieved the highest performance.

---

## System Requirements

- Python 3.7+
- Install required libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```

---

## Example Input and Output

**Input:**
```python
Enter amt: 50000
Enter city_pop: 5000
Enter category_food_dining: 0
Enter category_gas_transport: 0
Enter category_grocery_net: 0
Enter category_grocery_pos: 0
Enter category_health_fitness: 0
Enter category_home: 0
Enter category_kids_pets: 0
Enter category_misc_net: 1
Enter category_misc_pos: 0
Enter category_personal_care: 0
Enter category_shopping_net: 1
Enter category_shopping_pos: 0
Enter category_travel: 1
Enter gender_M: 1
```

**Output:**
```python
Probability of Fraud: 0.02

The transaction is: Legitimate
```



