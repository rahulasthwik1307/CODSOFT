# Customer Churn Prediction Using Machine Learning

## Project Overview

This project focuses on predicting customer churn using machine learning algorithms. It involves data preprocessing, feature engineering, model training, and evaluation to classify customers as likely to churn or remain loyal.

---

## Dataset

- **Source:** The dataset used for this project is provided within the notebook.
- **Target Variable:** `Churn` (1 for churned customers, 0 for non-churned customers).
- **Key Features (before preprocessing):**
  - `Gender`: Gender of the customer.
  - `Age`: Age of the customer.
  - `MonthlyCharges`: Monthly subscription charges.
  - `Tenure`: Duration of subscription in months.
  - Other features representing customer behavior and demographics were included.

---

## Features Used

After preprocessing, important features include:

- `Gender`: Encoded as numeric values.
- `Tenure`: Customer's subscription duration in months.
- `MonthlyCharges`: Monthly billing amount.
- Categorical features such as `Contract`, `PaymentMethod`, and `InternetService`, which were one-hot encoded.

---

## Libraries Used

- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:**
  - Models: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `XGBClassifier`
  - Preprocessing: `LabelEncoder`, `StandardScaler`
- **Evaluation Metrics:** `accuracy_score`, `classification_report`, `confusion_matrix`

---

## Workflow

1. **Data Loading:** Load the dataset.
2. **Data Cleaning:** Handle missing values and remove irrelevant columns.
3. **Feature Engineering:**
   - Encode categorical variables.
   - Standardize numerical features.
4. **Model Training:** Train Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost models.
5. **Model Evaluation:** Evaluate models using accuracy, confusion matrix, and classification reports.
6. **Results Visualization:** Plot confusion matrices and feature importance charts.

---

## Results

- **Accuracy:**
  - Gradient Boosting: 78.28%
  - Random Forest: 76.81%
  - XGB: 77.30%
  - ID3: 75.83%
  - Logistic Regression: 72.39%
  - SVC: 72.15%
- **Best Model:** Gradient Boosting achieved the highest performance.





---

## System Requirements

- Python 3.7+
- Install required libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost
  ```

---

## Example Input and Output

**Enter MonthlyCharges: 75.00**\
**Enter Tenure: 12**\
**Enter Contract\_Month-to-month: 1**\
**Enter Contract\_One year: 0**\
**Enter Contract\_Two year: 0**

**Probability of Churn: 0.85**

The customer is: Likely to Churn

---

