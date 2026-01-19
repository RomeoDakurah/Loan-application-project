# Loan Application Prediction Project

## Overview
This project focuses on predicting whether a loan application will be approved using machine learning. The goal is to classify applications as approved or not, based on applicant information such as income, credit history, and employment status.  

---

## Dataset
The dataset contains information about loan applicants, including:

- Gender  
- Marital status  
- Education  
- Self-employment status  
- Applicant and co-applicant income  
- Loan amount and term  
- Credit history  
- Property area  

**Output:** A binary label indicating loan approval (1 = approved, 0 = not approved).

---

## Models and Hyperparameters

We tested three machine learning models. Each model was trained using the same feature set and data splits, and tuned using **GridSearchCV with 5-fold cross-validation**.

### 1. Logistic Regression (Baseline)
**Hyperparameters tuned:**
- `C`: `[0.01, 0.1, 1, 10]`  
- `penalty`: `['l2']`  
- `solver`: `['lbfgs']`  
- `max_iter`: `[100, 500, 1000]`  

**Tuning method:** Grid search with **accuracy** as the scoring metric.

### 2. Random Forest
**Hyperparameters tuned:**
- `n_estimators`: `[50, 100, 200]`  
- `max_depth`: `[None, 10, 20]`  
- `min_samples_split`: `[2, 5]`  
- `min_samples_leaf`: `[1, 2]`  

**Tuning method:** Grid search with 5-fold cross-validation.

### 3. MLP (Multi-Layer Perceptron)
**Hyperparameters tuned:**
- `hidden_layer_sizes`: `[(50,), (100,), (100, 100)]`  
- `activation`: `['relu', 'tanh']`  
- `solver`: `['adam', 'sgd']`  
- `learning_rate`: `['constant', 'adaptive']`  
- `max_iter`: `500`  

**Tuning method:** Grid search with 5-fold cross-validation.

---

## Methodology
1. **Data Preprocessing**  
   - Handle missing values  
   - Encode categorical variables  
   - Scale numerical features  

2. **Model Training**  
   - Train each algorithm with optimized hyperparameters from grid search  

3. **Evaluation**  
   - Metrics: **accuracy, precision, recall, F1 score**  
   - Compare models on a separate test set  

---

## Results

| Model | Accuracy (%) | Precision | Recall | F1 Score |
|-------|-------------|----------|--------|----------|
| Logistic Regression | 82.29 | 0.80 | 1.00 | 0.89 |
| Random Forest | 82.29 | 0.81 | 0.97 | 0.89 |
| MLP (Neural Net) | 69.79 | 0.71 | 0.96 | 0.82 |

**Analysis:**

- Logistic Regression and Random Forest achieved the same overall accuracy, but Random Forest had slightly better precision and recall.  
- MLP had the lowest accuracy and F1 score, suggesting it did not generalize as well on this dataset.  
- Logistic Regression performed best in balancing performance and stability.  
- Random Forest excelled in recall but had a small increase in false negatives.  
- MLP misclassified more loan rejections as approvals, leading to higher false positives and lower overall performance.  
- Confusion matrices indicate Logistic Regression predicts approvals well (no false negatives) but with more false positives, while Random Forest slightly reduces false positives.

---

## How to Run
1. Open the notebook in Google Colab.  
2. Ensure the dataset (`loan_data.csv`) is uploaded to session storage.  
3. Run all cells to train models and evaluate performance.

---
