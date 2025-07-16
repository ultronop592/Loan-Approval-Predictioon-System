# Loan-Approval-Predictioon-System
Loan Approval Prediction System Using Support Vector Machine 

# 🏦 Loan Approval Prediction System using Support Vector Machine (SVM)

This machine learning project predicts whether a loan application will be approved or not based on applicant details. It uses a **Support Vector Machine (SVM)** classifier to help financial institutions make faster and more accurate loan approval decisions.

---

## 🔍 Overview

Loan approval is a critical task for banks and financial institutions. By using machine learning, this project automates the loan approval process using historical data of applicants and their loan statuses.

---

## 🎯 Objective

- To develop a binary classification model that predicts **Loan Approval Status** (`Y` = Approved, `N` = Rejected).
- To assist financial institutions in automating and accelerating the decision-making process.

---

## 🧠 Machine Learning Approach

- **Algorithm Used:** Support Vector Machine (SVM)
- **Task Type:** Binary Classification
- **Evaluation Metric:** Accuracy Score

---

## 🛠️ Technologies & Tools

- **Language:** Python
- **IDE/Notebook:** Jupyter Notebook / Google Colab
- **Libraries:**
  - `pandas`, `numpy` – Data manipulation
  - `matplotlib`, `seaborn` – Data visualization
  - `sklearn` – Machine learning modeling and preprocessing

---

## 📊 Dataset Details

- **Source:** [Open-source Loan Prediction Dataset]
- **Features Used:**
  - Categorical: `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`
  - Numerical: `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`
- **Target:**
  - `Loan_Status` – `Y` (Approved), `N` (Rejected)

---

## 🧼 Data Preprocessing

- Handled missing values using imputation (mean/mode strategy)
- Applied **Label Encoding** to categorical variables
- **Feature Scaling** performed to normalize numerical data for better SVM performance

---

##  Model Building

- Model Used:support Vector Classifier (`sklearn.svm.SVC`)
- **Data Split:** 80% Training, 20% Testing
- **Performance:**
  - **Training Accuracy:** ~80.27%
  - **Testing Accuracy:** ~78.08%

---

  Key Insights

- **Credit_History** and **ApplicantIncome** are the most influential features in determining loan approval.
- Feature scaling significantly improved the SVM model’s performance.
- The dataset had slight class imbalance but SVM handled it fairly well.

