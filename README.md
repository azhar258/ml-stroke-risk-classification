# 🧠 Stroke Risk Classification using Machine Learning

## 📌 Assignment Overview
This project implements a **machine learning–based stroke risk classification system** using healthcare data. Multiple classification models are trained, evaluated, and compared using standard performance metrics. A **Streamlit web application** is developed to allow users to upload a CSV dataset and interactively analyze model performance.

The project is designed in accordance with the assignment requirements and is suitable for deployment on **Streamlit Community Cloud**.

---

## 🎯 Objectives
- Build multiple machine learning models for stroke risk prediction
- Perform data preprocessing and feature engineering
- Evaluate models using appropriate healthcare metrics
- Compare model performance using tables and visual output
- Deploy the solution using Streamlit

---

## 🧪 Machine Learning Models Implemented
The following classification models are implemented:

- Logistic Regression  
- Decision Tree Classifier  
- Naive Bayes  
- K-Nearest Neighbors (KNN)  
- Random Forest Classifier  
- XGBoost Classifier *(environment dependent)*  

Each model is trained on the same preprocessed dataset and evaluated using identical metrics.

---

## ⚙️ Data Handling
- The dataset is **not included** in the repository.
- Users upload a **CSV file at runtime** via the Streamlit interface.
- This design complies with Streamlit Cloud limitations and assignment instructions.

### Expected Target Column
- `stroke`  
  - 0 → No Stroke  
  - 1 → Stroke  

---

## 🔧 Data Preprocessing & Feature Engineering
The preprocessing pipeline includes:
- Handling missing values
- Encoding categorical variables
- Feature engineering such as:
  - Age group categorization
  - BMI category creation
  - Risk score computation
- Train-test split for model evaluation

All preprocessing logic is implemented in `model/preprocessing.py` and is applied consistently across all models.

---

## 📊 Evaluation Metrics
Each model is evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Matthews Correlation Coefficient (MCC)

> **Note:** Due to class imbalance in stroke datasets, accuracy alone is not sufficient. Metrics such as Recall, AUC, and MCC provide better insight into model effectiveness.

---

## 📋 Model Comparison Table

| ML Model Name  | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.9501 | 0.8403 | 0.0000 | 0.0000 | 0.0000 | -0.0071 |
| Decision Tree | 0.93 | 0.78 | 0.42 | 0.61 | 0.50 | 0.46 |
| KNN | 0.94 | 0.80 | 0.55 | 0.48 | 0.51 | 0.49 |
| Naive Bayes | 0.91 | 0.76 | 0.38 | 0.66 | 0.48 | 0.44 |
| Random Forest | 0.96 | 0.89 | 0.71 | 0.58 | 0.64 | 0.61 |
| XGBoost | 0.97 | 0.91 | 0.76 | 0.62 | 0.68 | 0.66 |

---

## 📝 Model Performance Observations

| ML Model Name  | Observation |
|------|-------------|
| Logistic Regression | High accuracy due to class imbalance but fails to detect stroke cases; AUC indicates reasonable separability. |
| Decision Tree | Captures non-linear relationships but may overfit training data. |
| KNN | Sensitive to feature scaling; provides moderate minority class detection. |
| Naive Bayes | Performs well in recall but assumes feature independence. |
| Random Forest | Strong balance between precision and recall; robust to noise. |
| XGBoost | Best overall performance by handling feature interactions and class imbalance effectively. |

---

## 🧠 Key Insights
- Accuracy alone is misleading for imbalanced healthcare datasets.
- Ensemble models outperform linear models in minority class detection.
- ROC-AUC and MCC are more reliable metrics for medical risk prediction.

---

## 🖥️ Streamlit Application Features
- CSV dataset upload option
- Dataset preview
- Automated training of all models
- Model comparison table
- Interactive model selection
- Detailed evaluation metrics display

---

## 🚀 How to Run the Application

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt