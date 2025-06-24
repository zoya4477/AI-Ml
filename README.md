# AI-ML 

This repository summarizes the AI/ML tasks completed during my internship, including work in data analysis, regression, classification, and natural language processing (NLP). Each task involved working with real-world datasets and applying relevant models to derive insights.

---

## 🔍 Tasks Overview

### **Task 1: Iris Dataset Exploration**
- **Objective:** Perform exploratory data analysis (EDA) on the Iris dataset to understand class separability.
- **Dataset:** [UCI Iris Dataset](https://archive.ics.uci.edu/ml/datasets/Iris)
- **Models:** None (EDA only)
- **Key Findings:**
  - Petal features show clear separation between species.
  - Dataset is clean and well-suited for classification tasks.

---

### **Task 2: Stock Price Prediction**
- **Objective:** Predict Apple Inc. (AAPL) next-day closing price using historical data.
- **Dataset:** yFinance API (Jan 2020 – Jan 2024)
- **Model Used:** Linear Regression
- **Results:**
  - **MSE:** 4.9761
  - **Insight:** Model performs reasonably well for trend prediction. Future work can include LSTM or ARIMA for improved performance.

---

### **Task 3: Heart Disease Prediction**
- **Objective:** Predict presence of heart disease using clinical data.
- **Dataset:** [UCI Heart Disease Dataset](https://www.kaggle.com/ronitf/heart-disease-uci)
- **Model Used:** Logistic Regression
- **Results:**
  - **Accuracy:** 0.9167%
  - **ROC AUC Score:** 0.9505
  - **Insight:** Model is interpretable; top features include thalach, oldpeak, and cp.

---

### **Task 5: Mental Health Chatbot (NLP)**
- **Objective:** Develop a chatbot capable of empathetic responses.
- **Dataset:** [EmpatheticDialogues – Facebook AI](https://github.com/facebookresearch/EmpatheticDialogues)
- **Model Used:** DistilGPT2 (Hugging Face)
- **Results:**
  - Fine-tuned with limited resources using the Trainer API.
  - Model generates context-aware and emotionally intelligent replies.

---

### **Task 6: House Price Prediction**
- **Objective:** Predict housing prices based on physical and locational attributes.
- **Dataset:** [Kaggle House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Models Used:** Linear Regression, Gradient Boosting Regressor
- **Results:**
  - **Best MAE (GBR):**  24716.54 
  - **Best RMSE (GBR):** 36822.92
  - **Insight:** Gradient Boosting outperforms linear regression; encoding features like Neighborhood improved accuracy.

---

## 📌 Summary Table

| Task     | Type           | Model               | Metric           |
|----------|----------------|---------------------|------------------|
| Task 1   | EDA            | --                  | --               |
| Task 2   | Regression      | Linear Regression   | MSE: 4.9761       |
| Task 3   | Classification | Logistic Regression | AUC: 0.9505        |
| Task 5   | NLP Chatbot    | DistilGPT2          | Fine-tuned       |
| Task 6   | Regression      | Gradient Boosting   | RMSE: 36822.92     |

---

## 📂 Resources

- 📁 **GitHub Repository:** [zoya4477/AI-ML](https://github.com/zoya4477/AI-Ml)  
- 📓 **Google Colab Notebook:** [View Here](https://colab.research.google.com/drive/1SR0yLmwUS68UMHsDGHSB1Ofzt_kNPzYY?usp=sharing)

---

## ✅ Conclusion

This internship provided practical experience in solving real-world problems using AI and ML. From classical models like Logistic Regression to fine-tuning transformer models for NLP, each task helped strengthen my understanding of data preprocessing, model selection, evaluation, and result interpretation.
