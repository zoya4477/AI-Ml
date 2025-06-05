# Task 1: Iris Dataset Exploration and Visualization

## Task Objective
The objective of this task is to explore and visualize the Iris dataset to understand the underlying data distributions, feature relationships, and potential outliers. This foundational step in data analysis helps in building intuition about the data before applying any machine learning models.

## Dataset Used
- **Dataset Name:** Iris Dataset  
- **Description:** The Iris dataset contains 150 samples of iris flowers, with measurements of sepal length, sepal width, petal length, and petal width, along with species classification (Setosa, Versicolor, Virginica).  
- **Source:** Loaded using `pandas` library

## Models Applied
- No predictive models were applied in this task.  
- The focus was on exploratory data analysis (EDA) using statistical summaries and visualization techniques such as scatter plots, histograms, and box plots.

## Key Results and Findings
- The dataset consists of 150 instances and 5 attributes (4 numerical features + 1 categorical species label).
- Pairwise scatter plots reveal clear separability between species, especially when considering petal length and petal width.
- Histograms show distinct value distributions across species for each feature.
- Box plots highlight some outliers, particularly in sepal width, indicating variability within species.
- Overall, the petal measurements are more discriminative for species classification than sepal measurements.



## Tools and Libraries Used
- Python (Google Colab)
- pandas for data handling
- seaborn and matplotlib for visualization

This exploration provides the groundwork for subsequent model training and evaluation tasks.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Task 2: Predict Future Stock Prices (Short-Term)

## Task Objective
The goal of this task is to use historical stock market data to predict the next day’s closing price of a selected stock. This involves working with time series data, extracting relevant features, and applying regression models to make short-term forecasts.

## Dataset Used
- **Dataset Name:** Historical stock data from Yahoo Finance  
- **Source:** Retrieved dynamically using the `yfinance` Python library  
- **Example Stock:** Apple Inc. (Ticker: AAPL) or Tesla Inc. (Ticker: TSLA)  
- **Features Used:** Open, High, Low, Volume (predicting next day's Close price)

## Models Applied
- **Linear Regression:** A simple and interpretable regression model used to predict the next day’s closing price.

## Key Results and Findings
- Linear Regression model was trained on historical stock data using features Open, High, Low, and Volume to predict the next day’s Close price.
- The model achieved a **Mean Squared Error (MSE) of 4.9761**, indicating the average squared difference between predicted and actual closing prices.
- Visualization of actual vs. predicted closing prices shows how well the model captures stock price movements.
- While Linear Regression provides a straightforward baseline, further improvements may be achieved using more complex models.

---

## Tools and Libraries Used
- Python (Google Colab)  
- `yfinance` for fetching historical stock data  
- pandas for data manipulation  
- scikit-learn for Linear Regression modeling  
- matplotlib and seaborn for data visualization

This task highlights the process of handling financial time series data, performing regression, and evaluating model performance for short-term stock price prediction.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Task 3: Heart Disease Prediction

## Task Objective
The goal of this task is to build a machine learning model to predict whether a person is at risk of heart disease based on their health data. This involves data cleaning, exploratory data analysis (EDA), model training, and evaluation.

## Dataset Used
- **Dataset Name:** Heart Disease UCI Dataset  
- **Source:** Available on Kaggle  
- **Description:** Contains various health-related attributes such as age, sex, cholesterol levels, cp, fbs, exang and others to predict the presence or absence of heart disease.

## Models Applied
- **Logistic Regression:** A binary classification algorithm used to predict the likelihood of heart disease risk.

## Key Results and Findings
- The Logistic Regression model achieved an **accuracy of 91.67%**, indicating high correctness in classification.  
- The model’s **ROC AUC score of 0.9509** demonstrates excellent capability in distinguishing between patients with and without heart disease.  
- Exploratory Data Analysis revealed important health features correlated with heart disease risk.  
- Feature importance analysis highlighted key predictors, helping interpret the model and providing valuable medical insights.  
- Confusion matrix and ROC curve visualizations were used to evaluate model performance thoroughly.

## Tools and Libraries Used
- Python (Google Colab)  
- pandas and numpy for data processing  
- matplotlib and seaborn for visualization  
- scikit-learn for Logistic Regression modeling and evaluation metrics

This task emphasizes medical data understanding, binary classification, and model evaluation techniques such as ROC curves and confusion matrices to build a robust heart disease prediction system.
