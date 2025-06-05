# Task 1: Iris Dataset Exploration and Visualization

## Task Objective
The objective of this task is to explore and visualize the Iris dataset to understand the underlying data distributions, feature relationships, and potential outliers. This foundational step in data analysis helps in building intuition about the data before applying any machine learning models.

## Dataset Used
- **Dataset Name:** Iris Dataset  
- **Description:** The Iris dataset contains 150 samples of iris flowers, with measurements of sepal length, sepal width, petal length, and petal width, along with species classification (Setosa, Versicolor, Virginica).  
- **Source:** Loaded using `seaborn` library's built-in dataset loader.

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

