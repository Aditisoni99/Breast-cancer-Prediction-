# Python-final-project
This is a final project for python course
# Overview of Breast Cancer Prediction Using Machine Learning Models:
This project aims to predict whether a breast tumor is malignant or benign using two machine learning models: Decision Tree and Naïve Bayes. The dataset contains diagnostic measurements of breast tumors, and the models are trained to classify tumors based on these measurements.
**The project explores every stage, including data preprocessing, feature selection, model training, evaluation, and insights into performance metrics.**
# Dataset Description
The dataset contains 569 records of breast tumor diagnostics with 32 features. Each record is labeled as:
M (Malignant): Cancerous Tumors.
B (Benign): Non-cancerous Tumors.
Key Features:
# radius_mean: Mean distances from the center to points on the tumor boundary.
# texture_mean: Standard deviation of gray-scale values.
# area_mean: Mean area of the tumor.
# smoothness_mean: Smoothness of the tumor surface.
# Data Preprocessing
**1. Irrelevant Feature Removal:**
Columns id and Unnamed: 32 were removed as they were irrelevant for predictions.
**2. Target Encoding:**
The target variable diagnosis was encoded as:
M (Malignant) → 1
B (Benign) → 0
**3. Scaling:**
Continuous features (e.g., radius_mean, area_mean) were normalized using StandardScaler to ensure all features were on the same scale.
**4.Exploratory Data Analysis**
The dataset shows a class difference, with a greater number of benign cases than malignant ones. 
# Class Distribution:
Benign (B): 357 cases
Malignant (M): 212 cases
**Observation:** The dataset has a class imbalance, with more benign cases than malignant ones.
# Feature Significance:
Box and scatter plots showed that variables such as radius_mean and area_mean indicate remarkable differentiation between benign and malignant scenarios.
**Correlation Heatmap:**
High correlation between radius_mean, perimeter_mean, and area_mean.
Redundant features were retained for model interpretability but flagged for potential optimization.
# Machine Learning Models
**Model 1: Decision Tree**
Why Used:
Provides interpretable decision-making and handles complex patterns.
Strength:
Captures non-linear relationships and ranks feature importance.
Weakness:
Slightly prone to overfitting.
**Model 2: Naïve Bayes**
Why Used:
Computationally efficient and ideal for small datasets.
Strength:
Probabilistic model ( Statistical)  that outputs class probabilities.
Weakness:
Assumes independence between features, which may not always hold.
# Model Evaluation
Metrics:
**Accuracy:** Overall correctness of the model.
**Precision:** Fraction of accurate optimistic predictions among all positive predictions.
**Recall:** Ability to identify all actual positives.
**F1-Score:** Harmonic mean of precision and recall.
**Confusion Matrix**: Breakdown of true positives, false positives, true negatives, and false negatives.
# Results:
Decision Tree:
Accuracy: 94.74%
True Positives (Malignant Correctly Predicted): 68
False Negatives (Malignant Predicted as Benign): 3
False Positives (Benign Predicted as Malignant): 3
True Negatives (Benign Correctly Predicted): 40
Naïve Bayes:
Accuracy: 96.49%
Confusion Matrix:
True Positives: 70
False Negatives: 1
False Positives: 0
True Negatives: 43
Cross-Validation:
# 10-fold cross-validation was used for robust performance evaluation:
Decision Tree:
* Average Accuracy: 93.33%
* Average Recall: 90.43%
Naïve Bayes:
* Average Accuracy: 94.02%
* Average Recall: 89.32%
  
**6. Insights and Observations**
1.  # Feature Importance:
Features like radius_mean and area_mean were critical in distinguishing between benign and malignant cases.
2. # Class Imbalance:
The dataset had more benign cases than malignant ones, which could bias the models.
So As the dataset is imbalance, so data augmentation technique (SMOTE) would be better in this scenario
Benefits of SMOTE: 1. Unlike random oversampling (which duplicates data), SMOTE reduces the risk of overfitting by generating new, unique samples.2 Balances the dataset, allowing the model to learn equally well from both classes.
3. # Decision Tree vs. Naïve Bayes:
Naïve Bayes had higher accuracy and fewer misclassifications overall.
Decision Tree provided better interpretability and recall for malignant cases.
**7. Limitations**
# Class Imbalance:
Malignant cases were under-represented, potentially affecting recall.
# Feature Correlation:
Highly correlated features may introduce redundancy.
# Generalization:
Results are based on the given dataset and may not generalize well to new data without external validation.
**8. Recommendations**
# Handle Class Imbalance:
Use SMOTE to oversample minority cases or assign class weights in the model.
# Optimize Decision Tree:
Tune parameters like max_depth and min_samples_split to reduce overfitting.
# Test Advanced Models:
Implement ensemble methods like Random Forest or Gradient Boosting for improved accuracy and robustness.
