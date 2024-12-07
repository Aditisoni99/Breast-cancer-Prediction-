# Python-final-project
This is a final project for python course
# Overview of Breast Cancer Prediction Using Machine Learning Models:
This project aims to predict whether a breast tumor is malignant or benign using two machine learning models: Decision Tree and Naïve Bayes. The dataset contains diagnostic measurements of breast tumors, and the models are trained to classify tumors based on these measurements.

**The project explores every stage, including data preprocessing, feature selection, model training, evaluation, and insights into performance metrics.**

# Dataset Description
The dataset contains 569 records of breast tumor diagnostics with 33 features. Each record is labeled as:

M (Malignant): Cancerous Tumors that spread to other parts of the body

B (Benign): Cancerous Tumors that are self-limiting and comparatively less harmful.

Key Features:

 **radius_mean:** Mean distances from the center to points on the tumor boundary.
 
**texture_mean:** Standard deviation of gray-scale values.

**area_mean:** Mean area of the tumor.

**smoothness_mean:** Smoothness of the tumor surface.

# Data Preprocessing

**Irrelevant Feature Removal:**

Columns id and Unnamed: 32 were removed as they were irrelevant for predictions.

**Target Encoding:**

The target variable diagnosis was encoded as:

M (Malignant) → 1
B (Benign) → 0

**Scaling:**

Continuous features (e.g., radius_mean, area_mean) were normalized using StandardScaler to ensure all features were on the same scale, preventing larger-valued features from dominating the model.

**Exploratory Data Analysis:**

Visualizations, such as boxplots and scatter plots, revealed distinct patterns between benign and malignant cases, confirming the importance of certain features.

The dataset shows a class imbalance, with fewer malignant cases than benign ones, which could impact model performance.
# Class Distribution:

Benign (B): 357 cases

Malignant (M): 212 cases

**Observation:** The dataset has a class imbalance, with more benign cases than malignant ones.

# Exploratory Data Analysis (EDA): Key Feature Insights

**Critical Features:**

radius_mean, area_mean, and concavity_mean:

These features demonstrate clear class separation between benign and malignant cases, making them highly influential for predictions.

For example:

area_mean: Malignant tumors typically have larger mean areas (~2500) compared to benign tumors (~1000), as shown in the bar plots.

radius_mean: Malignant cases have significantly higher radius values compared to benign cases, with little overlap.

concavity_mean: Malignant tumors exhibit higher concavity values, indicating more irregular growth patterns.

**Impact on Models:**

These features were prioritized during model training due to their strong correlation with the target variable (diagnosis).
Moderate Contribution:

texture_mean:

While this feature shows overlap between benign and malignant cases, it still contributes valuable information when combined with other features.

**Outliers:**

Outliers are present in features like area_mean and concavity_mean, but they do not significantly disrupt the separation between classes.

Outliers were retained to avoid data loss, as they represent real-world variations.

**Visual Validation:**

Boxplots and scatterplots confirmed the effectiveness of key features, such as:

radius_mean and area_mean: Show significant separation, with little overlap between the two classes.

**Correlation Heatmap:**

Features like radius_mean, perimeter_mean, and area_mean were found to be highly correlated

**Feature Correlation:**

The correlation heatmap revealed clusters of highly correlated features (e.g., radius_mean and perimeter_mean), indicating redundancy.

**Impact on Feature Selection:**

Highly correlated features were retained in this project to maintain interpretability

# Feature Relationships: Radius Mean vs. Area Mean

**Strong Correlation:**

radius_mean and area_mean have a clear linear relationship, making them effective for predicting tumor types.

**Clustered Distribution:**

Malignant tumors generally have larger values for both features, while benign tumors are smaller.

Some data points **overlap**, which may lead to misclassification, suggesting room for improvement in the model.

# Data Normalization

Continuous features (e.g., radius_mean, texture_mean, area_mean) were normalized using StandardScaler to ensure all features are on the same scale.

This step is essential for models sensitive to feature magnitudes, such as those relying on distances or gradient calculations.

**Process:**

The selected features were transformed to have a mean of 0 and a standard deviation of 1.

This adjustment helps prevent features with larger scales from dominating the model training process.

**Summary Statistics:**

**Mean:** After normalization, the mean of each feature is close to 0, confirming the scaling was applied correctly.

**Standard Deviation:** Features now have a standard deviation near 1, indicating consistent scaling.

**Range:** The minimum and maximum values reflect the normalized range, making the data suitable for model input.

# Train-Test Split

Split the dataset into training and testing sets using an 80-20 ratio.

Ensured reproducibility by setting random_state=42.

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

# Predictions

**Predictions were made on the test dataset using both models:**

Decision Tree: y_pred_dt

Naive Bayes: y_pred_nb


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

# Cross-Validation:

k-fold cross-validation was used for robust performance evaluation:

**Decision Tree:**

Average Accuracy: 93.33%

Average Precision: 91.26%

Average Recall: 90.43%

Average F1 Score: 90.54%

**Note:** This approach reduces the risk of overfitting and ensures that the model is tested on all data points.
  
**Naïve Bayes:**

Average Accuracy: 94.02%

Average Precision: 94.46%

Average Recall: 89.32%

Average F1 Score: 91.58%
  
# 6. Insights and Observations

**1.Feature Importance:**
Features like radius_mean and area_mean were critical in distinguishing between benign and malignant cases.

# Class Imbalance:

More benign cases than malignant ones, which could bias model predictions.

SMOTE (Synthetic Minority Oversampling Technique) was suggested to handle this issue:

1) Creates unique synthetic samples for the minority class.
   
2) Balances the dataset to allow models to learn equally from both classes.
  
   **Decision Tree vs. Naïve Bayes:**

Naïve Bayes had higher accuracy and fewer misclassifications overall.

Decision Tree provided better interpretability and recall for malignant cases.

# 7. Limitations

**Class Imbalance:**
Malignant cases were under-represented, potentially affecting recall.

**Feature Correlation:**
Highly correlated features may introduce redundancy.

**Generalization:**
Results are based on the given dataset and may not generalize well to new data without external validation.

# 8. Recommendations

**Handle Class Imbalance:**
Use SMOTE to oversample minority cases or assign class weights in the model.

**Optimize Decision Tree:**
Tune parameters like max_depth and min_samples_split to reduce overfitting.

**Test Advanced Models:**
Implement ensemble methods like Random Forest or Gradient Boosting for improved accuracy and robustness.
