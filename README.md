## Overview
[Python ML Tutorials](https://youtube.com/playlist?list=PLzMcBGfZo4-mP7qA9cagf68V06sko5otr&si=QK_LtMF9TVEPL4nb) playlist by Tech with Tim

## Setup
```bash
conda create -n python-ml-tutorial python=3.11 -y
conda activate python-ml-tutorial
pip install scikit-learn numpy pandas matplotlib
```

## Topics

### Linear Regression
Linear regression is a supervised learning algorithm used for predicting continuous values. The implementation uses scikit-learn's LinearRegression model to predict student final grades (G3) based on features like previous grades (G1, G2), study time, failures, and absences. The model finds the optimal linear relationship by minimizing the sum of squared differences between actual and predicted values. The resulting model saves coefficients and intercept that define the best-fit line equation y = mx + b. Training involves splitting data into train/test sets, fitting the model on training data, and evaluating performance using R-squared score.

### K Nearest Neighbors (KNN) Classification
KNN is a non-parametric classification algorithm that classifies data points based on majority class among their k nearest neighbors. The implementation uses car evaluation data with categorical features (buying price, maintenance cost, doors, persons, luggage boot, safety) to predict car acceptability class. Since the algorithm requires numerical input, LabelEncoder transforms categorical values to integers. For each test sample, KNN calculates Euclidean distance to all training samples, selects the k closest neighbors (k=9), and predicts the class through majority voting. The distance calculation uses feature vectors representing all car characteristics.

### Support Vector Machines (SVM)
SVM is a supervised learning algorithm that finds an optimal hyperplane to separate data points into different classes. The tutorial applies SVM to breast cancer classification using scikit-learn's built-in dataset with 30 features extracted from digitized breast mass images. The SVM with linear kernel creates a decision boundary that maximizes the margin between malignant and benign classes. The algorithm works by finding support vectors - critical data points closest to the decision boundary - and using them to define the optimal separating hyperplane. Performance comparison with KNN demonstrates SVM's effectiveness for high-dimensional medical classification tasks.

### K-means Clustering
K-means is an unsupervised learning algorithm that partitions data into k clusters by minimizing within-cluster variance. The implementation uses scikit-learn's digits dataset containing 8x8 pixel images of handwritten digits. The algorithm initializes k centroids (k=10 for digits 0-9), assigns each data point to its nearest centroid using Euclidean distance, then iteratively updates centroid positions as the mean of assigned points. This process continues until centroids converge or maximum iterations reached. The tutorial includes performance evaluation using metrics like homogeneity score, completeness score, and silhouette score, plus visualization of clusters using PCA dimensionality reduction to 2D space.