# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Mohammmad Ismail Ashiq Aslam
"""
#Importing pandas library
import pandas as pd
#Importing sklearn library
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


# Set the directory path where the CSV files are located
directory_path = "E:\\Work\\Portfolio\\CaseStudy-MachineLearning\\CSV"

# Import gender_submission.csv
gender_submission = pd.read_csv(directory_path + '\\gender_submission.csv')

# Import test.csv
test_data = pd.read_csv(directory_path + '\\test.csv')

# Import train.csv
train_data = pd.read_csv(directory_path + '\\train.csv')

# Create copies of the datasets
gender_submission_copy = gender_submission.copy()
test_data_copy = test_data.copy()
train_data_copy = train_data.copy()

# Load the data
data = train_data

# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# One-hot encode categorical variables
data = pd.get_dummies(data, columns=["Sex", "Embarked"])

# Define features and target variable
X = data.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
y = data['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=10, 
                                min_samples_leaf=2, max_features='sqrt', random_state=42)
rf_clf.fit(X_train, y_train)

# Predict probabilities of positive class for the test set
y_pred_proba = rf_clf.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
