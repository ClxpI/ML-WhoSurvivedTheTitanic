# ML-WhoSurvivedTheTitanic?

This repository contains a machine learning case study focused on predicting passenger survival during the Titanic disaster. Using datasets from Kaggle, the project analyzes various factors such as age, gender, class, and family size to develop predictive models. The repository demonstrates the application of data preprocessing, exploratory data analysis (EDA), and the implementation of multiple machine learning algorithms to determine the most accurate survival predictions.

---

## Project Overview

### Datasets
- **Train.csv**: Contains labeled data to train the models.
- **Test.csv**: Used to evaluate model predictions.
- **Gender_Submission.csv**: Provides a baseline prediction for comparison.

### Key Stages
1. **Data Preprocessing**: Handling missing values, scaling, and encoding features.
2. **Exploratory Data Analysis (EDA)**: Understanding relationships between features and survival outcomes.
3. **Feature Engineering**: Creating new features like family size and categorizing fare and age.
4. **Model Training**: Testing various machine learning algorithms including:
   - Random Forest
   - Logistic Regression
   - K-Nearest Neighbors
   - Naïve Bayes
   - Stochastic Gradient Descent
   - Decision Tree
   - Linear Support Vector Machine
5. **Model Evaluation**: Identifying the best-performing model using metrics such as accuracy, precision, recall, and F-score.
6. **Hyperparameter Tuning**: Optimizing model performance using Grid Search Cross Validation.

---

## What I Learned

### Data Preprocessing
- Managing missing data with techniques like median imputation.
- Encoding categorical variables for model compatibility.
- Scaling numerical features for better performance.

### Model Building and Evaluation
- Implemented and compared the performance of different machine learning algorithms.
- Utilized Random Forest as the best-performing model with an accuracy of **82.12%**.
- Evaluated model performance using confusion matrices, ROC curves, and precision-recall metrics.

### Feature Engineering
- Created new features such as:
  - **Family Size**: Number of family members on board.
  - **Family Survival**: Probability of survival based on family outcomes.
  - Categorized fare and age for better predictions.

---

## Visualizations and Outcomes

### Example Output

#### Confusion Matrix
|                | Predicted Not Survived | Predicted Survived |
|----------------|-------------------------|---------------------|
| **Actual Not Survived** | 92                      | 13                  |
| **Actual Survived**     | 20                      | 54                  |

A confusion matrix evaluates the model's performance by comparing actual outcomes to predictions. It helps measure metrics like:

- **Accuracy**: Proportion of correct predictions.
- **Precision**: Proportion of true positive predictions out of all positive predictions.
- **Recall**: Proportion of true positives correctly identified out of all actual positives.

![image](https://github.com/user-attachments/assets/fd963f5c-9623-45b0-893d-a9f129fa2579)


---

#### Precision-Recall Curve
A plot showing the trade-off between precision and recall, demonstrating the model’s ability to classify survival effectively. It is especially useful for imbalanced datasets. The goal is to maximize both precision and recall, shown by a curve approaching the top-right corner.

![image](https://github.com/user-attachments/assets/0edb32e0-5281-4b1b-bed0-ca4bbc4ef306)



---

#### Feature Importance
Feature importance quantifies the influence of each feature on the predictions. It helps identify key variables that contribute to the model's performance.

| Feature        | Importance |
|----------------|------------|
| Passenger ID   | 0.187      |
| Fare           | 0.181      |
| Age            | 0.169      |
| Sex_Male       | 0.146      |
| Pclass         | 0.078      |

- Features like `Fare`, `Age`, and `Sex_Male` significantly impact survival predictions. However, the high importance of `Passenger ID` may indicate overfitting or a data artifact.

![image](https://github.com/user-attachments/assets/12ecbcfb-a5e1-41e0-ad1b-0cdce65ffd9b)


---

#### Example Visualizations

During the exploratory data analysis (EDA), several key trends were identified to better understand the factors influencing survival rates:

- **Survival by Age Group**: 
  Survival rates varied across age groups, with the highest survival rate observed among passengers aged 17-32. Children aged 10 also showed relatively high survival rates, while survival likelihood decreased significantly for passengers older than 50.    
  ![survival_by_age_group](https://github.com/user-attachments/assets/90e967de-2d6d-4e41-9377-afe806eeea82)


- **Survival by Class**: 
  A bar chart illustrating that first-class passengers had the highest survival rates, while third-class passengers had the lowest.  
  ![survival_by_class](https://github.com/user-attachments/assets/f913aea4-565c-42c5-8b49-c3ba0eddde38)




---

## References
- Kaggle Titanic Dataset: [Link](https://www.kaggle.com/c/titanic)
- Libraries used: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
