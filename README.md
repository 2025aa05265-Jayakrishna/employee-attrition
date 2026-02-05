a. Problem Statement
Employee attrition is a major challenge for organizations as it leads to increased recruitment cost and loss of experienced employees.  
The objective of this project is to build a machine learning–based classification system to predict whether an employee is likely to leave the organization (Attrition = Yes or No) based on various employee-related features.

 b. Dataset Description
The dataset used in this project is the **IBM HR Analytics Employee Attrition Dataset**.

- Total Records: 1470
- Number of Features: 34 (after preprocessing)
- Target Variable: Attrition  
- Yes → Employee will leave  
- No → Employee will stay  

### Features include:
- Demographic details (Age, Gender, Marital Status)
- Job-related attributes (Job Role, Job Level, Monthly Income)
- Work environment factors (OverTime, WorkLifeBalance)
- Performance and experience-related attributes

The dataset contains both categorical and numerical features, which were preprocessed before model training.

 c. Models Used and Evaluation Metrics

The following six machine learning classification models were implemented and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

### Evaluation Metrics Used
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## Model Comparison Table

========== FINAL COMPARISON TABLE ==========

                     Accuracy   AUC  Precision  Recall  F1 Score   MCC
Logistic Regression      0.86  0.81       0.62    0.34      0.44  0.39
Decision Tree            0.80  0.64       0.39    0.40      0.40  0.28
KNN                      0.84  0.64       0.56    0.11      0.18  0.19
Naive Bayes              0.62  0.69       0.25    0.70      0.37  0.23
Random Forest            0.83  0.79       0.36    0.09      0.14  0.11
XGBoost                  0.87  0.77       0.78    0.30      0.43  0.43
---

## Model Performance Observations

Logistic Regression:   Logistic Regression provides a good baseline model with stable performance. It shows reasonable accuracy but lower recall, indicating difficulty in identifying minority attrition cases. 

Decision Tree: Decision Tree captures non-linear patterns but tends to overfit, resulting in lower generalization performance compared to ensemble models. 

KNN: KNN performance depends on distance calculation and scaling. It performs moderately but struggles with recall for attrition cases. 

Naive Bayes: Naive Bayes is fast and simple but assumes feature independence, which limits its predictive performance on this dataset. 

Random Forest (Ensemble) : Random Forest improves performance by combining multiple decision trees, leading to better accuracy and balanced metrics. 

XGBoost (Ensemble): XGBoost achieves the best overall performance with higher AUC, F1 Score, and MCC, making it the most effective model for employee attrition prediction. 


## Deployment
The trained models were saved using `joblib` and deployed using a **Streamlit web application**.  
The application allows users to:
- Upload a dataset
- Select a machine learning model
- View evaluation metrics
- Visualize confusion matrices
- Predict employee attrition (Yes / No)


