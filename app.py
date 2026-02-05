
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# Page Title

st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")
st.title("Employee Attrition Prediction App")
st.write("Machine Learning Assignment - 2")
st.markdown("---")

# Load Models Safely

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))
models = {
    "Logistic Regression": joblib.load(os.path.join(BASE_DIR, "model", "logistic.pkl")),
    "Decision Tree": joblib.load(os.path.join(BASE_DIR, "model", "tree.pkl")),
    "KNN": joblib.load(os.path.join(BASE_DIR, "model", "knn.pkl")),
    "Naive Bayes": joblib.load(os.path.join(BASE_DIR, "model", "nb.pkl")),
    "Random Forest": joblib.load(os.path.join(BASE_DIR, "model", "rf.pkl")),
    "XGBoost": joblib.load(os.path.join(BASE_DIR, "model", "xgb.pkl"))
}


# Model Selection
st.subheader("Select Machine Learning Model")
model_name = st.selectbox("Choose a model", list(models.keys()))
model = models[model_name]

# Dataset Upload

st.subheader("Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader("Upload Employee Attrition CSV File", type=["csv"])
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Preview of Uploaded Dataset:")
    st.dataframe(df.head())

    
    # Data Preparation
   
    if "Attrition" not in df.columns:
        st.error("Uploaded dataset must contain 'Attrition' column.")
        st.stop()

    # Convert target
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Scaling
    X_scaled = scaler.transform(X)

        # Predictions
    
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # ================================
    # Metrics Calculation
    # ================================
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.markdown("---")
    st.subheader("Model Evaluation Metrics")

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
        "Value": [acc, auc, prec, rec, f1, mcc]
    })

    metrics_df["Value"] = metrics_df["Value"].round(2)
    st.table(metrics_df)

    # ================================
    # Confusion Matrix (Figure Only)
    # ================================
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")

    st.pyplot(fig)

    # ================================
    # Prediction Output
    # ================================
    st.subheader("Prediction Output")

    result_df = pd.DataFrame({
        "Actual Attrition": y.map({1: "Yes", 0: "No"}),
        "Predicted Attrition": pd.Series(y_pred).map({1: "Yes", 0: "No"})
    })

    st.dataframe(result_df.head(10))

else:
    st.info("Please upload a CSV file to proceed.")
