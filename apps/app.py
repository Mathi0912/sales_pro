import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Sales Regression Project", layout="wide")

st.title("📊 Sales Analytics – Regression Use Cases")

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:

    # ========== LOAD DATA ==========
    df = pd.read_csv(uploaded_file)

    st.success("Dataset loaded successfully")

    # Show columns (IMPORTANT for debugging)
    st.subheader("Dataset Columns")
    st.write(df.columns.tolist())

    # ========== CLEANING ==========
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna()

    st.subheader("Cleaned Dataset Preview")
    st.dataframe(df.head())

    # =====================================================
    # USE CASE 1 – SALES AMOUNT REGRESSION
    # =====================================================
    st.header("Use Case 1: Sales Amount Prediction")

    # 🔴 CHANGE COLUMN NAMES IF YOUR DATA IS DIFFERENT
    features_uc1 = ['price', 'quantity', 'discount']
    target_uc1 = 'sales'

    if all(col in df.columns for col in features_uc1 + [target_uc1]):

        X1 = df[features_uc1]
        y1 = df[target_uc1]

        X1_train, X1_test, y1_train, y1_test = train_test_split(
            X1, y1, test_size=0.2, random_state=42
        )

        # 🔥 MODEL TRAINING (NO joblib, NO pkl)
        model_uc1 = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )

        model_uc1.fit(X1_train, y1_train)

        preds_uc1 = model_uc1.predict(X1_test)

        # ========== METRICS ==========
        r2 = r2_score(y1_test, preds_uc1)
        mae = mean_absolute_error(y1_test, preds_uc1)

        col1, col2 = st.columns(2)
        col1.metric("R² Score", round(r2, 3))
        col2.metric("MAE", round(mae, 2))

        # ========== VISUAL ==========
        fig, ax = plt.subplots()
        sns.scatterplot(x=y1_test, y=preds_uc1, ax=ax)
        ax.set_xlabel("Actual Sales")
        ax.set_ylabel("Predicted Sales")
        ax.set_title("Actual vs Predicted Sales")
        st.pyplot(fig)

    else:
        st.error("Required columns for Use Case 1 not found in dataset")

    # =====================================================
    # USE CASE 2 – REVENUE REGRESSION
    # =====================================================
    st.header("Use Case 2: Revenue Prediction")

    features_uc2 = ['price', 'quantity']
    target_uc2 = 'revenue'

    if all(col in df.columns for col in features_uc2 + [target_uc2]):

        X2 = df[features_uc2]
        y2 = df[target_uc2]

        X2_train, X2_test, y2_train, y2_test = train_test_split(
            X2, y2, test_size=0.2, random_state=42
        )

        model_uc2 = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.07,
            max_depth=3,
            random_state=42
        )

        model_uc2.fit(X2_train, y2_train)

        preds_uc2 = model_uc2.predict(X2_test)

        r2_2 = r2_score(y2_test, preds_uc2)
        mae_2 = mean_absolute_error(y2_test, preds_uc2)

        col3, col4 = st.columns(2)
        col3.metric("R² Score", round(r2_2, 3))
        col4.metric("MAE", round(mae_2, 2))

    else:
        st.error("Required columns for Use Case 2 not found in dataset")

else:
    st.warning("Please upload a CSV file to start")
