import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Retail Demand & Sales Analytics",
    layout="wide"
)

st.title("📊 Retail Demand & Sales Analytics Dashboard")

# -----------------------------
# DATA UPLOAD
# -----------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload the dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# BASIC CLEANING
# -----------------------------
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna()

# Encode categorical columns
cat_cols = ['Category', 'Region', 'Weather_Condition',
            'Promotion', 'Seasonality', 'Epidemic']

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# -----------------------------
# USE CASE 1 – DEMAND PREDICTION
# -----------------------------
st.header("🟦 Use Case 1: Demand Prediction")

features_uc1 = [
    'Price',
    'Discount',
    'Competitor_Pricing',
    'Inventory_Level',
    'Units_Ordered'
]

X1 = df_encoded[features_uc1]
y1 = df_encoded['Demand']

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

model_uc1 = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model_uc1.fit(X1_train, y1_train)

y1_pred = model_uc1.predict(X1_test)

st.metric("R² Score", round(r2_score(y1_test, y1_pred), 3))
st.metric("MAE", round(mean_absolute_error(y1_test, y1_pred), 2))

# Demand Visualization
fig1 = px.scatter(
    x=y1_test,
    y=y1_pred,
    labels={"x": "Actual Demand", "y": "Predicted Demand"},
    title="Actual vs Predicted Demand"
)
st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# USE CASE 2 – UNITS SOLD PREDICTION
# -----------------------------
st.header("🟩 Use Case 2: Units Sold Prediction")

features_uc2 = [
    'Demand',
    'Price',
    'Discount',
    'Inventory_Level'
]

X2 = df_encoded[features_uc2]
y2 = df_encoded['Units_Sold']

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

model_uc2 = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model_uc2.fit(X2_train, y2_train)

y2_pred = model_uc2.predict(X2_test)

st.metric("R² Score", round(r2_score(y2_test, y2_pred), 3))
st.metric("MAE", round(mean_absolute_error(y2_test, y2_pred), 2))

# Units Sold Visualization
fig2 = px.scatter(
    x=y2_test,
    y=y2_pred,
    labels={"x": "Actual Units Sold", "y": "Predicted Units Sold"},
    title="Actual vs Predicted Units Sold"
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# BUSINESS INSIGHTS
# -----------------------------
st.header("📌 Business Insights")
st.markdown("""
- Demand is influenced strongly by price, discounts, and competition  
- Promotions significantly increase demand and sales  
- Inventory misalignment leads to stock-out risks  
- Regression models help improve planning accuracy  
""")

   
    
    
       
