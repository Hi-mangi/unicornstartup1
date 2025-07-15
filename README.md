
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ----------------------------
# Load & Preprocess Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\himan\Downloads\Unicorn_Companies.csv")
    df["Valuation"] = df["Valuation ($B)"].replace('[\$,]', '', regex=True).astype(float)
    df["Investors Count"] = pd.to_numeric(df["Investors Count"], errors='coerce')
    df["Founded Year"] = pd.to_numeric(df["Founded Year"], errors='coerce')
    df["Total Raised"] = df["Total Raised"].replace('[\$,]', '', regex=True)
    df["Total Raised"] = df["Total Raised"].str.replace('B', 'e9').str.replace('M', 'e6')
    df["Total Raised"] = pd.to_numeric(df["Total Raised"], errors='coerce')
    df = df.dropna(subset=["Valuation", "Country", "Industry", "Founded Year", "Total Raised", "Investors Count"])
    return df

df = load_data()

# ----------------------------
# Train Model
# ----------------------------
features = ["Country", "Industry", "Founded Year", "Total Raised", "Investors Count"]
target = "Valuation"

X = df[features]
y = df[target]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Country", "Industry"])
], remainder='passthrough')

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X, y)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Unicorn Valuation Predictor", layout="centered")
st.title("🦄 Unicorn Startup Valuation Predictor")

st.markdown("Enter details about a startup to predict its valuation (in **Billion USD**):")

country = st.selectbox("Country", sorted(df["Country"].unique()))
industry = st.selectbox("Industry", sorted(df["Industry"].unique()))
founded_year = st.slider("Founded Year", int(df["Founded Year"].min()), int(df["Founded Year"].max()), 2015)
total_raised = st.number_input("Total Raised ($)", value=50000000, step=1000000, format="%d")
investor_count = st.slider("Number of Investors", 1, 100, 5)

# Prediction
if st.button("Predict Valuation"):
    input_df = pd.DataFrame([{
"Country": country, 
"Industry": industry,
        "Founded Year": founded_year,
        "Total Raised": total_raised,
        "Investors Count": investor_count
    }])
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Valuation: **${round(prediction, 2)} Billion**")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


