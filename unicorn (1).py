
import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Unicorn Dashboard", page_icon="🦄", layout="wide")

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv("Unicorn_Companies.csv")

df.rename(columns={"Valuation ($B)": "Valuation", "Date Joined": "Date_Joined"}, inplace=True)
df["Year_Joined"] = pd.to_datetime(df["Date_Joined"], errors='coerce').dt.year

# ----------------------------
# App Header
# ----------------------------
st.markdown("""
    <h1 style='text-align: center; color: #FF4B4B;'>🦄 Global Unicorn Startups Explorer</h1>
    <h4 style='text-align: center;'>Explore trends, valuations, and industries of billion-dollar startups</h4>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🔎 Filter the Data")
countries = st.sidebar.multiselect("Select Countries", sorted(df["Country"].dropna().unique()))
industries = st.sidebar.multiselect("Select Industries", sorted(df["Industry"].dropna().unique()))

filtered_df = df.copy()
if countries:
    filtered_df = filtered_df[filtered_df["Country"].isin(countries)]
if industries:
    filtered_df = filtered_df[filtered_df["Industry"].isin(industries)]

# ----------------------------
# Metrics Display
# ----------------------------
st.subheader("📊 Key Metrics")
valuation_series = pd.to_numeric(filtered_df["Valuation"], errors='coerce')

col1, col2, col3 = st.columns(3)
col1.metric("Total Unicorns", len(filtered_df))
col2.metric("Top Country", filtered_df["Country"].value_counts().idxmax() if not filtered_df.empty else "N/A")
# Ensure Valuation is numeric
filtered_df["Valuation"] = pd.to_numeric(filtered_df["Valuation"], errors="coerce")

avg_valuation = round(valuation_series.mean(), 2) if not valuation_series.dropna().empty else "N/A"
col3.metric("Avg Valuation ($B)", round(filtered_df["Valuation"].mean(), 2) if not filtered_df.empty else "N/A")
# Ensure Valuation is numeric
filtered_df["Valuation"] = pd.to_numeric(filtered_df["Valuation"], errors="coerce")

# ----------------------------
# Raw Data Preview
# ----------------------------
with st.expander("📂 View Raw Dataset"):
    st.dataframe(filtered_df)

# ----------------------------
# Valuation by Country Box Plot
# ----------------------------
st.subheader("💰 Valuation Distribution by Country")
if not filtered_df.empty:
    fig1 = px.box(filtered_df, x="Country", y="Valuation", color="Country", points="all")
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("No data available for selected filters.")

# ----------------------------
# Unicorns Over Time
# ----------------------------
st.subheader("📈 Unicorns Joined per Year")
if not filtered_df.empty:
    yearly = filtered_df.groupby("Year_Joined").size().reset_index(name="Count")
    fig2 = px.bar(yearly, x="Year_Joined", y="Count", color="Count", title="Unicorns per Year")
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Industry Pie Chart
# ----------------------------
st.subheader("🏭 Industry Distribution")
if not filtered_df.empty:
    industry_count = filtered_df["Industry"].value_counts().reset_index()
    industry_count.columns = ["Industry", "Count"]
    fig3 = px.pie(industry_count, values="Count", names="Industry", title="Industry Breakdown")
    st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# Animated Bar Chart
# ----------------------------
st.subheader("🎞️ Animated Valuation by Country Over Years")
if not filtered_df.empty:
    animated_df = filtered_df.dropna(subset=["Year_Joined"])
    fig4 = px.bar(
        animated_df.sort_values("Year_Joined"),
        x="Country",
        y="Valuation",
        color="Country",
        animation_frame="Year_Joined",
        range_y=[0, animated_df["Valuation"].max() + 10],
        title="Valuation Growth Over Time"
    )
    st.plotly_chart(fig4, use_container_width=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
---
Made with ❤️ using Streamlit and Plotly | Data: Unicorn_Companies.csv
""")

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
    df = pd.read_csv("Unicorn_Companies.csv")

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



