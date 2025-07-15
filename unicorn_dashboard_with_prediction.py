
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
col1, col2, col3 = st.columns(3)
col1.metric("Total Unicorns", len(filtered_df))
col2.metric("Top Country", filtered_df["Country"].value_counts().idxmax() if not filtered_df.empty else "N/A")
col3.metric("Avg Valuation ($B)", round(filtered_df["Valuation"].mean(), 2) if not filtered_df.empty else "N/A")

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
# 💡 Valuation Prediction Section
# ----------------------------
st.subheader("🧮 Valuation Predictor")

model_df = df.dropna(subset=["Valuation", "Year_Joined", "Country", "Industry"])
X = model_df[["Year_Joined", "Country", "Industry"]]
y = model_df["Valuation"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Country", "Industry"])
], remainder='passthrough')

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])
model.fit(X, y)

with st.form("prediction_form"):
    st.markdown("#### Enter Startup Details:")
    year_input = st.slider("Year Joined", min_value=2000, max_value=2025, value=2020)
    country_input = st.selectbox("Country", sorted(df["Country"].dropna().unique()))
    industry_input = st.selectbox("Industry", sorted(df["Industry"].dropna().unique()))
    submit = st.form_submit_button("Predict Valuation")

    if submit:
        input_df = pd.DataFrame([{
            "Year_Joined": year_input,
            "Country": country_input,
            "Industry": industry_input
        }])
        prediction = model.predict(input_df)[0]
        st.success(f"📈 Predicted Valuation: **${prediction:.2f} Billion**")

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
---
Made with ❤️ using Streamlit, Plotly, and Scikit-learn | Data: Unicorn_Companies.csv
""")
