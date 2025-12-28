# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ml_toolkit   # import your class file

# Ocean theme styling
def ocean_style():
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(to bottom, #E6F7FF, #B3E5FC); }
        [data-testid="stSidebar"] { background: linear-gradient(to bottom, #1CA9C9, #006994); color: white; }
        h1, h2, h3 { color: #003366; }
        .stButton>button { background-color: #1CA9C9; color: white; border-radius: 10px; border: none; }
        .stButton>button:hover { background-color: #006994; color: #E6F7FF; }
        div[data-testid="stMetricValue"] { color: #006994; font-weight: bold; }
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="🌊 Credit Scoring ML — Ocean Edition", layout="wide")
ocean_style()
st.title("🌊 Credit Scoring ML — Ocean Edition")

ml = ml_toolkit.MachineLearning()

menu = st.sidebar.radio("🐬 Navigation", ["Data Upload", "Exploration", "Preprocessing", "Feature Selection", "Modeling", "Evaluation"])

if menu == "Data Upload":
    file = st.file_uploader("Upload dataset 🌊", type=["csv", "xlsx"])
    if file:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
        st.session_state.df = df
        st.write("Preview", df.head())
        st.dataframe(ml.data_information(df))

elif menu == "Exploration":
    if "df" in st.session_state:
        col = st.selectbox("Select column", st.session_state.df.columns)
        fig = ml.histogram_plot(col, st.session_state.df)
        st.pyplot(fig)

elif menu == "Preprocessing":
    st.write("Handle nulls, outliers, scaling, encoding here...")

elif menu == "Feature Selection":
    st.write("Run ANOVA and Chi² feature selection...")

elif menu == "Modeling":
    st.write("Train models and show metrics...")

elif menu == "Evaluation":
    st.write("Confusion matrix, ROC, PR curves...")
