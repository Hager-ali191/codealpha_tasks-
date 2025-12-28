import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import io

# === FULL CSS with blue/light blue/purple theme ===
full_css = """
/* GENERAL */
body, .css-18e3th9 {
    background-color: #e0f7fa; /* Light ocean blue */
    color: #004d40;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    transition: all 0.3s ease;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #b2ebf2; /* Soft teal */
    color: #004d40;
    padding: 30px 20px;
    font-weight: 600;
    border-radius: 0 15px 15px 0;
    box-shadow: 0 0 15px rgba(0, 150, 170, 0.2);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #006064;
}

/* RADIO BUTTONS */
[data-baseweb="radio"] > div {
    background-color: transparent !important;
}
.css-1kyxreq {
    color: #00796b !important;
}

/* MAIN TITLE */
.css-10trblm {
    font-weight: 700 !important;
    color: #006064 !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(90deg, #4dd0e1, #40c4ff); /* Ocean gradient */
    color: white;
    font-weight: 700;
    border-radius: 12px;
    border: none;
    padding: 10px 25px;
    box-shadow: 0 0 10px #80deea88;
    transition: all 0.3s ease-in-out;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #40c4ff, #4dd0e1);
    box-shadow: 0 0 15px #80deea;
    transform: scale(1.03);
    cursor: pointer;
}

/* FILE UPLOADER */
.css-1hynsf2 {
    background-color: #e0f7fa !important;
    border-radius: 12px;
    border: 2px dashed #4dd0e1 !important;
    padding: 20px !important;
    color: #004d40 !important;
    font-weight: 600;
    transition: all 0.3s ease;
}
.css-1hynsf2:hover {
    background-color: #b2ebf2 !important;
    border-color: #00acc1;
}

/* DATAFRAME */
.stDataFrame > div {
    background-color: #ffffff !important;
    color: #004d40 !important;
    border-radius: 15px;
    padding: 10px;
    box-shadow: 0 0 10px rgba(0, 150, 170, 0.1);
}
.stDataFrame table {
    color: #004d40 !important;
    border-collapse: separate !important;
    border-spacing: 0 10px !important;
    font-weight: 500;
}
.stDataFrame th {
    background-color: #b2ebf2 !important;
    color: #004d40 !important;
    padding: 8px !important;
    border-radius: 12px !important;
}
.stDataFrame td {
    background-color: #e0f7fa !important;
    padding: 8px !important;
    border-radius: 10px !important;
}

/* HEADERS */
h1, h2, h3, h4 {
    color: #00796b !important;
    font-weight: 700 !important;
}

/* METRICS */
[data-testid="stMetricValue"] {
    color: #00796b !important;
    font-weight: 700 !important;
    font-size: 2.2rem !important;
}

/* TEXT */
.stText, .stMarkdown {
    color: #004d40 !important;
}

/* SLIDERS and NUMBER INPUTS */
.css-1fv8s86, .css-1avcm0n {
    background-color: #e0f7fa !important;
    border-radius: 10px !important;
    color: #004d40 !important;
    font-weight: 600 !important;
    border: 1px solid #80deea;
}
.css-1fv8s86:focus, .css-1avcm0n:focus {
    border-color: #00acc1 !important;
    box-shadow: 0 0 5px #00acc188;
}

/* SELECTBOX */
.css-1v0mbdj, .css-1r6slb0 {
    background-color: #e0f7fa !important;
    color: #004d40 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: 1px solid #80deea;
}
.css-1v0mbdj:focus, .css-1r6slb0:focus {
    border-color: #00acc1 !important;
    box-shadow: 0 0 5px #00acc188;
}

/* CONFUSION MATRIX HEATMAP */
svg text {
    fill: #004d40 !important;
    font-weight: 600 !important;
}

/* CHECKBOX */
.stCheckbox > label > div {
    color: #00796b !important;
    font-weight: 600 !important;
}

/* PROGRESS BAR */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00acc1, #4dd0e1) !important;
    border-radius: 12px !important;
}

/* SCROLLBAR */
::-webkit-scrollbar {
  width: 8px;
}
::-webkit-scrollbar-track {
  background: #e0f7fa;
  border-radius: 10px;
}
::-webkit-scrollbar-thumb {
  background: #80deea;
  border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
  background: #4dd0e1;
}

/* MISC */
.css-ffhzg2 {
    background-color: transparent !important;
}

/* CONTAINER PADDING */
.css-1d391kg {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-top: 1rem !important;
}

/* ANIMATIONS */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in {
    animation: fadeIn 0.5s ease forwards;
}
"""

# Inject CSS
st.markdown(f"<style>{full_css}</style>", unsafe_allow_html=True)

# App sidebar and navigation
st.sidebar.title("💓 Heart Disease Predictor")

page = st.sidebar.radio("🌌 Navigation", ["Upload & Overview", "Preprocessing", "Feature Selection", "Modeling", "Prediction"])

if "df" not in st.session_state:
    st.session_state.df = None

if page == "Upload & Overview":
    st.title("💓 Heart Disease Dataset Overview")
    file = st.file_uploader("📤 Upload your heart.csv file", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.session_state.df = df.copy()
        st.subheader("📊 Raw Data Preview")
        st.dataframe(df.head())
        st.subheader("📌 Summary Statistics")
        st.write(df.describe())
        if st.checkbox("🔍 Show DataFrame Info"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

elif page == "Preprocessing" and st.session_state.df is not None:
    st.title("🧹 Data Preprocessing")
    df = st.session_state.df.copy()
    st.subheader("🔍 Missing Values (Before)")
    st.write(df.isnull().sum())
    df.replace({'chol': {'twenty': '20', 'ss': '0', np.nan: '0'}}, inplace=True)
    df["chol"] = pd.to_numeric(df["chol"], errors='coerce')
    for col in ["cp", "sex", "fbs", "restecg", "thal", "slope", "exang"]:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in ["age", "ca", "trestbps", "oldpeak", "thalach", "chol"]:
        df[col].fillna(df[col].median(), inplace=True)
    st.subheader("✅ Missing Values (After)")
    st.write(df.isnull().sum())
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    after = df.shape[0]
    st.subheader("📦 Duplicate Records Removed")
    st.write(f"Removed {before - after} duplicate rows")
    st.subheader("📉 Outlier Treatment")
    st.write("Before Outlier Removal")
    fig1, ax1 = plt.subplots()
    sns.boxplot(data=df.select_dtypes(include=np.number), ax=ax1)
    st.pyplot(fig1)
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        extreme_lower = Q1 - 3 * IQR
        extreme_upper = Q3 + 3 * IQR
        df = df[(df[col] >= extreme_lower) & (df[col] <= extreme_upper)]
        df[col] = np.clip(df[col], lower, upper)
    st.write("After Outlier Treatment")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df.select_dtypes(include=np.number), ax=ax2)
    st.pyplot(fig2)
    st.session_state.cleaned_df = df

elif page == "Feature Selection" and st.session_state.get("cleaned_df") is not None:
    st.title("🧠 Feature Selection")
    df = st.session_state.cleaned_df.copy()
    target = st.selectbox("🎯 Choose Target Column", df.columns, index=len(df.columns) - 1)
    x = df.drop(columns=[target])
    y = df[target]
    selector = SelectKBest(score_func=f_classif, k=10)
    x_selected = selector.fit_transform(x, y)
    selected_features = x.columns[selector.get_support()].tolist()
    st.subheader("✅ Top Features:")
    st.write(selected_features)
    X_train, X_test, y_train, y_test = train_test_split(df[selected_features], y, test_size=0.2, random_state=42)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.selected_features = selected_features
    st.session_state.scaler = scaler

elif page == "Modeling" and "X_train" in st.session_state:
    st.title("📈 Model Training & Evaluation")
    model_name = st.selectbox("🤖 Select Classifier", ["Logistic Regression", "SVM", "Decision Tree", "KNN"])
    if st.button("🔍 Find Best k for KNN"):
        scores = []
        for k in range(1, 21):
            knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='manhattan')
            knn.fit(st.session_state.X_train, st.session_state.y_train)
            y_pred = knn.predict(st.session_state.X_test)
            scores.append(accuracy_score(st.session_state.y_test, y_pred))
        best_k = np.argmax(scores) + 1
        st.session_state.best_k = best_k
        st.success(f"Best k = {best_k} with accuracy = {max(scores) * 100:.2f}%")
        fig, ax = plt.subplots()
        ax.plot(range(1, 21), scores, marker='o', color="#7366ff")
        ax.set_title("Accuracy vs k (KNN)")
        ax.set_xlabel("k")
        ax.set_ylabel("Accuracy")
        st.pyplot(fig)
    if st.button("🚀 Train & Evaluate"):
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "SVM":
            model = SVC(kernel='rbf')
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(criterion="entropy")
        elif model_name == "KNN":
            best_k = st.session_state.get("best_k", 7)
            model = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='manhattan')
        model.fit(st.session_state.X_train, st.session_state.y_train)
        y_pred = model.predict(st.session_state.X_test)
        st.session_state.model = model
        acc = accuracy_score(st.session_state.y_test, y_pred)
        st.subheader("📊 Accuracy")
        st.metric("Accuracy", f"{acc * 100:.2f}%")
        st.subheader("🧾 Classification Report")
        st.text(classification_report(st.session_state.y_test, y_pred))
        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

elif page == "Prediction" and "model" in st.session_state:
    st.title("🔮 Make a Prediction")
    st.write("👨‍🚀 Enter values for the features below...")
    input_data = {}
    cols = st.columns(3)
    for i, feature in enumerate(st.session_state.selected_features):
        with cols[i % 3]:
            input_data[feature] = st.number_input(f"{feature}", value=0.0)
    input_df = pd.DataFrame([input_data])
    scaled_input = st.session_state.scaler.transform(input_df)
    if st.button("📡 Predict"):
        prediction = st.session_state.model.predict(scaled_input)[0]
        if prediction == 1:
            st.error("💔 Prediction: Heart Disease Detected")
        else:
            st.success("💖 Prediction: No Heart Disease")

else:
    st.info("Please upload and preprocess data first.")
