import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score

# Sidebar for navigation
st.sidebar.title("ML App")
menu = st.sidebar.radio("Navigation", ["Upload Data", "Data Visualization", "Preprocessing", "Model Training"])

# Global variables
uploaded_file = None
df = None

# Upload Data
if menu == "Upload Data":
    st.title("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df)

# Data Visualization
elif menu == "Data Visualization":
    st.title("Data Visualization")
    if uploaded_file:
        st.write("Select columns to visualize:")
        columns = st.multiselect("Select Columns", df.columns)
        if columns:
            st.write("Correlation Heatmap:")
            corr = df[columns].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            st.pyplot()
            st.write("Pairplot:")
            sns.pairplot(df[columns])
            st.pyplot()
    else:
        st.warning("Please upload a dataset first.")

# Preprocessing
elif menu == "Preprocessing":
    st.title("Data Preprocessing")
    if uploaded_file:
        st.write("Select columns to preprocess:")
        target = st.selectbox("Select Target Variable", df.columns)
        features = st.multiselect("Select Features", [col for col in df.columns if col != target])
        if target and features:
            st.write("Preprocessing Data...")
            X = df[features]
            y = df[target]
            le = LabelEncoder()
            if y.dtypes == 'object':
                y = le.fit_transform(y)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            st.success("Preprocessing Complete!")
    else:
        st.warning("Please upload a dataset first.")

# Model Training
elif menu == "Model Training":
    st.title("Model Training")
    if uploaded_file:
        task = st.radio("Select Task", ["Classification", "Regression"])
        target = st.selectbox("Select Target Variable", df.columns)
        features = st.multiselect("Select Features", [col for col in df.columns if col != target])
        if target and features:
            X = df[features]
            y = df[target]
            le = LabelEncoder()
            if task == "Classification" and y.dtypes == 'object':
                y = le.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if task == "Classification":
                model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVC"])
                model = None
                if model_choice == "Logistic Regression":
                    model = LogisticRegression()
                elif model_choice == "Random Forest":
                    model = RandomForestClassifier()
                elif model_choice == "Gradient Boosting":
                    model = GradientBoostingClassifier()
                elif model_choice == "SVC":
                    model = SVC()

                if model:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.write(f"Accuracy: {acc}")
                    st.write("Classification Report:")
                    st.text(classification_report(y_test, y_pred))

            elif task == "Regression":
                model_choice = st.selectbox("Select Model", ["Linear Regression", "Ridge Regression", "Random Forest Regressor", "Gradient Boosting Regressor", "SVR"])
                model = None
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                elif model_choice == "Ridge Regression":
                    model = Ridge()
                elif model_choice == "Random Forest Regressor":
                    model = RandomForestRegressor()
                elif model_choice == "Gradient Boosting Regressor":
                    model = GradientBoostingRegressor()
                elif model_choice == "SVR":
                    model = SVR()

                if model:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"Mean Squared Error: {mse}")
                    st.write(f"R2 Score: {r2}")
    else:
        st.warning("Please upload a dataset first.")
        