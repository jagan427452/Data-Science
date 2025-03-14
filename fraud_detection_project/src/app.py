import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/fraud_detection_model.pkl")

st.title("💳 Credit Card Fraud Detector")

uploaded_file = st.file_uploader("Upload a CSV file for fraud detection", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    predictions = model.predict(df)
    df["Fraud Prediction"] = predictions
    st.write(df)
