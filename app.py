
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦")

st.title("🏦 Bank Customer Churn Predictor")
st.write("Enter customer information to predict whether they may leave the bank.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.keras")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()

st.subheader("Customer Information")

credit_score = st.number_input("Credit Score", 300, 900, 650)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 38)
tenure = st.slider("Tenure (Years)", 0, 10, 5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
active = st.selectbox("Is Active Member?", ["Yes", "No"])
salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0)

if st.button("Predict Churn", type="primary"):

    customer = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if active == "Yes" else 0,
        "EstimatedSalary": salary
    }])

    processed = preprocessor.transform(customer)

    probability = float(model.predict(processed, verbose=0)[0][0])

    st.metric("Churn Probability", f"{probability * 100:.2f}%")

    if probability >= 0.5:
        st.error("Customer is likely to LEAVE.")
    else:
        st.success("Customer is likely to STAY.")
