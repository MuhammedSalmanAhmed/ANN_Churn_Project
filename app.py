
import streamlit as st
import pandas as pd
import numpy as np
import json
import tensorflow as tf

st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦"
)

st.title("🏦 Bank Customer Churn Predictor")

st.write(
    "Enter customer information to predict whether they may leave the bank."
)

# Load model and preprocessing configuration
@st.cache_resource
def load_model():

    model = tf.keras.models.load_model("model.keras")

    with open("preprocessor_config.json", "r") as f:
        config = json.load(f)

    return model, config

model, config = load_model()

st.subheader("Customer Information")

credit_score = st.number_input("Credit Score", 300, 900, 650)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.slider("Age", 18, 100, 38)

tenure = st.slider("Tenure (Years)", 0, 10, 5)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

num_products = st.selectbox(
    "Number of Products",
    [1, 2, 3, 4]
)

has_card = st.selectbox(
    "Has Credit Card?",
    ["Yes", "No"]
)

active = st.selectbox(
    "Is Active Member?",
    ["Yes", "No"]
)

salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=100000.0
)

if st.button("Predict Churn", type="primary"):

    # Original customer information
    customer = {
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
    }

    # Apply the same StandardScaler transformation
    numeric_values = np.array([
        customer[col]
        for col in config["numeric_columns"]
    ], dtype=np.float64)

    means = np.array(config["means"], dtype=np.float64)
    scales = np.array(config["scales"], dtype=np.float64)

    scaled_numeric = (numeric_values - means) / scales

    # Apply the same OneHotEncoder transformation
    encoded_categorical = []

    for col, categories in zip(
        config["categorical_columns"],
        config["categories"]
    ):

        value = str(customer[col])

        encoded_values = [
            1.0 if value == str(category) else 0.0
            for category in categories
        ]

        encoded_categorical.extend(encoded_values)

    # Combine numeric and categorical features
    processed = np.array(
        [list(scaled_numeric) + encoded_categorical],
        dtype=np.float32
    )

    # Predict churn probability
    probability = float(
        model.predict(processed, verbose=0)[0][0]
    )

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if probability >= 0.5:
        st.error("Customer is likely to LEAVE.")
    else:
        st.success("Customer is likely to STAY.")
