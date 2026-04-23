import pickle
import pandas as pd
import numpy as np
import streamlit as st
import os

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Energy Predictor", layout="centered")

st.title(" Global Energy & dept shock")
st.markdown("Predict energy/emission values using ML model")

# -------------------------------
# Load Models Safely
# -------------------------------
def load_model(path):
    if not os.path.exists(path):
        st.error(f"❌ File not found: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

# -------------------------------
# Prediction Function
# -------------------------------
def predict_energy(country, year, oil, gas, coal, renew):

    scaler = load_model("models/scaler.pkl")
    model = load_model("models/model.pkl")
    encoder = load_model("models/encoder.pkl")

    if scaler is None or model is None or encoder is None:
        return None

    try:
        country_enc = encoder.transform([country])[0]
    except:
        st.error("❌ Country not present in training data")
        return None

    # IMPORTANT: must match training features
    data = pd.DataFrame({
        "Country": [country_enc],
        "Year": [year],
        "oilcons_ej": [oil],
        "gascons_ej": [gas],
        "coalcons_ej": [coal],
        "renewables_ej": [renew]
    })

    try:
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)[0]
        return prediction
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None


# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("📥 Input Parameters")

country = st.sidebar.selectbox(
    "Select Country",
    ["India", "United States", "China", "Japan", "Germany"]
)

year = st.sidebar.slider("Year", 1965, 2023, 2020)

oil = st.sidebar.number_input("Oil Consumption (EJ)", min_value=0.0)
gas = st.sidebar.number_input("Gas Consumption (EJ)", min_value=0.0)
coal = st.sidebar.number_input("Coal Consumption (EJ)", min_value=0.0)
renew = st.sidebar.number_input("Renewables (EJ)", min_value=0.0)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔍 Predict"):

    result = predict_energy(country, year, oil, gas, coal, renew)

    if result is not None:
        st.success(f"🌱 Predicted Value: {result:.2f}")

        # progress visualization
        st.progress(min(result / 10000, 1.0))

        st.info("Higher values indicate greater energy/emission output")

    else:
        st.error("❌ Prediction failed. Check model files.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("AI/ML Project: Global Energy & Debt Shock Analysis")