import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Breast Cancer Prediction App",
    page_icon="🎗️",
    layout="centered"
)

# Load pre-trained PCA and Classifier models (.sav format)
@st.cache_resource
def load_models():
    pca = joblib.load('pca_model.sav')
    classifier = joblib.load('classifier_model.sav')
    return pca, classifier

try:
    pca_model, classifier_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error loading models: {e}. Please make sure 'pca_model.sav' and 'classifier_model.sav' are present in the directory.")

# App Header
st.title("🎗️ Breast Cancer Prediction Platform")
st.write("""
This application uses a Machine Learning pipeline containing **Principal Component Analysis (PCA)**
for dimension reduction and a **Logistic Regression Classifier** to predict whether a breast mass tumor is **Malignant** or **Benign**.
""")

if models_loaded:
    st.sidebar.header("User Input Features")

    # Setup a simple way for users to manually input some primary parameters
    st.subheader("Input Mass Measurements manually:")

    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=40.0, value=14.0)
        texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=40.0, value=19.0)
        perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=200.0, value=92.0)
    with col2:
        area_mean = st.number_input("Area Mean", min_value=0.0, max_value=2500.0, value=650.0)
        smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=0.5, value=0.1)
        compactness_mean = st.number_input("Compactness Mean", min_value=0.0, max_value=0.5, value=0.1)
    with col3:
        concavity_mean = st.number_input("Concavity Mean", min_value=0.0, max_value=0.5, value=0.09)
        concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, max_value=0.5, value=0.05)
        symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, max_value=0.5, value=0.18)

    # Create a full mock vector of 30 standardized features required by our PCA
    raw_input_data = np.zeros((1, 30))
    raw_input_data[0, 0] = radius_mean
    raw_input_data[0, 1] = texture_mean
    raw_input_data[0, 2] = perimeter_mean
    raw_input_data[0, 3] = area_mean
    raw_input_data[0, 4] = smoothness_mean
    raw_input_data[0, 5] = compactness_mean
    raw_input_data[0, 6] = concavity_mean
    raw_input_data[0, 7] = concave_points_mean
    raw_input_data[0, 8] = symmetry_mean

    if st.button("Predict Diagnosis"):
        # Apply PCA reduction
        pca_input = pca_model.transform(raw_input_data)

        # Make prediction
        prediction = classifier_model.predict(pca_input)[0]
        prediction_proba = classifier_model.predict_proba(pca_input)[0]

        st.subheader("Prediction Results")
        if prediction == 1:
            st.error("⚠️ **Prediction: Malignant (M)**")
            st.write(f"Confidence Level: {prediction_proba[1]:.2%}")
        else:
            st.success("✅ **Prediction: Benign (B)**")
            st.write(f"Confidence Level: {prediction_proba[0]:.2%}")

        st.info("Note: For a fully integrated deployment, ensure that user inputs are passed through the correct StandardScaler sequence before dimension reduction.")
