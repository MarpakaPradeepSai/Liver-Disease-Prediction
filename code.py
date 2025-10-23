import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io

# ======================================================================================
# Function to Load Model and Features from GitHub
# ======================================================================================
# Use Streamlit's caching to load the model only once, improving performance.
@st.cache_resource
def load_model_from_github():
    """
    Loads the trained RandomForest model and the list of selected features
    directly from a public GitHub repository.
    """
    # URLs to the raw model and features files on GitHub
    model_url = 'https://raw.githubusercontent.com/MarpakaPradeepSai/Liver-Disease-Prediction/main/LDP_RFC_Model/final_random_forest_model.joblib'
    features_url = 'https://raw.githubusercontent.com/MarpakaPradeepSai/Liver-Disease-Prediction/main/LDP_RFC_Model/selected_features_list.joblib'

    try:
        # Download the model file
        model_response = requests.get(model_url)
        model_response.raise_for_status()  # Raise an exception for bad status codes
        model_file = io.BytesIO(model_response.content)
        model = joblib.load(model_file)

        # Download the features list file
        features_response = requests.get(features_url)
        features_response.raise_for_status()
        features_file = io.BytesIO(features_response.content)
        features = joblib.load(features_file)

        return model, features
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading files from GitHub: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading model/features: {e}")
        return None, None

# Load the model and feature list
final_model, best_features = load_model_from_github()

# ======================================================================================
# Streamlit App User Interface
# ======================================================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="🩺",
    layout="centered" # Use 'centered' layout for a cleaner look with inputs in the main area
)

# --- App Title ---
st.title("🩺 Liver Disease Prediction App")
st.markdown("Enter the patient's test results below to predict the likelihood of liver disease.")

# Check if model and features loaded successfully before creating input fields
if final_model and best_features:

    st.subheader("Patient Data Input")

    # Create a dictionary to hold user inputs
    input_data = {}

    # Define tooltips for better user understanding
    tooltips = {
        'Alkphos_Alkaline_Phosphotase': 'Measures the amount of alkaline phosphatase enzyme in your blood. (IU/L)',
        'Sgot_Aspartate_Aminotransferase': 'Measures the enzyme AST (SGOT) in your blood. (IU/L)',
        'Sgpt_Alamine_Aminotransferase': 'Measures the enzyme ALT (SGPT) in your blood. (IU/L)',
        'Total_Bilirubin': 'Measures the total amount of bilirubin in your blood. (mg/dL)',
        'Total_Proteins': 'Measures the total amount of protein in your blood. (g/dL)',
        'Direct_Bilirubin': 'Measures the amount of direct (conjugated) bilirubin. (mg/dL)',
        'ALB_Albumin': 'Measures the amount of albumin in your blood. (g/dL)',
        'A/G_Ratio_Albumin_and_Globulin_Ratio': 'The ratio of albumin to globulin in the blood.'
    }

    # Use columns to organize the input fields neatly in the main interface
    col1, col2 = st.columns(2)

    with col1:
        for feature in best_features[:4]: # First 4 features in the first column
            label = feature.replace('_', ' ').title()
            input_data[feature] = st.number_input(
                label=label,
                value=None, # Set default value to None to keep the box empty
                placeholder="Enter value...", # Placeholder text
                min_value=0.0,
                max_value=2500.0,
                step=None, # Allow any float value
                help=tooltips.get(feature, "Enter the measured value.")
            )

    with col2:
        for feature in best_features[4:]: # Last 4 features in the second column
            label = feature.replace('_', ' ').title()
            input_data[feature] = st.number_input(
                label=label,
                value=None, # Set default value to None
                placeholder="Enter value...",
                min_value=0.0,
                max_value=2500.0,
                step=None, # Allow any float value
                help=tooltips.get(feature, "Enter the measured value.")
            )
            
    st.markdown("---") # Visual separator

    # --- Prediction Button ---
    if st.button("Predict Liver Disease Status", type="primary", use_container_width=True):
        # Check if all fields are filled
        if None in input_data.values():
            st.warning("Please fill in all the required fields before making a prediction.", icon="⚠️")
        else:
            # 1. Convert input data to a DataFrame
            input_df = pd.DataFrame([input_data])
            # Ensure the column order matches the model's training order
            input_df = input_df[best_features]

            # 2. Make Prediction
            prediction = final_model.predict(input_df)
            prediction_proba = final_model.predict_proba(input_df)

            # 3. Display Results
            st.subheader("Prediction Result")

            res_col1, res_col2 = st.columns(2)

            with res_col1:
                if prediction[0] == 1:
                    st.error("Prediction: **LIVER DISEASE DETECTED**", icon="💔")
                else:
                    st.success("Prediction: **NO LIVER DISEASE DETECTED**", icon="💚")

            with res_col2:
                st.info("**Confidence Score**")
                # Display probability for the predicted class
                confidence_score = prediction_proba[0][prediction[0]] * 100
                st.write(f"The model is **{confidence_score:.2f}%** confident.")
                st.progress(int(confidence_score))

            # 4. Show input data for confirmation
            with st.expander("Show Input Data Used for Prediction"):
                st.dataframe(input_df)
else:
    st.error("Model could not be loaded. The application cannot proceed with predictions. Please check the logs or the source files.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed for demonstration purposes only. **This application is not a substitute for professional medical advice.**")
