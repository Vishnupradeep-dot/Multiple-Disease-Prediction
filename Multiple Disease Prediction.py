import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ------------------------
# Load Models and Scalers
# ------------------------
liver_model = joblib.load("models/liver_logistic.pkl")
liver_scaler = joblib.load("models/liver_scaler.pkl")

kidney_model = joblib.load("models/kidney_logistic.pkl")
kidney_scaler = joblib.load("models/kidney_scaler.pkl")

parkinson_model = joblib.load("models/parkinson_logistic.pkl")
parkinson_scaler = joblib.load("models/parkinson_scaler.pkl")

# Load training columns for Kidney & Parkinson
kidney_columns = pd.read_csv("D://min_project.4//kidney_disease - kidney_disease.csv").drop('classification', axis=1).columns.tolist()
parkinson_columns = pd.read_csv("D://min_project.4//parkinsons - parkinsons.csv").drop('status', axis=1).columns.tolist()

# ------------------------
# Title
# ------------------------
st.title("Multiple Disease Prediction System")
st.write("Predict **Liver Disease**, **Kidney Disease**, or **Parkinson's Disease**.")

# ------------------------
# Tabs
# ------------------------
tabs = st.tabs(["Liver Disease", "Kidney Disease", "Parkinson's Disease"])

# ------------------------
# LIVER DISEASE TAB
# ------------------------
with tabs[0]:
    st.header("Liver Disease Prediction")

    liver_inputs = {}
    liver_features_list = [
        "Age", "Gender", "Total Bilirubin", "Direct Bilirubin",
        "Alkaline Phosphotase", "Alamine Aminotransferase",
        "Aspartate Aminotransferase", "Total Proteins",
        "Albumin", "Albumin and Globulin Ratio"
    ]

    for feature in liver_features_list:
        if feature == "Gender":
            liver_inputs[feature] = st.selectbox(feature, ["Male", "Female"])
        else:
            liver_inputs[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict Liver Disease"):
        gender_val = 1 if liver_inputs["Gender"] == "Male" else 0
        input_data = np.array([[
            liver_inputs["Age"], gender_val, liver_inputs["Total Bilirubin"],
            liver_inputs["Direct Bilirubin"], liver_inputs["Alkaline Phosphotase"],
            liver_inputs["Alamine Aminotransferase"], liver_inputs["Aspartate Aminotransferase"],
            liver_inputs["Total Proteins"], liver_inputs["Albumin"],
            liver_inputs["Albumin and Globulin Ratio"]
        ]])
        scaled_data = liver_scaler.transform(input_data)
        prediction = liver_model.predict(scaled_data)
        st.success("**Liver Disease Detected**" if prediction[0] == 1 else "**No Liver Disease**")

# ------------------------
# KIDNEY DISEASE TAB
# ------------------------
with tabs[1]:
    st.header("Kidney Disease Prediction")

    kidney_inputs = {}
    for feature in kidney_columns:
        if feature in ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]:
            options = ["yes", "no"] if feature in ["htn", "dm", "cad", "pe", "ane"] else ["normal", "abnormal"]
            if feature in ["pcc", "ba"]:
                options = ["present", "notpresent"]
            if feature == "appet":
                options = ["good", "poor"]
            kidney_inputs[feature] = st.selectbox(feature, options)
        else:
            kidney_inputs[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict Kidney Disease"):
        def encode(val):
            return 1 if val in ["yes", "present", "normal", "good"] else 0

        # Prepare input as DataFrame
        input_data = []
        for col in kidney_columns:
            val = kidney_inputs[col]
            if col in ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]:
                val = encode(val)
            input_data.append(val)

        kidney_df = pd.DataFrame([input_data], columns=kidney_columns)

        # Drop 'id' if exists to match training
        if 'id' in kidney_df.columns:
            kidney_df = kidney_df.drop(columns=['id'])

        scaled_data = kidney_scaler.transform(kidney_df)
        prediction = kidney_model.predict(scaled_data)
        st.success("**Kidney Disease Detected**" if prediction[0] == 1 else "**No Kidney Disease**")

# ------------------------
# PARKINSON'S DISEASE TAB
# ------------------------
with tabs[2]:
    st.header("Parkinson's Disease Prediction")

    parkinson_inputs = {}
    for feature in parkinson_columns:
        parkinson_inputs[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict Parkinson's Disease"):
        parkinson_df = pd.DataFrame([[parkinson_inputs[col] for col in parkinson_columns]],
                                    columns=parkinson_columns)

        # Drop 'name' if exists to match training
        if 'name' in parkinson_df.columns:
            parkinson_df = parkinson_df.drop(columns=['name'])

        scaled_data = parkinson_scaler.transform(parkinson_df)
        prediction = parkinson_model.predict(scaled_data)
        st.success("**Parkinson's Disease Detected**" if prediction[0] == 1 else "**No Parkinson's Disease**")
