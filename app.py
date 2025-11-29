import gradio as gr
import joblib
import pandas as pd
import numpy as np
import shap

# --- 1. LOAD THE SAVED ARTIFACTS ---

# Load the trained model, scaler, feature names, and SHAP data
try:
    model = joblib.load('svm_model.joblib')
    scaler = joblib.load('scaler.joblib')
    feature_names = joblib.load('feature_names.joblib')
    shap_data = joblib.load('shap_data.joblib')
except FileNotFoundError:
    print("Error: One or more required .joblib files not found. Make sure they are in the same directory as app.py.")
    exit()

# --- 2. CREATE THE SHAP EXPLAINER ---

# The explainer tells us how features contribute to the prediction
explainer = shap.KernelExplainer(model.predict_proba, shap_data)


# --- 3. DEFINE THE PREDICTION FUNCTION ---

# This function will take user inputs, process them, and return the results
def predict_cancer(*feature_values):
    try:
        # Convert the 30 input values into a NumPy array
        input_data = np.array([float(val) if val is not None else 0 for val in feature_values]).reshape(1, -1)
    except (ValueError, TypeError):
        return "Error", "Invalid input. Please ensure all 30 fields are filled with numbers.", "", ""

    # Scale the input data using the loaded scaler
    scaled_data = scaler.transform(input_data)

    # --- GET PREDICTION AND CONFIDENCE SCORE ---
    prediction_proba = model.predict_proba(scaled_data)[0]
    prediction = model.predict(scaled_data)[0]

    if prediction == 0:
        diagnosis = "Benign"
        confidence_score = f"{prediction_proba[0] * 100:.2f}%"
    else:
        diagnosis = "Malignant"
        confidence_score = f"{prediction_proba[1] * 100:.2f}%"

    # --- GET TOP CONTRIBUTING FEATURES USING SHAP ---
    shap_values = explainer.shap_values(scaled_data)[1] # Get values for the "Malignant" class

    # Create a DataFrame for easy analysis
    feature_shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': abs(shap_values)
    })

    # Get the top 3 features with the highest impact
    top_features_df = feature_shap_df.sort_values(by='shap_value', ascending=False).head(3)
    top_features = "\n".join(top_features_df['feature'].tolist())

    # --- GENERATE THE EXPLANATORY NOTE ---
    if diagnosis == "Benign":
        note = "The model predicts the tumor is **Benign**. This means it is likely non-cancerous. This prediction is based on the diagnostic features provided."
    else:
        note = "The model predicts the tumor is **Malignant**. This indicates a high likelihood of being cancerous. Please consult a medical professional for confirmation and further steps."

    return diagnosis, confidence_score, top_features, note


# --- 4. CREATE THE GRADIO INTERFACE ---

# Create a list of input components for the 30 features
input_components = [gr.Number(label=name) for name in feature_names]

# Create the user interface
app = gr.Interface(
    fn=predict_cancer,
    inputs=input_components,
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence Score"),
        gr.Textbox(label="Top 3 Contributing Features"),
        gr.Markdown(label="What This Prediction Means")
    ],
    title="Breast Cancer Diagnosis Predictor",
    description="Enter the 30 diagnostic feature values below to get a prediction from the SVM model.",
    allow_flagging="never"
)

# --- 5. LAUNCH THE APP ---
if __name__ == "__main__":
    app.launch()
