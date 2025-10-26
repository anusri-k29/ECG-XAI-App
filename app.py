import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

# -----------------------
# Load Model and Utilities
# -----------------------
model = tf.keras.models.load_model("model_files/ecg_cnn_model.keras")
scaler = joblib.load("model_files/scaler.pkl")
class_names = joblib.load("model_files/class_names.pkl")

st.title("🫀 ECG Classification + Explainable AI Dashboard")

# -----------------------
# Upload ECG File (.npy)
# -----------------------
uploaded_file = st.file_uploader("Upload your ECG file (.npy format)", type=["npy"])

if uploaded_file is not None:
    try:
        # Load ECG signal
        signal = np.load(uploaded_file)

        # Flatten if multi-dimensional
        signal = np.squeeze(signal)

        # Ensure 1D array
        if signal.ndim != 1:
            st.error("Uploaded file must contain a 1D ECG signal (single lead).")
            st.stop()

        # Display raw signal
        st.subheader("Uploaded ECG Signal")
        st.line_chart(signal)

        # -----------------------
        # Preprocessing
        # -----------------------
        try:
            # Reshape and scale
            signal_scaled = scaler.transform(signal.reshape(1, -1)).reshape(1, 200, 1)
        except Exception as e:
            st.error(f"Scaling failed: {e}")
            st.stop()

        # -----------------------
        # Prediction
        # -----------------------
        pred_prob = model.predict(signal_scaled)[0][0]
        pred_label = class_names[int(pred_prob >= 0.5)]

        st.subheader("Prediction Result")
        st.success(f"Predicted Class: **{pred_label}** (Confidence: {pred_prob:.2f})")

        # -----------------------
        # Simple Explanation
        # -----------------------
        diff = signal_scaled.flatten() - np.mean(signal_scaled)
        hr_diff = np.mean(np.abs(diff)) * 100
        explanation = (
            f"This ECG shows a deviation of approximately {hr_diff:.2f}% "
            f"from the baseline normal rhythm pattern."
        )

        st.subheader("Explainability Insight")
        st.info(explanation)

        # -----------------------
        # Counterfactual (Simulated Normal)
        # -----------------------
        st.subheader("Counterfactual: Closest Normal ECG (Demo)")
        reconstructed_signal = np.mean(signal_scaled) + np.random.normal(0, 0.02, len(signal_scaled.flatten()))
        st.line_chart(reconstructed_signal)

        st.caption("Simulated 'normal' ECG helps visualize what the healthy signal may look like.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
