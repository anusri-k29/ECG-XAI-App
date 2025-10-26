import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import scipy.io
import wfdb

# -----------------------
# Load Model and Utilities
# -----------------------
model = tf.keras.models.load_model("model_files/ecg_cnn_model.keras")
scaler = joblib.load("model_files/scaler.pkl")
class_names = joblib.load("model_files/class_names.pkl")

st.title("🫀 ECG Classification + Explainable AI")

# -----------------------
# Upload .mat File
# -----------------------


uploaded_file = st.file_uploader("Upload ECG record (.dat + .hea)", type=["dat"])
if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.dat", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read the ECG signal
    record = wfdb.rdrecord("temp", sampto=5000)  # sampto for demo segment
    signal = record.p_signal[:, 0]  # first lead

    st.line_chart(signal)

    # Preprocess like training
    signal_scaled = scaler.transform(signal.reshape(-1, 1)).reshape(1, -1, 1)

    # Prediction
    pred_prob = model.predict(signal_scaled)[0][0]
    pred_label = class_names[int(pred_prob >= 0.5)]
    st.success(f"Predicted Class: **{pred_label}** ({pred_prob:.2f})")

    # Optional: Generate simple text explanation
    diff = signal_scaled.flatten() - np.mean(signal_scaled)
    hr_diff = np.mean(diff)
    explanation = f"ECG shows deviation from normal pattern (mean diff: {hr_diff:.3f})."
    st.info("Explanation:\n" + explanation)
    
    # Optional: Plot "closest normal" counterfactual (from your autoencoder)
    # reconstruct = autoencoder.predict(signal_scaled)  # add if you save autoencoder
    # st.line_chart(reconstruct.flatten())
