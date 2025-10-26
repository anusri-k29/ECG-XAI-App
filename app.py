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

st.title("🫀 ECG Classification + Explainable AI (NumPy Version)")
st.write("Upload a `.npy` ECG signal file to classify and explain results.")

# -----------------------
# Upload ECG File
# -----------------------
uploaded_file = st.file_uploader("Upload your ECG file (.npy)", type=["npy"])

if uploaded_file is not None:
    # Load the signal
    signal = np.load(uploaded_file)

    # Handle different shapes safely
    signal = np.squeeze(signal)  # remove extra dimensions
    st.subheader("📈 Raw ECG Signal (first 2000 samples)")
    st.line_chart(signal[:2000])

    # -----------------------
    # Preprocess
    # -----------------------
    # Model expects 200 points
    if len(signal) > 200:
        start = np.random.randint(0, len(signal) - 200)
        signal_segment = signal[start:start + 200]
    else:
        signal_segment = np.pad(signal, (0, 200 - len(signal)), 'constant')

    # Scale (reshape to 1x200 for scaler)
    signal_scaled = scaler.transform(signal_segment.reshape(1, -1)).reshape(1, 200, 1)

    # -----------------------
    # Prediction
    # -----------------------
    pred_prob = model.predict(signal_scaled)[0][0]
    pred_label = class_names[int(pred_prob >= 0.5)]

    st.subheader("🧠 Prediction")
    st.success(f"**Predicted Class:** {pred_label}")
    st.metric(label="Confidence", value=f"{pred_prob:.2f}")

    # -----------------------
    # Simple Explanation
    # -----------------------
    diff = signal_segment - np.mean(signal_segment)
    mean_diff = np.mean(np.abs(diff))

    explanation = f"Signal deviation from baseline mean: {mean_diff:.3f}. " \
                  f"Pattern corresponds to **{pred_label}** according to model."
    st.info("🩺 **Explanation:**\n" + explanation)

    # Plot processed 200-sample segment
    st.subheader("🔍 Processed Segment (used for prediction)")
    fig, ax = plt.subplots()
    ax.plot(signal_segment)
    ax.set_title("ECG Segment (200 samples)")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)
