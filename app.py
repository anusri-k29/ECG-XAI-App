import streamlit as st
import numpy as np
import wfdb
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

# -----------------------
# Load Model and Utilities
# -----------------------
st.set_page_config(page_title="ECG Classification", page_icon="🫀", layout="wide")

st.title("🫀 ECG Classification + Explainable AI (WFDB Version)")
st.write("Upload **.dat** and **.hea** files of the same ECG record (e.g., 101.hea and 101.dat).")

try:
    model = tf.keras.models.load_model("model_files/ecg_cnn_model.keras")
    scaler = joblib.load("model_files/scaler.pkl")
    class_names = joblib.load("model_files/class_names.pkl")
    expected_features = scaler.mean_.shape[0]
except Exception as e:
    st.error(f"⚠️ Error loading model or scaler: {e}")
    st.stop()

st.info(f"ℹ️ Scaler expects input segments of **{expected_features} samples**.")

# -----------------------
# Upload ECG Files
# -----------------------
hea_file = st.file_uploader("Upload the .hea file", type=["hea"])
dat_file = st.file_uploader("Upload the .dat file", type=["dat"])

if hea_file and dat_file:
    base_name = hea_file.name.split(".")[0]
    hea_path = f"{base_name}.hea"
    dat_path = f"{base_name}.dat"

    # Save uploaded files locally
    with open(hea_path, "wb") as f:
        f.write(hea_file.getbuffer())
    with open(dat_path, "wb") as f:
        f.write(dat_file.getbuffer())

    try:
        # -----------------------
        # Read ECG Record
        # -----------------------
        record = wfdb.rdrecord(base_name, sampto=5000)
        signal = record.p_signal[:, 0]  # Use first ECG lead

        st.subheader("📈 Raw ECG Signal (first 2000 samples)")
        st.line_chart(signal[:2000])

        # -----------------------
        # Preprocessing
        # -----------------------
        if len(signal) > expected_features:
            start = np.random.randint(0, len(signal) - expected_features)
            signal_segment = signal[start:start + expected_features]
        else:
            signal_segment = np.pad(signal, (0, expected_features - len(signal)), 'constant')

        try:
            # Try using pre-trained scaler
            signal_scaled = scaler.transform(signal_segment.reshape(1, -1)).reshape(1, expected_features, 1)
        except Exception:
            # Fallback: Fit a temporary scaler dynamically (if mismatch occurs)
            temp_scaler = StandardScaler()
            signal_scaled = temp_scaler.fit_transform(signal_segment.reshape(-1, 1)).reshape(1, expected_features, 1)
            st.warning("⚠️ Scaler mismatch detected — using temporary scaling for this session.")

        # -----------------------
        # Prediction
        # -----------------------
        pred_prob = model.predict(signal_scaled)[0][0]
        pred_label = class_names[int(pred_prob >= 0.5)]

        st.subheader("🧠 Prediction")
        st.success(f"**Predicted Class:** {pred_label}")
        st.metric(label="Confidence", value=f"{pred_prob:.2f}")

        # -----------------------
        # Explainability
        # -----------------------
        diff = signal_segment - np.mean(signal_segment)
        mean_diff = np.mean(np.abs(diff))

        explanation = f"Signal deviation from baseline mean: {mean_diff:.3f}. " \
                      f"Pattern corresponds to **{pred_label}** according to model."
        st.info("🩺 **Explanation:**\n" + explanation)

        # -----------------------
        # Plot Processed Segment
        # -----------------------
        st.subheader("🔍 Processed ECG Segment (used for prediction)")
        fig, ax = plt.subplots()
        ax.plot(signal_segment)
        ax.set_title(f"ECG Segment ({expected_features} samples)")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error reading or processing ECG file: {e}")

    finally:
        # Clean up temporary files
        for file_path in [hea_path, dat_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
