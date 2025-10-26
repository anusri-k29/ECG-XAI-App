import streamlit as st
import numpy as np
import wfdb
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# -----------------------
# Load Model and Utilities
# -----------------------
model = tf.keras.models.load_model("model_files/ecg_cnn_model.keras")
scaler = joblib.load("model_files/scaler.pkl")
class_names = joblib.load("model_files/class_names.pkl")

st.title("🫀 ECG Classification + Explainable AI (WFDB Version)")
st.write("Upload **.dat** and **.hea** files of the same ECG record (e.g., 101.hea and 101.dat).")

# -----------------------
# Upload ECG Files
# -----------------------
hea_file = st.file_uploader("Upload the .hea file", type=["hea"])
dat_file = st.file_uploader("Upload the .dat file", type=["dat"])

if hea_file and dat_file:
    # Save both files under the same name base
    with open("uploaded_record.hea", "wb") as f:
        f.write(hea_file.getbuffer())
    with open("uploaded_record.dat", "wb") as f:
        f.write(dat_file.getbuffer())

    try:
        # Read the record using the base name
        record = wfdb.rdrecord("uploaded_record", sampto=5000)
        signal = record.p_signal[:, 0]  # use only the first ECG lead

        st.subheader("📈 Raw ECG Signal (first 2000 samples)")
        st.line_chart(signal[:2000])

        # -----------------------
        # Preprocess
        # -----------------------
        if len(signal) > 200:
            start = np.random.randint(0, len(signal) - 200)
            signal_segment = signal[start:start + 200]
        else:
            signal_segment = np.pad(signal, (0, 200 - len(signal)), 'constant')

        # Scale and reshape
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

        # Plot processed segment
        st.subheader("🔍 Processed Segment (used for prediction)")
        fig, ax = plt.subplots()
        ax.plot(signal_segment)
        ax.set_title("ECG Segment (200 samples)")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error reading the ECG file: {e}")

    finally:
        # Clean up temporary files
        if os.path.exists("uploaded_record.hea"):
            os.remove("uploaded_record.hea")
        if os.path.exists("uploaded_record.dat"):
            os.remove("uploaded_record.dat")
