import streamlit as st
import numpy as np
import wfdb
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks

# -----------------------
# Streamlit Page Config
# -----------------------
st.set_page_config(page_title="ECG Classification + Explainability", page_icon="🫀", layout="wide")
st.title("🫀 ECG Classification + Explainable AI (WFDB Version)")
st.write("Upload your **.hea** and **.dat** ECG record files (e.g., `108.hea` and `108.dat`).")

# -----------------------
# Load Model and Artifacts
# -----------------------
try:
    model = tf.keras.models.load_model("model_files/ecg_cnn_model.keras")
    scaler = joblib.load("model_files/scaler.pkl")
    class_names = joblib.load("model_files/class_names.pkl")
    expected_features = scaler.mean_.shape[0]
except Exception as e:
    st.error(f"⚠️ Error loading model or scaler: {e}")
    st.stop()

st.info(f"Model expects input segments of **{expected_features} samples** per ECG window.")

# -----------------------
# File Upload
# -----------------------
hea_file = st.file_uploader("Upload .hea file", type=["hea"])
dat_file = st.file_uploader("Upload .dat file", type=["dat"])

# -----------------------
# Compute Helper Functions
# -----------------------
def compute_basic_diff_metrics(orig, recon, fs=360):
    orig, recon = orig.flatten(), recon.flatten()
    diff = orig - recon
    l2 = np.linalg.norm(diff)
    mean_abs = np.mean(np.abs(diff))
    peaks_o, _ = find_peaks(orig, distance=int(0.2*fs), prominence=0.3)
    peaks_r, _ = find_peaks(recon, distance=int(0.2*fs), prominence=0.3)
    duration_sec = len(orig)/fs
    hr_o = len(peaks_o)*(60.0/duration_sec)
    hr_r = len(peaks_r)*(60.0/duration_sec)
    return {
        'l2': float(l2),
        'mean_abs': float(mean_abs),
        'num_peaks_orig': len(peaks_o),
        'num_peaks_recon': len(peaks_r),
        'hr_orig_bpm': hr_o,
        'hr_recon_bpm': hr_r
    }

# -----------------------
# If Files Uploaded
# -----------------------
if hea_file and dat_file:
    base_name = hea_file.name.split(".")[0]
    hea_path, dat_path = f"{base_name}.hea", f"{base_name}.dat"

    with open(hea_path, "wb") as f:
        f.write(hea_file.getbuffer())
    with open(dat_path, "wb") as f:
        f.write(dat_file.getbuffer())

    try:
        record = wfdb.rdrecord(base_name)
        signal = record.p_signal[:, 0]
        fs = int(record.fs) if hasattr(record, 'fs') else 360

        st.subheader("📈 Raw ECG Signal")
        st.line_chart(signal[:2000])

        # Segment handling
        if len(signal) > expected_features:
            start = np.random.randint(0, len(signal)-expected_features)
            segment = signal[start:start+expected_features]
        else:
            segment = np.pad(signal, (0, expected_features-len(signal)))

        # Scaling
        try:
            segment_scaled = scaler.transform(segment.reshape(1, -1)).reshape(1, expected_features, 1)
        except Exception:
            temp = StandardScaler()
            segment_scaled = temp.fit_transform(segment.reshape(-1,1)).reshape(1, expected_features, 1)
            st.warning("⚠️ Temporary scaling applied (mismatch detected).")

        # Prediction
        pred_prob = model.predict(segment_scaled)[0][0]
        pred_label = class_names[int(pred_prob >= 0.5)]
        conf = pred_prob if pred_label == class_names[1] else 1 - pred_prob

        st.subheader("🧠 Prediction Results")
        st.metric("Predicted Class", pred_label)
        st.metric("Confidence", f"{conf*100:.2f}%")

        # Plot processed ECG
        fig1, ax1 = plt.subplots()
        ax1.plot(segment)
        ax1.set_title("Processed ECG Segment (used for prediction)")
        ax1.set_xlabel("Samples")
        ax1.set_ylabel("Amplitude")
        st.pyplot(fig1)

        # -----------------------
        # XAI 1: Autoencoder Reconstruction
        # -----------------------
        st.subheader("🔍 Autoencoder Reconstruction Comparison")

        auto_model = tf.keras.models.load_model("model_files/autoencoder_model.keras") if os.path.exists("model_files/autoencoder_model.keras") else None
        if auto_model:
            recon = auto_model.predict(segment_scaled)
            metrics = compute_basic_diff_metrics(segment_scaled.flatten(), recon.flatten(), fs)

            fig2, ax2 = plt.subplots()
            ax2.plot(segment_scaled.flatten(), label="Original")
            ax2.plot(recon.flatten(), label="Reconstructed", alpha=0.7)
            ax2.legend()
            ax2.set_title("Abnormal vs Reconstructed (Closest Normal)")
            st.pyplot(fig2)

            st.write("📊 **Difference Metrics:**")
            st.json(metrics)
        else:
            st.warning("Autoencoder model not found — skipping reconstruction explainability.")

        # -----------------------
        # XAI 2: SHAP Feature Importance
        # -----------------------
        import shap
        st.subheader("📊 SHAP Feature Importance")
        try:
            background = segment_scaled + np.random.normal(0, 0.01, size=segment_scaled.shape)
            explainer = shap.GradientExplainer(model, background)
            shap_vals = explainer.shap_values(segment_scaled)
            shap_vals = shap_vals[0].flatten() if isinstance(shap_vals, list) else shap_vals.flatten()

            fig3, ax3 = plt.subplots()
            ax3.plot(segment, label="ECG Signal", color='black')
            ax3.fill_between(range(len(segment)), 0, shap_vals, where=shap_vals>0,
                             color='green', alpha=0.3, label='Positive Influence')
            ax3.fill_between(range(len(segment)), 0, shap_vals, where=shap_vals<0,
                             color='red', alpha=0.3, label='Negative Influence')
            ax3.set_title("SHAP Contribution Over Time")
            ax3.legend()
            st.pyplot(fig3)
        except Exception as e:
            st.warning(f"SHAP analysis skipped: {e}")

        # -----------------------
        # XAI 3: Saliency Map
        # -----------------------
        st.subheader("🔥 Saliency Map (Model Focus)")
        try:
            with tf.GradientTape() as tape:
                inp = tf.convert_to_tensor(segment_scaled, dtype=tf.float32)
                tape.watch(inp)
                out = model(inp)
            grads = tape.gradient(out, inp).numpy().squeeze()
            saliency = np.abs(grads)
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

            fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 6))
            ax4a.plot(segment_scaled.flatten(), color='black')
            ax4a.set_title("ECG Signal")
            ax4b.plot(saliency, color='red')
            ax4b.set_title("Saliency (Feature Importance)")
            plt.tight_layout()
            st.pyplot(fig4)
        except Exception as e:
            st.warning(f"Saliency computation skipped: {e}")

    except Exception as e:
        st.error(f"⚠️ Error reading or processing ECG file: {e}")

    finally:
        for p in [hea_path, dat_path]:
            if os.path.exists(p):
                os.remove(p)
