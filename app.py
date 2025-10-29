import streamlit as st
import numpy as np
import wfdb
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import shap
plt.rcParams["figure.figsize"] = (8, 3)
# ----------------------------------
# Streamlit setup
# ----------------------------------
st.set_page_config(page_title="ECG Classification + Explainability", page_icon="🫀", layout="wide")
st.title("🫀 ECG Classiifcation using XAI")
st.write("Upload your **.hea** and **.dat** ECG record files.")
st.caption("ℹ️ `.hea` (header) and `.dat` (data) files together store the ECG recording — the header has metadata like sampling rate, and the `.dat` file has the raw waveform signal.")


# ----------------------------------
# Load model
# ----------------------------------
try:
    model = tf.keras.models.load_model("model_files/ecg_cnn_model_new.keras")
    scaler = joblib.load("model_files/scaler_new.pkl")
    class_names = joblib.load("model_files/class_names_new.pkl")
    expected_features = scaler.mean_.shape[0]
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.info(f"Model expects input segments of **{expected_features} samples** per ECG window.")

# ----------------------------------
# Helper functions
# ----------------------------------
def compute_basic_diff_metrics(orig, recon, fs=360):
    orig, recon = orig.flatten(), recon.flatten()
    diff = orig - recon
    l2 = np.linalg.norm(diff)
    mean_abs = np.mean(np.abs(diff))
    peaks_o, _ = find_peaks(orig, distance=int(0.2 * fs), prominence=0.3)
    peaks_r, _ = find_peaks(recon, distance=int(0.2 * fs), prominence=0.3)
    duration_sec = len(orig) / fs
    hr_o = len(peaks_o) * (60.0 / duration_sec)
    hr_r = len(peaks_r) * (60.0 / duration_sec)
    return {
        "l2": float(l2),
        "mean_abs": float(mean_abs),
        "num_peaks_orig": len(peaks_o),
        "num_peaks_recon": len(peaks_r),
        "hr_orig_bpm": hr_o,
        "hr_recon_bpm": hr_r,
    }

# ----------------------------------
# File uploader
# ----------------------------------
hea_file = st.file_uploader("Upload .hea file", type=["hea"])
dat_file = st.file_uploader("Upload .dat file", type=["dat"])

# ----------------------------------
# Main workflow
# ----------------------------------
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
        fs = int(record.fs) if hasattr(record, "fs") else 360

        st.subheader("📈 Raw ECG Signal")
        st.line_chart(signal[:2000])

        # segment
        if len(signal) > expected_features:
            start = np.random.randint(0, len(signal) - expected_features)
            segment = signal[start : start + expected_features]
        else:
            segment = np.pad(signal, (0, expected_features - len(signal)))

        # scaling
        try:
            segment_scaled = scaler.transform(segment.reshape(1, -1)).reshape(
                1, expected_features, 1
            )
        except Exception:
            temp = StandardScaler()
            segment_scaled = temp.fit_transform(segment.reshape(-1, 1)).reshape(
                1, expected_features, 1
            )
            st.warning("Temporary scaling applied.")

        # prediction
        pred_prob = model.predict(segment_scaled)[0][0]
        pred_label = class_names[int(pred_prob >= 0.5)]
        conf = pred_prob if pred_label == class_names[1] else 1 - pred_prob

        st.subheader(" Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Class", pred_label)
        with col2:
            st.metric("Confidence", f"{conf*100:.2f}%")

        # ----------------------------------
        # XAI 1: Autoencoder Reconstruction
        # ----------------------------------
        st.subheader(" Autoencoder Reconstruction Comparison")
        st.caption(" The Autoencoder compares your ECG with its learned 'normal' pattern to highlight how far it differs from a healthy heartbeat.")
        auto_path = "model_files/autoencoder_model.keras"
        if os.path.exists(auto_path):
            auto_model = tf.keras.models.load_model(auto_path)
            recon = auto_model.predict(segment_scaled)
            metrics = compute_basic_diff_metrics(segment_scaled.flatten(), recon.flatten(), fs)

            fig2, ax2 = plt.subplots()
            ax2.plot(segment_scaled.flatten(), label="Original")
            ax2.plot(recon.flatten(), label="Reconstructed (closest normal)", alpha=0.7)
            ax2.legend()
            ax2.set_title("Original vs Reconstructed Signal")
            st.pyplot(fig2)

            st.write(" **Difference Metrics:**")
            st.json(metrics)
        else:
            st.warning("Autoencoder model not found — skipping reconstruction explainability.")

        # ----------------------------------
        # XAI 2: SHAP Feature Importance
        # ----------------------------------
        st.subheader("SHAP Feature Importance (Enhanced)")
        st.caption(" SHAP explains which parts of your ECG most influenced the model’s decision.")

        try:
            background = segment_scaled + np.random.normal(0, 0.005, size=segment_scaled.shape)
            explainer = shap.GradientExplainer(model, background)
            shap_vals = explainer.shap_values(segment_scaled)
            shap_vals = shap_vals[0].flatten() if isinstance(shap_vals, list) else shap_vals.flatten()
            shap_vals_scaled = shap_vals / np.max(np.abs(shap_vals)) * np.max(np.abs(segment))

            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.plot(segment, label="ECG Signal", color="black", linewidth=1.2)
            ax3.plot(shap_vals_scaled, label="SHAP Influence", color="red", alpha=0.7, linewidth=1.5)
            ax3.fill_between(range(len(segment)), 0, shap_vals_scaled,
                             where=shap_vals_scaled>0, color='green', alpha=0.3, label='Positive Influence')
            ax3.fill_between(range(len(segment)), 0, shap_vals_scaled,
                             where=shap_vals_scaled<0, color='red', alpha=0.3, label='Negative Influence')
            ax3.set_title("SHAP Feature Importance Over ECG Time Steps", fontsize=13)
            ax3.legend()
            ax3.grid(alpha=0.3)
            st.pyplot(fig3)
        except Exception as e:
            st.warning(f"SHAP analysis skipped: {e}")

        # ----------------------------------
        # XAI 3: Saliency Map
        # ----------------------------------
        st.subheader("Saliency Map (Model Focus)")
        st.caption("The Saliency Map shows which time points the model focused on most when making its prediction.")

        try:
            with tf.GradientTape() as tape:
                inp = tf.convert_to_tensor(segment_scaled, dtype=tf.float32)
                tape.watch(inp)
                out = model(inp)
            grads = tape.gradient(out, inp).numpy().squeeze()
            saliency = np.abs(grads)
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

            fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 6))
            ax4a.plot(segment_scaled.flatten(), color="black")
            ax4a.set_title("ECG Signal")
            ax4b.plot(saliency, color="red")
            ax4b.set_title("Saliency (Model Focus Regions)")
            plt.tight_layout()
            st.pyplot(fig4)
        except Exception as e:
            st.warning(f"Saliency computation skipped: {e}")

        # ----------------------------------
        # P-Q-R-S-T Section Graph with SHAP Overlay
        # ----------------------------------
        st.subheader("ECG P–Q–R–S–T Section Visualization (with SHAP Overlay)")
        st.caption("This highlights key heartbeat components — P wave, QRS complex, and T wave — and overlays model attention to show which region mattered most.")

        try:
            regions = {
                "P_wave": (0, 50),
                "QRS_complex": (50, 120),
                "T_wave": (120, 180)
            }

            # Prepare SHAP overlay safely
            if 'shap_vals' in locals() and len(shap_vals) > 0:
                shap_curve = shap_vals
            else:
                shap_curve = np.zeros_like(segment)

            signal = segment[:200]
            shap_scaled = (
                (shap_curve[:200] / (np.max(np.abs(shap_curve)) + 1e-8))
                * np.max(np.abs(signal))
                * 0.8
            )

            # Compute region-wise SHAP importance
            region_importance = {}
            for region, (start, end) in regions.items():
                mean_val = np.mean(np.abs(shap_curve[start:end]))
                region_importance[region] = float(mean_val)

            st.write("**Region-wise |SHAP| importance:**")
            st.json(region_importance)

            fig5, ax5 = plt.subplots(figsize=(12, 4))
            ax5.plot(signal, color='black', linewidth=1.2, label='ECG Signal')
            ax5.plot(shap_scaled, color='red', alpha=0.6, label='SHAP Influence (scaled)')
            ax5.axvspan(*regions["P_wave"], color='blue', alpha=0.1, label='P wave')
            ax5.axvspan(*regions["QRS_complex"], color='yellow', alpha=0.2, label='QRS complex')
            ax5.axvspan(*regions["T_wave"], color='green', alpha=0.1, label='T wave')
            ax5.legend()
            ax5.set_title("ECG Signal with SHAP Importance by Region")
            ax5.set_xlabel("Time Step")
            ax5.set_ylabel("Amplitude")
            st.pyplot(fig5)

        except Exception as e:
            st.warning(f"Section visualization skipped: {e}")

    except Exception as e:
        st.error(f"Error reading or processing ECG file: {e}")

    finally:
        for p in [hea_path, dat_path]:
            if os.path.exists(p):
                os.remove(p)
