import streamlit as st
import numpy as np
import librosa
import joblib
import sounddevice as sd
import wavio
import tempfile
import os

# === Load model dan scaler ===
model = joblib.load("model_knn_voice.pkl")
scaler = joblib.load("scaler.pkl")

# === Fungsi ekstraksi fitur ===
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# === Header Aplikasi ===
st.title("🎙️ Voice Command Classifier (BUKA / TUTUP)")
st.write("Upload atau rekam suara, lalu sistem akan memprediksi apakah suara termasuk **BUKA** atau **TUTUP**.")

# === Pilihan Input ===
option = st.radio("Pilih metode input suara:", ["🎧 Upload file", "🎤 Rekam langsung"])

file_path = None

if option == "🎧 Upload file":
    uploaded_file = st.file_uploader("Upload file suara (format .mp3 atau .wav)", type=["mp3", "wav"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name
        st.audio(uploaded_file)

elif option == "🎤 Rekam langsung":
    duration = st.slider("Durasi rekaman (detik)", 1, 10, 3)
    if st.button("Mulai Rekam"):
        st.info("🎙️ Merekam suara... Silakan bicara.")
        recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
        sd.wait()
        st.success("✅ Rekaman selesai!")

        # Simpan rekaman sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wavio.write(tmp.name, recording, 44100, sampwidth=2)
            file_path = tmp.name
        st.audio(file_path)

# === Tombol Prediksi ===
if file_path and st.button("🔍 Prediksi"):
    try:
        features = extract_features(file_path).reshape(1, -1)
        features = scaler.transform(features)

        pred = model.predict(features)[0]
        result = "🟢 BUKA" if pred == 0 else "🔴 TUTUP"

        st.subheader("Hasil Prediksi:")
        st.success(f"Model memprediksi: **{result}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# === Footer ===
st.markdown("---")
st.caption("Dibuat dengan ❤️ menggunakan Streamlit dan KNN")
