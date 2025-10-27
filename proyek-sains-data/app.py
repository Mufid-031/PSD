import streamlit as st
import numpy as np
import joblib
import librosa
import soundfile as sf
import os
import tempfile
from pydub import AudioSegment
import sounddevice as sd
import time

# ===============================
# 🔹 Judul & Deskripsi
# ===============================
st.title("🎙️ Voice Command Classifier: BUKA / TUTUP")
st.write("Pilih salah satu metode di bawah untuk memberikan input suara:")

# ===============================
# 🔹 Load Model & Scaler
# ===============================
model_path = os.path.join(os.path.dirname(__file__), "model_knn_voice.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ===============================
# 🔹 Pilihan Metode Input
# ===============================
mode = st.radio("🎧 Pilih metode input suara:", ["🎙️ Rekam selama 3 detik", "📁 Upload file (.wav / .mp3)"])

# ===============================
# 1️⃣ MODE REKAM SELAMA 3 DETIK
# ===============================
if mode == "🎙️ Rekam selama 3 detik":
    st.info("Klik tombol di bawah ini untuk merekam suara Anda selama 3 detik...")

    if st.button("🎙️ Rekam Suara"):
        sr = 16000  # sample rate
        duration = 3  # detik

        st.write("⏺️ Merekam suara selama 3 detik...")
        recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()  # tunggu sampai selesai merekam
        st.success("✅ Rekaman selesai!")

        # Simpan hasil rekaman sementara
        temp_wav = "recorded_audio.wav"
        sf.write(temp_wav, recording, sr)

        # Tampilkan player audio
        st.audio(temp_wav, format="audio/wav")

        # Ekstraksi MFCC dan Prediksi
        y, _ = librosa.load(temp_wav, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)
        features = np.mean(mfcc.T, axis=0).reshape(1, -1)
        features = scaler.transform(features)
        pred = model.predict(features)
        result = "BUKA" if pred[0] == 0 else "TUTUP"

        st.success(f"🎧 Prediksi: {result}")

# ===============================
# 2️⃣ MODE UPLOAD FILE
# ===============================
else:
    uploaded_file = st.file_uploader("📁 Upload file suara (.wav / .mp3)", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Simpan sementara ke file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            temp_path = tmp.name

        # Gunakan pydub agar mendukung .mp3 dan .wav
        audio = AudioSegment.from_file(uploaded_file)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(temp_path, format="wav")

        # Tampilkan player
        st.audio(temp_path, format="audio/wav")

        # Ekstraksi MFCC dan Prediksi
        y, sr = librosa.load(temp_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.mean(mfcc.T, axis=0).reshape(1, -1)
        features = scaler.transform(features)
        pred = model.predict(features)
        result = "BUKA" if pred[0] == 0 else "TUTUP"

        st.success(f"🎧 Prediksi: {result}")
