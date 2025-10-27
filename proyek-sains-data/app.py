import streamlit as st
import numpy as np
import librosa
import joblib
from st_audiorec import st_audiorec

# Judul aplikasi
st.title("🎙️ Voice Command Classifier: BUKA / TUTUP")
st.write("Rekam suara kamu (‘buka’ atau ‘tutup’) dan sistem akan memprediksi perintahnya.")

# Load model dan scaler
model = joblib.load("model_knn_voice.pkl")
scaler = joblib.load("scaler.pkl")

def extract_features(file):
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Komponen perekam audio
st.write("🎧 Klik tombol di bawah untuk merekam suara:")
wav_audio_data = st_audiorec()

# Jika ada hasil rekaman
if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')
    st.write("⏳ Sedang menganalisis suara...")

    try:
        # Simpan file sementara
        with open("temp.wav", "wb") as f:
            f.write(wav_audio_data)

        # Ekstraksi fitur & prediksi
        features = extract_features("temp.wav").reshape(1, -1)
        features = scaler.transform(features)
        pred = model.predict(features)[0]

        result = "🔓 BUKA" if pred == 0 else "🔒 TUTUP"
        st.success(f"Prediksi suara: **{result}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
