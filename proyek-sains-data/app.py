import streamlit as st
import numpy as np
import librosa
import joblib
import os

st.title("🎙️ Voice Command Classifier: BUKA / TUTUP")

model_path = os.path.join(os.path.dirname(__file__), "model_knn_voice.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def extract_features(file):
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

uploaded_file = st.file_uploader("🎤 Upload file audio (mp3/wav)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    features = extract_features("temp.wav").reshape(1, -1)
    features = scaler.transform(features)
    pred = model.predict(features)[0]

    result = "🔓 BUKA" if pred == 0 else "🔒 TUTUP"
    st.success(f"Prediksi suara: **{result}**")
