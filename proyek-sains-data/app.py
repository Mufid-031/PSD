import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration, WebRtcMode
import numpy as np
import joblib
import librosa
import os
import av
import soundfile as sf

st.title("🎙️ Voice Command Classifier: BUKA / TUTUP")
st.write("Pilih salah satu metode di bawah untuk memberikan input suara:")

# Load model dan scaler
model_path = os.path.join(os.path.dirname(__file__), "model_knn_voice.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Pilih metode input
mode = st.radio("🎧 Pilih metode input suara:", ["🎙️ Rekam langsung", "📁 Upload file .wav"])

# ===============================
# 1️⃣ MODE REKAM LANGSUNG
# ===============================
if mode == "🎙️ Rekam langsung":
    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.buffer = np.array([], dtype=np.float32)

        def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
            audio = frame.to_ndarray()
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            audio = audio.astype(np.float32)
            self.buffer = np.concatenate((self.buffer, audio))
            return frame

    ctx = webrtc_streamer(
        key="voice-cmd",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"audio": True, "video": False},
    )

    if ctx.state.playing and ctx.audio_processor:
        if st.button("🔍 Analisis Voice"):
            audio_data = ctx.audio_processor.buffer

            if len(audio_data) < 16000:
                st.warning("⚠️ Suara terlalu singkat. Coba ucapkan lebih lama 'Buka' atau 'Tutup'.")
            else:
                sr = 16000
                mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
                features = np.mean(mfcc.T, axis=0).reshape(1, -1)
                features = scaler.transform(features)
                pred = model.predict(features)
                result = "BUKA" if pred[0] == 0 else "TUTUP"

                st.success(f"🎧 Prediksi: {result}")
                ctx.audio_processor.buffer = np.array([], dtype=np.float32)

# ===============================
# 2️⃣ MODE UPLOAD FILE
# ===============================
else:
    uploaded_file = st.file_uploader("📁 Upload file suara (.wav)", type=["wav"])

    if uploaded_file is not None:
        # Simpan sementara
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())

        # Tampilkan player
        st.audio("temp.wav", format="audio/wav")

        # Analisis
        if st.button("🔍 Analisis Voice"):
            y, sr = librosa.load("temp.wav", sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features = np.mean(mfcc.T, axis=0).reshape(1, -1)
            features = scaler.transform(features)
            pred = model.predict(features)
            result = "BUKA" if pred[0] == 0 else "TUTUP"

            st.success(f"🎧 Prediksi: {result}")
