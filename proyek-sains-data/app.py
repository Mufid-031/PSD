import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration, WebRtcMode
import numpy as np
import joblib
import librosa
import os


model_path = os.path.join(os.path.dirname(__file__), "model_knn_voice.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
# Load model dan scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.title("🎙️ Voice Command Classifier: BUKA / TUTUP")
st.write("Tekan tombol **Start** lalu ucapkan kata 'Buka' atau 'Tutup'...")

# Konfigurasi WebRTC
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = []

    def recv_audio(self, frames):
        # Konversi audio ke array numpy
        audio_data = np.frombuffer(frames.to_ndarray().tobytes(), np.int16)
        self.buffer.extend(audio_data)
        return frames

# Jalankan WebRTC streamer
ctx = webrtc_streamer(
    key="voice-cmd",
    mode=WebRtcMode.SENDONLY,  # ✅ gunakan konstanta yang benar
    audio_processor_factory=AudioProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={"audio": True, "video": False},
)

if ctx.audio_processor:
    if st.button("🔍 Analisis Voice"):
        if len(ctx.audio_processor.buffer) == 0:
            st.warning("⚠️ Tidak ada suara yang terekam. Coba ucapkan lagi 'Buka' atau 'Tutup'.")
        else:
            # Konversi buffer audio ke numpy array
            audio = np.array(ctx.audio_processor.buffer, dtype=np.float32)
            sr = 16000  # sample rate
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features = np.mean(mfcc.T, axis=0).reshape(1, -1)
            features = scaler.transform(features)
            pred = model.predict(features)
            result = "BUKA" if pred[0] == 0 else "TUTUP"
            st.success(f"🎧 Prediksi: {result}")