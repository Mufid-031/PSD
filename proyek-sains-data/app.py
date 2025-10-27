import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration, WebRtcMode
import numpy as np
import joblib
import librosa
import os
import av

st.title("🎙️ Voice Command Classifier: BUKA / TUTUP")
st.write("Tekan tombol **Start** lalu ucapkan kata 'Buka' atau 'Tutup'...")

# Load model dan scaler
model_path = os.path.join(os.path.dirname(__file__), "model_knn_voice.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = np.array([], dtype=np.float32)

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Ambil audio numpy array dari frame
        audio = frame.to_ndarray()
        # Jika stereo → ubah jadi mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        # Normalisasi ke float32
        audio = audio.astype(np.float32)
        # Tambahkan ke buffer
        self.buffer = np.concatenate((self.buffer, audio))
        return frame

ctx = webrtc_streamer(
    key="voice-cmd",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={"audio": True, "video": False},
)

# Analisis jika audio sudah direkam
if ctx.state.playing and ctx.audio_processor:
    if st.button("🔍 Analisis Voice"):
        audio_data = ctx.audio_processor.buffer

        if len(audio_data) < 16000:
            st.warning("⚠️ Suara terlalu singkat. Coba ucapkan lebih lama 'Buka' atau 'Tutup'.")
        else:
            # Proses MFCC
            sr = 16000
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            features = np.mean(mfcc.T, axis=0).reshape(1, -1)
            features = scaler.transform(features)
            pred = model.predict(features)
            result = "BUKA" if pred[0] == 0 else "TUTUP"

            st.success(f"🎧 Prediksi: {result}")
            # Reset buffer agar tidak menumpuk
            ctx.audio_processor.buffer = np.array([], dtype=np.float32)
