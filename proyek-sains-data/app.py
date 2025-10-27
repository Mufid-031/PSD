import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration, WebRtcMode
import numpy as np
import joblib
import librosa
import os
import av
import soundfile as sf
from pydub import AudioSegment  # untuk konversi mp3 → wav

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
# 🔹 Konfigurasi WebRTC
# ===============================
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ===============================
# 🔹 Pilihan Metode Input
# ===============================
mode = st.radio("🎧 Pilih metode input suara:", ["🎙️ Rekam langsung", "📁 Upload file (.wav / .mp3)"])

# ===============================
# 1️⃣ MODE REKAM LANGSUNG
# ===============================
if mode == "🎙️ Rekam langsung":
    st.info("Tekan **Start** untuk mulai merekam suara Anda. Setelah selesai, tekan **Analisis Voice**.")

    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.frames = []

        def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
            audio = frame.to_ndarray()
            # Konversi ke mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            self.frames.append(audio.astype(np.float32))
            return frame

    ctx = webrtc_streamer(
        key="voice-cmd",
        mode=WebRtcMode.SENDRECV,  # Gunakan SENDRECV agar audio stream lebih stabil
        audio_processor_factory=AudioProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"audio": True, "video": False},
    )

    # Setelah user selesai bicara
    if ctx and ctx.state.playing and ctx.audio_processor:
        if st.button("🔍 Analisis Voice"):
            if not ctx.audio_processor.frames:
                st.warning("⚠️ Tidak ada suara yang terekam. Coba ulangi.")
            else:
                # Gabungkan frame audio
                audio_data = np.concatenate(ctx.audio_processor.frames)
                sr = 16000

                # Normalisasi panjang minimal
                if len(audio_data) < sr:
                    st.warning("⚠️ Suara terlalu singkat. Ucapkan 'Buka' atau 'Tutup' dengan lebih lama.")
                else:
                    # Simpan sementara hasil rekaman (opsional)
                    sf.write("recorded_audio.wav", audio_data, sr)
                    st.audio("recorded_audio.wav", format="audio/wav")

                    # Ekstraksi MFCC
                    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
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
        temp_path = "uploaded_audio.wav"

        # Jika file MP3 → konversi ke WAV
        if uploaded_file.type == "audio/mpeg":
            audio = AudioSegment.from_mp3(uploaded_file)
            audio.export(temp_path, format="wav")
        else:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

        # Tampilkan player
        st.audio(temp_path, format="audio/wav")

        # Analisis
        if st.button("🔍 Analisis Voice"):
            y, sr = librosa.load(temp_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features = np.mean(mfcc.T, axis=0).reshape(1, -1)
            features = scaler.transform(features)
            pred = model.predict(features)
            result = "BUKA" if pred[0] == 0 else "TUTUP"

            st.success(f"🎧 Prediksi: {result}")
