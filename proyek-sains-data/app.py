import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration, WebRtcMode
import numpy as np
import joblib
import librosa
import os
import av
import soundfile as sf
from pydub import AudioSegment
import queue

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
    st.info("Tekan **Start** untuk mulai merekam suara Anda. Ucapkan 'BUKA' atau 'TUTUP' dengan jelas.")
    
    # Queue untuk menyimpan audio frames
    if "audio_queue" not in st.session_state:
        st.session_state.audio_queue = queue.Queue()
    
    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.frames = []
            self.sample_rate = 48000  # Default WebRTC sample rate
            
        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            # Konversi frame ke numpy array
            sound = frame.to_ndarray()
            
            # Konversi ke mono jika stereo
            if len(sound.shape) == 2:
                sound = sound.mean(axis=1)
            
            # Simpan ke frames
            self.frames.append(sound.flatten())
            
            # Update queue untuk monitoring
            try:
                st.session_state.audio_queue.put(len(sound), block=False)
            except queue.Full:
                pass
            
            return frame

    ctx = webrtc_streamer(
        key="voice-cmd",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    # Tampilkan status recording
    if ctx.state.playing:
        st.success("🔴 Sedang merekam... Ucapkan 'BUKA' atau 'TUTUP'")
        
        # Tampilkan jumlah frame yang terekam
        if ctx.audio_processor:
            frame_count = len(ctx.audio_processor.frames)
            st.metric("Frame Audio Terekam", frame_count)
    
    # Tombol analisis (bisa ditekan kapan saja)
    if st.button("🔍 Analisis Voice", disabled=not ctx.state.playing):
        if ctx.audio_processor and len(ctx.audio_processor.frames) > 0:
            try:
                # Gabungkan semua frame
                audio_data = np.concatenate(ctx.audio_processor.frames)
                original_sr = ctx.audio_processor.sample_rate
                
                st.info(f"📊 Audio terekam: {len(audio_data)} samples, SR: {original_sr} Hz")
                
                # Resample ke 16kHz untuk model
                target_sr = 16000
                audio_resampled = librosa.resample(
                    audio_data.astype(np.float32), 
                    orig_sr=original_sr, 
                    target_sr=target_sr
                )
                
                # Normalisasi amplitude
                audio_resampled = audio_resampled / np.max(np.abs(audio_resampled) + 1e-6)
                
                # Cek panjang minimal (minimal 0.5 detik)
                min_length = int(0.5 * target_sr)
                if len(audio_resampled) < min_length:
                    st.warning("⚠️ Suara terlalu singkat. Minimal 0.5 detik. Coba ucapkan lebih lama.")
                else:
                    # Simpan untuk preview
                    sf.write("recorded_audio.wav", audio_resampled, target_sr)
                    st.audio("recorded_audio.wav", format="audio/wav")
                    
                    # Ekstraksi MFCC
                    mfcc = librosa.feature.mfcc(y=audio_resampled, sr=target_sr, n_mfcc=13)
                    features = np.mean(mfcc.T, axis=0).reshape(1, -1)
                    
                    # Debug info
                    st.write(f"🔍 Shape MFCC: {mfcc.shape}")
                    st.write(f"🔍 Shape Features: {features.shape}")
                    
                    # Prediksi
                    features_scaled = scaler.transform(features)
                    pred = model.predict(features_scaled)
                    proba = model.predict_proba(features_scaled)
                    
                    result = "BUKA" if pred[0] == 0 else "TUTUP"
                    confidence = np.max(proba) * 100
                    
                    st.success(f"🎧 Prediksi: **{result}** (Confidence: {confidence:.1f}%)")
                    
                    # Reset frames setelah analisis
                    ctx.audio_processor.frames = []
                    
            except Exception as e:
                st.error(f"❌ Error saat analisis: {str(e)}")
                st.write("Detail error:", e)
        else:
            st.warning("⚠️ Tidak ada audio yang terekam. Pastikan mikrofon aktif dan izin akses diberikan.")

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
            try:
                y, sr = librosa.load(temp_path, sr=16000)
                
                # Normalisasi
                y = y / np.max(np.abs(y) + 1e-6)
                
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                features = np.mean(mfcc.T, axis=0).reshape(1, -1)
                features_scaled = scaler.transform(features)
                pred = model.predict(features_scaled)
                proba = model.predict_proba(features_scaled)
                
                result = "BUKA" if pred[0] == 0 else "TUTUP"
                confidence = np.max(proba) * 100
                
                st.success(f"🎧 Prediksi: **{result}** (Confidence: {confidence:.1f}%)")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")