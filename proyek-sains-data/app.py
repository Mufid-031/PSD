import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration, WebRtcMode
import numpy as np
import joblib
import librosa
import os
import av
import soundfile as sf
from pydub import AudioSegment
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
# 🔹 Konfigurasi WebRTC
# ===============================
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ===============================
# 🔹 Session State untuk Audio Buffer
# ===============================
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = []
if "recording_start" not in st.session_state:
    st.session_state.recording_start = None
if "total_samples" not in st.session_state:
    st.session_state.total_samples = 0

# ===============================
# 🔹 Pilihan Metode Input
# ===============================
mode = st.radio("🎧 Pilih metode input suara:", ["🎙️ Rekam langsung", "📁 Upload file (.wav / .mp3)"])

# ===============================
# 1️⃣ MODE REKAM LANGSUNG
# ===============================
if mode == "🎙️ Rekam langsung":
    st.info("1️⃣ Tekan **START** → 2️⃣ Ucapkan 'BUKA' atau 'TUTUP' selama 2-3 detik → 3️⃣ Tekan **Analisis Voice**")
    
    class AudioProcessor(AudioProcessorBase):
        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            # Konversi frame ke numpy array
            sound = frame.to_ndarray()
            
            # Konversi ke mono jika stereo
            if len(sound.shape) == 2:
                sound = sound.mean(axis=1)
            
            sound = sound.flatten().astype(np.float32)
            
            # Simpan ke session state buffer (persistent across reruns)
            st.session_state.audio_buffer.append(sound)
            st.session_state.total_samples += len(sound)
            
            return frame

    # WebRTC Streamer
    ctx = webrtc_streamer(
        key="voice-cmd",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    # Update recording start time
    if ctx.state.playing and st.session_state.recording_start is None:
        st.session_state.recording_start = time.time()
    elif not ctx.state.playing:
        st.session_state.recording_start = None

    # Status Recording
    col1, col2 = st.columns(2)
    with col1:
        if ctx.state.playing:
            duration = time.time() - st.session_state.recording_start if st.session_state.recording_start else 0
            st.success(f"🔴 **MEREKAM** ({duration:.1f}s)")
        else:
            st.info("⚪ Tekan START untuk mulai")
    
    with col2:
        samples_collected = st.session_state.total_samples
        duration_seconds = samples_collected / 48000 if samples_collected > 0 else 0
        st.metric("Audio Terekam", f"{duration_seconds:.2f} detik")
    
    # Progress bar
    min_duration = 1.0  # Minimal 1 detik
    progress = min(duration_seconds / min_duration, 1.0)
    st.progress(progress)
    
    if duration_seconds < min_duration:
        st.warning(f"⏳ Rekam minimal {min_duration} detik. Sekarang: {duration_seconds:.2f} detik")
    
    # Tombol Reset
    if st.button("🔄 Reset Recording"):
        st.session_state.audio_buffer = []
        st.session_state.total_samples = 0
        st.rerun()
    
    # Tombol Analisis
    analyze_button = st.button(
        "🔍 Analisis Voice", 
        disabled=(duration_seconds < min_duration),
        type="primary"
    )
    
    if analyze_button:
        if len(st.session_state.audio_buffer) == 0:
            st.error("❌ Buffer audio kosong! Pastikan mikrofon aktif.")
        else:
            try:
                with st.spinner("Menganalisis audio..."):
                    # Gabungkan semua audio chunks
                    audio_data = np.concatenate(st.session_state.audio_buffer)
                    original_sr = 48000  # WebRTC default sample rate
                    
                    st.info(f"📊 Total audio: {len(audio_data)} samples ({len(audio_data)/original_sr:.2f} detik)")
                    
                    # Resample ke 16kHz
                    target_sr = 16000
                    audio_resampled = librosa.resample(
                        audio_data, 
                        orig_sr=original_sr, 
                        target_sr=target_sr
                    )
                    
                    # Normalisasi
                    audio_resampled = audio_resampled / (np.max(np.abs(audio_resampled)) + 1e-8)
                    
                    # Simpan untuk preview
                    sf.write("recorded_audio.wav", audio_resampled, target_sr)
                    st.audio("recorded_audio.wav", format="audio/wav")
                    
                    # Trim silence (opsional tapi membantu)
                    audio_trimmed, _ = librosa.effects.trim(audio_resampled, top_db=20)
                    
                    st.write(f"✂️ Audio setelah trim: {len(audio_trimmed)/target_sr:.2f} detik")
                    
                    # Ekstraksi MFCC
                    mfcc = librosa.feature.mfcc(y=audio_trimmed, sr=target_sr, n_mfcc=13)
                    features = np.mean(mfcc.T, axis=0).reshape(1, -1)
                    
                    st.write(f"🔍 MFCC shape: {mfcc.shape}, Features shape: {features.shape}")
                    
                    # Prediksi
                    features_scaled = scaler.transform(features)
                    pred = model.predict(features_scaled)
                    
                    # Cek apakah model punya predict_proba
                    try:
                        proba = model.predict_proba(features_scaled)
                        confidence = np.max(proba) * 100
                        
                        result = "BUKA" if pred[0] == 0 else "TUTUP"
                        st.success(f"# 🎧 Prediksi: **{result}**")
                        st.info(f"Confidence: {confidence:.1f}%")
                        
                        # Tampilkan probabilitas untuk kedua kelas
                        st.write("📊 Probabilitas:")
                        st.write(f"- BUKA: {proba[0][0]*100:.1f}%")
                        st.write(f"- TUTUP: {proba[0][1]*100:.1f}%")
                    except:
                        result = "BUKA" if pred[0] == 0 else "TUTUP"
                        st.success(f"# 🎧 Prediksi: **{result}**")
                    
                    # Reset buffer setelah analisis
                    st.session_state.audio_buffer = []
                    st.session_state.total_samples = 0
                    
            except Exception as e:
                st.error(f"❌ Error saat analisis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

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
        if st.button("🔍 Analisis Voice", type="primary"):
            try:
                y, sr = librosa.load(temp_path, sr=16000)
                
                st.info(f"📊 Audio duration: {len(y)/sr:.2f} detik")
                
                # Normalisasi
                y = y / (np.max(np.abs(y)) + 1e-8)
                
                # Trim silence
                y_trimmed, _ = librosa.effects.trim(y, top_db=20)
                
                mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
                features = np.mean(mfcc.T, axis=0).reshape(1, -1)
                features_scaled = scaler.transform(features)
                pred = model.predict(features_scaled)
                
                try:
                    proba = model.predict_proba(features_scaled)
                    confidence = np.max(proba) * 100
                    
                    result = "BUKA" if pred[0] == 0 else "TUTUP"
                    st.success(f"# 🎧 Prediksi: **{result}**")
                    st.info(f"Confidence: {confidence:.1f}%")
                    
                    st.write("📊 Probabilitas:")
                    st.write(f"- BUKA: {proba[0][0]*100:.1f}%")
                    st.write(f"- TUTUP: {proba[0][1]*100:.1f}%")
                except:
                    result = "BUKA" if pred[0] == 0 else "TUTUP"
                    st.success(f"# 🎧 Prediksi: **{result}**")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())