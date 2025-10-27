import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration, WebRtcMode
import numpy as np
import joblib
import librosa
import os
import av
import soundfile as sf
from pydub import AudioSegment
import threading

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
    st.info("1️⃣ Tekan **START** → 2️⃣ Ucapkan 'BUKA' atau 'TUTUP' 2-3 detik → 3️⃣ Tekan **STOP** → 4️⃣ Klik **Analisis Voice**")
    
    # Audio Processor dengan buffer internal dan lock
    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.frames = []
            self.lock = threading.Lock()
            self.sample_rate = 48000
            
        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            sound = frame.to_ndarray()
            
            # Konversi ke mono
            if len(sound.shape) == 2:
                sound = sound.mean(axis=1)
            
            sound = sound.flatten().astype(np.float32)
            
            # Thread-safe append
            with self.lock:
                self.frames.append(sound)
            
            return frame
        
        def get_frames(self):
            """Ambil semua frames dengan thread-safe"""
            with self.lock:
                return self.frames.copy()
        
        def get_total_samples(self):
            """Hitung total samples"""
            with self.lock:
                return sum(len(f) for f in self.frames)
        
        def clear_frames(self):
            """Clear frames"""
            with self.lock:
                self.frames = []

    # WebRTC Streamer
    ctx = webrtc_streamer(
        key="voice-cmd",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    # Status dan monitoring
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if ctx.state.playing:
            st.success("🔴 **MEREKAM**")
        else:
            st.info("⚪ Tidak merekam")
    
    with col2:
        if ctx.audio_processor:
            total_samples = ctx.audio_processor.get_total_samples()
            duration = total_samples / 48000
            st.metric("Durasi", f"{duration:.2f}s")
        else:
            st.metric("Durasi", "0.00s")
    
    with col3:
        if ctx.audio_processor:
            frame_count = len(ctx.audio_processor.get_frames())
            st.metric("Frames", frame_count)
        else:
            st.metric("Frames", 0)
    
    # Progress bar
    if ctx.audio_processor:
        duration = ctx.audio_processor.get_total_samples() / 48000
        min_duration = 1.0
        progress = min(duration / min_duration, 1.0)
        st.progress(progress)
        
        if duration < min_duration and ctx.state.playing:
            st.warning(f"⏳ Rekam minimal {min_duration:.0f} detik. Sekarang: {duration:.2f} detik")
        elif duration >= min_duration and ctx.state.playing:
            st.success(f"✅ Audio cukup! Tekan STOP lalu klik Analisis Voice")
    
    # Tombol kontrolu
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🔄 Clear Buffer"):
            if ctx.audio_processor:
                ctx.audio_processor.clear_frames()
                st.rerun()
    
    with col_btn2:
        can_analyze = (ctx.audio_processor and 
                      ctx.audio_processor.get_total_samples() > 0 and
                      not ctx.state.playing)
        
        analyze_clicked = st.button(
            "🔍 Analisis Voice", 
            disabled=not can_analyze,
            type="primary"
        )
    
    # Proses analisis
    if analyze_clicked and ctx.audio_processor:
        frames = ctx.audio_processor.get_frames()
        
        if len(frames) == 0:
            st.error("❌ Tidak ada audio! Pastikan:")
            st.write("- Mikrofon sudah diizinkan")
            st.write("- Tombol START sudah ditekan")
            st.write("- Ada suara yang masuk")
        else:
            try:
                with st.spinner("🔄 Memproses audio..."):
                    # Gabungkan frames
                    audio_data = np.concatenate(frames)
                    original_sr = 48000
                    
                    duration = len(audio_data) / original_sr
                    st.info(f"📊 Audio terekam: {duration:.2f} detik ({len(audio_data)} samples)")
                    
                    # Resample ke 16kHz
                    target_sr = 16000
                    audio_resampled = librosa.resample(
                        audio_data, 
                        orig_sr=original_sr, 
                        target_sr=target_sr
                    )
                    
                    # Normalisasi
                    audio_resampled = audio_resampled / (np.max(np.abs(audio_resampled)) + 1e-8)
                    
                    # Simpan dan tampilkan
                    sf.write("recorded_audio.wav", audio_resampled, target_sr)
                    st.audio("recorded_audio.wav", format="audio/wav")
                    
                    # Trim silence
                    audio_trimmed, _ = librosa.effects.trim(audio_resampled, top_db=20)
                    st.write(f"✂️ Setelah trim: {len(audio_trimmed)/target_sr:.2f} detik")
                    
                    if len(audio_trimmed) < 0.3 * target_sr:
                        st.warning("⚠️ Audio terlalu pendek setelah trim. Mungkin hanya noise. Coba ucapkan lebih keras.")
                    else:
                        # Ekstraksi MFCC
                        mfcc = librosa.feature.mfcc(y=audio_trimmed, sr=target_sr, n_mfcc=13)
                        features = np.mean(mfcc.T, axis=0).reshape(1, -1)
                        
                        st.write(f"🔍 MFCC shape: {mfcc.shape}")
                        
                        # Prediksi
                        features_scaled = scaler.transform(features)
                        pred = model.predict(features_scaled)
                        
                        result = "BUKA" if pred[0] == 0 else "TUTUP"
                        
                        # Tampilkan hasil
                        st.success(f"# 🎧 Prediksi: **{result}**")
                        
                        # Coba ambil confidence
                        try:
                            proba = model.predict_proba(features_scaled)
                            confidence = np.max(proba) * 100
                            st.info(f"**Confidence:** {confidence:.1f}%")
                            
                            with st.expander("📊 Detail Probabilitas"):
                                st.write(f"- BUKA: {proba[0][0]*100:.1f}%")
                                st.write(f"- TUTUP: {proba[0][1]*100:.1f}%")
                        except:
                            pass
                        
                        # Clear buffer setelah analisis
                        ctx.audio_processor.clear_frames()
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("🐛 Debug Info"):
                    st.code(traceback.format_exc())

# ===============================
# 2️⃣ MODE UPLOAD FILE
# ===============================
else:
    uploaded_file = st.file_uploader("📁 Upload file suara (.wav / .mp3)", type=["wav", "mp3"])

    if uploaded_file is not None:
        temp_path = "uploaded_audio.wav"

        # Konversi MP3 → WAV jika perlu
        if uploaded_file.type == "audio/mpeg":
            audio = AudioSegment.from_mp3(uploaded_file)
            audio.export(temp_path, format="wav")
        else:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

        st.audio(temp_path, format="audio/wav")

        if st.button("🔍 Analisis Voice", type="primary"):
            try:
                y, sr = librosa.load(temp_path, sr=16000)
                
                st.info(f"📊 Durasi: {len(y)/sr:.2f} detik")
                
                # Normalisasi
                y = y / (np.max(np.abs(y)) + 1e-8)
                
                # Trim
                y_trimmed, _ = librosa.effects.trim(y, top_db=20)
                
                # MFCC
                mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
                features = np.mean(mfcc.T, axis=0).reshape(1, -1)
                features_scaled = scaler.transform(features)
                
                # Prediksi
                pred = model.predict(features_scaled)
                result = "BUKA" if pred[0] == 0 else "TUTUP"
                
                st.success(f"# 🎧 Prediksi: **{result}**")
                
                try:
                    proba = model.predict_proba(features_scaled)
                    confidence = np.max(proba) * 100
                    st.info(f"**Confidence:** {confidence:.1f}%")
                    
                    with st.expander("📊 Detail"):
                        st.write(f"- BUKA: {proba[0][0]*100:.1f}%")
                        st.write(f"- TUTUP: {proba[0][1]*100:.1f}%")
                except:
                    pass
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")