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
import logging

# Setup logging untuk debug
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
# 🔹 Session State untuk Auto-Analyze
# ===============================
if "was_playing" not in st.session_state:
    st.session_state.was_playing = False
if "should_analyze" not in st.session_state:
    st.session_state.should_analyze = False

# ===============================
# 🔹 Pilihan Metode Input
# ===============================
mode = st.radio("🎧 Pilih metode input suara:", ["🎙️ Rekam langsung", "📁 Upload file (.wav / .mp3)"])

# ===============================
# 1️⃣ MODE REKAM LANGSUNG
# ===============================
if mode == "🎙️ Rekam langsung":
    st.info("1️⃣ Tekan **START** → 2️⃣ Ucapkan 'BUKA' atau 'TUTUP' 2-3 detik → 3️⃣ Tekan **STOP** (otomatis analisis)")
    
    # Audio Processor dengan debugging
    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.frames = []
            self.lock = threading.Lock()
            self.sample_rate = 48000
            self.frame_count = 0
            logger.info("AudioProcessor initialized")
            
        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            try:
                self.frame_count += 1
                
                # Log setiap 50 frames
                if self.frame_count % 50 == 0:
                    logger.info(f"Received frame #{self.frame_count}")
                
                # Konversi frame ke numpy array
                sound = frame.to_ndarray()
                
                # Debug info pertama kali
                if self.frame_count == 1:
                    logger.info(f"First frame - shape: {sound.shape}, dtype: {sound.dtype}, format: {frame.format.name}, layout: {frame.layout.name}")
                
                # PENTING: PyAV mengembalikan array dengan shape (channels, samples)
                # Kita perlu transpose jika channels ada di axis 0
                if len(sound.shape) == 2:
                    # Jika shape (2, 480) -> transpose ke (480, 2) lalu rata-rata
                    if sound.shape[0] < sound.shape[1]:
                        sound = sound.T  # Transpose
                    # Konversi ke mono dengan rata-rata channels
                    sound = sound.mean(axis=1)
                elif len(sound.shape) == 1:
                    # Sudah mono
                    pass
                else:
                    logger.warning(f"Unexpected shape: {sound.shape}")
                
                # Pastikan 1D array
                sound = sound.flatten().astype(np.float32)
                
                # Thread-safe append
                with self.lock:
                    self.frames.append(sound)
                
                return frame
            except Exception as e:
                logger.error(f"Error in recv: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return frame
        
        def get_frames(self):
            with self.lock:
                return self.frames.copy()
        
        def get_total_samples(self):
            with self.lock:
                return sum(len(f) for f in self.frames)
        
        def get_frame_count(self):
            return self.frame_count
        
        def clear_frames(self):
            with self.lock:
                self.frames = []
                self.frame_count = 0

    # WebRTC Streamer dengan konfigurasi lebih eksplisit
    ctx = webrtc_streamer(
        key="voice-cmd-v2",  # Ganti key untuk reset
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "audio": {
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True,
            },
            "video": False
        },
        async_processing=True,
    )

    # Debug info
    with st.expander("🐛 Debug Info", expanded=True):
        st.write(f"**WebRTC State:** {ctx.state}")
        st.write(f"**Playing:** {ctx.state.playing}")
        st.write(f"**Audio Processor exists:** {ctx.audio_processor is not None}")
        
        if ctx.audio_processor:
            st.write(f"**Frame count (internal):** {ctx.audio_processor.get_frame_count()}")
            st.write(f"**Frames list length:** {len(ctx.audio_processor.get_frames())}")
            st.write(f"**Total samples:** {ctx.audio_processor.get_total_samples()}")

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
    
    # Auto-refresh untuk update real-time
    if ctx.state.playing:
        import time
        time.sleep(0.1)
        st.session_state.was_playing = True
        st.rerun()
    
    # Deteksi transisi dari playing ke stopped -> auto analyze
if st.session_state.was_playing and not ctx.state.playing:
    st.session_state.was_playing = False
    if ctx.audio_processor and ctx.audio_processor.get_total_samples() > 0:
        duration = ctx.audio_processor.get_total_samples() / 48000
        if duration >= 1.0:  # Minimal duration check
            st.session_state.should_analyze = True
            st.info("⏳ Memulai analisis audio setelah STOP...")
            st.rerun()
        else:
            st.warning(f"⚠️ Rekaman terlalu pendek ({duration:.2f}s). Minimal 1 detik. Silakan rekam ulang.")

# Trigger analisis (otomatis atau manual)
analyze_clicked = st.session_state.should_analyze or manual_analyze

if analyze_clicked and not st.session_state.get("analysis_done", False):
    st.session_state.analysis_done = True
    st.session_state.should_analyze = False  # Reset flag
    if ctx.audio_processor and ctx.audio_processor.get_total_samples() > 0:
        frames = ctx.audio_processor.get_frames()
        
        if len(frames) == 0:
            st.error("❌ Tidak ada audio! Troubleshooting:")
            st.write("✓ Apakah tombol START sudah ditekan?")
            st.write("✓ Apakah ada tulisan 'MEREKAM' berwarna hijau?")
            st.write("✓ Apakah browser meminta izin mikrofon?")
            st.write("✓ Apakah Frame count > 0 saat merekam?")
        else:
            try:
                with st.spinner("🔄 Memproses audio..."):
                    # Gabungkan frames
                    audio_data = np.concatenate(frames)
                    original_sr = 48000
                    
                    duration = len(audio_data) / original_sr
                    st.info(f"📊 Audio terekam: {duration:.2f} detik ({len(audio_data)} samples, {len(frames)} frames)")
                    
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
                        try:
                            pred = model.predict(features_scaled)
                            result = "BUKA" if pred[0] == 0 else "TUTUP"
                            st.success(f"# 🎧 Prediksi: **{result}**")
                            
                            # Confidence
                            try:
                                proba = model.predict_proba(features_scaled)
                                confidence = np.max(proba) * 100
                                st.info(f"**Confidence:** {confidence:.1f}%")
                                
                                with st.expander("📊 Detail Probabilitas"):
                                    st.write(f"- BUKA: {proba[0][0]*100:.1f}%")
                                    st.write(f"- TUTUP: {proba[0][1]*100:.1f}%")
                            except:
                                pass
                        except Exception as e:
                            st.error(f"❌ Gagal memprediksi: {str(e)}")
                            logger.error(f"Prediction error: {e}")
                        
                        # Clear buffer setelah analisis
                        ctx.audio_processor.clear_frames()
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("🐛 Debug Info"):
                    st.code(traceback.format_exc())
    
    st.session_state.analysis_done = False  # Reset flag after analysis

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