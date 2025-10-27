import streamlit as st
from audio_recorder_streamlit import audio_recorder
import numpy as np
import joblib
import librosa
import os
import soundfile as sf
from pydub import AudioSegment
import io

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
# 🔹 Session State
# ===============================
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

# ===============================
# 🔹 Pilihan Metode Input
# ===============================
mode = st.radio("🎧 Pilih metode input suara:", ["🎙️ Rekam langsung", "📁 Upload file (.wav / .mp3)"])

# ===============================
# 🔹 Fungsi Analisis Audio
# ===============================
def analyze_audio(audio_bytes, source="rekaman"):
    """Fungsi untuk analisis audio dari bytes"""
    try:
        # Konversi bytes ke audio array
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
        
        # Convert ke numpy array
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        
        # Normalize berdasarkan bit depth
        if audio_segment.sample_width == 2:  # 16-bit
            samples = samples / 32768.0
        elif audio_segment.sample_width == 4:  # 32-bit
            samples = samples / 2147483648.0
        
        # Convert stereo ke mono jika perlu
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        
        sr = audio_segment.frame_rate
        
        st.info(f"📊 Audio {source}: {len(samples)/sr:.2f} detik, SR: {sr} Hz")
        
        # Resample ke 16kHz jika perlu
        if sr != 16000:
            samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Normalisasi
        samples = samples / (np.max(np.abs(samples)) + 1e-8)
        
        # Simpan untuk preview
        sf.write("temp_audio.wav", samples, sr)
        st.audio("temp_audio.wav", format="audio/wav")
        
        # Trim silence
        samples_trimmed, _ = librosa.effects.trim(samples, top_db=20)
        st.write(f"✂️ Setelah trim silence: {len(samples_trimmed)/sr:.2f} detik")
        
        # Cek durasi minimal
        if len(samples_trimmed) < 0.3 * sr:
            st.warning("⚠️ Audio terlalu pendek setelah trim. Mungkin hanya noise. Coba ucapkan lebih keras dan lebih lama.")
            return None
        
        # Ekstraksi MFCC
        mfcc = librosa.feature.mfcc(y=samples_trimmed, sr=sr, n_mfcc=13)
        features = np.mean(mfcc.T, axis=0).reshape(1, -1)
        
        with st.expander("🔍 Detail Teknis"):
            st.write(f"MFCC shape: {mfcc.shape}")
            st.write(f"Features shape: {features.shape}")
            st.write(f"Sample rate: {sr} Hz")
            st.write(f"Duration: {len(samples_trimmed)/sr:.2f} seconds")
        
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
            
            # Progress bar untuk confidence
            st.progress(confidence / 100)
            
            with st.expander("📊 Detail Probabilitas"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("BUKA", f"{proba[0][0]*100:.1f}%")
                with col2:
                    st.metric("TUTUP", f"{proba[0][1]*100:.1f}%")
        except:
            pass
        
        return result
        
    except Exception as e:
        st.error(f"❌ Error saat analisis: {str(e)}")
        import traceback
        with st.expander("🐛 Debug Info"):
            st.code(traceback.format_exc())
        return None

# ===============================
# 1️⃣ MODE REKAM LANGSUNG
# ===============================
if mode == "🎙️ Rekam langsung":
    st.info("🎙️ Tekan tombol mikrofon di bawah, ucapkan **'BUKA'** atau **'TUTUP'** dengan jelas, lalu tekan stop.")
    
    # Audio recorder widget
    audio_bytes = audio_recorder(
        text="Klik untuk merekam",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="3x",
        pause_threshold=2.0,
        sample_rate=16000
    )
    
    # Jika ada audio baru yang direkam
    if audio_bytes:
        # Cek apakah ini audio baru (berbeda dari sebelumnya)
        if audio_bytes != st.session_state.last_audio:
            st.session_state.last_audio = audio_bytes
            
            st.success("✅ Audio berhasil direkam! Menganalisis...")
            
            # Analisis otomatis
            with st.spinner("🔄 Memproses audio..."):
                analyze_audio(audio_bytes, source="rekaman")
        else:
            # Audio sama dengan sebelumnya, tampilkan tombol analisis ulang
            st.info("ℹ️ Audio sudah dianalisis. Rekam ulang untuk prediksi baru.")
            
            if st.button("🔄 Analisis Ulang", type="secondary"):
                with st.spinner("🔄 Memproses audio..."):
                    analyze_audio(audio_bytes, source="rekaman")
    else:
        st.info("👆 Klik tombol mikrofon di atas untuk mulai merekam")

# ===============================
# 2️⃣ MODE UPLOAD FILE
# ===============================
else:
    st.info("📁 Upload file audio Anda di bawah ini")
    
    uploaded_file = st.file_uploader(
        "Upload file suara (.wav / .mp3)", 
        type=["wav", "mp3"],
        help="Pilih file audio yang berisi ucapan 'BUKA' atau 'TUTUP'"
    )

    if uploaded_file is not None:
        # Baca file
        audio_bytes = uploaded_file.read()
        
        # Preview audio
        st.audio(audio_bytes, format=f"audio/{uploaded_file.type.split('/')[-1]}")
        
        # Tombol analisis
        if st.button("🔍 Analisis Audio", type="primary"):
            with st.spinner("🔄 Memproses audio..."):
                # Konversi ke WAV jika MP3
                if uploaded_file.type == "audio/mpeg":
                    audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                    wav_io = io.BytesIO()
                    audio_segment.export(wav_io, format="wav")
                    audio_bytes = wav_io.getvalue()
                
                analyze_audio(audio_bytes, source="upload")

# ===============================
# 🔹 Footer & Tips
# ===============================
st.divider()
with st.expander("💡 Tips untuk Hasil Terbaik"):
    st.write("""
    **Untuk hasil prediksi yang akurat:**
    1. 🔊 Ucapkan kata dengan **jelas dan lantang**
    2. ⏱️ Durasi minimal **1 detik**, ideal **2-3 detik**
    3. 🤫 Rekam di **tempat yang tenang** (minimal noise)
    4. 🎤 Jarak mikrofon **10-30 cm** dari mulut
    5. 🗣️ Intonasi **natural**, tidak terlalu cepat/lambat
    6. 🔁 Jika confidence rendah, coba **rekam ulang**
    """)

with st.expander("❓ Troubleshooting"):
    st.write("""
    **Jika mengalami masalah:**
    - **Audio tidak terekam**: Pastikan browser mengizinkan akses mikrofon
    - **Prediksi salah**: Coba ucapkan lebih jelas atau rekam di tempat lebih tenang
    - **Confidence rendah**: Audio mungkin terlalu pendek atau terlalu banyak noise
    - **Error saat analisis**: Coba refresh halaman (F5) atau gunakan mode Upload File
    """)