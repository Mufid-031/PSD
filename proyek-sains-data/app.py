import streamlit as st
from audio_recorder_streamlit import audio_recorder
import numpy as np
import pickle
import librosa
import tsfel
import io
import os
import soundfile as sf
from pydub import AudioSegment

# ===============================
# 🧠 Load Model & Scaler
# ===============================
@st.cache_resource
def load_models():
    """Load semua model dan scaler dengan path dinamis"""
    base_dir = os.path.dirname(__file__)   # Folder tempat app.py berada
    model_dir = os.path.join(base_dir, "model")

    # Buat path dinamis
    model_action_path = os.path.join(model_dir, "model_action.pkl")
    model_person_path = os.path.join(model_dir, "model_person.pkl")
    scaler_action_path = os.path.join(model_dir, "scaler_action.pkl")
    scaler_person_path = os.path.join(model_dir, "scaler_person.pkl")

    # Cek apakah semua file ada
    required_files = [model_action_path, model_person_path, scaler_action_path, scaler_person_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        st.error("❌ Beberapa file model/scaler tidak ditemukan:")
        for mf in missing_files:
            st.write(f"- `{os.path.basename(mf)}` tidak ada di `{model_dir}`")
        st.stop()

    # Load model dan scaler
    with open(model_action_path, "rb") as f:
        model_action = pickle.load(f)
    with open(model_person_path, "rb") as f:
        model_person = pickle.load(f)
    with open(scaler_action_path, "rb") as f:
        scaler_action = pickle.load(f)
    with open(scaler_person_path, "rb") as f:
        scaler_person = pickle.load(f)

    st.success("✅ Model dan scaler berhasil dimuat!")
    return model_action, model_person, scaler_action, scaler_person


# ===============================
# ⚙️ Ekstraksi Fitur TSFEL
# ===============================
def extract_tsfel_features(samples, sr):
    """Ekstraksi fitur dari audio"""
    features = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain(), samples, fs=sr, verbose=0
    )
    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    return features


# ===============================
# 🔍 Analisis Audio (Versi Aman)
# ===============================
def analyze_audio(audio_bytes, source="rekaman"):
    import pandas as pd
    try:
        # Konversi bytes ke AudioSegment
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

        # Normalisasi berdasarkan bit depth
        if audio_segment.sample_width == 2:
            samples = samples / 32768.0
        elif audio_segment.sample_width == 4:
            samples = samples / 2147483648.0

        # Stereo → Mono
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)

        sr = audio_segment.frame_rate

        # Simpan dan preview audio
        sf.write("temp_audio.wav", samples, sr)
        st.audio("temp_audio.wav", format="audio/wav")
        st.info(f"📊 Audio {source}: {len(samples)/sr:.2f} detik | SR: {sr} Hz")

        # Trim silence
        samples_trimmed, _ = librosa.effects.trim(samples, top_db=20)
        if len(samples_trimmed) < 0.3 * sr:
            st.warning("⚠️ Audio terlalu pendek setelah trim. Ucapkan lebih keras atau lebih lama.")
            return

        # Ekstraksi fitur TSFEL
        with st.spinner("🔍 Mengekstraksi fitur..."):
            df_features = extract_tsfel_features(samples_trimmed, sr)

        # Pastikan df_features adalah DataFrame dan bersih
        if not isinstance(df_features, pd.DataFrame):
            df_features = pd.DataFrame(df_features)

        df_features = df_features.replace([np.inf, -np.inf], 0).fillna(0)

        # Load model & scaler
        model_action, model_person, scaler_action, scaler_person = load_models()

        # Pastikan kolom sesuai dengan fitur yang digunakan saat training
        try:
            df_features = df_features[scaler_action.feature_names_in_]
        except AttributeError:
            # Jika scaler tidak punya atribut feature_names_in_, abaikan
            pass
        except KeyError:
            # Jika kolom tidak cocok, isi kolom yang hilang dengan 0
            missing_cols = [c for c in scaler_action.feature_names_in_ if c not in df_features.columns]
            for col in missing_cols:
                df_features[col] = 0
            df_features = df_features[scaler_action.feature_names_in_]

        # Scaling
        X_action_scaled = scaler_action.transform(df_features)
        X_person_scaled = scaler_person.transform(df_features)

        # Prediksi
        pred_action = model_action.predict(X_action_scaled)[0]
        pred_person = model_person.predict(X_person_scaled)[0]

        # Confidence (jika model mendukung predict_proba)
        confidence_action, confidence_person = None, None
        try:
            confidence_action = np.max(model_action.predict_proba(X_action_scaled)) * 100
            confidence_person = np.max(model_person.predict_proba(X_person_scaled)) * 100
        except Exception:
            pass

        # Tampilkan hasil
        st.success("✅ Prediksi Berhasil!")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Aksi")
            st.markdown(f"**{pred_action.upper()}**")
            if confidence_action:
                st.progress(confidence_action / 100)
                st.caption(f"Confidence: {confidence_action:.1f}%")
        with col2:
            st.subheader("🧑 Orang")
            st.markdown(f"**{pred_person.upper()}**")
            if confidence_person:
                st.progress(confidence_person / 100)
                st.caption(f"Confidence: {confidence_person:.1f}%")

        return pred_action, pred_person

    except Exception as e:
        st.error(f"❌ Error saat analisis: {e}")
        import traceback
        with st.expander("🐛 Debug Info"):
            st.code(traceback.format_exc())

# ===============================
# 🎛️ Tampilan Utama
# ===============================
st.set_page_config(page_title="Voice Classifier", page_icon="🎙️", layout="centered")

st.title("🎙️ Voice Classifier (BUKA/TUTUP + IMAM/MUFID)")
st.write("Kamu bisa **merekam suara langsung** atau **upload file audio (.mp3 / .wav)** untuk diprediksi oleh model TSFEL.")

mode = st.radio("🎧 Pilih metode input:", ["🎙️ Rekam langsung", "📁 Upload file (.mp3 / .wav)"])

# ===============================
# MODE 1: REKAM LANGSUNG
# ===============================
if mode == "🎙️ Rekam langsung":
    st.info("Tekan tombol mikrofon di bawah untuk mulai merekam.")
    audio_bytes = audio_recorder(
        text="Klik untuk merekam",
        recording_color="#e74c3c",
        neutral_color="#2ecc71",
        icon_name="microphone",
        icon_size="3x"
    )

    if audio_bytes:
        st.success("✅ Audio berhasil direkam! Menganalisis...")
        with st.spinner("🔄 Memproses audio..."):
            analyze_audio(audio_bytes, source="rekaman")

# ===============================
# MODE 2: UPLOAD FILE
# ===============================
else:
    uploaded_file = st.file_uploader("📁 Upload file audio", type=["mp3", "wav"])
    if uploaded_file:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format=f"audio/{uploaded_file.type.split('/')[-1]}")
        if st.button("🔍 Analisis Audio", type="primary"):
            with st.spinner("🔄 Memproses audio..."):
                analyze_audio(audio_bytes, source="upload")


# ===============================
# 📘 Footer
# ===============================
st.divider()
with st.expander("💡 Tips Penggunaan"):
    st.markdown("""
    - Ucapkan **BUKA** atau **TUTUP** dengan jelas.
    - Gunakan di **tempat yang tenang**.
    - Durasi ideal: **1–3 detik**.
    - Confidence rendah? Coba ulangi dengan volume lebih tinggi.
    """)
