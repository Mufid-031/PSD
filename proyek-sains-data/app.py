import streamlit as st
import cv2
import numpy as np
import joblib
from scipy.signal import resample
import tempfile
import os

# ==============================
# Load Model & PCA
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

svm_model = joblib.load(
    os.path.join(BASE_DIR, "models", "svm_car_model.pkl")
)


label_map = {
    1: "Sedan",
    2: "Pickup",
    3: "Minivan",
    4: "SUV"
}

# ==============================
# Image → Time Series
# ==============================
def image_to_timeseries(img_path, target_length=577):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Preprocessing
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        raise ValueError("Kontur tidak ditemukan")

    cnt = max(contours, key=cv2.contourArea)
    center = cnt.mean(axis=0)

    ts = np.linalg.norm(cnt - center, axis=2).flatten()

    ts_resampled = resample(ts, target_length)

    # ✅ Z-Normalization per sample
    ts_norm = (ts_resampled - ts_resampled.mean()) / ts_resampled.std()

    return ts_norm.reshape(1, -1), cnt, img

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Car Type Classification", layout="centered")

st.title("🚗 Klasifikasi Jenis Kendaraan")
st.write("Upload gambar kendaraan untuk melakukan prediksi")

uploaded_file = st.file_uploader(
    "Upload gambar (jpg / png)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        img_path = tmp.name

    try:
        ts, contour, img = image_to_timeseries(img_path)

        prediction = svm_model.predict(ts)[0]
        class_name = label_map[prediction]


        # ==========================
        # OUTPUT
        # ==========================
        st.subheader("✅ Hasil Prediksi")
        st.success(f"Jenis Kendaraan: **{class_name}**")

        # ==========================
        # Visualisasi
        # ==========================
        st.subheader("🖼️ Visualisasi Kontur Kendaraan")
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_color, [contour], -1, (0, 255, 0), 2)
        st.image(img_color, channels="BGR", caption="Kontur Kendaraan")

        st.subheader("📈 Time Series Kontur")
        st.line_chart(ts.flatten())

    except Exception as e:
        st.error(f"Gagal memproses gambar: {e}")
