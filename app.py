import streamlit as st
import tf_keras as keras 
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(
    page_title="Palm Oil AI Detector",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/188/188333.png", width=100)
    st.title("Panel Kontrol")
    st.info("Aplikasi ini menggunakan Deep Learning untuk membedakan tingkat kematangan sawit.")
    st.caption("© 2025 Palm AI Project")

@st.cache_resource
def load_prediction_model():
    model = keras.models.load_model("keras_model.h5", compile=False)
    return model

st.title("🌴 Klasifikasi Kematangan Sawit")
st.write("---")

try:
    model = load_prediction_model()
    class_names = ["Unripe (Mentah)", "Ripe (Matang)", "Overripe (Lewat Matang)"]
except Exception as e:
    st.error(f"Gagal memuat model. Error: {e}")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Input Gambar")
    file_upload = st.file_uploader("Upload gambar di sini...", type=["jpg", "png", "jpeg"])
    
    if file_upload is not None:
        image = Image.open(file_upload).convert("RGB")
        st.image(image, caption="Gambar Original", use_container_width=True)

with col2:
    st.subheader("🔍 Hasil Analisis")
    
    if file_upload is None:
        st.info("⬅️ Silakan upload gambar di kolom sebelah kiri.")
    
    else:
        # Pre-processing
        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image_resized)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        if model:
            with st.spinner('Sedang memindai tekstur dan warna...'):
                prediction = model.predict(data)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence = prediction[0][index]
                
            if index == 0:
                color = "#D32F2F"
                msg = "Belum siap panen."
            elif index == 1:
                color = "#388E3C"
                msg = "Siap panen (Kualitas Optimal)."
            else:
                color = "#FBC02D"
                msg = "Terlalu matang (Overripe)."

            st.markdown(f"""
            <div style="background-color:{color};padding:20px;border-radius:10px;color:white;text-align:center;margin-bottom:20px;">
                <h2 style='margin:0;color:white;'>{class_name}</h2>
                <h4 style='margin:0;color:white;'>Akurasi: {confidence * 100:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.success(msg)

            st.write("---")
            for i, name in enumerate(class_names):
                score = prediction[0][i]
                st.write(f"{name}")
                st.progress(int(score * 100))