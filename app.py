import streamlit as st
import tf_keras as keras 
from PIL import Image, ImageOps
import numpy as np

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Palm Oil Smart Detector",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS DARK MODE (TEMA FINAL) ---
st.markdown("""
    <style>
    /* Background & Text Color */
    [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    [data-testid="stSidebar"] {
        background-color: #262730;
        border-right: 1px solid #464B5C;
    }
    
    /* Font Global */
    html, body, [class*="css"], h1, h2, h3, p {
        font-family: 'Segoe UI', sans-serif;
        color: #FAFAFA !important;
    }
    
    /* File Uploader */
    .stFileUploader {
        background-color: #262730;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #464B5C;
    }
    
    /* Result Card Styling */
    .result-card {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        border: 1px solid rgba(255,255,255,0.1);
        animation: fadeIn 0.5s;
    }
    
    /* Progress Bar Color */
    .stProgress > div > div > div > div {
        background-color: #00C853;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_model():
    # Load model tanpa compile agar lebih cepat dan kompatibel
    model = keras.models.load_model("keras_model.h5", compile=False)
    return model

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1598/1598196.png", width=80)
    st.title("Palm AI")
    st.caption("v1")
    st.markdown("---")
    st.info("💡 **Panduan:** Pastikan foto diambil dengan pencahayaan cukup agar akurasi maksimal.")
    st.markdown("---")
    st.write("© 2025")

# --- 5. UI UTAMA ---
col_title, col_logo = st.columns([4, 1])
with col_title:
    st.title("Deteksi Kualitas Sawit")
    st.markdown("##### 🔍 Sistem Klasifikasi Kematangan Buah Berbasis AI")

st.write("---")

col_left, col_right = st.columns([1, 1.2], gap="large")

# --- KOLOM KIRI (INPUT) ---
with col_left:
    st.subheader("📷 Unggah Foto")
    file_upload = st.file_uploader("Format: JPG, PNG", type=["jpg", "png", "jpeg"])
    
    if file_upload:
        image = Image.open(file_upload).convert("RGB")
        st.markdown('<style>img {border-radius: 12px; border: 2px solid #555;}</style>', unsafe_allow_html=True)
        st.image(image, caption="Gambar Input", use_container_width=True)

# --- KOLOM KANAN (OUTPUT) ---
with col_right:
    st.subheader("📊 Hasil Analisa")
    
    if file_upload is None:
        # Tampilan Placeholder (Menunggu Gambar)
        st.markdown("""
        <div style="background-color:#262730;padding:40px;border-radius:15px;text-align:center;border: 2px dashed #464B5C;">
            <h1 style="font-size:50px;">🖼️</h1>
            <h3 style="color:#FAFAFA;">Menunggu Gambar...</h3>
            <p style="color:#90A4AE;">Silakan upload foto buah sawit di panel sebelah kiri.</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        try:
            model = load_model()
            
            # --- PRE-PROCESS IMAGE ---
            size = (224, 224)
            image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image_resized)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # --- PREDICTION ---
            prediction = model.predict(data)
            index = np.argmax(prediction)
            confidence = prediction[0][index]
            
            # Label Kelas (Sesuai Teachable Machine 4 Kelas)
            labels = ["Mentah (Unripe)", "Matang (Ripe)", "Lewat Matang (Overripe)", "⛔ Bukan Sawit"]

            # --- LOGIKA PENENTUAN STATUS ---
            
            # 1. Cek jika "Bukan Sawit" (Index 3)
            if index == 3:
                bg_color = "#37474F" # Abu-abu Gelap
                text_color = "#ECEFF1"
                status = "TIDAK DIKENALI"
                desc = "Objek tidak teridentifikasi sebagai buah sawit."
                icon = "🚫"
            
            # 2. Cek jika Confidence Rendah (Threshold 65%)
            elif confidence < 0.65:
                bg_color = "#FF6F00" # Oranye Gelap
                text_color = "#FFFFFF"
                status = "MERAGUKAN"
                desc = "Mirip sawit, namun AI kurang yakin. Coba foto ulang."
                icon = "⚠️"
            
            # 3. Klasifikasi Sawit Valid
            else:
                if index == 0:
                    bg_color = "#D50000" # Merah Menyala
                    text_color = "#FFFFFF"
                    status = "MENTAH (UNRIPE)"
                    desc = "Buah berwarna hitam/ungu. Belum siap panen."
                    icon = "❌"
                elif index == 1:
                    bg_color = "#00C853" # Hijau Neon
                    text_color = "#FFFFFF"
                    status = "MATANG (RIPE)"
                    desc = "Warna oranye kemerahan. Kualitas optimal."
                    icon = "✅"
                elif index == 2:
                    bg_color = "#FFD600" # Kuning Emas
                    text_color = "#212121" # Teks Hitam
                    status = "LEWAT MATANG"
                    desc = "Buah mulai membrondol. Segera angkut."
                    icon = "⚠️"

            # --- TAMPILAN KARTU UTAMA ---
            st.markdown(f"""
            <div class="result-card" style="background-color:{bg_color};">
                <div style="font-size: 50px; margin-bottom: 10px;">{icon}</div>
                <h1 style="color:{text_color}; margin:0; font-weight:800; font-size:28px;">{status}</h1>
                <p style="color:{text_color}; font-size:16px; margin-top:5px; opacity:0.9;">Tingkat Keyakinan: {confidence*100:.1f}%</p>
                <hr style="border-color:rgba(255,255,255,0.2); margin: 15px 0;">
                <p style="color:{text_color}; font-style:italic; font-size:14px;">"{desc}"</p>
            </div>
            """, unsafe_allow_html=True)
            
            # --- STATISTIK ---
            with st.expander("📊 Lihat Detail Probabilitas"):
                st.markdown("Rincian prediksi untuk setiap kategori:")
                for i, label in enumerate(labels):
                    # Jangan tampilkan bar 'Bukan Sawit' agar user fokus ke kualitas sawit saja
                    # kecuali jika memang terdeteksi bukan sawit
                    if i < 3 or index == 3: 
                        score = prediction[0][i]
                        st.write(f"**{label}**")
                        st.progress(int(score * 100))
                        st.caption(f"{score*100:.1f}%")

        except Exception as e:
            st.error("Terjadi kesalahan pada sistem AI.")
            st.code(f"Error Log: {e}")