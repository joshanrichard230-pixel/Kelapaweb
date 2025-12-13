import streamlit as st
import tf_keras as keras 
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(
    page_title="Palm AI",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Global Styles */
    [data-testid="stAppViewContainer"] { background-color: #0E1117; color: #FAFAFA; }
    [data-testid="stSidebar"] { background-color: #262730; border-right: 1px solid #464B5C; }
    
    /* Font */
    html, body, h1, h2, h3, p, li { font-family: 'Segoe UI', sans-serif; color: #FAFAFA !important; }
    
    /* Tombol Navigasi Sidebar */
    .stRadio > div { background-color: #262730; padding: 10px; border-radius: 10px; }
    
    /* Card Styling */
    .info-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #464B5C;
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .info-card:hover { transform: scale(1.02); border-color: #00C853; }
    
    /* Result Card (Scanner) */
    .result-card {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    
    /* Custom Button */
    .stButton>button {
        background-color: #00C853; color: white; border-radius: 10px; border: none; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = keras.models.load_model("keras_model.h5", compile=False)
    return model

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1598/1598196.png", width=80)
    st.title("Palm AI")
    
    selected_menu = st.radio(
        "Navigasi Menu:",
        ["🏠 Beranda & Panduan", "📷 Mulai Deteksi"],
        index=0
    )
    
    st.markdown("---")
    st.info("💡 **Status Sistem:** Online")
    st.caption("© 2025 Palm Tech Solutions")

if selected_menu == "🏠 Beranda & Panduan":
    
    st.title("Selamat Datang di Palm AI")
    st.markdown("#### Sistem Cerdas Klasifikasi Kematangan Kelapa Sawit")
    st.write("---")
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 📖 Panduan Penggunaan")
        st.markdown("""
        Ikuti langkah mudah berikut untuk menggunakan aplikasi:
        
        1.  Buka menu **"📷 Mulai Deteksi"** di sebelah kiri.
        2.  Klik tombol **Browse Files** untuk mengunggah foto buah sawit.
        3.  Tunggu beberapa detik, AI akan menganalisis gambar.
        4.  Hasil klasifikasi (Mentah/Matang/Lewat) akan muncul otomatis.
        """)
        
        st.info("💡 **Tips Akurasi:** Pastikan foto diambil dari jarak dekat, pencahayaan cukup, dan hanya menampilkan satu tandan buah.")

    with col2:
        st.markdown("""
        <div class="info-card" style="text-align:center;">
            <h3>🎯 Teknologi AI</h3>
            <p>Menggunakan Deep Learning (CNN) untuk mengenali pola warna dan tekstur kulit buah sawit.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    st.subheader("📚 Kriteria Kematangan")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="info-card" style="border-top: 5px solid #D50000;">
            <h3 style="color:#FF5252;">Mentah (Unripe)</h3>
            <p>Warna dominan hitam atau ungu gelap. Daging buah masih keras. Kadar minyak sangat rendah.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="info-card" style="border-top: 5px solid #00C853;">
            <h3 style="color:#69F0AE;">Matang (Ripe)</h3>
            <p>Warna oranye kemerahan cerah. Buah mulai lepas dari tandan (brondol). Kadar minyak optimal.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
        <div class="info-card" style="border-top: 5px solid #FFD600;">
            <h3 style="color:#FFE57F;">Lewat (Overripe)</h3>
            <p>Warna merah tua pudar atau coklat. Banyak buah lepas. Kadar Asam Lemak Bebas (ALB) tinggi.</p>
        </div>
        """, unsafe_allow_html=True)
        
elif selected_menu == "📷 Mulai Deteksi":
    col_title, col_stat = st.columns([3,1])
    with col_title:
        st.title("Scanner Kualitas Sawit")
    with col_stat:
        st.write("")

    st.write("---")
    
    col_left, col_right = st.columns([1, 1.2], gap="large")
    
    with col_left:
        st.subheader("📷 Unggah Foto")
        file_upload = st.file_uploader("Format: JPG, PNG", type=["jpg", "png", "jpeg"])
        
        if file_upload:
            image = Image.open(file_upload).convert("RGB")
            st.markdown('<style>img {border-radius: 12px; border: 2px solid #555;}</style>', unsafe_allow_html=True)
            st.image(image, caption="Gambar Input", use_container_width=True)

    with col_right:
        st.subheader("📊 Hasil Analisa")
        
        if file_upload is None:
            st.markdown("""
            <div style="background-color:#262730;padding:40px;border-radius:15px;text-align:center;border: 2px dashed #464B5C;">
                <h1 style="font-size:50px;">🖼️</h1>
                <h3 style="color:#FAFAFA;">Area Hasil</h3>
                <p style="color:#90A4AE;">Hasil prediksi akan muncul di sini setelah Anda mengupload gambar.</p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            try:
                model = load_model()
                
                size = (224, 224)
                image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                image_array = np.asarray(image_resized)
                normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                data[0] = normalized_image_array
                
                prediction = model.predict(data)
                index = np.argmax(prediction)
                confidence = prediction[0][index]
                
                labels = ["Mentah (Unripe)", "Matang (Ripe)", "Lewat Matang (Overripe)", "⛔ Bukan Sawit"]

                if index == 3:
                    bg_color = "#37474F"
                    status = "TIDAK DIKENALI"
                    desc = "Objek tidak teridentifikasi sebagai buah sawit."
                    icon = "🚫"
                elif confidence < 0.65:
                    bg_color = "#E65100"
                    status = "MERAGUKAN"
                    desc = "Mirip sawit, namun AI kurang yakin. Coba foto ulang."
                    icon = "⚠️"
                else:
                    if index == 0:
                        bg_color = "#D50000"
                        status = "MENTAH (UNRIPE)"
                        desc = "Belum siap panen."
                        icon = "❌"
                    elif index == 1:
                        bg_color = "#00C853"
                        status = "MATANG (RIPE)"
                        desc = "Siap panen (Kualitas Optimal)."
                        icon = "✅"
                    elif index == 2:
                        bg_color = "#FFD600"
                        status = "LEWAT MATANG"
                        desc = "Segera angkut ke pabrik."
                        icon = "⚠️"

                text_col = "#212121" if index == 2 and confidence >= 0.65 else "#FFFFFF"

                st.markdown(f"""
                <div class="result-card" style="background-color:{bg_color};">
                    <div style="font-size: 50px; margin-bottom: 10px;">{icon}</div>
                    <h1 style="color:{text_col}; margin:0; font-weight:800; font-size:28px;">{status}</h1>
                    <p style="color:{text_col}; font-size:16px; margin-top:5px; opacity:0.9;">Akurasi: {confidence*100:.1f}%</p>
                    <hr style="border-color:rgba(255,255,255,0.2); margin: 15px 0;">
                    <p style="color:{text_col}; font-style:italic; font-size:14px;">"{desc}"</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📊 Lihat Detail Probabilitas"):
                    for i, label in enumerate(labels):
                        if i < 3 or index == 3:
                            score = prediction[0][i]
                            st.write(f"**{label}**")
                            st.progress(int(score * 100))
                            st.caption(f"{score*100:.1f}%")

            except Exception as e:
                st.error("Terjadi kesalahan teknis.")
                st.code(e)
