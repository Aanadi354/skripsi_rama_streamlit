import os
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Klasifikasi Alzheimer (Axial MRI)",
    page_icon="🧠",
    layout="wide"
)

# Styling CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .result-card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .badge-ad { background-color: #FEE2E2; color: #DC2626; border: 1px solid #F87171; }
    .badge-cn { background-color: #DCFCE7; color: #16A34A; border: 1px solid #4ADE80; }
    .badge-emci { background-color: #FEF3C7; color: #D97706; border: 1px solid #FBBF24; }
    .badge-lmci { background-color: #FFEDD5; color: #EA580C; border: 1px solid #FB923C; }
</style>
""", unsafe_allow_html=True)

KELAS = ["AD", "CN", "EMCI", "LMCI"]
KELAS_DESC = {
    "AD": "Alzheimer's Disease (Kondisi lanjut kerusakan kognitif / demensia)",
    "CN": "Cognitive Normal (Kondisi otak normal, kontrol sehat)",
    "EMCI": "Early Mild Cognitive Impairment (Penurunan kognitif tahap awal)",
    "LMCI": "Late Mild Cognitive Impairment (Penurunan kognitif tahap lanjut menuju AD)"
}
KELAS_WARNA = {
    "AD": "#EF4444",
    "CN": "#10B981",
    "EMCI": "#F59E0B",
    "LMCI": "#F97316"
}

# ==============================================================================
# 2. DEFINISI MODEL (Sesuai Pelatihan)
# ==============================================================================
class DinamisSinglePlaneViT(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(DinamisSinglePlaneViT, self).__init__()
        self.vit = models.vit_b_16(weights=None)
        self.vit.heads = nn.Identity() 
        self.klasifikasi_akhir = nn.Sequential(
            nn.Linear(768, 512), 
            nn.ReLU(),
            nn.Dropout(p=dropout_rate), 
            nn.Linear(512, num_classes) 
        )

    def forward(self, x):
        return self.klasifikasi_akhir(self.vit(x))

class MinMaxNormalize(object):
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 0:
            return (tensor - min_val) / (max_val - min_val)
        return tensor - min_val

# ==============================================================================
# 3. PEMUATAN MODEL AXIAL DENGAN CACHE
# ==============================================================================
@st.cache_resource
def muat_model_axial():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cek lokasi file bobot (di folder saat ini atau di subfolder models/)
    path_kemungkinan = [
        "Axial_UNFREEZE_KFold_1.pth",
        os.path.join("models", "Axial_UNFREEZE_KFold_1.pth")
    ]
    
    model_path = None
    for p in path_kemungkinan:
        if os.path.isfile(p):
            model_path = p
            break
            
    if model_path is None:
        return None, device
        
    model = DinamisSinglePlaneViT(num_classes=4, dropout_rate=0.5)
    state_dict = torch.load(model_path, map_location=device)
    
    # Bersihkan prefix 'module.' jika model dilatih menggunakan DataParallel / multi-GPU
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v
            
    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()
    return model, device

# ==============================================================================
# 4. PREPROCESSING CITRA
# ==============================================================================
def proses_gambar(file_upload):
    if file_upload.name.endswith(".npy"):
        arr = np.load(file_upload)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=0)
        elif arr.ndim == 3 and arr.shape[0] != 3:
            arr = np.transpose(arr, (2, 0, 1))
        img_tensor = torch.tensor(arr, dtype=torch.float32)
    else:
        pil_img = Image.open(file_upload).convert("RGB")
        pil_img = pil_img.resize((224, 224))
        arr = np.array(pil_img, dtype=np.float32).transpose((2, 0, 1))
        img_tensor = torch.tensor(arr, dtype=torch.float32)

    norm = MinMaxNormalize()
    img_tensor = norm(img_tensor)
    return img_tensor.unsqueeze(0)  # Menambah dimensi batch [1, 3, 224, 224]

# ==============================================================================
# 5. TAMPILAN APLIKASI
# ==============================================================================
st.markdown('<div class="main-header">🧠 Deteksi Alzheimer MRI (Axial View)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Model: <b>Vision Transformer (ViT-B/16 Full Fine-Tuning - Fold 1)</b></div>', unsafe_allow_html=True)

# Muat Model
model, device = muat_model_axial()

if model is None:
    st.error("⚠️ File `Axial_UNFREEZE_KFold_1.pth` belum ditemukan!")
    st.info("💡 Letakkan file `Axial_UNFREEZE_KFold_1.pth` di dalam folder yang sama dengan file `app.py` ini.")
    st.stop()

# Layout Kolom
col_kiri, col_kanan = st.columns([1, 1.2], gap="large")

with col_kiri:
    st.subheader("📤 Unggah Citra Axial")
    file_upload = st.file_uploader(
        "Pilih file MRI Axial (JPG, PNG, JPEG, atau .npy)",
        type=["png", "jpg", "jpeg", "npy"]
    )
    
    if file_upload is not None:
        if file_upload.name.endswith(".npy"):
            st.success(f"File NumPy termuat: `{file_upload.name}`")
        else:
            gambar = Image.open(file_upload)
            st.image(gambar, caption="Pratinjau Citra MRI Axial", use_container_width=True)

with col_kanan:
    st.subheader("📊 Hasil Diagnosis")
    
    if file_upload is None:
        st.info("Silakan unggah citra irisan Axial di sebelah kiri untuk melihat hasil analisis.")
    else:
        with st.spinner("Memproses citra dengan Vision Transformer..."):
            input_tensor = proses_gambar(file_upload).to(device)
            
            with torch.no_grad():
                logits = model(input_tensor)
                probabilitas = F.softmax(logits, dim=1).cpu().numpy()[0]
                
            pred_idx = int(np.argmax(probabilitas))
            pred_label = KELAS[pred_idx]
            pred_conf = probabilitas[pred_idx] * 100

        # Kartu Hasil
        badge_class = f"badge-{pred_label.lower()}"
        st.markdown(f"""
        <div class="result-card">
            <span style="font-size: 0.9rem; color: #64748B; font-weight: 600;">HASIL PREDIKSI:</span><br>
            <div style="margin-top: 8px;">
                <span class="badge {badge_class}">{pred_label}</span>
                <span style="font-size: 1.4rem; font-weight: 700; margin-left: 12px; color: #1E293B;">
                    {pred_conf:.2f}% Confidence
                </span>
            </div>
            <p style="margin-top: 12px; font-size: 0.95rem; color: #475569;">
                <b>Keterangan:</b> {KELAS_DESC[pred_label]}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        # Diagram Batang Probabilitas
        fig = go.Figure(go.Bar(
            x=[p * 100 for p in probabilitas],
            y=KELAS,
            orientation='h',
            marker=dict(
                color=[KELAS_WARNA[k] for k in KELAS],
                line=dict(color='#000000', width=1)
            ),
            text=[f"{p*100:.1f}%" for p in probabilitas],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Distribusi Probabilitas Kelas",
            xaxis_title="Tingkat Keyakinan (%)",
            yaxis_title="Kelas",
            xaxis=dict(range=[0, 110]),
            height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
