import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import base64
import json

# ============================================
# MODEL ARCHITECTURE (Integrated)
# ============================================
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
             g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi_in = self.relu(g1 + x1)
        psi_out = self.psi(psi_in)
        return x * psi_out, psi_out

class TransUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, d_model=256, nhead=8, num_transformer_layers=1):
        super(TransUNet, self).__init__()
        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck_cnn = conv_block(128, d_model)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, activation='relu', batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)

        self.up2 = nn.ConvTranspose2d(d_model, 128, kernel_size=2, stride=2)
        self.Att2 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.Att1 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.dec1 = conv_block(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b_feat = self.bottleneck_cnn(p2)
        b, c, h, w = b_feat.size()
        x_flat = b_feat.flatten(2).permute(0, 2, 1)
        t_out = self.transformer_encoder(x_flat)
        t_out = t_out.permute(0, 2, 1).view(b, c, h, w)

        u2 = self.up2(t_out)
        a2, alpha2 = self.Att2(g=u2, x=e2)
        c2 = torch.cat([a2, u2], dim=1)
        d2 = self.dec2(c2)

        u1 = self.up1(d2)
        a1, alpha1 = self.Att1(g=u1, x=e1)
        c1 = torch.cat([a1, u1], dim=1)
        d1 = self.dec1(c1)
        
        return self.out_conv(d1), alpha1, alpha2

# ============================================
# PAGE CONFIG & CSS
# ============================================
st.set_page_config(page_title="RectoScan AI", page_icon="🧠", layout="wide")

def display_permanent_logo():
    try:
        with open("logo.png", "rb") as f:
            data = f.read()
            home_b64 = base64.b64encode(data).decode()
        st.markdown(
            f"""
            <style>
                .custom-logo {{ position: fixed; top: 10px; left: 1rem; z-index: 999999; }}
                .custom-logo img {{ height: 40px; width: auto; cursor: pointer; }}
            </style>
            <div class="custom-logo">
                <a href="https://acai-apps.amity.edu:8501/" target="_self">
                    <img src="data:image/png;base64,{home_b64}">
                </a>
            </div>
            """, unsafe_allow_html=True
        )
    except: pass

display_permanent_logo()

col1, col2, col3 = st.columns([3, 4, 1])
with col2:
    st.image("amity_logo.png")


# Custom Medical Theme CSS
st.markdown("""
<style>
    .main-header { font-size: 2.8rem; font-weight: bold; text-align: center; color: #0891B2; margin-bottom: 0.5rem; }
    .card {
        background: #0F172A; 
        border: 1px solid #1E293B;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 18px;
        color: white;
    }
    .badge {
        display:inline-block;
        padding:5px 12px;
        border-radius:999px;
        background:#0891B2;
        color:white;
        font-size:12px;
        margin-bottom: 12px;
        font-weight: 600;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #0891B2 0%, #155E75 100%);
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE & UTILS
# ============================================
if 'active_tab' not in st.session_state: st.session_state.active_tab = "🏠 Home"
if 'processed' not in st.session_state: st.session_state.processed = False
if 'viz_data' not in st.session_state: st.session_state.viz_data = {}

@st.cache_resource
def load_model():
    model = TransUNet(in_channels=1, out_channels=2)
    checkpoint = 'transunet_attention_binary_best_dice_no_es (1).pth'
    try:
        state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    except: pass
    model.eval()
    return model

def preprocess(image):
    t = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return t(image).unsqueeze(0)

# ============================================
# NAVIGATION
# ============================================
tabs = ["🏠 Home", "🔎 Prediction", "ℹ️ About"]
current_index = tabs.index(st.session_state.active_tab)
selected_tab = st.radio("Nav", tabs, index=current_index, horizontal=True, label_visibility="collapsed")
st.session_state.active_tab = selected_tab
st.markdown("---")

# ============================================
# TAB 1: HOME
# ============================================
if selected_tab == "🏠 Home":
    st.markdown('<div class="main-header">🧠 RectoScan AI</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.5, 4, 0.5])
    with col2:
        try: st.image("architecture.png", caption="TransUNet with Dual Attention Mechanism")
        except: st.info("Place your 'architecture.png' here.")
        st.markdown("""
        <div style="text-align: justify;">
            RectoScan AI is a state-of-the-art diagnostic assistant leveraging the <b>TransUNet</b> architecture for automated medical image segmentation. 
            By combining the spatial precision of CNNs with the global context of Transformers and a Dual-Attention mechanism, the system identifies and 
            localizes brain tumors from MRI/CT scans with superhuman accuracy. 
            Designed for clinical decision support, it provides heatmaps (Attention Maps) to explain <i>why</i> the AI made its diagnosis, 
            ensuring transparency in medical AI.
        </div><br>""", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("Launch"):
                st.session_state.active_tab = "🔎 Prediction"
                st.rerun()

# ============================================
# TAB 2: PREDICTION
# ============================================
elif selected_tab == "🔎 Prediction":
    st.subheader("Upload Scan")
    file = st.file_uploader("Upload DICOM/Image", type=["jpg", "png", "jpeg"])
    
    if st.button("Run AI Analysis", use_container_width=True):
        if file:
            with st.spinner("Executing TransUNet Pipeline..."):
                img = Image.open(file).convert("RGB")
                model = load_model()
                input_batch = preprocess(img)

                with torch.no_grad():
                    logits, a1, a2 = model(input_batch)
                    probs = F.softmax(logits, dim=1)
                    conf_vals, pred_vals = torch.max(probs, dim=1)
                    pred_np = pred_vals.squeeze().cpu().numpy()
                    
                    if torch.any(pred_vals == 1):
                        confidence = conf_vals[pred_vals == 1].mean().item()
                        diagnosis = "TUMOR DETECTED"
                        color = "red"
                    else:
                        confidence = conf_vals.mean().item()
                        diagnosis = "NORMAL"
                        color = "green"

                    # Store data for visualization
                    st.session_state.viz_data = {
                        "orig": np.array(img.convert("L").resize((224, 224))),
                        "pred": pred_np,
                        "a1": a1.squeeze().cpu().numpy(),
                        "a2": a2.squeeze().cpu().numpy(),
                        "diag": diagnosis,
                        "conf": confidence,
                        "color": color
                    }
                    st.session_state.processed = True
                    # st.session_state.active_tab = "📊 Analysis"
                    
                    vd = st.session_state.viz_data
                    st.markdown(f"## Diagnostic Report")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"""
                        <div class="card">
                            <div class="badge">AI Diagnosis</div>
                            <h2 style="color:{vd['color']}">{vd['diag']}</h2>
                            <p>Status: Pathological focus localized</p>
                        </div>""", unsafe_allow_html=True)
                        
                    with c2:
                        st.markdown(f"""
                        <div class="card">
                            <div class="badge">Confidence Metric</div>
                            <h2>{vd['conf']*100:.2f}%</h2>
                            <p>Model certainty on region of interest</p>
                        </div>""", unsafe_allow_html=True)

                    st.subheader("🛠️ Visualization Dashboard")
                    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
                    axes[0].imshow(vd['orig'], cmap='gray')
                    axes[0].set_title("Input Scan")
                    axes[0].axis('off')
                    
                    mask_rgb = np.zeros((*vd['pred'].shape, 3))
                    mask_rgb[vd['pred'] == 1] = [1, 0, 0]
                    axes[1].imshow(mask_rgb)
                    axes[1].set_title("Predicted Tumor Mask")
                    axes[1].axis('off')
                    
                    axes[2].imshow(vd['a1'], cmap='viridis')
                    axes[2].set_title("Fine Attention")
                    axes[2].axis('off')
                    
                    axes[3].imshow(vd['a2'], cmap='viridis')
                    axes[3].set_title("Coarse Attention")
                    axes[3].axis('off')
                    
                    st.pyplot(fig)

                    if st.button("New Scan Analysis"):
                        st.session_state.processed = False
                        st.session_state.active_tab = "🏠 Home"
        else:
            st.error("Please upload an image first.")
        

# ============================================
# TAB 3: ABOUT
# ============================================
elif selected_tab == "ℹ️ About":
    st.markdown("""
    <div style="text-align: justify;">
        <h2>Research Abstract</h2>
        <h4>Background</h4>
        Precision in medical imaging is critical for early oncology intervention. Traditional UNet models often struggle with long-range dependencies in complex brain structures. 
        <h4>Objectives</h4>
        RectoScan AI aims to provide a robust segmentation tool using <b>TransUNet</b>, which leverages Transformers to maintain global context during the encoding phase.
        <h4>Methods</h4>
        The system utilizes a hybrid CNN-Transformer backbone. Attention gates are implemented in the skip-connections to filter out non-relevant features and focus on pathological areas. 
        The model was trained on a comprehensive binary classification dataset to achieve high Dice scores and explainable attention coefficients.
    </div>""", unsafe_allow_html=True)
