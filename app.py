import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
st.set_page_config(page_title="RectoScan AI", layout="wide")
NUM_CLASSES = 2 

# --- STEP 1: MODEL ARCHITECTURE ---

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
    def __init__(self, in_channels=1, out_channels=NUM_CLASSES, d_model=256, nhead=8, num_transformer_layers=1):
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

# --- STEP 2: UTILITIES ---

@st.cache_resource
def load_model():
    model = TransUNet(in_channels=1, out_channels=NUM_CLASSES)
    # Ensure the filename matches exactly
    checkpoint = 'transunet_attention_binary_best_dice_no_es (1).pth'
    state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
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

def create_viz(orig, pred, a1, a2):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    # 1. Original
    axes[0].imshow(orig, cmap='gray')
    axes[0].set_title("Input Scan")
    axes[0].axis('off')

    # 2. Predicted Mask (Overlay Style)
    mask_rgb = np.zeros((*pred.shape, 3))
    mask_rgb[pred == 1] = [1, 0, 0] # Red for Tumor
    axes[1].imshow(mask_rgb)
    axes[1].set_title("Predicted Label (Tumor)")
    axes[1].axis('off')

    # 3. Attention Map 1
    im1 = axes[2].imshow(a1, cmap='viridis')
    axes[2].set_title("Attention Map 1 (Fine)")
    plt.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis('off')

    # 4. Attention Map 2
    im2 = axes[3].imshow(a2, cmap='viridis')
    axes[3].set_title("Attention Map 2 (Coarse)")
    plt.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].axis('off')

    plt.tight_layout()
    return fig

# --- STEP 3: STREAMLIT UI ---

st.title("🧠 RectoScan AI: TransUNet Diagnostic")
st.write("Upload a brain MRI/CT slice for automated tumor segmentation.")

file = st.file_uploader("Upload Medical Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    model = load_model()
    input_batch = preprocess(img)

    with torch.no_grad():
        logits, alpha1, alpha2 = model(input_batch)
        probs = F.softmax(logits, dim=1)
        
        # Category Logic
        conf_vals, pred_vals = torch.max(probs, dim=1)
        pred_np = pred_vals.squeeze().cpu().numpy()
        
        # Confidence Score
        # We calculate mean confidence specifically for predicted tumor pixels if any, else whole image
        if torch.any(pred_vals == 1):
            confidence = conf_vals[pred_vals == 1].mean().item()
            diagnosis = "TUMOR DETECTED (Category 1)"
            diag_color = "red"
        else:
            confidence = conf_vals.mean().item()
            diagnosis = "NORMAL (Category 0)"
            diag_color = "green"

    # Display Results
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"### Diagnosis: :{diag_color}[{diagnosis}]")
    with c2:
        st.metric("Detection Confidence", f"{confidence*100:.2f}%")

    st.subheader("Analysis Visualization")
    # Resize original for plotting overlay
    orig_np = np.array(img.convert("L").resize((224, 224)))
    a1_np = alpha1.squeeze().cpu().numpy()
    a2_np = alpha2.squeeze().cpu().numpy()
    
    st.pyplot(create_viz(orig_np, pred_np, a1_np, a2_np))
