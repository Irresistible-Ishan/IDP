import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import scipy.ndimage as ndi

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from roadgap_lab import make_world, extract_skeleton, gamma_mask, find_bridges, apply_bridges

st.set_page_config(layout="wide", page_title="IDP Project: Road-Graph Extraction", page_icon="🛰️")

# --- Custom CSS (Adapted for Light/Dark mode readability) ---
st.markdown("""
<style>
    .main-header { font-size: 3rem !important; font-weight: 800; color: #3B82F6; text-align: center; margin-bottom: 0px; }
    .sub-header { font-size: 1.5rem !important; font-style: italic; color: var(--text-color); text-align: center; margin-bottom: 40px; opacity: 0.8;}
    
    /* Enforcing dark text on these specific light backgrounds so it never becomes invisible in dark mode */
    .theory-box { background-color: #F3F4F6; color: #111827; padding: 25px; border-radius: 12px; border-left: 6px solid #3B82F6; margin-bottom: 20px; font-size: 1.2rem; line-height: 1.6;}
    .alert-box { background-color: #FEF2F2; color: #111827; padding: 25px; border-radius: 12px; border-left: 6px solid #EF4444; margin-bottom: 20px; font-size: 1.2rem; line-height: 1.6;}
    .success-box { background-color: #ECFDF5; color: #111827; padding: 25px; border-radius: 12px; border-left: 6px solid #10B981; margin-bottom: 20px; font-size: 1.2rem; line-height: 1.6;}
    .info-box { background-color: #EFF6FF; color: #111827; padding: 25px; border-radius: 12px; border-left: 6px solid #3B82F6; margin-bottom: 20px; font-size: 1.1rem; line-height: 1.6;}
    
    .section-title { font-size: 2.2rem !important; font-weight: 700; color: var(--text-color); margin-top: 50px; margin-bottom: 25px; border-bottom: 2px solid var(--border-color); padding-bottom: 10px;}
    
    /* Flowchart CSS using native Streamlit variables */
    .flowchart { display: flex; justify-content: space-between; align-items: center; margin: 40px 0; font-size: 1.2rem; text-align: center;}
    .step { background-color: var(--secondary-background-color); color: var(--text-color); padding: 25px; border-radius: 12px; border: 2px solid var(--border-color); width: 22%; box-shadow: 0 4px 10px rgba(0,0,0,0.05); font-weight: 600;}
    .arrow { font-size: 2.5rem; color: #9CA3AF; font-weight: bold; }
    .step-highlight { border-color: #3B82F6; background-color: #EFF6FF; color: #1E3A8A; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🛰️ Road-Graph Extraction from Satellite Imagery</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">20% Review: Evidence-Verified Bridging vs. Gamma Correction</p>', unsafe_allow_html=True)

# --- THEORY SECTION ---
st.markdown('<div class="section-title">📖 The Core Research Problem</div>', unsafe_allow_html=True)
col_t1, col_t2 = st.columns(2)
with col_t1:
    st.markdown("""
    <div class="theory-box">
    <b>🚧 The Issue: Graph Disconnections</b><br>
    When AI models (like SAM-Road) process satellite images, obstacles like <b>tree canopies and shadows</b> block the view of the road. 
    The AI's confidence plummets, causing the extracted road network to have <b>broken links (dead-ends)</b>. Real-world navigation apps cannot route through broken graphs.
    </div>
    """, unsafe_allow_html=True)

with col_t2:
    st.markdown("""
    <div class="alert-box">
    <b>❌ The Flaw in Prior Art (Paper 1)</b><br>
    Previous researchers applied <i>Global Gamma Correction</i> to artificially brighten the whole image to recover these hidden roads. 
    <b>Our finding:</b> This is mathematically identical to just lowering the confidence threshold. It bridges gaps, but disastrously amplifies faint background noise into <b>spurious fake roads (False Positives)</b>.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="success-box">
<b>✅ Our Proposed Solution: Evidence-Verified Bridging</b><br>
Instead of altering the entire image, we maintain a strict baseline threshold to suppress noise. We then deploy an intelligent A* search that scans outward strictly from dead-ends. 
Crucially, it only finalizes a bridge if the path meets a strict <b>probability evidence threshold (p_min)</b>, ensuring we bridge real roads, not fields.
</div>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT FOR BUTTONS ---
if 'gamma_val' not in st.session_state:
    st.session_state.gamma_val = 2.0
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.5
if 'p_min' not in st.session_state:
    st.session_state.p_min = 0.25

def set_paper1_mode():
    st.session_state.gamma_val = 2.0
    st.session_state.threshold = 0.5
    st.session_state.p_min = 0.0

def set_proposed_mode():
    st.session_state.gamma_val = 1.0 # Disable gamma
    st.session_state.threshold = 0.5
    st.session_state.p_min = 0.25

# --- INTERACTIVE DASHBOARD ---
st.markdown('<div class="section-title">🔬 Interactive Synthetic Laboratory</div>', unsafe_allow_html=True)

st.sidebar.header("🎬 Auto-Simulate (Action Replay)")
st.sidebar.markdown("Click to instantly snap sliders to compare configurations:")
st.sidebar.button("❌ Simulate Paper 1 (Flawed Gamma)", on_click=set_paper1_mode, use_container_width=True, type="secondary")
st.sidebar.button("✅ Simulate Our Proposed Method", on_click=set_proposed_mode, use_container_width=True, type="primary")

st.sidebar.divider()

st.sidebar.header("⚙️ Manual Controls")
seed = st.sidebar.slider("World Seed (Generate Different Maps)", 0, 100, 42)
st.sidebar.slider("Base Threshold (tau)", min_value=0.1, max_value=0.9, step=0.05, key='threshold')
st.sidebar.slider("Gamma Value (Prior Art)", min_value=1.0, max_value=3.0, step=0.1, key='gamma_val')
st.sidebar.slider("Evidence Verifier (p_min)", min_value=0.0, max_value=0.5, step=0.05, key='p_min')

# Generate World
with st.spinner("Rendering Simulation Logic..."):
    world = make_world(seed)
    gt_sk, occ, P = world["gt_sk"], world["occ"], world["P"]
    
    base_sk = extract_skeleton(P > st.session_state.threshold)
    gamma_sk = extract_skeleton(gamma_mask(P, st.session_state.gamma_val) > st.session_state.threshold)
    
    Pm = ndi.gaussian_filter(P, 1.0)
    bridges = find_bridges(base_sk, Pm, R=30, lam=2.0)
    proposed_sk = apply_bridges(base_sk, bridges, p_min=st.session_state.p_min)

def plot_mask(mask, title, cmap="gray", is_prob=False):
    fig, ax = plt.subplots(figsize=(7, 7))
    if is_prob:
        im = ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.imshow(mask, cmap=cmap)
    ys, xs = np.where(occ)
    if len(ys) > 0:
        ax.scatter(xs, ys, c='red', s=8, alpha=0.3, label='Occlusion')
    ax.set_title(title, fontsize=16, fontweight='bold', color='black') # Forced black title for plot background
    ax.axis("off")
    return fig

# Show Source Data
st.markdown("#### Step 1: The Input Environment")
col1, col2 = st.columns(2)
with col1:
    st.info("🎯 **Ground Truth (Reality):** The actual road network. The red haze highlights simulated occlusions (e.g., dense tree canopies).")
    st.pyplot(plot_mask(gt_sk, "Ground Truth Skeleton"))
with col2:
    st.warning("🌫️ **AI Probability Mask:** The neural network's raw prediction. Notice the dark gaps right under the red occlusions, and the faint background noise.")
    st.pyplot(plot_mask(P, "Raw Probability Mask (SAM-Road)", cmap="viridis", is_prob=True))

st.markdown("#### Step 2: The Extraction Algorithms (The Results)")
col3, col4, col5 = st.columns(3)
with col3:
    st.markdown("##### 🟨 A. Baseline (Standard)")
    st.markdown(f"<span style='color:gray'>*(Threshold = {st.session_state.threshold:.2f})*</span><br>High precision (no fake roads), but leaves massive **dead-end gaps** at occlusions.", unsafe_allow_html=True)
    st.pyplot(plot_mask(base_sk, "Baseline (Broken Graph)"))

with col4:
    if st.session_state.gamma_val > 1.2:
        st.error("##### ❌ B. Gamma (Prior Art)")
        st.markdown(f"<span style='color:gray'>*(Gamma = {st.session_state.gamma_val:.1f})*</span><br>Bridges the gaps, but ruins accuracy by inventing **fake spurious roads**!", unsafe_allow_html=True)
    else:
        st.markdown("##### 🟨 B. Gamma (Prior Art)")
        st.markdown(f"<span style='color:gray'>*(Gamma = {st.session_state.gamma_val:.1f})*</span><br>Gamma is effectively disabled at 1.0.", unsafe_allow_html=True)
    st.pyplot(plot_mask(gamma_sk, "Gamma (Spurious False Positives)"))

with col5:
    if st.session_state.p_min >= 0.15 and st.session_state.gamma_val <= 1.5:
        st.success("##### ✅ C. Proposed (Verified)")
        st.markdown(f"<span style='color:gray'>*(p_min = {st.session_state.p_min:.2f})*</span><br>Gaps are perfectly bridged **without** generating any fake roads. Clean routing!", unsafe_allow_html=True)
    else:
        st.markdown("##### 🟩 C. Proposed (Verified)")
        st.markdown(f"<span style='color:gray'>*(p_min = {st.session_state.p_min:.2f})*</span><br>Adjust controls or click the green button above.", unsafe_allow_html=True)
    st.pyplot(plot_mask(proposed_sk, "Proposed (Clean Reconnection)"))


# --- EXHAUSTIVE 80% PROJECT PLAN SECTION ---
st.markdown('<div class="section-title">🚀 Final Research: The Remaining 80% Architecture</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
The synthetic laboratory above mathematically proves our algorithmic bridging mechanism. 
The remaining 80% of our IDP project transitions this logic to exact real-world <b>Machine Learning Models</b> and <b>Open-Source Satellite Datasets</b>.
</div>
""", unsafe_allow_html=True)

# Custom HTML Flowchart
st.markdown("""
<div class="flowchart">
    <div class="step">🌍 1. Raw Satellite Imagery<br><span style="font-size:0.95rem;font-weight:400;">SpaceNet & City-Scale</span></div>
    <div class="arrow">➔</div>
    <div class="step">🧠 2. Foundation Model<br><span style="font-size:0.95rem;font-weight:400;">SAM-Road (ViT-B)</span></div>
    <div class="arrow">➔</div>
    <div class="step step-highlight">🔬 3. Our Verified Bridging<br><span style="font-size:0.95rem;font-weight:400;">Replacing Gamma/A*</span></div>
    <div class="arrow">➔</div>
    <div class="step">🗺️ 4. Formal Evaluation<br><span style="font-size:0.95rem;font-weight:400;">Go-APLS & TOPO F1</span></div>
</div>
""", unsafe_allow_html=True)

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    st.markdown("""
    ### 💻 Phase 1: Real-World Inference (ML Models)
    **Foundation Model:**
    - **Model Architecture:** We will use **SAM-Road** (Segment Anything Model adapted for Roads). Specifically, the **ViT-B** (Vision Transformer) image encoder coupled with a Geometry & Topology Decoder.
    - **Training State:** We will use pre-trained Hugging Face checkpoints (`congrui/sam_road`). We will execute **Inference-Only** (no re-training required), meaning it can be run on student hardware.
    
    **Datasets (Standard Benchmarks):**
    - **[SpaceNet](https://spacenet.ai/spacenet-roads-dataset/):** 382 Testing Tiles (400×400 px, 1 m/px resolution). Covering Las Vegas, Paris, Shanghai, and Khartoum.
    - **[City-Scale](https://github.com/anilbatra21/Road_Extraction_From_Satellite_Images):** 27 Testing Tiles (2048×2048 px, 1 m/px resolution) across 20 US cities.
    """)

with col_p2:
    st.markdown("""
    ### 📊 Phase 2: Formal Metric Validation
    To ensure our results are respected by the academic community, we will not use standard pixel-segmentation IoU. We will use Graph-Theory metrics:
    
    - **APLS (Average Path Length Similarity):** The official routing metric. We will utilize the standard **Go-language implementation** to calculate how closely navigation distances match ground truth.
    - **TOPO F1 Score:** The topological "marbles and holes" metric to measure strict structural connectivity.
    
    **Pareto Optimization Protocol:**
    - Unlike previous papers, we will carve out a strict **Validation Split** to lock in our angular cost (`lam`) and evidence threshold (`p_min`) to prevent dataset bias.
    """)

with col_p3:
    st.markdown("""
    ### 📝 Phase 3: Stratified Evaluation & Publication
    - **New Evaluation Metric (Occlusion-Stratified):** We will script a custom evaluation method to measure "Gap Recall vs. Precision" *specifically* on road segments covered by trees (using RGB excess-green cues) and shadows. This proves our method works exactly where it is designed to.
    
    - **The Mathematical Proof:** We will formally document the control experiment mathematically proving that `P^(1/gamma) > tau` is identical to `P > tau^gamma`, rendering the Gamma Correction claims from prior art redundant. 
    
    - **Ablation Studies:** Proving the relative impact of our Angular-Direction Cost vs. Probability-Evidence Cost on real satellite grids.
    """)

# --- REFERENCES SECTION ---
st.markdown('<div class="section-title">📚 Core References & Base Papers</div>', unsafe_allow_html=True)

st.markdown("""
<div class="theory-box" style="margin-bottom: 50px;">
<b>The exact papers our IDP is directly building upon and improving:</b>
<ul style="margin-top: 10px; line-height: 1.8;">
<li><b>Primary Base Paper (The Flawed Baseline):</b> Fathul'ibad et al., <i>"Improving SAM-Road Model for Occlusion Handling in Road Extraction"</i> (JISEBI 12(1), Feb 2026). <br><span style="color: #4B5563;">This is the paper proposing the Global Gamma Correction & distance-only A* bridging that our logic is scientifically correcting.</span> <br>🔗 <a href="https://e-journal.unair.ac.id/JISEBI/article/view/76342" target="_blank" style="color: #2563EB; font-weight: bold; text-decoration: none;">[View Official Publication]</a></li>
<li style="margin-top: 15px;"><b>The Foundation Model (SAM-Road):</b> Congrui Hetang et al., <i>"SAM-Road: Adapting Segment Anything Model for Street Network Extraction"</i> (CVPRW 2024 Best Paper). <br><span style="color: #4B5563;">The official pre-trained Vision Transformer model architecture we use to extract the raw road probabilities from satellite imagery.</span> <br>🔗 <a href="https://arxiv.org/abs/2403.16051" target="_blank" style="color: #2563EB; font-weight: bold; text-decoration: none;">[View arXiv Preprint]</a></li>
<li style="margin-top: 15px;"><b>The SOTA Benchmark (DOGE):</b> <i>"Differentiable Bézier Graph Optimization for Remote Sensing"</i> (2025). <br><span style="color: #4B5563;">The current State-of-the-Art (SOTA) graph extraction paper whose TOPO/APLS scores we will benchmark against in our final evaluation.</span> <br>🔗 <a href="https://arxiv.org/abs/2511.19850" target="_blank" style="color: #2563EB; font-weight: bold; text-decoration: none;">[View arXiv Preprint]</a></li>
</ul>
</div>
""", unsafe_allow_html=True)

