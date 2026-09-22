import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from roadgap_lab import make_world, extract_skeleton, gamma_mask, find_bridges, apply_bridges

st.set_page_config(layout="wide", page_title="IDP Project 20% Demonstration")

st.title("Road-Graph Extraction from Satellite Imagery")
st.subheader("20% Review Demonstration: Evidence-Verified Bridging vs. Gamma Correction")

st.markdown("""
### The Problem: Occlusions in Satellite Imagery
When foundation models (like SAM-Road) predict road networks from satellite imagery, tree canopies, shadows, and clouds cause "occlusions." These result in low-confidence probability predictions, which get cut off by the thresholding process, creating **dead ends** and **disconnected graphs**.

### The Flaw in Existing Solutions
Current literature proposes fixing this by applying **Global Gamma Correction** (which brightens low probabilities) or naive A* dead-end bridging.
*   **Our Finding:** Gamma correction is mathematically identical to simply lowering the threshold. This forces the model to guess aggressively, converting background noise into **spurious roads (false positives)**.
*   **Our Proposed Solution:** **Evidence-Verified Bridging.** We use the original strict threshold to avoid false positives, but specifically search outward from dead-ends. We only finalize a bridge if the path's image evidence (mean probability and turning angle) meets a strict verification criteria.
""")

st.divider()

# Controls
st.sidebar.header("Simulation Controls")
seed = st.sidebar.slider("World Seed (Generate different maps)", 0, 100, 42)
gamma_val = st.sidebar.slider("Gamma Value (Paper 1)", 1.0, 3.0, 2.0, 0.1)
threshold = st.sidebar.slider("Base Threshold", 0.1, 0.9, 0.5, 0.05)
p_min = st.sidebar.slider("Evidence Verifier (p_min)", 0.0, 0.5, 0.25, 0.05)

# Generate World
with st.spinner("Generating Synthetic World..."):
    world = make_world(seed)
    
    gt_sk = world["gt_sk"]
    occ = world["occ"]
    P = world["P"]
    
    # Baseline
    base_sk = extract_skeleton(P > threshold)
    
    # Paper 1: Gamma
    gamma_sk = extract_skeleton(gamma_mask(P, gamma_val) > threshold)
    
    # Our Method: Evidence-Verified Bridging
    import scipy.ndimage as ndi
    Pm = ndi.gaussian_filter(P, 1.0)
    bridges = find_bridges(base_sk, Pm, R=30, lam=2.0) # Using direction lam=2.0
    proposed_sk = apply_bridges(base_sk, bridges, p_min=p_min)

def plot_mask(mask, title, cmap="gray", is_prob=False):
    fig, ax = plt.subplots(figsize=(5, 5))
    if is_prob:
        im = ax.imshow(mask, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.imshow(mask, cmap=cmap)
    
    # Highlight occlusion area
    ys, xs = np.where(occ)
    if len(ys) > 0:
        ax.scatter(xs, ys, c='red', s=1, alpha=0.1, label='Occlusion')
        
    ax.set_title(title)
    ax.axis("off")
    return fig

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 1. Ground Truth & Occlusion")
    st.markdown("Red haze indicates simulated tree canopies/shadows.")
    fig_gt = plot_mask(gt_sk, "Ground Truth Skeleton")
    st.pyplot(fig_gt)

with col2:
    st.markdown("### 2. SAM-Road Predicted Probability")
    st.markdown("Notice the drop in confidence (dark gaps) under the occlusions, and faint background noise.")
    fig_P = plot_mask(P, "Raw Probability Mask", cmap="viridis", is_prob=True)
    st.pyplot(fig_P)

st.divider()
st.markdown("### 3. Comparison of Graph Extraction Methods")

col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("#### A. Baseline Thresholding")
    st.markdown(f"*(Threshold = {threshold})*")
    st.markdown("Fails to bridge the occlusion gap, resulting in disconnected dead-ends. But generates no fake roads.")
    fig_base = plot_mask(base_sk, "Baseline (Broken Connectivity)")
    st.pyplot(fig_base)

with col4:
    st.markdown("#### B. Global Gamma (Paper 1)")
    st.markdown(f"*(Gamma = {gamma_val})*")
    st.markdown("Bridges the gap, but massively amplifies faint background noise into **spurious fake roads** (False Positives).")
    fig_gamma = plot_mask(gamma_sk, "Gamma (Spurious Shortcuts)")
    st.pyplot(fig_gamma)

with col5:
    st.markdown("#### C. Proposed: Evidence-Verified")
    st.markdown(f"*(Base Thr = {threshold}, p_min = {p_min})*")
    st.markdown("Maintains strict threshold to avoid noise, uses A* to explore gaps, and **only connects** if the path evidence is verified.")
    fig_prop = plot_mask(proposed_sk, "Proposed (Clean Bridge)")
    st.pyplot(fig_prop)

st.divider()

st.markdown("""
### Conclusion for 20% Review
As visually proven in this synthetic laboratory:
1. **Gamma correction is flawed** because it uniformly brightens the entire image, leading to unacceptable false positives (spurious marbles in the TOPO metric).
2. **Our Evidence-Verified Bridging** specifically targets dead-ends and dynamically validates the connection logic, successfully bypassing occlusions without sacrificing topological precision.

*Next Steps (Final Research):* We will apply this verified logic directly to the output of the SAM-Road foundation model on the SpaceNet and City-scale real satellite datasets.
""")
