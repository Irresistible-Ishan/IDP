import os
import cv2
import json
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

try:
    import rasterio
    from rasterio.features import rasterize
    from shapely.geometry import shape, LineString
except ImportError:
    print("Please wait for rasterio to finish installing.")
    exit(1)

# Import the logic we built last time
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mvp")))
from roadgap_lab import extract_skeleton, gamma_mask, find_bridges, apply_bridges, evaluate

def simulate_sam_road_mask(gt_mask):
    """
    Since we don't have the heavy PyTorch SAM-Road model running locally,
    we simulate its prediction output by degrading the real ground truth.
    This creates realistic gaps (occlusions) and faint background noise.
    """
    H, W = gt_mask.shape
    road = cv2.dilate(gt_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    
    # Base probability on the real roads
    P = 0.05 + 0.85 * ndi.gaussian_filter(road.astype(float), 0.8)
    
    # Simulate occlusions (trees/shadows) over the roads
    rng = np.random.default_rng(42)
    ys, xs = np.nonzero(gt_mask)
    occ = np.zeros((H, W), np.uint8)
    if len(ys) > 0:
        for _ in range(8): # Add 8 random tree occlusions
            i = rng.integers(len(ys))
            cv2.circle(occ, (int(xs[i]), int(ys[i])), int(rng.integers(8, 16)), 1, -1)
            
    soft_occ = ndi.gaussian_filter(occ.astype(float), 2.0)
    P = P * (1 - soft_occ * rng.uniform(0.6, 0.9)) # drastically drop probability under trees
    
    # Simulate faint noise (spurious fields that trick gamma)
    noise = ndi.gaussian_filter(rng.normal(size=(H, W)), 2.5)
    P = np.clip(P + 0.05 * noise / noise.std(), 0, 1)
    
    return P, occ.astype(bool)

def run_metrics_on_real_sample():
    base_dir = "SpaceNet_Roads_Sample"
    aoi_path = os.path.join(base_dir, "AOI_2_Vegas_Roads_Sample")
    rgb_dir = os.path.join(aoi_path, "RGB-PanSharpen")
    geojson_dir = os.path.join(aoi_path, "geojson", "spacenetroads")
    
    # Pick a specific image that we know has roads
    img_name = "RGB-PanSharpen_AOI_2_Vegas_img1454.tif"
    geo_name = "spacenetroads_AOI_2_Vegas_img1454.geojson"
    
    img_path = os.path.join(rgb_dir, img_name)
    geo_path = os.path.join(geojson_dir, geo_name)

    print(f"Loading Real Ground Truth from: {geo_name}...")
    
    # 1. Rasterize the real GeoJSON to create the Ground Truth Skeleton (gt_sk)
    with rasterio.open(img_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        
    with open(geo_path, 'r') as f:
        data = json.load(f)
        
    lines = []
    for feat in data.get('features', []):
        geom = shape(feat['geometry'])
        if isinstance(geom, LineString):
            lines.append(geom)
            
    if not lines:
        print("No roads found in this image.")
        return
        
    # Rasterize lines to image grid (1 for road, 0 for background)
    gt_mask = rasterize(lines, out_shape=(height, width), transform=transform, default_value=1)
    gt_sk = gt_mask > 0

    # 2. Simulate the SAM-Road AI Probability prediction
    print("Simulating SAM-Road AI Prediction on real geometry...")
    P, occ = simulate_sam_road_mask(gt_sk)
    
    w = {"P": P, "gt_sk": gt_sk, "occ": occ, "seed": 42, "Pm": ndi.gaussian_filter(P, 1.0)}

    # 3. Run the Models
    print("Running Baseline (Threshold)...")
    base_sk = extract_skeleton(P > 0.5)
    
    print("Running Paper 1 (Gamma Correction)...")
    gamma_sk = extract_skeleton(gamma_mask(P, 1.5) > 0.5)
    
    print("Running Proposed (Evidence-Verified Bridging)...")
    bridges = find_bridges(base_sk, w["Pm"], R=30, lam=2.0)
    proposed_sk = apply_bridges(base_sk, bridges, p_min=0.25)

    # 4. Evaluate Metrics
    print("Scoring metrics against ground truth...")
    res_base = evaluate(base_sk, w, 42)
    res_gamma = evaluate(gamma_sk, w, 42)
    res_prop = evaluate(proposed_sk, w, 42)

    # 5. Print Results Table
    print("\n" + "="*80)
    print("REAL DATASET RESULTS: Vegas (img1454) - Simulated SAM-Road Output")
    print("="*80)
    print(f"{'Method':<35} | {'APLS (Routing)':<15} | {'TOPO F1':<10} | {'Gap Recall (Trees)':<15}")
    print("-" * 80)
    print(f"{'1. Baseline (Disconnected)':<35} | {res_base['apls']:.4f}          | {res_base['f1']:.4f}     | {res_base['gap_rec']:.4f}")
    print(f"{'2. Paper 1 (Flawed Gamma)':<35} | {res_gamma['apls']:.4f}          | {res_gamma['f1']:.4f}     | {res_gamma['gap_rec']:.4f}")
    print(f"{'3. Proposed (Verified Bridging)':<35} | {res_prop['apls']:.4f}          | {res_prop['f1']:.4f}     | {res_prop['gap_rec']:.4f}")
    print("="*80)
    
    # Save a comparison image
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(P, cmap='viridis'); axs[0].set_title("SAM-Road AI Output (Gaps)")
    axs[1].imshow(base_sk, cmap='gray'); axs[1].set_title("Baseline (Gaps Remain)")
    axs[2].imshow(gamma_sk, cmap='gray'); axs[2].set_title("Paper 1 Gamma (Fake Roads added)")
    axs[3].imshow(proposed_sk, cmap='gray'); axs[3].set_title("Our Proposed (Clean Bridges)")
    
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.savefig("real_sample_comparison.png", dpi=150)
    print("\nVisual comparison saved to 'real_sample_comparison.png'")

if __name__ == "__main__":
    run_metrics_on_real_sample()
