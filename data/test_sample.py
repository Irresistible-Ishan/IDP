import os
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

def test_sample():
    # The structure will likely be inside 'SN3_roads_sample' or directly AOI folders
    # Let's search for an image directory
    base_dir = "."
    if os.path.exists("SpaceNet_Roads_Sample"):
        base_dir = "SpaceNet_Roads_Sample"
        
    aoi_dirs = [d for d in os.listdir(base_dir) if "AOI" in d and os.path.isdir(os.path.join(base_dir, d))]
    
    if not aoi_dirs:
        print("No AOI directories found yet. Make sure the extraction is complete.")
        return

    # Just pick the first AOI (e.g. AOI_2_Vegas)
    test_aoi = aoi_dirs[0]
    aoi_path = os.path.join(base_dir, test_aoi)
    
    rgb_dir = os.path.join(aoi_path, "RGB-PanSharpen")
    geojson_dir = os.path.join(aoi_path, "geojson", "spacenetroads")
    
    if not os.path.exists(rgb_dir):
        # SpaceNet 3 sometimes uses PS-RGB
        rgb_dir = os.path.join(aoi_path, "PS-RGB")
        
    if not os.path.exists(rgb_dir):
        print(f"Could not find RGB images in {aoi_path}")
        return
        
    images = [f for f in os.listdir(rgb_dir) if f.endswith('.tif')]
    if not images:
        print("No .tif images found.")
        return
        
    img_name = images[0]
    img_path = os.path.join(rgb_dir, img_name)
    
    # Matching GeoJSON name (typically replaces 'PS-RGB' or 'RGB-PanSharpen' with 'spacenetroads' and .tif with .geojson)
    # E.g. SN3_roads_train_AOI_2_Vegas_PS-RGB_img10.tif -> SN3_roads_train_AOI_2_Vegas_geojson_spacenetroads_img10.geojson
    geo_name = img_name.replace("PS-RGB_", "spacenetroads_").replace("RGB-PanSharpen_", "spacenetroads_").replace(".tif", ".geojson")
    geo_path = os.path.join(geojson_dir, geo_name)

    print(f"Loading Image: {img_path}")
    print(f"Loading Labels: {geo_path}")

    # Load Image
    img = cv2.imread(img_path)
    # Convert BGR to RGB (OpenCV default is BGR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print("Failed to load image using OpenCV. It might be a 16-bit TIFF requiring rasterio/gdal.")
        import tifffile
        img = tifffile.imread(img_path)
        # Normalize if 16-bit
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    
    # Try to load and overlay GeoJSON (this is spatial data, but SpaceNet GeoJSONs have pixel coordinates?
    # Actually, SpaceNet geojsons usually have real Lat/Lon. Wait! SpaceNet 3 usually provides pixel coordinates 
    # in a separate CSV or a specific geojson, but often the default geojson is lat/lon. 
    # Let's check if we can plot it. If it's Lat/Lon, we can't easily overlay it on the image matrix without rasterizing.
    # But just loading them proves the data is downloaded and structured.
    try:
        with open(geo_path, 'r') as f:
            data = json.load(f)
            features = data.get('features', [])
            print(f"Found {len(features)} road segments in GeoJSON.")
            
            # Simple alert for presentation
            plt.title(f"SpaceNet Sample Loaded\n{test_aoi} - {len(features)} road segments")
    except Exception as e:
        print(f"Could not parse geojson: {e}")
        plt.title(f"SpaceNet Sample Loaded\n{test_aoi}")

    ax.axis("off")
    plt.tight_layout()
    plt.savefig("sample_test_output.png", dpi=150)
    print("Saved preview to 'sample_test_output.png'. Check the data folder!")

if __name__ == "__main__":
    test_sample()
