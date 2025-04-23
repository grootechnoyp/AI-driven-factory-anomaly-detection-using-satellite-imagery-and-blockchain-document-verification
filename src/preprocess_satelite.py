import rasterio
import numpy as np
import cv2
import os

def preprocess_sentinel(image_dir, output_path, size=(224, 224)):
    # Find RGB bands in Sentinel-2 folder
    red_path = next(f for f in os.listdir(image_dir) if "B04" in f)
    green_path = next(f for f in os.listdir(image_dir) if "B03" in f)
    blue_path = next(f for f in os.listdir(image_dir) if "B02" in f)
    
    # Read bands
    with rasterio.open(os.path.join(image_dir, red_path)) as red:
        r = red.read(1)
    with rasterio.open(os.path.join(image_dir, green_path)) as green:
        g = green.read(1)
    with rasterio.open(os.path.join(image_dir, blue_path)) as blue:
        b = blue.read(1)
    
    # Stack into RGB
    rgb = np.dstack((r, g, b))
    
    # Normalize to 0-255
    rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Resize
    rgb_resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, rgb_resized)
    print(f"Preprocessed image saved to {output_path}")

if __name__ == "__main__":
    image_dir = "/Users/yashmandaviya/esca/esca-mvp/data/S2B_MSIL2A_20241207T054129_N0511_R005_T42QZJ_20241207T074448.SAFE/GRANULE/L2A_T42QZJ_A040500_20241207T054906/IMG_DATA/R10m"
    output_path = "/Users/yashmandaviya/esca/esca-mvp/data/preprocessed/factory_image.jpg"
    preprocess_sentinel(image_dir, output_path)