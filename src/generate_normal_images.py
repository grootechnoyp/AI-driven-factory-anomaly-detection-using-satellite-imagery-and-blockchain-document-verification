import cv2
import numpy as np
import os

def generate_normal_images(base_image_path, output_dir, num_images=5):
    # Read base image
    img = cv2.imread(base_image_path)
    if img is None:
        raise FileNotFoundError(f"Base image not found: {base_image_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate variations with slight noise
    for i in range(num_images):
        # Add random brightness noise
        noise = np.random.uniform(-20, 20, img.shape).astype(np.float32)
        noisy_img = img.astype(np.float32) + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        # Save
        output_path = os.path.join(output_dir, f"normal{i+1}.jpg")
        cv2.imwrite(output_path, noisy_img)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    base_image_path = "/Users/yashmandaviya/esca/esca-mvp/data/preprocessed/factory_image.jpg"
    output_dir = "/Users/yashmandaviya/esca/esca-mvp/data/train/normal"
    try:
        generate_normal_images(base_image_path, output_dir)
    except Exception as e:
        print(f"Error: {e}")