import cv2
import numpy as np
import os

def generate_anomalous_images(base_image_path, output_dir, num_images=5):
    # Read base image
    img = cv2.imread(base_image_path)
    if img is None:
        raise FileNotFoundError(f"Base image not found: {base_image_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate images with random bright spots
    for i in range(num_images):
        anomalous_img = img.copy()
        # Add 1-3 random white circles
        num_spots = np.random.randint(1, 4)
        for _ in range(num_spots):
            center = (np.random.randint(50, 174), np.random.randint(50, 174))
            radius = np.random.randint(5, 15)
            cv2.circle(anomalous_img, center, radius, (255, 255, 255), -1)
        
        # Save
        output_path = os.path.join(output_dir, f"anomalous{i+1}.jpg")
        cv2.imwrite(output_path, anomalous_img)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    base_image_path = "/Users/yashmandaviya/esca/esca-mvp/data/preprocessed/factory_image.jpg"
    output_dir = "/Users/yashmandaviya/esca/esca-mvp/data/train/anomalous"
    try:
        generate_anomalous_images(base_image_path, output_dir)
    except Exception as e:
        print(f"Error: {e}")