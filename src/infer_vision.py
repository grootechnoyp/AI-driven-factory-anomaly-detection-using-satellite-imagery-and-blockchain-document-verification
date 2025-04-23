import torch
from torchvision import models, transforms
import cv2
import numpy as np

def load_model(model_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def infer_image(model, image_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0)
    
    device = torch.device("cpu")
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        classes = ["Normal", "Anomalous"]
        return classes[predicted.item()]

if __name__ == "__main__":
    model_path = "/Users/yashmandaviya/esca/esca-mvp/models/resnet_anomaly.pth"
    image_path = "/Users/yashmandaviya/esca/esca-mvp/data/preprocessed/factory_image.jpg"
    try:
        model = load_model(model_path)
        result = infer_image(model, image_path)
        print(f"Prediction: {result}")
    except Exception as e:
        print(f"Error: {e}")