import torch
from torchvision import models, transforms
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import cv2
import numpy as np
import os

# Load vision model (ResNet)
def load_vision_model(model_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Infer vision (Normal/Anomalous)
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

# Load NLP model (DistilBERT)
def load_nlp_model(model_path):
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

# Infer NLP (Authentic/Suspicious)
def infer_text(model, tokenizer, text, max_len=128):
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=1).item()
        classes = ["Authentic", "Suspicious"]
        return classes[predicted]

# Combine vision and NLP for anomaly detection
def detect_anomaly(vision_model, nlp_model, nlp_tokenizer, image_path, text, factory_name="Unknown"):
    # Run vision inference
    vision_result = infer_image(vision_model, image_path)
    
    # Run NLP inference
    nlp_result = infer_text(nlp_model, nlp_tokenizer, text)
    
    # Decision rule: High risk if both Anomalous and Suspicious
    if vision_result == "Anomalous" and nlp_result == "Suspicious":
        risk_level = "High"
        message = f"High risk: Anomalous activity and suspicious record detected for Factory {factory_name}."
    elif vision_result == "Anomalous" or nlp_result == "Suspicious":
        risk_level = "Medium"
        message = f"Medium risk: {vision_result} activity and {nlp_result.lower()} record for Factory {factory_name}."
    else:
        risk_level = "Low"
        message = f"Low risk: Normal activity and authentic record for Factory {factory_name}."
    
    return {
        "factory": factory_name,
        "vision_result": vision_result,
        "nlp_result": nlp_result,
        "risk_level": risk_level,
        "message": message
    }

if __name__ == "__main__":
    # Paths to models
    vision_model_path = "/Users/yashmandaviya/esca/esca-mvp/models/resnet_anomaly.pth"
    nlp_model_path = "/Users/yashmandaviya/esca/esca-mvp/models/distilbert_verification"
    
    # Test inputs
    image_path = "/Users/yashmandaviya/esca/esca-mvp/data/preprocessed/test_anomalous.jpg"
    text = "Factory A claimed 1000 garments on 2025-04-12 with only 2 workers, no machinery details."
    factory_name = "A"
    
    try:
        # Load models
        vision_model = load_vision_model(vision_model_path)
        nlp_model, nlp_tokenizer = load_nlp_model(nlp_model_path)
        
        # Run anomaly detection
        result = detect_anomaly(vision_model, nlp_model, nlp_tokenizer, image_path, text, factory_name)
        
        # Print results
        print(f"Factory: {result['factory']}")
        print(f"Vision Result: {result['vision_result']}")
        print(f"NLP Result: {result['nlp_result']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Message: {result['message']}")
    except Exception as e:
        print(f"Error: {e}")