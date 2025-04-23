import streamlit as st
import torch
from torchvision import models, transforms
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import cv2
import numpy as np
import os
from PIL import Image

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
def detect_anomaly(vision_model, nlp_model, nlp_tokenizer, image_path, text, factory_name):
    vision_result = infer_image(vision_model, image_path)
    nlp_result = infer_text(nlp_model, nlp_tokenizer, text)
    
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

# Streamlit app
def main():
    st.set_page_config(page_title="ESCA Anomaly Detection", page_icon="🛰️")
    st.title("🛰️ ESCA: Factory Anomaly Detection")
    st.markdown("Upload a satellite image and enter a production record to detect anomalies in factory operations.")

    # Sidebar for inputs
    st.sidebar.header("Input Parameters")
    factory_name = st.sidebar.text_input("Factory Name", value="A")
    record_text = st.sidebar.text_area(
        "Production Record",
        value="Factory A claimed 1000 garments on 2025-04-12 with only 2 workers, no machinery details.",
        height=100
    )
    uploaded_file = st.sidebar.file_uploader("Upload Satellite Image (JPG)", type=["jpg", "jpeg"])

    # Load models
    vision_model_path = "/Users/yashmandaviya/esca/esca-mvp/models/resnet_anomaly.pth"
    nlp_model_path = "/Users/yashmandaviya/esca/esca-mvp/models/distilbert_verification"
    
    try:
        vision_model = load_vision_model(vision_model_path)
        nlp_model, nlp_tokenizer = load_nlp_model(nlp_model_path)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    # Process inputs
    if uploaded_file is not None and record_text:
        # Save uploaded image temporarily
        temp_image_path = "/Users/yashmandaviya/esca/esca-mvp/data/temp/temp_image.jpg"
        os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display inputs
        st.subheader("Input Data")
        col1, col2 = st.columns(2)
        with col1:
            st.image(temp_image_path, caption="Uploaded Satellite Image", use_column_width=True)
        with col2:
            st.write("**Factory Name**:")
            st.write(factory_name)
            st.write("**Production Record**:")
            st.write(record_text)
        
        # Run anomaly detection
        with st.spinner("Analyzing..."):
            try:
                result = detect_anomaly(
                    vision_model, nlp_model, nlp_tokenizer, temp_image_path, record_text, factory_name
                )
                
                # Display results
                st.subheader("Anomaly Detection Results")
                st.write(f"**Factory**: {result['factory']}")
                st.write(f"**Vision Result**: {result['vision_result']}")
                st.write(f"**NLP Result**: {result['nlp_result']}")
                
                # Color-coded risk level
                if result['risk_level'] == "High":
                    st.markdown(f"**Risk Level**: <span style='color:red'>{result['risk_level']}</span>", unsafe_allow_html=True)
                elif result['risk_level'] == "Medium":
                    st.markdown(f"**Risk Level**: <span style='color:orange'>{result['risk_level']}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Risk Level**: <span style='color:green'>{result['risk_level']}</span>", unsafe_allow_html=True)
                
                st.write(f"**Summary**: {result['message']}")
                
                # Clean up
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                    
            except Exception as e:
                st.error(f"Error during analysis: {e}")
    else:
        st.info("Please upload an image and enter a production record to analyze.")

if __name__ == "__main__":
    main()