import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def load_model(model_path):
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

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

if __name__ == "__main__":
    model_path = "/Users/yashmandaviya/esca/esca-mvp/models/distilbert_verification"
    # Test with a sample suspicious record
    test_text = "Factory C claimed 1000 garments on 2025-04-12 with only 2 workers, no machinery details."
    try:
        model, tokenizer = load_model(model_path)
        result = infer_text(model, tokenizer, test_text)
        print(f"Prediction: {result}")
    except Exception as e:
        print(f"Error: {e}")