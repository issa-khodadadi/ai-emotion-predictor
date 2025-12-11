import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

class EmotionPredictor:
    def __init__(self):
        model_dir = "model_output"

        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        label_encoder_path = os.path.join(model_dir, "label_encoder.pt")
        safe_globals = [LabelEncoder]
        self.label_encoder = torch.load(label_encoder_path, map_location="cpu", weights_only=False)
        
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
        return self.label_encoder.inverse_transform([pred_idx])[0]
