import torch
from config import Config

def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def infer(model, features, tokenizer):
    inputs = tokenizer(features, return_tensors="pt", padding=True, truncation=True, max_length=Config.MAX_SEQ_LENGTH)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs
