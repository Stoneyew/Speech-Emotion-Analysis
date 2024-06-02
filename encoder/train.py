import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from transformer import load_pretrained_model
from config import Config

class AudioDataset(Dataset):
    def __init__(self, file_paths, tokenizer):
        self.file_paths = file_paths
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'r') as f:
            features_str = f.read()
        inputs = self.tokenizer.encode(features_str, max_length=Config.MAX_SEQ_LENGTH, truncation=True, padding='max_length')
        label = float(idx) / len(self.file_paths)  # Example: Simulating a regression target
        return torch.tensor(inputs).long(), torch.tensor(label).float()  # Return inputs and label

def load_pretrained_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer

def train_model():
    preprocessed_dir = '/Users/raycheng/Desktop/AI_final/19/198_preprocessed'
    file_paths = [os.path.join(preprocessed_dir, f) for f in os.listdir(preprocessed_dir) if f.endswith('.txt')]
    train_paths, val_paths = train_test_split(file_paths, test_size=0.2)
    
    tokenizer, model = load_pretrained_model()
    
    train_dataset = AudioDataset(train_paths, tokenizer)
    val_dataset = AudioDataset(val_paths, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # The first element is the loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs, labels=labels)
                loss = outputs[0]  # The first element is the loss
                val_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_dataloader)}, Validation Loss: {val_loss / len(val_dataloader)}")

    torch.save(model.state_dict(), Config.MODEL_PATH)

# Call this function to start training
if __name__ == "__main__":
    train_model()
