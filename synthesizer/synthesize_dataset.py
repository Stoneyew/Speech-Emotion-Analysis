import os
import torch
from torch.utils.data import Dataset
import numpy as np

class SynthesizerDataset(Dataset):
    def __init__(self, text_files, mel_files, hparams):
        self.text_files = text_files
        self.mel_files = mel_files
        self.hparams = hparams

        assert len(text_files) == len(mel_files), "Text and mel files must be of the same length"

    def __len__(self):
        return len(self.text_files)

    def __getitem__(self, idx):
        text = np.load(self.text_files[idx])
        mel = np.load(self.mel_files[idx])
        return {"text": torch.tensor(text, dtype=torch.long), "mel": torch.tensor(mel, dtype=torch.float32)}

def collate_fn(batch):
    # Get the maximum length of text and mel in the batch
    max_text_len = max(len(item['text']) for item in batch)
    max_mel_len = max(item['mel'].shape[0] for item in batch)
    
    # Pad sequences to the maximum length
    for item in batch:
        text_len = len(item['text'])
        mel_len = item['mel'].shape[0]
        
        # Pad text
        text_pad = torch.zeros(max_text_len, dtype=torch.long)
        text_pad[:text_len] = item['text']
        item['text'] = text_pad
        
        # Pad mel
        mel_pad = torch.zeros(max_mel_len, item['mel'].shape[1], dtype=torch.float32)
        mel_pad[:mel_len, :] = item['mel']
        item['mel'] = mel_pad
    
    # Stack all items in the batch
    texts = torch.stack([item['text'] for item in batch])
    mels = torch.stack([item['mel'] for item in batch])
    
    return {"text": texts, "mel": mels}

def get_file_paths(directory, extension=".npy"):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]
