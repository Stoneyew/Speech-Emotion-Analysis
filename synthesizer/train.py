# synthesizer/train.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tacotron import Tacotron2WaveNet
from synthesize_dataset import SynthesizerDataset, get_file_paths, collate_fn
from hparams import HParams
from preprocess import preprocess_data
from vocab import vocab

def train_model(model, dataset, hparams, checkpoint_path):
    dataloader = DataLoader(dataset, batch_size=hparams.batch_size, shuffle=True, collate_fn=collate_fn)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    model.train()
    
    for epoch in range(hparams.epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            text, mel_inputs = batch['text'], batch['mel']
            outputs = model(text, mel_inputs)
            loss = criterion(outputs[0], mel_inputs)  # Adjust this line based on your output structure
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        # Save model checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Free up GPU memory

def validate_model(model, dataset, criterion):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            text, mel_inputs = batch['text'], batch['mel']
            outputs = model(text, mel_inputs)
            loss = criterion(outputs[0], mel_inputs)  # Adjust this line based on your output structure
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate_mos(model, dataset):
    # Placeholder for MOS evaluation
    # In a real-world scenario, this function would collect MOS scores from human listeners
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    model.eval()
    mos_scores = []
    with torch.no_grad():
        for batch in dataloader:
            text, mel_inputs = batch['text'], batch['mel']
            outputs = model(text, mel_inputs)
            # For simplicity, assume all outputs have a MOS of 4.0
            mos_scores.extend([4.0] * len(outputs[0]))
    
    avg_mos = sum(mos_scores) / len(mos_scores)
    return avg_mos

if __name__ == "__main__":
    hparams = HParams()

    # Adjust learning rate
    hparams.learning_rate = 0.0001  # Reduce learning rate

    # Paths to input data directories
    input_dir = "/Users/angus/Desktop/college/Intro_to_ai/Real-time-voice-cloning/LibriSpeech/train-clean-100/19"
    output_dir = "/Users/angus/Desktop/college/Intro_to_ai/Real-time-voice-cloning/19/19"
    checkpoint_path = "/Users/angus/Desktop/college/Intro_to_ai/Real-time-voice-cloning/synthesizer/models/tacotron2_wavenet.pth"
    
    # Ensure checkpoint directory exists
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path))
    
    # Preprocess data
    #preprocess_data(input_dir, output_dir, hparams)

    # Paths to preprocessed data
    train_text_files = get_file_paths(os.path.join(output_dir, "text_seq"))
    train_mel_files = get_file_paths(os.path.join(output_dir, "mel"))

    # Ensure the number of text and mel files match
    assert len(train_text_files) == len(train_mel_files), "Text and mel files must be of the same length"

    # Create datasets
    train_dataset = SynthesizerDataset(train_text_files, train_mel_files, hparams)
    
    # Split dataset into training and validation sets
    val_split = 0.1
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Initialize models
    tacotron_config = {
        "embed_dims": 512,
        "num_chars": len(vocab.characters),
        "encoder_dims": 512,
        "decoder_dims": 512,
        "n_mels": hparams.tacotron_num_mels,
        "fft_bins": 1025,
        "postnet_dims": 512,
        "encoder_K": 16,
        "lstm_dims": 1024,
        "postnet_K": 8,
        "num_highways": 4,
        "dropout": 0.5,
        "stop_threshold": 0.5,
        "speaker_embedding_size": 128,
    }

    wavenet_config = {
        "in_channels": 1,
        "out_channels": 1,
        "residual_channels": 512,
        "gate_channels": 512,
        "skip_channels": 256,
        "aux_channels": 80,
        "num_blocks": 3,
        "num_layers": 10,
        "kernel_size": 3,
        "dropout": 0.05,
    }

    model = Tacotron2WaveNet(tacotron_config, wavenet_config)
    
    # Train model
    train_model(model, train_dataset, hparams, checkpoint_path)
    
    # Validate model
    criterion = torch.nn.MSELoss()
    val_loss = validate_model(model, val_dataset, criterion)
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Evaluate model using MOS
    mos_score = evaluate_mos(model, val_dataset)
    print(f"MOS: {mos_score:.2f}")


