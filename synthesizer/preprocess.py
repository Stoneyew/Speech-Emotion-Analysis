# synthesizer/preprocess.py
import os
import numpy as np
from audio import load_audio, melspectrogram
from hparams import HParams
from vocab import vocab

def preprocess_data(input_dir, output_dir, hparams):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    audio_dir = os.path.join(input_dir, "198")
    text_dir = os.path.join(input_dir,"preprocess")
    mel_dir = os.path.join(output_dir, "mel")
    text_seq_dir = os.path.join(output_dir, "text_seq")

    if not os.path.exists(mel_dir):
        os.makedirs(mel_dir)
    if not os.path.exists(text_seq_dir):
        os.makedirs(text_seq_dir)

    missing_files = []
    
    for file_name in os.listdir(audio_dir):
        if file_name.endswith(".flac"):
            audio_path = os.path.join(audio_dir, file_name)
            text_path = os.path.join(text_dir, file_name.replace('.flac', '.txt'))

            if not os.path.exists(text_path):
                missing_files.append(file_name)
                continue

            # Load and preprocess audio
            audio, _ = load_audio(audio_path, sr=hparams.tacotron_sample_rate)
            mel = melspectrogram(audio, sr=hparams.tacotron_sample_rate, n_mels=hparams.tacotron_num_mels)

            # Save mel spectrogram
            mel_file_path = os.path.join(mel_dir, file_name.replace('.flac', '.npy'))
            np.save(mel_file_path, mel.T)  # Transpose to match expected shape (time, mel_dim)

            # Load and preprocess text
            with open(text_path, 'r') as f:
                text = f.read().strip()
            text_sequence = vocab.text_to_sequence(text)

            # Save text sequence
            text_seq_file_path = os.path.join(text_seq_dir, file_name.replace('.flac', '.npy'))
            np.save(text_seq_file_path, text_sequence)

    if missing_files:
        print(f"Missing text files for the following audio files: {missing_files}")


input_dir = '/Users/angus/Desktop/college/Intro_to_ai/Real-time-voice-cloning/19/19'
output_dir = '/Users/angus/Desktop/college/Intro_to_ai/Real-time-voice-cloning/19/19'
hparams = HParams()
preprocess_data(input_dir, output_dir, hparams)