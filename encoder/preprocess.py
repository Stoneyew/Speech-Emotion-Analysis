import os
import numpy as np
from audio import *
from config import *
BERT_MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5
SAMPLE_RATE = 22050
N_MFCC = 13
FRAME_SIZE = 512

def preprocess_data(input_dir, output_dir, config):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.flac'):
            file_path = os.path.join(input_dir, file_name)
            audio, _ = load_audio(file_path, sr=config.SAMPLE_RATE)
            features = extract_features(audio, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC)
            features_str = ' '.join(map(str, features.flatten()))
            output_file_path = os.path.join(output_dir, file_name.replace('.flac', '.txt'))
            with open(output_file_path, 'w') as f:
                f.write(features_str)

# Call this function 
preprocess_data('/Users/raycheng/Desktop/AI_final/19/198', '/Users/raycheng/Desktop/AI_final/19/198_preprocessed', Config)
