from transformer_model import TransformerSER
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import librosa
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import tensorflow as tf
import warnings
import configs
warnings.filterwarnings("ignore")


class TQDMProgressBar(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.tqdm = tqdm(total=self.epochs, desc='Epoch', unit='epoch')
        
    def on_epoch_end(self, epoch, logs=None):
        self.tqdm.update(1)
        self.tqdm.set_postfix(logs)
        
    def on_train_end(self, logs=None):
        self.tqdm.close()

def extract_audio_features(audio_path, sr=22050, min_signal_length=1024, n_mfcc=39):
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Pad signal if it's too short
    if len(y) < min_signal_length:
        padding = min_signal_length - len(y)
        y = np.pad(y, (0, padding), mode='constant')
    
    n_fft = min(len(y), min_signal_length)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=n_fft//2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=n_fft//2)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=n_fft//2)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    
    features = np.vstack([mfccs, chroma, spectral_contrast, tonnetz])
    return features.T


def load_data(data_folder):
    X, y = [], []
    for emotion in os.listdir(data_folder):
        emotion_folder = os.path.join(data_folder, emotion)
        for file in os.listdir(emotion_folder):
            if file.endswith('.wav'):
                file_path = os.path.join(emotion_folder, file)
                features = extract_audio_features(file_path)
                X.append(features)
                y.append(emotion)
    return X, np.array(y)

def pad_features(features_list):
    max_len = max(len(feature) for feature in features_list)
    return pad_sequences(features_list, maxlen=max_len, padding='post', dtype='float32'), max_len


def main():
    config = {
        'data_folder': "datasets",
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 1e-4,
        'checkpoint_path': 'checkpoints',
        'checkpoint_name': 'ser_transformer_model',
    }
    
    # Load and preprocess data
    print('----- start data preprocessing -----')
    X, y = load_data(config['data_folder'])
    X, max_len = pad_features(X)  # Padding sequences to the same length

    # Encode the labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded, num_classes=len(le.classes_))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
    print("Unique labels in training set:", np.unique(y_train.argmax(axis=1)))
    print("Unique labels in test set:", np.unique(y_test.argmax(axis=1)))

    # Create transformer model using TransformerSER class
    input_shape = (X_train.shape[1], X_train.shape[2])  # Time steps and feature dimensions
    transformer_ser = TransformerSER.make(input_shape, num_heads=8, ff_dim=128, num_classes=len(le.classes_), learning_rate=configs.lr)

    # Train the model
    print(f'----- start training {transformer_ser.__class__.__name__} -----')
    transformer_ser.train(X_train, y_train, x_val=X_test, y_val=y_test, batch_size=config['batch_size'], n_epochs=config['epochs'])
    print(f'----- end training {transformer_ser.__class__.__name__} -----')

    # Evaluate the model
    print('----- start evaluating -----')
    transformer_ser.evaluate(X_test, y_test)
    print('----- end evaluating -----')

    # Save the trained model
    transformer_ser.save(configs.checkpoint_path, configs.checkpoint_name)

if __name__ == "__main__":
    main()
