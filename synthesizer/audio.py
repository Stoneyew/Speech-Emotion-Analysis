import numpy as np
import librosa

def load_audio(file_path, sr=22050):
    return librosa.load(file_path, sr=sr)

def melspectrogram(y, sr, n_mels=80, hop_length=256):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    return librosa.power_to_db(S, ref=np.max)
