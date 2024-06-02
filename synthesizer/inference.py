import os
import torch
import numpy as np
import librosa
from tacotron import Tacotron2WaveNet
from hparams import HParams
from vocab import vocab

class Synthesizer:
    def __init__(self, tacotron_config, wavenet_config, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Tacotron2WaveNet(tacotron_config, wavenet_config).to(self.device)
        self.load_model(checkpoint_path)
        self.model.eval()

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")

    def synthesize(self, text):
        text_sequence = torch.tensor([vocab.text_to_sequence(text)]).to(self.device)
        mel_outputs, linear_outputs, attn_scores, stop_tokens = self.model.tacotron2.generate(text_sequence)
        waveform = self.model.wavenet(mel_outputs)
        return waveform.squeeze().cpu().detach().numpy()

    def save_wav(self, wav, path, sample_rate):
        librosa.output.write_wav(path, wav, sr=sample_rate)
        print(f"Saved wav file at {path}")

if __name__ == "__main__":
    hparams = HParams()
    checkpoint_path = "path/to/checkpoints/tacotron2_wavenet.pth"
    output_wav_path = "output.wav"
    sample_rate = 22050

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

    text = "Your input text for synthesis."

    synthesizer = Synthesizer(tacotron_config, wavenet_config, checkpoint_path)
    wav = synthesizer.synthesize(text)
    synthesizer.save_wav(wav, output_wav_path, sample_rate)
