import torch
from hparams import HParams
def synthesize(text, tacotron_model, wavenet_model, Hparams):
    tacotron_output = tacotron_model(text)
    waveform = wavenet_model(tacotron_output)
    return waveform
