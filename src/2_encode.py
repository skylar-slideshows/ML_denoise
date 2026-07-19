############################################################

# EECE 5644 - Machine Learning
# Final Project - Speech Denoising with Neural Networks 

# Skylar Denno
# July 19, 2026

# 2_encode.py

# Encode audio files to 1D (mono) arrays of samples. STFT

############################################################

# REQUIRES: numpy, soundfile, soxr (Sound eXchange Resampler)

# OUTPUTS:


############################################################


import mlflow
import os
import sys
import numpy as np
import yaml
import soundfile as sf
import soxr
from scipy.signal import stft as scipy_stft


mlflow.set_tracking_uri("http://localhost:5000")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config", "2_encode.yaml")
  
def load_config(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("Config file not found")
    with open(path, "r") as fh:
        return yaml.safe_load(fh)
    

############################################################






# load_mono_16k: filepath -> 1D array of samples (mono, 16kHz)
def load_mono_16k(path, cfg):
    # read audio file as array of samples (x) and sample rate (sr)
    # always 2D to avoid different shapes for mono and stereo files
    x, sr = sf.read(path, always_2d=True)

    x = x.mean(axis=1).astype(np.float32) # downmix to mono, float 32 = less data for training

    # resample to target sample rate
    if sr != cfg["target_sr"]:
        x = soxr.resample(x, sr, cfg["target_sr"]).astype(np.float32)
    return x

# do_stft: 1D waveform -> complex spectrogram [F, T]
def do_stft(x, cfg):
    _, _, Z = scipy_stft(
        x,
        fs=cfg["target_sr"],
        window=cfg["window"],           # "hann"
        nperseg=cfg["n_fft"],           # 512
        noverlap=cfg["n_fft"] - cfg["hop"],   # 512-128 = 384
        nfft=cfg["n_fft"],
        boundary="zeros",
        padded=True,
    )
    return Z.astype(np.complex64)


def main():
    cfg = load_config(CONFIG_PATH)
