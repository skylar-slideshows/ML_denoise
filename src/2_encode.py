############################################################

# EECE 5644 - Machine Learning
# Final Project - Speech Denoising with Neural Networks 

# Skylar Denno
# July 19, 2026

# 2_encode.py

# Encode audio files to 1D (mono) arrays of samples.
# Use short-time Fourier transform (STFT) to convert time-domain
# signals into frequency-domain representations
# The ML model will work directly with these.

############################################################

# REQUIRES: numpy, scipy, soundfile, soxr (Sound eXchange Resampler), mlflow

# OUTPUTS:

# USAGE:
#    python src/2_encode.py

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


############################################################


# do_stft: 1D waveform -> complex spectrogram [F, T]
def do_stft(x, cfg):
    _, _, Z = scipy_stft(
        x,
        fs=cfg["target_sr"],
        window=cfg["win"],
        nperseg=cfg["n_fft"],
        noverlap=cfg["n_fft"] - cfg["n_hop"],   # 512-128 = 384
        nfft=cfg["n_fft"],
        boundary="zeros",
        padded=True,
    )
    return Z.astype(np.complex64)


############################################################


def main():
    cfg = load_config(CONFIG_PATH)
    os.makedirs(cfg["cache_out"], exist_ok=True)

    # find the kept pairs (both folders have identical filenames after stage 1)
    names = sorted(f for f in os.listdir(cfg["clean_in"]) if f.lower().endswith(".wav"))

    for name in names:
        # 1. load + resample both
        clean = load_mono_16k(os.path.join(cfg["clean_in"], name), cfg)
        noisy = load_mono_16k(os.path.join(cfg["noisy_in"], name), cfg)

        # 2. match lengths (resampling can differ by a sample)
        m = min(len(clean), len(noisy))
        clean, noisy = clean[:m], noisy[:m]

        # 3. peak-normalize on noisy, same gain to clean (preserves the pair)
        peak = np.max(np.abs(noisy)) + 1e-8
        gain = 1.0 / peak
        clean, noisy = clean * gain, noisy * gain

        # 4. STFT both -> complex [F, T]
        S = do_stft(clean, cfg)
        Y = do_stft(noisy, cfg)

        # 5. align frame counts
        T = min(S.shape[1], Y.shape[1])
        S, Y = S[:, :T], Y[:, :T]

        # 6. cache both complex spectrograms + the gain
        out_path = os.path.join(cfg["cache_out"], name.replace(".wav", ".npz"))
        np.savez_compressed(out_path, clean_stft=S, noisy_stft=Y, gain=np.float32(gain))

if __name__ == "__main__":
    main()