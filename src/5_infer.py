############################################################

# EECE 5644 - Machine Learning
# Final Project - Speech Denoising with Neural Networks

# Skylar Denno
# July 26, 2026

# 5_infer.py

# Run the trained LSTM denoiser on a noisy audio file (WAV) and produce a denoised version.

############################################################

# USAGE:
#    python src/5_infer.py --input path/to/noisy.wav --output path/to/denoised.wav

############################################################

import os
import argparse
import numpy as np
import yaml
import torch
import torch.nn as nn
import soundfile as sf
import soxr
from scipy.signal import stft as scipy_stft
from scipy.signal import istft as scipy_istft
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config", "5_infer.yaml")


def load_config(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def parse_args():
    parser = argparse.ArgumentParser(description="Run speech denoising inference on one audio file")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to inference config YAML")
    parser.add_argument("--input", dest="input_wav", default=None, help="Input noisy WAV path")
    parser.add_argument("--output", dest="output_wav", default=None, help="Output denoised WAV path")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Optional device override",
    )
    return parser.parse_args()


############################################################


# I/O and setupfunctions:

# instantiate lstm
class LSTMDenoiser(nn.Module):
    def __init__(self, n_freq, hidden_size, num_layers, dropout, prediction_target="direct_mag", mask_max=1.0):
        super().__init__()
        self.prediction_target = str(prediction_target)
        self.mask_max = float(mask_max)
        self.lstm = nn.LSTM(
            input_size=n_freq,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        if self.prediction_target == "irm_mask":
            self.head = nn.Sequential(
                nn.Linear(hidden_size * 2, n_freq),
                nn.Sigmoid(),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_size * 2, n_freq),
                nn.Softplus(),
            )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.head(out)
        if self.prediction_target == "irm_mask":
            out = out * self.mask_max
        return out


# load the user's wav file and resample it if needed
def load_mono_resampled(path, target_sr):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input audio file not found. Path set in config is {path}")

    x, sr = sf.read(path, always_2d=True)
    x = x.mean(axis=1).astype(np.float32)

    if sr != target_sr:
        x = soxr.resample(x, sr, target_sr).astype(np.float32)

    return x


# stft encode
def do_stft(x, cfg):
    _, _, z = scipy_stft(
        x,
        fs=cfg["target_sr"],
        window=cfg["win"],
        nperseg=cfg["n_fft"],
        noverlap=cfg["n_fft"] - cfg["n_hop"],
        nfft=cfg["n_fft"],
        boundary="zeros",
        padded=True,
    )
    return z.astype(np.complex64)


# inverse stft
def do_istft(z, cfg):
    _, x = scipy_istft(
        z,
        fs=cfg["target_sr"],
        window=cfg["win"],
        nperseg=cfg["n_fft"],
        noverlap=cfg["n_fft"] - cfg["n_hop"],
        nfft=cfg["n_fft"],
        input_onesided=True,
        boundary=True,
    )
    return x.astype(np.float32)


def infer_model_dims_from_checkpoint(ckpt):
    state = ckpt["model_state_dict"]

    if "head.0.weight" not in state:
        raise KeyError("Checkpoint model_state_dict missing head.0.weight")

    n_freq = int(state["head.0.weight"].shape[0])
    hidden_size = int(state["lstm.weight_ih_l0"].shape[0] // 4)

    num_layers = 0
    while f"lstm.weight_ih_l{num_layers}" in state:
        num_layers += 1

    return n_freq, hidden_size, num_layers


# load model from best training checkpoint (MUST BE SET IN CONFIG)
def load_model(checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    train_cfg = ckpt.get("config", {})

    n_freq, hidden_size_inferred, num_layers_inferred = infer_model_dims_from_checkpoint(ckpt)
    hidden_size = int(train_cfg.get("hidden_size", hidden_size_inferred))
    num_layers = int(train_cfg.get("num_layers", num_layers_inferred))
    dropout = float(train_cfg.get("dropout", 0.0))
    mask_max = float(train_cfg.get("mask_max", 1.0))
    prediction_target = train_cfg.get("prediction_target", "direct_mag")

    model = LSTMDenoiser(
        n_freq=n_freq,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        prediction_target=prediction_target,
        mask_max=mask_max,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, prediction_target


############################################################


def denoise_waveform(model, noisy_wave, cfg, device, prediction_target="direct_mag"):
    peak = float(np.max(np.abs(noisy_wave)) + 1e-8)
    gain = 1.0 / peak
    noisy_norm = noisy_wave * gain

    z_noisy = do_stft(noisy_norm, cfg)

    noisy_log_mag = np.log1p(np.abs(z_noisy)).T.astype(np.float32)
    t_len = noisy_log_mag.shape[0]

    x = torch.from_numpy(noisy_log_mag).unsqueeze(0).to(device)
    lengths = torch.tensor([t_len], dtype=torch.long, device=device)

    with torch.no_grad():
        pred_out = model(x, lengths)[0, :t_len, :].cpu().numpy().T

    if prediction_target == "irm_mask":
        noisy_mag = np.abs(z_noisy).astype(np.float32)
        pred_mag = pred_out.astype(np.float32) * noisy_mag
    else:
        pred_mag = np.expm1(pred_out).astype(np.float32)
        pred_mag = np.maximum(pred_mag, 0.0)

    noisy_phase = np.angle(z_noisy)
    z_hat = pred_mag * np.exp(1j * noisy_phase)

    denoised_norm = do_istft(z_hat, cfg)

    if denoised_norm.shape[0] > noisy_norm.shape[0]:
        denoised_norm = denoised_norm[: noisy_norm.shape[0]]
    elif denoised_norm.shape[0] < noisy_norm.shape[0]:
        denoised_norm = np.pad(denoised_norm, (0, noisy_norm.shape[0] - denoised_norm.shape[0]))

    denoised = denoised_norm / gain

    if cfg.get("normalize_output_peak", True):
        out_peak = float(np.max(np.abs(denoised)) + 1e-8)
        if out_peak > 1.0:
            denoised = denoised / out_peak

    return denoised.astype(np.float32)


############################################################


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    input_wav = args.input_wav or cfg.get("input_wav")
    output_wav = args.output_wav or cfg.get("output_wav")
    checkpoint = args.checkpoint or cfg.get("checkpoint")

    if input_wav is None or output_wav is None or checkpoint is None:
        raise ValueError("input_wav, output_wav, and checkpoint must be set via args or config")

    print("#### Running denoising inference ####")
    print(f"Input file: {input_wav}")
    print(f"Output file: {output_wav}")
    print(f"Checkpoint: {checkpoint}")

    noisy = load_mono_resampled(input_wav, cfg["target_sr"])
    model, prediction_target = load_model(checkpoint, device)
    denoised = denoise_waveform(model, noisy, cfg, device, prediction_target=prediction_target)

    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    sf.write(output_wav, denoised, cfg["target_sr"])

    print("\n#### Denoising complete ####")
    print(f"Wrote denoised audio to {output_wav}")


if __name__ == "__main__":
    main()
