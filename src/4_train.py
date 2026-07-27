############################################################

# EECE 5644 - Machine Learning
# Final Project - Speech Denoising with Neural Networks

# Skylar Denno
# July 26, 2026

# 4_train.py

# Train the speech denoising neural network using the prepared dataset.

############################################################

# USAGE:
#    python src/4_train.py

############################################################


import os
import argparse
import numpy as np
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config", "4_train.yaml")

# init/utilty functions:

def load_config(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as fh:
        return yaml.safe_load(fh)

# test mode overrides to make it run really fast (only if test_mode enabled)
def apply_test_overrides(cfg):
    overrides = cfg.get("test_overrides", {})
    merged = dict(cfg)
    merged.update(overrides)
    return merged

def read_split_list(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path, "r") as fh:
        names = [line.strip() for line in fh if line.strip()]
    if len(names) == 0:
        raise ValueError(f"Split file is empty: {path}")
    return names

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM speech denoiser")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable fast test mode with lightweight settings",
    )
    return parser.parse_args()


############################################################

# CLASS: For loading sequences from the cached STFT NPZ files (/data/encode/cache)
class SpectrogramDataset(Dataset):
    
    def __init__(self, cache_dir, file_names, max_frames=None, random_crop=False):
        self.cache_dir = cache_dir
        self.file_names = file_names
        self.max_frames = max_frames
        self.random_crop = random_crop

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        path = os.path.join(self.cache_dir, name)

        # failsafes
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Cached feature file not found: {path}")

        data = np.load(path)
        if "noisy_stft" not in data or "clean_stft" not in data:
            raise KeyError(f"Expected keys clean_stft and noisy_stft in {path}")

        noisy_stft = data["noisy_stft"]
        clean_stft = data["clean_stft"]

        # model takes [T, F]: log-magnitude spectrums
        noisy_mag = np.log1p(np.abs(noisy_stft)).T.astype(np.float32)
        clean_mag = np.log1p(np.abs(clean_stft)).T.astype(np.float32)

        if self.max_frames is not None and noisy_mag.shape[0] > self.max_frames:
            if self.random_crop:
                start = np.random.randint(0, noisy_mag.shape[0] - self.max_frames + 1)
            else:
                start = 0
            end = start + self.max_frames
            noisy_mag = noisy_mag[start:end]
            clean_mag = clean_mag[start:end]

        return {
            "x": torch.from_numpy(noisy_mag),
            "y": torch.from_numpy(clean_mag),
            "length": noisy_mag.shape[0],
        }

############################################################

# CLASS: for instantiating the LSTMs 
class LSTMDenoiser(nn.Module):
    def __init__(self, n_freq, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_freq,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, n_freq),
            nn.Softplus(),
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        return self.head(out)


############################################################


# Outer functions

# batched tensor - need to pad STFT sequences to the same length.
def collate_batch(batch):
    xs = [item["x"] for item in batch]
    ys = [item["y"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)

    x_padded = pad_sequence(xs, batch_first=True)
    y_padded = pad_sequence(ys, batch_first=True)

    return {
        "x": x_padded,
        "y": y_padded,
        "lengths": lengths,
    }


# compute loss only on valid frames and ignore right-padding
def masked_mse(pred, target, lengths):
    batch_size, max_t, _ = pred.shape
    time_ids = torch.arange(max_t, device=pred.device).unsqueeze(0).expand(batch_size, -1)
    mask = (time_ids < lengths.unsqueeze(1)).unsqueeze(2).float()
    sq = (pred - target) ** 2
    return (sq * mask).sum() / mask.sum().clamp(min=1.0)


# instantiate data loaders
def make_loader(cache_dir, split_path, batch_size, shuffle, num_workers, max_files=None, max_frames=None):
    names = read_split_list(split_path)
    if max_files is not None:
        names = names[:max_files]
    dataset = SpectrogramDataset(
        cache_dir,
        names,
        max_frames=max_frames,
        random_crop=shuffle,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )

# infer the number of frequency bins from the first sample in the split
def infer_n_freq(cache_dir, split_path):
    first_name = read_split_list(split_path)[0]
    path = os.path.join(cache_dir, first_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cached feature file not found: {path}")
    sample = np.load(path)
    if "noisy_stft" not in sample:
        raise KeyError(f"Expected noisy_stft key in {path}")
    return int(sample["noisy_stft"].shape[0])


############################################################


def run_epoch(model, loader, optimizer, device, grad_clip=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running = 0.0
    n_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            lengths = batch["lengths"].to(device)

            pred = model(x, lengths)
            loss = masked_mse(pred, y, lengths)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            running += float(loss.item())
            n_batches += 1

    return running / max(n_batches, 1)


############################################################


def main():

    print("#### Initializing training program ####\n")

    args = parse_args()
    cfg = load_config(CONFIG_PATH)

    # use test mode?
    if cfg.get("test_mode", True):
        cfg = apply_test_overrides(cfg)
        print("\n\n#### TEST MODE IS ENABLED PLEASE DON'T FORGET TO DISABLE IT!!!!!! ####\n\n")

    cache_dir = cfg["cache_dir"]
    train_split = cfg["train_split"]
    val_split = cfg["val_split"]
    out_dir = cfg.get("out_dir", "artifacts/train")
    os.makedirs(out_dir, exist_ok=True)

    random.seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))
    torch.manual_seed(cfg.get("seed", 42))

    # use best avaliable device for training
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(cfg.get("seed", 42))
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # make the data loaders for training and validation
    train_loader = make_loader(
        cache_dir=cache_dir,
        split_path=train_split,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=cfg.get("num_workers", 0),
        max_files=cfg.get("max_train_files"),
        max_frames=cfg.get("max_frames"),
    )

    val_loader = make_loader(
        cache_dir=cache_dir,
        split_path=val_split,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 0),
        max_files=cfg.get("max_val_files"),
        max_frames=cfg.get("max_frames"),
    )

    n_freq = infer_n_freq(cache_dir, train_split)

    model = LSTMDenoiser(
        n_freq=n_freq,
        hidden_size=cfg.get("hidden_size", 256),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)

    # adaptive moment estimator optimizer.
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    epochs = cfg.get("epochs", 30)
    grad_clip = cfg.get("grad_clip", 1.0)
    best_val = float("inf")


    #--------------------------------------------------------


    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    print(f"Output directory: {out_dir}")
    print(f"EPOCHS: {epochs}")

    print("\n\n#### Starting training loop ####\n")

    # training loop
    for epoch in range(1, epochs + 1):

        print(f"STARTING epoch {epoch} of {epochs}")

        train_loss = run_epoch(model, train_loader, optimizer, device, grad_clip=grad_clip)
        val_loss = run_epoch(model, val_loader, optimizer=None, device=device)

        print(f"COMPLETED epoch {epoch} of {epochs}")
        print(f"Train loss: {train_loss}")
        print(f"Validation loss: {val_loss}")

        # save checkpoint file
        ckpt_latest = os.path.join(out_dir, "latest.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config": cfg,
            },
            ckpt_latest,
        )

        if val_loss < best_val:
            best_val = val_loss
            ckpt_best = os.path.join(out_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": cfg,
                },
                ckpt_best,
            )
            print(f"!! NEW BEST checkpoint !!: {ckpt_best}")

        print("Checkpoint saved\n")

    print(f"\n\n#### Training complete with best validation loss {best_val}")


if __name__ == "__main__":
    main()
