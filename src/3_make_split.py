############################################################

# EECE 5644 - Machine Learning
# Final Project - Speech Denoising with Neural Networks 

# Skylar Denno
# July 19, 2026

# 3_make_split.py

# Encode audio files to 1D (mono) arrays of samples.
# Use short-time Fourier transform (STFT) to convert time-domain
# signals into frequency-domain representations
# The ML model will work directly with these.

############################################################

# USAGE:
#    python src/3_make_split.py

############################################################


import os
import numpy as np
import random
import yaml

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config", "3_make_split.yaml")

def load_config(path):
    with open(path, "r") as fh:
        return yaml.safe_load(fh)
    

############################################################


def speaker_of(filename):
    # "p226_001.npz" -> "p226"  (speaker ID is the first part of the filename)
    return filename.split("_")[0]


############################################################


def make_split(cfg, cache_dir, files):
    n_val_speakers = cfg.get("n_val_speakers", 4)
    seed = cfg.get("split_seed", 42)

    # group filenames by speaker
    by_speaker = {}
    for f in files:
        by_speaker.setdefault(speaker_of(f), []).append(f)
    speakers = sorted(by_speaker.keys())
    print(f"\nFound {len(files)} files from {len(speakers)} speakers.")

    # pick validation speakers deterministically (reproducible)
    rng = random.Random(seed)
    val_speakers = set(rng.sample(speakers, n_val_speakers))
    train_speakers = [s for s in speakers if s not in val_speakers]

    train_files, val_files = [], []
    for s in speakers:
        (val_files if s in val_speakers else train_files).extend(by_speaker[s])
    train_files.sort()
    val_files.sort()

    # write lists next to the cache
    train_path = os.path.join(cache_dir, "train_files.txt")
    val_path   = os.path.join(cache_dir, "val_files.txt")
    with open(train_path, "w") as fh:
        fh.write("\n".join(train_files) + "\n")
    with open(val_path, "w") as fh:
        fh.write("\n".join(val_files) + "\n")

    print("\n===== SPLIT =====")
    print(f"Val speakers ({n_val_speakers}): {sorted(val_speakers)}")
    print(f"Train: {len(train_files)} utterances from {len(train_speakers)} speakers")
    print(f"Val  : {len(val_files)} utterances from {len(val_speakers)} speakers")
    overlap = set(map(speaker_of, train_files)) & set(map(speaker_of, val_files))
    print(f"Speaker overlap (must be empty): {overlap}")
    print(f"\nWrote:\n  {train_path}\n  {val_path}")


############################################################


def main():
    cfg = load_config(CONFIG_PATH)
    cache_dir = cfg["cache_out"]
    split_dir = cfg["split_out"]
    # list the cache once, shared by both phases
    files = sorted(f for f in os.listdir(cache_dir)
                   if f.endswith(".npz") and not f.startswith("_"))

    files = [f for f in files]
    make_split(cfg, split_dir, files)


if __name__ == "__main__":
    main()