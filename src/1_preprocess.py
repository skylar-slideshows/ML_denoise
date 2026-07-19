############################################################

# EECE 5644 - Machine Learning
# Final Project - Speech Denoising with Neural Networks 

# Skylar Denno
# July 19, 2026

# 1_preprocess.py

# Preprocessing and file checking script, quality control
# for audio data before encoding, training

############################################################

# REQUIRES: numpy, soundfile, pyyaml

# OUTPUTS: 
#    Processed clean and noisy audio files in the specified output folders
#    CSV report summarizing the quality checks for each file

# USAGE:
#    python src/1_preprocess.py

############################################################


import os
import yaml
import shutil
import numpy as np
import soundfile as sf
import csv

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config", "1_preprocess.yaml")
  
def load_config(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("Config file not found")
    with open(path, "r") as fh:
        return yaml.safe_load(fh)

    

############################################################


# gives the decibel full scale (dBFS) value of a signal amplitude
def dbfs(x):
    x = float(x)
    return -np.inf if x <= 0 else 20.0 * np.log10(x)


############################################################


# flag_issues: filepath -> flags, dictionary of file metadata
def flag_issues(path, cfg):

    # read audio file as an array of samples
    x, sr = sf.read(path, always_2d=True)

    flags = []

    # flag if sample rate is not as expected
    if sr != cfg["expected_sr"]:
        flags.append("bad_sr")

    # flag if file is not mono (rows > 1) - does not drop the file
    if x.shape[1] != cfg["expected_channels"]:
        flags.append(f"channels={x.shape[1]}")

    # flag short if the duration is below the minimum threshold
    duration = x.shape[0] / sr
    if duration < cfg["min_duration"]:
        flags.append("short")

    # mono for level checks
    mono = x.mean(axis=1)

    # flag quiet if the RMS level is below the minimum threshold
    rms = np.sqrt(np.mean(mono**2))
    rms_db = dbfs(rms)
    if rms_db < cfg["min_dbfs"]:
        flags.append("quiet")

    peak = np.max(np.abs(mono))
    clip_frac = np.mean(np.abs(mono) >= 0.999)

    # flag clipping if the peak is above the clipping threshold and the fraction of clipped samples is too high
    peak_db = dbfs(peak)
    if peak_db > cfg["clip_dbfs"] and clip_frac > cfg["clip_frac"]:
        flags.append("clipping")

    # flag DC offset if the average of the samples is more than max allowed offset
    dc = float(np.mean(mono))
    if abs(dc) > cfg["max_dc_offset"]:
        flags.append("offset")

    stats = dict(
        sr=sr,
        channels=x.shape[1],
        duration_s=round(duration, 3),
        rms_dbfs=round(rms_db, 2),
        peak_dbfs=round(peak_db, 2),
        dc_offset=round(dc, 5),
        clip_frac=round(clip_frac, 6),
    )

    return flags, stats


############################################################


# fails: flags -> boolean
def fails(flags, cfg):
    return any(f in cfg["drop_flags"] for f in flags)


############################################################
 
 
def main():

    cfg = load_config(CONFIG_PATH)
 
    # make folders for output files
    os.makedirs(cfg["clean_out"], exist_ok=True)
    os.makedirs(cfg["noisy_out"], exist_ok=True)
 
    # make sure clean/noisy pairs for training exist for each file
    clean_names = {f for f in os.listdir(cfg["clean_in"]) if f.lower().endswith(".wav")}
    noisy_names = {f for f in os.listdir(cfg["noisy_in"]) if f.lower().endswith(".wav")}
    paired = sorted(clean_names & noisy_names) # names present in both clean and noisy folder
    unpaired = (clean_names ^ noisy_names) # names in only one folder -> excluded
 
    if unpaired:
        print(f"WARNING: {len(unpaired)} unpaired files (no partner) will be skipped.")
 
    # CREATE PROCESSING REPORT
    report_rows = []
    kept, dropped = 0, 0
    drop_reasons = {}
 
    for name in paired:
        c_path = os.path.join(cfg["clean_in"], name)
        n_path = os.path.join(cfg["noisy_in"], name)
        c_flags, c_stats = flag_issues(c_path, cfg)
        n_flags, n_stats = flag_issues(n_path, cfg)
 
        c_fail, n_fail = fails(c_flags, cfg), fails(n_flags, cfg)
        pair_ok = not (c_fail or n_fail)
 
        report_rows.append(dict(set="clean", file=name, kept=pair_ok,
                                flags="|".join(c_flags), **c_stats))
        report_rows.append(dict(set="noisy", file=name, kept=pair_ok,
                                flags="|".join(n_flags), **n_stats))
 
        if pair_ok:
            shutil.copy2(c_path, os.path.join(cfg["clean_out"], name))
            shutil.copy2(n_path, os.path.join(cfg["noisy_out"], name))
            kept += 1
        else:
            dropped += 1
            # tally why (union of drop-flags from either file)
            reasons = {f for f in (c_flags + n_flags) if f in cfg["drop_flags"]}
            for r in reasons:
                drop_reasons[r] = drop_reasons.get(r, 0) + 1
 
    # write report file
    if report_rows:
        fields = ["set", "file", "kept", "flags", "sr", "channels",
                  "duration_s", "rms_dbfs", "peak_dbfs", "dc_offset", "clip_frac"]
        with open(cfg["report"], "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(report_rows)
 
    # print summary to CLI
    print("\n===== SUMMARY =====")
    print(f"Paired inputs      : {len(paired)}")
    print(f"Pairs KEPT         : {kept}")
    print(f"Pairs DROPPED      : {dropped}")
    print(f"Unpaired skipped   : {len(unpaired)}")
    if drop_reasons:
        print("Drop reasons (pairs affected):")
        for r, c in sorted(drop_reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {r:12s}: {c}")
    print(f"\nCopied to:\n  {cfg['clean_out']}\n  {cfg['noisy_out']}")
    print(f"Report: {cfg['report']}")


if __name__ == "__main__":
    main()