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
#    python src/1_preprocess.py \
#        --clean-in  [path_to_folder_of_clean_files] \
#        --noisy-in  [path_to_folder_of_noisy_files] \
#        --clean-out [path_to_output_folder_for_clean_files] \
#        --noisy-out [path_to_output_folder_for_noisy_files] \
#        --report    [path_to_save_report_csv]/report.csv

############################################################


import mlflow
import os
import sys
import shutil
import numpy as np
import soundfile as sf
import csv
import argparse

mlflow.set_tracking_uri("http://localhost:5000")


# FILE FILTERING CONFIG:
MIN_DURATION = 1.0 # minimum file duration in seconds
MIN_DBFS = -40.0 # minimum signal level in decibels full scale
CLIP_DBFS = -0.1 # maximum signal level before flagging as clipping
CLIP_FRAC = 0.001 # maximum fraction of samples allowed to exceed CLIP_DBFS
MAX_DC_OFFSET = 0.05 # maximum allowed DC offset in the signal
EXPECTED_CHANNELS = 1 # expected number of audio channels (mono)
EXPECTED_SR = 48000 # expected sample rate of the audio files 48kHz

# which flags cause a file (and therefore its whole pair) to be dropped.
DROP_FLAGS = {"short", "quiet", "clipping", "offset", "bad_sr"}


############################################################


# gives the decibel full scale (dBFS) value of a signal amplitude
def dbfs(x):
    x = float(x)
    return -np.inf if x <= 0 else 20.0 * np.log10(x)


############################################################


# flag_issues: filepath -> flags, dictionary of file metadata
def flag_issues(path):

    # read audio file as an array of samples
    x, sr = sf.read(path, always_2d=True)

    flags = []

    # flag if sample rate is not as expected
    if sr != EXPECTED_SR:
        flags.append("bad_sr")

    # flag if file is not mono (rows > 1) - does not drop the file
    if x.shape[1] != EXPECTED_CHANNELS:
        flags.append(f"channels={x.shape[1]}")

    # flag short if the duration is below the minimum threshold
    duration = x.shape[0] / sr
    if duration < MIN_DURATION:
        flags.append("short")

    # mono for level checks
    mono = x.mean(axis=1)

    # flag quiet if the RMS level is below the minimum threshold
    rms = np.sqrt(np.mean(mono**2))
    rms_db = dbfs(rms)
    if rms_db < MIN_DBFS:
        flags.append("quiet")

    peak = np.max(np.abs(mono))
    clip_frac = np.mean(np.abs(mono) >= 0.999)

    # flag clipping if the peak is above the clipping threshold and the fraction of clipped samples is too high
    peak_db = dbfs(peak)
    if peak_db > CLIP_DBFS and clip_frac > CLIP_FRAC:
        flags.append("clipping")

    # flag DC offset if the average of the samples is more than max allowed offset
    dc = float(np.mean(mono))
    if abs(dc) > MAX_DC_OFFSET:
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
def fails(flags):
    return any(f in DROP_FLAGS for f in flags)


############################################################
 
 
def main():

    # parse CLI call
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-in", required=True)
    ap.add_argument("--noisy-in", required=True)
    ap.add_argument("--clean-out", required=True)
    ap.add_argument("--noisy-out", required=True)
    ap.add_argument("--report", required=True, default="report.csv")
    args = ap.parse_args()
 
    # make folders for output files
    os.makedirs(args.clean_out, exist_ok=True)
    os.makedirs(args.noisy_out, exist_ok=True)
 
    # make sure clean/noisy pairs for training exist for each file
    clean_names = {f for f in os.listdir(args.clean_in) if f.lower().endswith(".wav")}
    noisy_names = {f for f in os.listdir(args.noisy_in) if f.lower().endswith(".wav")}
    paired = sorted(clean_names & noisy_names) # names present in both clean and noisy folder
    unpaired = (clean_names ^ noisy_names) # names in only one folder -> excluded
 
    if unpaired:
        print(f"WARNING: {len(unpaired)} unpaired files (no partner) will be skipped.")
 
    # CREATE PROCESSING REPORT
    report_rows = []
    kept, dropped = 0, 0
    drop_reasons = {}
 
    for name in paired:
        c_path = os.path.join(args.clean_in, name)
        n_path = os.path.join(args.noisy_in, name)
        c_flags, c_stats = flag_issues(c_path)
        n_flags, n_stats = flag_issues(n_path)
 
        c_fail, n_fail = fails(c_flags), fails(n_flags)
        pair_ok = not (c_fail or n_fail)
 
        report_rows.append(dict(set="clean", file=name, kept=pair_ok,
                                flags="|".join(c_flags), **c_stats))
        report_rows.append(dict(set="noisy", file=name, kept=pair_ok,
                                flags="|".join(n_flags), **n_stats))
 
        if pair_ok:
            shutil.copy2(c_path, os.path.join(args.clean_out, name))
            shutil.copy2(n_path, os.path.join(args.noisy_out, name))
            kept += 1
        else:
            dropped += 1
            # tally why (union of drop-flags from either file)
            reasons = {f for f in (c_flags + n_flags) if f in DROP_FLAGS}
            for r in reasons:
                drop_reasons[r] = drop_reasons.get(r, 0) + 1
 
    # write report file
    if report_rows:
        fields = ["set", "file", "kept", "flags", "sr", "channels",
                  "duration_s", "rms_dbfs", "peak_dbfs", "dc_offset", "clip_frac"]
        with open(args.report, "w", newline="") as fh:
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
    print(f"\nCopied to:\n  {args.clean_out}\n  {args.noisy_out}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()