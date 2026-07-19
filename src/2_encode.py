############################################################

# EECE 5644 - Machine Learning
# Final Project - Speech Denoising with Neural Networks 

# Skylar Denno
# July 19, 2026

# 1_preprocess.py

# Preprocessing and file checking script, quality control
# for audio data before encoding, training

############################################################

# REQUIRES: numpy, soundfile, soxr (Sound eXchange Resampler)

# OUTPUTS:


############################################################



import mlflow
import os
import sys
import numpy as np
import soundfile as sf
import csv
import argparse
import soxr

mlflow.set_tracking_uri("http://localhost:5000")



# FILE FILTERING CONFIG
MIN_DURATION = 1.0 # minimum file duration in seconds
MIN_DBFS = -40.0 # minimum signal level in decibels full scale
CLIP_DBFS = -0.1 # maximum signal level before flagging as clipping
CLIP_FRAC = 0.001 # maximum fraction of samples allowed to exceed CLIP_DBFS
MAX_DC_OFFSET = 0.05 # maximum allowed DC offset in the signal
EXPECTED_CHANNELS = 1 # expected number of audio channels (mono)
EXPECTED_SR = 48000 # expected sample rate of the audio files 48kHz
TARGET_SR = 16000 # target sample rate for the audio files 16kHz downsample (max audio freq = 8kHz)
N_FFT = 512 # number of samples for the STFT window
HOP = 128 # number of samples of overlap
WIN = "hann" # DSP window type for the FFT
EPS = 1e-8 # avoid log 0



# gives the decibel full scale (dBFS) value of a signal amplitude
def dbfs(x):
    x = float(x)
    return -np.inf if x <= 0 else 20.0 * np.log10(x)


# load_mono_16k: filepath -> 1D array of samples (mono, 16kHz)
def load_mono_16k(path):
    # read audio file as array of samples (x) and sample rate (sr)
    # always 2D to avoid different shapes for mono and stereo files
    x, sr = sf.read(path, always_2d=True)

    x = x.mean(axis=1).astype(np.float32) # downmix to mono, float 32 = less data for training

    # resample to target sample rate
    if sr != TARGET_SR:
        x = soxr.resample(x, sr, TARGET_SR).astype(np.float32)
    return x


# flag_issues: 1D array of samples -> sys.flags
def flag_issues(x):

    # flag short if the duration is below the minimum threshold
    duration = x.shape[0] / TARGET_SR
    if duration < MIN_DURATION:
        sys.flags.append("short")

    # flag quiet if the RMS level is below the minimum threshold
    rms = np.sqrt(np.mean(x**2))
    rms_db = dbfs(rms)
    if rms_db < MIN_DBFS:
        sys.flags.append("quiet")

    peak = np.max(np.abs(x))
    clip_frac = np.mean(np.abs(x) >= 0.999)

    # flag clipping if the peak is above the clipping threshold and the fraction of clipped samples is too high
    peak_db = dbfs(peak)
    if peak_db > CLIP_DBFS and clip_frac > CLIP_FRAC:
        sys.flags.append("clipping")

    # flag DC offset if the average of the samples is more than max allowed offset
    dc = float(np.mean(x))
    if abs(dc) > MAX_DC_OFFSET:
        sys.flags.append("offset")

def main():
