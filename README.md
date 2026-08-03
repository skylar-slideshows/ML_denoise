#### FILE ORGANIZATION
The full training pipeline is organized into 4 files in src/
These are:
   1_preprocess.py [Preprocessing / file validation (See FINAL_REPORT.pdf, section 2.1)]
   2_encode.py [Short-time Fourier transform (STFT) encoding and caching STFT arrays (see FINAL_REPORT.pdf, section 2.2)]
   3_make_split.py [Separate (by speaker / at the speaker level, so no speaker is included in both the training and validation
                    sets to avoid elevated validation scores). Creates data/encode/split/train_files.txt and Creates data/encode/split/val_files.txt]
   4_train.py [Data load-in and LSTM training loop, organized by epochs and including hyperparameter tuning]


#### DATASET
The dataset used for training is not included in the repository due to its size.
It is the Noisy speech database from the University of Edinburgh's DataShare program:
Download from: https://doi.org/10.7488/ds/2117

Only the 28 speaker dataset (folders named clean_trainset_28spk_wav and noisy_trainset_28spk_wav) are used
The software will look for data in (from the project root directory) data/raw/clean_trainset_28spk_wav and data/raw/noisy_trainset_28spk_wav

Also run mkdir data/preprocess and mkdir data/encode.


#### CONFIGURATION
Each python script has a corresponding config .yaml file in config/
For this dataset, expected_sr (expected sample rate) is 48000.
target_sr (Target sample rate) MUST be the same across all configs.

ALL input/output file paths may be changed in the config.
Short-time Fourier transform (STFT) encoding settings may be changed in 2_encode.yaml.
ALL training parameters (batch size, epochs, etc.) may be edited in 4_train.yaml.

The current config files are set up for a 44.1kHz sample rate (high-quality, bidirectional) training loop.
If run as is, the result at epoch 32 will output files with the same denoising performance as the current ones in WAVs/ (not 16k)

The random seed used in making the training/validation split is '42' by default. It may be changed in
3_make_split.yaml. Files are split at the speaker level (the same speaker will not appear in both the training and validation data),
and the current filename detection is based off the DataShare dataset's file naming conventions.


#### USAGE
From the project root directory:
   python src/1_preprocess.py
   python src/2_encode.py
   python src/3_make_split.py
   python src/4_train.py

OR, to run a full orchestration script that runs every script:
   python run.py


#### USAGE (INFER SCRIPT)
To run the completed, trained model on any .wav audio file, (noisy speech recording),
run 5_infer.py (python src/5_infer.py) or 5_infer_BATCH.py (python src/5_infer_BATCH.py).

The batch version runs the model on all files in a folder. The individual file inference time and
average time it takes to encode each file will be reported.

TO CHANGE THE .PT (PYTORCH ARTIFACT / CHECKPOINT FILE) BEING USED:
Edit the file path in 5_infer.yaml or 5_infer_BATCH.yaml.

All PyTorch artifact files, from every epoch of the 16kHz standard quality and 44.1kHz high quality runs
are in artifacts/. The current config uses the best epoch from the 44.1kHz run.

Once the PyTorch artifact is selected in the config, the input and output file locations may be chosen
when running the script using the optional arguments:
   python src/5_infer.py --input path/to/noisy.wav --output path/to/denoised.wav
   OR
   python src/5_infer_BATCH.py --input path/to/noisy/folder --output path/to/denoised/folder
For individual files and a folder of .wav files respectively

If you file is not in the .wav audio format (ogg, flac, mp3, m4a, etc.), this online free file converter may be used:
https://www.freeconvert.com/wav-converter

On most systems, the time it takes to infer the clean audio should be much shorter than the length of the audio itself,
even with the high quality 44.1kHz model.


#### CREDIT
CODE: This entire project was developed by Skylar Denno
There are no other group members or collaborators.
All code written by Skylar Denno.

DATASET: The dataset used is the property of University of Edinburgh, provided by their DataShare program and the
School of Informatics, Centre for Speech Technology Research (CSTR)
https://doi.org/10.7488/ds/2117


#### LICENSE
This project is open source, under the MIT License:

Copyright 2026 Skylar Denno

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
