import librosa
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
import torch
import scipy.fftpack as fft
import torch.nn as nn
import copy
import pickle
import os 
folder_name = "10s_clips/AUDIO_CLEAN/"
folders = os.listdir(folder_name)

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 512
def audio_to_log_mel(
    file_path: str,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """
    Load audio and convert to standardized log-Mel spectrogram.
    Output shape: (n_mels, time_frames)
    """
    try:
        y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {file_path}\n{e}")

    if len(y) == 0:
        raise ValueError(f"Empty audio file: {file_path}")

    # Normalize waveform
    max_val = np.max(np.abs(y)) + 1e-9
    y = y / max_val

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Per-sample normalization
    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-9
    log_mel = (log_mel - mean) / std

    return log_mel.astype(np.float32)


def cqcc_extract(filename):
    audio,sr = librosa.load(filename)
    cqt = np.abs(librosa.cqt(audio, sr=sr))

    log_cqt = librosa.amplitude_to_db(cqt)

    cqcc = fft.dct(log_cqt, axis=0, type=2, norm='ortho')
    return cqcc

data_array = []
for label,folder in enumerate(folders):
    if folder==".DS_Store":
        continue
    files = os.listdir(folder_name+folder)
    for audio_name in files:
        print(audio_name)
        cqcc_val  = audio_to_log_mel(folder_name+folder+"/"+audio_name)
        bot = label
        print(cqcc_val.shape,bot)
        data_array.append ([cqcc_val,bot])

save_file = open("10s_log_mel_unshuffle.pkl","wb")

pickle.dump(data_array, save_file)


import random
shuffled_array = random.sample(data_array, len(data_array))

save_file = open("10s_log_mel_shuffled.pkl","wb")
pickle.dump(shuffled_array, save_file)