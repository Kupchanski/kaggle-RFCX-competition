import os
import cv2
import glob
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.io import imread, imsave

F_MIN = 84
F_MAX = 15056
N_MELS = 320
AUDIO_LENGTH = 60  # sec

DATA_DIR = f"../data/mel_{N_MELS}_p2/train/"
CSV_PATH = "../data/train_tp_folds.csv"

def freq_to_y(f, n_mels=224):
    m_min = librosa.hz_to_mel(F_MIN)
    m_max = librosa.hz_to_mel(F_MAX)
    m = librosa.hz_to_mel(f)
    y = (m - m_min) / (m_max - m_min)
    return int(round((y * n_mels)))

def time_to_x(t, x_shape):
    return int(round((t / AUDIO_LENGTH * x_shape)))

def main():

    df = pd.read_csv(CSV_PATH)
    recodrings = df["recording_id"].unique()

    for recording_id in tqdm(recodrings):
        sub_df = df[df["recording_id"] == recording_id]
        image_path = os.path.join(DATA_DIR, recording_id + ".tif")
        image = imread(image_path)
        mask = np.zeros_like(image) - 1

        for i, row in sub_df.iterrows():
            
            id = row["recording_id"]
            cls = row["species_id"]
            t_min = row["t_min"]
            t_max = row["t_max"]
            f_min = row["f_min"]
            f_max = row["f_max"]

            x1 = time_to_x(t_min, image.shape[1])
            x2 = time_to_x(t_max, image.shape[1])
            y1 = freq_to_y(f_min, n_mels=N_MELS)
            y2 = freq_to_y(f_max, n_mels=N_MELS)

            mask[y1:y2, x1:x2] = cls

        mask_path = image_path.replace("train", "train_tp_masks")
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        imsave(mask_path, mask)

if __name__ == "__main__":
    main()