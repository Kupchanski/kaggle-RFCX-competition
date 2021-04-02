import os
import glob
import librosa
import argparse
import numpy as np
import multiprocessing as mp
from skimage.io import imsave

from tqdm import tqdm

# configure params
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 320
SR = 48000
F_MIN = 84 
F_MAX = 15056
POWER = 2

DATA_DIR = "../data/"

def read_as_mel(path, p=1):
    audio, sr = librosa.load(path, sr=SR)   
    mel_spec = librosa.feature.melspectrogram(
        audio, n_fft=N_FFT, hop_length=HOP_LENGTH, sr=sr, 
        fmin=F_MIN, fmax=F_MAX, power=p, n_mels=N_MELS,
    )
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = mel_spec.astype("int8")
    return mel_spec

def convert(args):
    input_path, output_path = args
    mel = read_as_mel(input_path, p=POWER)
    imsave(output_path, mel)

def main(args):

    num_workers = mp.cpu_count()

    train_files = glob.glob(os.path.join(DATA_DIR, "train", "*.flac"))
    test_files = glob.glob(os.path.join(DATA_DIR, "test", "*.flac"))

    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "test"), exist_ok=True)

    output_train_files = [os.path.join(args.output_dir, "train", x.split("/")[-1].replace("flac", "tif")) for x in train_files]
    output_test_files = [os.path.join(args.output_dir, "test", x.split("/")[-1].replace("flac", "tif")) for x in test_files]

    with tqdm(total=len(train_files)) as pbar:
        with mp.Pool(num_workers) as p:
            for _ in p.imap_unordered(convert, zip(train_files, output_train_files)):
                pbar.update()

    with tqdm(total=len(test_files)) as pbar:
        with mp.Pool(num_workers) as p:
            for _ in p.imap_unordered(convert, zip(test_files, output_test_files)):
                pbar.update()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)