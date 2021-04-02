import os
import glob
import librosa
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

DATA_DIR = "../data/"

def convert(args):
    input_path, output_path = args
    audio, sr = librosa.load(input_path, sr=None)
    np.save(output_path, audio)


def main():

    num_workers = mp.cpu_count()

    train_files = glob.glob(os.path.join(DATA_DIR, "train", "*.flac"))
    test_files = glob.glob(os.path.join(DATA_DIR, "test", "*.flac"))

    output_train_files = [x.replace("flac", "npy").replace("train", "train_np") for x in train_files]
    output_test_files = [x.replace("flac", "npy").replace("test", "test_np") for x in test_files]

    os.makedirs(os.path.dirname(output_train_files[0]), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_files[0]), exist_ok=True)

    with tqdm(total=len(train_files)) as pbar:
        with mp.Pool(num_workers) as p:
            for _ in p.imap_unordered(convert, zip(train_files, output_train_files)):
                pbar.update()

    with tqdm(total=len(test_files)) as pbar:
        with mp.Pool(num_workers) as p:
            for _ in p.imap_unordered(convert, zip(test_files, output_test_files)):
                pbar.update()

if __name__ == "__main__":
    main()