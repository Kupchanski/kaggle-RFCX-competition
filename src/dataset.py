import os
import copy
import torch
import random
import librosa
import torchaudio
import numpy as np
from skimage.io import imread


class MelTrainDataset(torch.utils.data.Dataset):

    def __init__(self, df, images_root, masks_root, num_classes=24, transform=None):
        assert "recording_id" in df.columns

        self.df = df
        self.images_root = images_root
        self.masks_root = masks_root
        self.transform = transform
        self.num_classes = num_classes

    def read_image(self, path):
        return imread(path)

    def __getitem__(self, i):
        
        row = self.df.iloc[i]
        recording_id = row["recording_id"]
        image_path = os.path.join(self.images_root, recording_id + ".tif")
        mask_path = os.path.join(self.masks_root, recording_id + ".tif")

        image = self.read_image(image_path)
        image = image if image.ndim == 3 else image[..., None]
        mask = self.read_image(mask_path)
        sample = dict(image=image, mask=mask)
        
        if self.transform is not None:
            # Crop Mask in albumentations ignore 0 index for classes
            # but our mask have -1 as background and classes started from 0
            sample = self.transform(image=image, mask=mask + 1)
            image = sample["image"]
            mask = sample["mask"] - 1
            
        label_ids = np.unique(mask)[1:]
        label = np.zeros(self.num_classes)
        label[label_ids] = 1
        
        return dict(recording_id=recording_id, image=image, label=label, mask=mask)

    def __len__(self):
        return self.df.shape[0]


class MelInferenceDataset(torch.utils.data.Dataset):

    def __init__(self, paths: list, image_width: int, window_width: int, step: int, transform=None, labels=None):
        
        self.paths = paths
        self.ids = [p.split("/")[-1].split(".")[0] for p in paths]
        self.id_to_path = {id: path for id, path in zip(self.ids, self.paths)}
        self.image_width = image_width
        self.window_width = window_width
        self.step = step
        self.transform = transform
        self.blocks = self.compute_bloks()
        self.labels = labels

    def compute_bloks(self):
        """Precompute slices for all images to itearate over them"""
        crops_per_image = (self.image_width - self.window_width + 1) // self.step + 2
        starts = [i * self.step for i in range(crops_per_image)]
        starts[-1] = self.image_width - self.window_width
        stops = [start + self.window_width for start in starts]
        blocks = []
        for id in self.ids:
            for start, stop in zip(starts, stops):
                block = dict(recording_id=id, start=start, stop=stop)
                blocks.append(block)
        return blocks

    def read_image(self, path):
        return imread(path)

    def __getitem__(self, i):
        
        # deepcopy somehow prevent memory leak!
        sample = copy.deepcopy(self.blocks[i])
        path = copy.deepcopy(self.id_to_path[sample["recording_id"]])

        image = self.read_image(path)
        image = image if image.ndim == 3 else image[..., None]
        image = image[:, sample["start"]:sample["stop"]]
        
        sample["image"] = image
        sample["t_min"] = round(sample["start"] / self.image_width * 60, 2)
        sample["t_max"] = round(sample["stop"] / self.image_width * 60, 2)

        if self.transform is not None:
            sample = self.transform(**sample)
        
        if self.labels is not None:
            labels_ids = self.labels[sample["recording_id"]]
            labels = np.zeros(24)
            labels[np.array(labels_ids)] = 1
            sample["label"] = labels

        return sample

    def __len__(self):
        return len(self.blocks)


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, audio_transform=None, transform=None, sample_size=512, n_mels=224, power=2,
        hop_length=512, n_fft=2048, f_min=84, f_max=15056, sr=48000):

        self.n_mels = n_mels
        self.power = power
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.sr = sr
        self.n_fft = n_fft
        self.sample_size = sample_size
        self.transform = transform
        self.audio_transform = audio_transform

    def read_audio(self, path):
        audio = np.load(path)
        return audio

    def time_to_length(self, t):
        return t * self.sr

    def length_to_time(self, length):
        return length / self.sr
    
    def pix_to_length(self, pix):
        return (pix - 1) * self.hop_length

    def to_mel(self, x):
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            power=self.power,
            f_min=self.f_min,
            f_max=self.f_max,
        )
        tensor = torch.from_numpy(x)
        mel_tensor = transform(tensor)
        mel_spec = mel_tensor.numpy()
        mel_spec = librosa.power_to_db(mel_spec, top_db=80.)
        return mel_spec


class TrainDataset(BaseDataset):

    N_CLASSES = 24

    def __init__(self, root, df, fp_df=None, audio_transform=None, transform=None, mixup_fp=0., sample_size=512,
        n_mels=224, power=2, hop_length=512, n_fft=2048, f_min=84, f_max=15056, sr=48000):

        super().__init__(audio_transform=audio_transform, transform=transform, sample_size=sample_size,
            n_mels=n_mels, power=power, hop_length=hop_length, n_fft=n_fft, f_min=f_min, f_max=f_max, sr=sr)

        self.df = df
        self.fp_df = fp_df
        self.root = root
        self.mixup_fp = mixup_fp
        if mixup_fp > 0:
            assert self.fp_df is not None

    def crop_audio(self, audio, t_min, t_max):
        """Make random audio crop, saving all signal inside crop"""
        l_min = self.time_to_length(t_min)
        l_max = self.time_to_length(t_max)
        l_crop = (self.sample_size - 1) * self.hop_length
        
        l_signal = l_max - l_min
        shift_range = int(l_crop - l_signal)
        
        l_start = l_min - random.randint(min(0, shift_range), max(0, shift_range))
        l_start = np.clip(l_start, 0, audio.shape[0] - l_crop)
        l_start = int(l_start)
        l_stop = l_start + l_crop

        return audio[l_start:l_stop], self.length_to_time(l_start), self.length_to_time(l_stop)

    # def crop_audio(self, audio, *args):
    #     l_crop = (self.sample_size - 1) * self.hop_length
    #     l_start = random.randint(0, len(audio) - l_crop - 1)
    #     l_stop = l_start + l_crop
    #     return audio[l_start:l_stop], self.length_to_time(l_start), self.length_to_time(l_stop)

    def __getitem__(self, i):

        row = self.df.iloc[i]

        # read and crop audio file 
        path = os.path.join(self.root, row["recording_id"] + ".npy")
        audio = self.read_audio(path)
        audio_crop, t_start, t_stop = self.crop_audio(audio, row["t_min"], row["t_max"])

        mixup = True if random.random() < self.mixup_fp else False

        if mixup:
            mix_row = self.fp_df.iloc[random.randint(0, len(self.fp_df) - 1)]
            mix_path = os.path.join(self.root, mix_row["recording_id"] + ".npy")
            mix_audio = self.read_audio(mix_path)
            mix_audio_crop, _, _ = self.crop_audio(mix_audio, mix_row["t_min"], mix_row["t_max"])
            audio_crop = audio_crop + mix_audio_crop

        # apply audio augs
        if self.audio_transform is not None:
            audio_crop = self.audio_transform(audio_crop)

        # convert to MEL spectrogram
        mel_spec = self.to_mel(audio_crop)[..., None]
        
        # apply image augs
        if self.transform is not None:
            mel_spec = self.transform(image=mel_spec)["image"]

        label = np.zeros(self.N_CLASSES)
        label[row["species_id"]] = 1.0

        # if mixup:
        #     label[mix_row["species_id"]] = 1.0

        sample = dict(image=mel_spec, label=label)
        return sample

    def __len__(self):
        return self.df.shape[0]


class InferenceDataset(BaseDataset):

    N_CLASSES = 24
    AUDIO_LENGTH = 60  # sec
    
    def __init__(self, paths, labels=None, audio_transform=None, transform=None, sample_size=512,
        n_mels=224, power=2, hop_length=512, n_fft=2048, f_min=84, f_max=15056, sr=48000):

        super().__init__(audio_transform=audio_transform, transform=transform, sample_size=sample_size,
            n_mels=n_mels, power=power, hop_length=hop_length, n_fft=n_fft, f_min=f_min, f_max=f_max, sr=sr)
            
        self.paths = paths
        self.labels = labels
        
        self.ids = [p.split("/")[-1].split(".")[0] for p in paths]
        self.id_to_path = {id: path for id, path in zip(self.ids, self.paths)}
        self.blocks = self.compute_bloks()

    def compute_bloks(self):
        """Precompute slices for all images to itearate over them"""
        length = self.time_to_length(self.AUDIO_LENGTH)
        sample_length = self.pix_to_length(self.sample_size)
        step = sample_length // 2

        crops_per_image = (length - sample_length + 1) // step + 2

        starts = [i * step for i in range(crops_per_image)]
        starts[-1] = length - sample_length
        stops = [start + sample_length for start in starts]
        blocks = []
        for id in self.ids:
            for start, stop in zip(starts, stops):
                block = dict(recording_id=id, start=start, stop=stop)
                blocks.append(block)
        return blocks

    def __getitem__(self, i):
        
        # deepcopy somehow prevent memory leak!
        sample = copy.deepcopy(self.blocks[i])
        path = copy.deepcopy(self.id_to_path[sample["recording_id"]])

        audio = self.read_audio(path)
        cropped_audio = audio[sample["start"]:sample["stop"]]

        if self.audio_transform is not None:
            cropped_audio = self.audio_transform(cropped_audio)

        mel = self.to_mel(cropped_audio)[..., None]

        if self.transform is not None:
            mel = self.transform(image=mel)["image"]

        sample["image"] = mel
        
        if self.labels is not None:
            labels_ids = self.labels[sample["recording_id"]]
            labels = np.zeros(24)
            labels[np.array(labels_ids)] = 1
            sample["label"] = labels

        return sample

    def __len__(self):
        return len(self.blocks)
