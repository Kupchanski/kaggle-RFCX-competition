import librosa
import librosa.display
import random

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from IPython.display import Audio

"""
AddGaussianNoise
GaussianNoiseSNR
PinkNoiseSNR
PitchShift
TimeStretch
TimeShift
VolumeControl

"""

class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class OneOf:
    def __init__(self, transforms: list, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, y: np.ndarray):
        if np.random.rand() < self.p:
            n_trns = len(self.transforms)
            trns_idx = np.random.choice(n_trns)
            trns = self.transforms[trns_idx]
            y = trns(y)
        return y
    
class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, trg=None):
        if trg is None:
            if random.random() < self.prob:
                image = self.transform(image)
            return image
        else:
            if random.random() < self.prob:
                image, trg = self.transform(image, trg)
            return image, trg
    
<<<<<<< HEAD
    
def spec_augment(spec: np.ndarray,
                 num_mask=2,
                 freq_masking=0.15,
                 time_masking=0.20,
                 value=0):
    spec = spec.copy()
    num_mask = random.randint(1, num_mask)
    for i in range(num_mask):
        all_freqs_num, all_frames_num  = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[f0:f0 + num_freqs_to_mask, :] = value

        time_percentage = random.uniform(0.0, time_masking)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[:, t0:t0 + num_frames_to_mask] = value
    return spec


class SpecAugment:
    def __init__(self,
                 num_mask=2,
                 freq_masking=0.15,
                 time_masking=0.20):
        self.num_mask = num_mask
        self.freq_masking = freq_masking
        self.time_masking = time_masking

    def __call__(self, image):
        return spec_augment(image,
                            self.num_mask,
                            self.freq_masking,
                            self.time_masking,
                            image.min())

=======
>>>>>>> 3c6e4f10b9c31cf58847d555ffb684e28db5913b
#AddGaussianNoise

"""
Add noise that follows normal distribution (a.k.a whitenoise). The amplitude of noise is randomly decided.
"""
class AddGaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_amplitude=0.5, **kwargs):
        super().__init__(always_apply, p)

        self.noise_amplitude = (0.0, max_noise_amplitude)

    def apply(self, y: np.ndarray, **params):
        noise_amplitude = np.random.uniform(*self.noise_amplitude)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_amplitude).astype(y.dtype)
        return augmented

    
#GaussianNoiseSNR
"""
The problem with Data Augmentation introduced above is that if you specify an amplitude of noise, it can be masked by noise when the original signal is weak.

To prevent this, it is easier to adaptively set an appropriate noise level based on the amplitude of the signal in the original sound.

The ratio of the signal-to-noise level is called Signal-to-Noise Ratio (SNR). The signal-to-noise ratio (SNR) is expressed as the ratio of the actual amplitude to the logarithm of the signal's amplitude and is calculated by the following formula.

SNR=20log10AsignalAnoise
 
The larger this amount is, the stronger the signal, or the more audible the sound is, and it is expressed in dB (decibel), where 0dB means that the strength of the signal is balanced with the strength of the noise, when it is negative, the noise is stronger, and when it is positive, the signal is stronger.

There may be several ways to estimate the strength of the signal sound, but in this case we will treat the absolute maximum of the amplitude in the clip as the amplitude of the signal.
"""

class GaussianNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented
    

#PinkNoiseSNR

"""
Gaussian Noise is a so-called white noise, which is a noise over the whole frequency range. Pink noise, which we introduce here, is noise with a gradual decrease in noise intensity from low frequency to low frequency bands. The noise in the natural world is said to be such noise.

The noise other than white noise is called "colored noise", and various noises such as brown noise and blue noise have been proposed.

The colorednoise library is used to generate pink noise, and its name comes from the above.

In the previous article, we introduced an implementation of white noise that directly specifies the intensity of the noise.
"""

import colorednoise as cn


class PinkNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented
    
    
#PitchShift

"""
PitchShift is a data augmentation that adjusts the pitch of the sound (high and low), making the sound heard as an effect higher/lower. On the Meru spectrogram, certain frequency bands in the pattern will be shifted up or down.

PitchShift takes more time than the previously introduced Data Augmentation because of resampling, and you should be careful not to change the pitch too much because it may cause the sound to crack."""


class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_steps=5, sr=32000):
        super().__init__(always_apply, p)

        self.max_steps = max_steps
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        n_steps = np.random.randint(-self.max_steps, self.max_steps)
        augmented = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)
        return augmented
    
    
#TimeStretch

"""
TimeStretch stretches and compresses the original sound in time. As a result, the speed of the sound may be increased or decreased.

TimeStretch is another time-consuming form of data augmentation.

"""

class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1.2):
        super().__init__(always_apply, p)

        self.max_rate = max_rate

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented
    
    
#TimeShift

"""
TimeShift is such an operation that shifts a sound event in time. As for dealing with the part of the sound clip that goes out of the original length as a result of shifting, you can bring it forward (or backward) and stick it to the front (or backward), or ignore it and throw it away.
ref - https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6
"""

class TimeShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_shift_second=2, sr=32000, padding_mode="replace"):
        super().__init__(always_apply, p)
    
        assert padding_mode in ["replace", "zero"], "`padding_mode` must be either 'replace' or 'zero'"
        self.max_shift_second = max_shift_second
        self.sr = sr
        self.padding_mode = padding_mode

    def apply(self, y: np.ndarray, **params):
        shift = np.random.randint(-self.sr * self.max_shift_second, self.sr * self.max_shift_second)
        augmented = np.roll(y, shift)
        if self.padding_mode == "zero":
            if shift > 0:
                augmented[:shift] = 0
            else:
                augmented[shift:] = 0
        return augmented
    
    
#VolumeControl

"""
VolumeControl controls the volume. I think I mentioned before that the SNR has more influence on sound perception than the volume itself, but adjusting the volume causes a very small change in the mel spectrogram. Adjusting the volume according to a sine curve, cosine curve, etc. is also useful because it causes a big change in the mel spectrogram.
"""

class VolumeControl(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, db_limit=10, mode="uniform"):
        super().__init__(always_apply, p)

        assert mode in ["uniform", "fade", "cosine", "sine"], \
            "`mode` must be one of 'uniform', 'fade', 'cosine', 'sine'"

        self.db_limit= db_limit
        self.mode = mode

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.db_limit, self.db_limit)
        if self.mode == "uniform":
            db_translated = 10 ** (db / 20)
        elif self.mode == "fade":
            lin = np.arange(len(y))[::-1] / (len(y) - 1)
            db_translated = 10 ** (db * lin / 20)
        elif self.mode == "cosine":
            cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * cosine / 20)
        else:
            sine = np.sin(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * sine / 20)
        augmented = y * db_translated
        return augmented
    
    
"""
example

transform = Compose([
  OneOf([
    GaussianNoiseSNR(min_snr=10),
    PinkNoiseSNR(min_snr=10)
  ]),
  PitchShift(max_steps=2, sr=sr),
  TimeStretch(),
  TimeShift(sr=sr),
  VolumeControl(mode="sine")
])


"""

#TODO: Codec augment (format to different codecs)

