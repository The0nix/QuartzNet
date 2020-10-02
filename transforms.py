from typing import Union

import numpy as np
import torch
import torchaudio


class MelSpectrogram(torchaudio.transforms.MelSpectrogram):
    """
    torchaudio MelSpectrogram wrapper for audiomentations's Compose
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int) -> torch.Tensor:
        return super(MelSpectrogram, self).forward(torch.tensor(samples))


class Squeeze:
    """
    Transform to squeeze monochannel waveform
    """
    def __call__(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int):
        return samples.squeeze(0)


class ToNumpy:
    """
    Transform to make numpy array
    """
    def __call__(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int):
        return np.array(samples)


class LogTransform:
    """
    Transform for taking logarithm of mel spectrograms (or anything else)
    :param fill_value: value to substitute non-positive numbers with before applying log
    """
    def __init__(self, fill_value: float = 1e-5) -> None:
        self.fill_value = fill_value

    def __call__(self, samples: torch.Tensor, sample_rate: int):
        samples = samples + torch.full_like(samples, self.fill_value) * (samples <= 0)
        return torch.log(samples)
