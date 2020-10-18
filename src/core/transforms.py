from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import youtokentome as yttm


class MelSpectrogram(torchaudio.transforms.MelSpectrogram):
    """
    torchaudio MelSpectrogram wrapper for audiomentations's Compose
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int) -> torch.Tensor:
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples)
        return super(MelSpectrogram, self).forward(samples)


class Resample(torchaudio.transforms.Resample):
    """
    torchaudio Resample wrapper for audiomentations's Compose
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, samples: Union[np.ndarray, torch.Tensor], sample_rate: int) -> torch.Tensor:
        return super(Resample, self).forward(samples)


class BPETransform:
    """
    Byte Pair Encoding transform
    :param: model_path: path to YTTM BPE model
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.bpe = yttm.BPE(model=self.model_path)

    def __call__(self, utterance):
        return torch.tensor(self.bpe.encode(utterance))

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['bpe']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        self.bpe = yttm.BPE(model=self.model_path)


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


class LogTransform(nn.Module):
    """
    Transform for taking logarithm of mel spectrograms (or anything else)
    :param fill_value: value to substitute non-positive numbers with before applying log
    """
    def __init__(self, fill_value: float = 1e-5) -> None:
        super().__init__()
        self.fill_value = fill_value

    def __call__(self, samples: torch.Tensor, sample_rate: int):
        samples = samples + torch.full_like(samples, self.fill_value) * (samples <= 0)
        return torch.log(samples)
