import os
import csv
import itertools
from typing import Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as torchdata
import torchaudio
import sklearn.model_selection
from num2words import num2words


def get_split(dataset: torchdata.Dataset, random_state: int, train_size: float):
    """
    Get train and test indices for dataset
    :param dataset: torch.Dataset (or any object with length)
    :param random_state: random state for dataset
    :param train_size: fraction of indices to use for training
    :return:
    """
    idxs = np.arange(len(dataset))
    train_idx, test_idx = sklearn.model_selection.train_test_split(idxs, train_size=train_size,
                                                                   random_state=random_state)
    return train_idx, test_idx


class LIBRISPEECH(torchaudio.datasets.LIBRISPEECH):
    """
    torchaudio.datasets.LIBRISPEECH wrapper to pass transforms
    """
    def __init__(self, waveform_transform=None, utterance_transform=None, *args, **kwargs):
        super().__init__(url="train-clean-100", download=True, *args, **kwargs)
        self.waveform_transform = waveform_transform
        self.utterance_transform = utterance_transform

    def __getitem__(self, idx) -> Union[Tuple[torch.Tensor, torch.Tensor, int, str],
                                        Tuple[torch.Tensor, int, str]]:
        waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = super().__getitem__(idx)
        if self.utterance_transform is not None:
            utterance = self.utterance_transform(utterance)
        if self.waveform_transform is not None:
            spectrogram = self.waveform_transform(samples=waveform, sample_rate=sample_rate)
            return waveform, spectrogram, sample_rate, utterance
        else:
            return waveform, sample_rate, utterance

    def get_utterance(self, idx: int) -> str:
        """
        Get utterance only for bytepair encoding
        (copypaste from LIBRISPEECH source)
        """
        fileid = self._walker[idx]
        speaker_id, chapter_id, utterance_id = fileid.split("-")

        file_text = speaker_id + "-" + chapter_id + self._ext_txt
        file_text = os.path.join(self._path, speaker_id, chapter_id, file_text)
        fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id

        # Load text
        with open(file_text) as ft:
            for line in ft:
                fileid_text, utterance = line.strip().split(" ", 1)
                if fileid_audio == fileid_text:
                    break
            else:
                # Translation not found
                raise FileNotFoundError("Translation not found for " + fileid_audio)

        return utterance


class Numbers(torchdata.Dataset):
    """
    Simple dataset with spoken numbers:
    https://drive.google.com/file/d/1HKtLLbiEk0c3l1mKz9LUXRAmKd3DvD0P/view?usp=sharing
    """
    def __init__(self, root, waveform_transform=None, utterance_transform=None):
        self.root = Path(root)
        self.waveform_transform = waveform_transform
        self.utterance_transform = utterance_transform
        with open(self.root / "train.csv") as f:
            d = csv.reader(f)
            self.paths, self.genders, self.targets = zip(*itertools.islice(d, 1, None))

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.root / self.paths[idx])
        utterance = num2words(self.targets[idx], lang="ru")

        if self.utterance_transform is not None:
            utterance = self.utterance_transform(utterance)
        if self.waveform_transform is not None:
            spectrogram = self.waveform_transform(samples=waveform, sample_rate=sample_rate)
            return waveform, spectrogram, sample_rate, utterance

        return waveform, sample_rate, utterance

    def get_utterance(self, idx: int) -> str:
        """
        Get utterance only for bytepair encoding
        """
        utterance = num2words(self.targets[idx], lang="ru")
        return utterance

    def __len__(self):
        return len(self.paths)


def get_dataset(name: str, root: str, waveform_transform=None, utterance_transform=None, *args) \
        -> Union[LIBRISPEECH, Numbers]:
    """
    Get dataset by its name
    :param name: One of {"librispeech", "numbers"}
    :param root:
    :param waveform_transform: audiomentations transform for waveform
    :param utterance_transform: transform for utterance
    :return: either LIBRISPEECH or Numbers dataset
    """
    if name == "librispeech":
        return LIBRISPEECH(root=root, waveform_transform=waveform_transform, utterance_transform=utterance_transform,
                           *args)
    elif name == "numbers":
        return Numbers(root=root, waveform_transform=waveform_transform, utterance_transform=utterance_transform)
    else:
        raise ValueError(f"Unknown dataset name: \"{name}\"")
