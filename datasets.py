import os
from typing import Tuple

import numpy as np
import torch.utils.data
import torchaudio
import sklearn.model_selection


def get_split(dataset: torch.utils.data.Dataset, random_state: int, train_size: float):
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
    torchaudio.datasets.LIBRISPEECH
    waveform_transform
    """
    def __init__(self, waveform_transform=None, utterance_transform=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waveform_transform = waveform_transform
        self.utterance_transform = utterance_transform

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, str, int, int, int]:
        waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = super().__getitem__(idx)
        if self.waveform_transform is not None:
            waveform = self.waveform_transform(samples=waveform, sample_rate=sample_rate)
        if self.utterance_transform is not None:
            utterance = self.utterance_transform(utterance)
        return waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id

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
