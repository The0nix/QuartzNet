"""
This script gets librispeech dataset, calculates lengths of
waveforms and utterances and saves them for later use
"""
import os

import numpy as np
import hydra
from omegaconf import DictConfig
from tqdm.auto import trange

import core


@hydra.main(config_path="../config", config_name="config")
def build_dataset_lengths(cfg: DictConfig):
    dataset = core.datasets.get_dataset(name=cfg.dataset.name,
                                        root=hydra.utils.to_absolute_path(cfg.dataset.path))

    waveform_lengths = []
    utterance_lengths = []
    for i in trange(len(dataset)):
        waveform, sample_rate, utterance = dataset[i]
        waveform_lengths.append(waveform.shape[1])
        utterance_lengths.append(len(utterance))
    path_name = os.path.join(hydra.utils.to_absolute_path(os.path.join("../..", cfg.dataset.path)),
                             cfg.dataset.lengths_filename)
    np.save(path_name.format("waveform"), np.array(waveform_lengths))
    np.save(path_name.format("utterance"), np.array(utterance_lengths))


if __name__ == "__main__":
    build_dataset_lengths()
