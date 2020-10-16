"""
This script gets librispeech dataset, calculates lengths of
waveforms and utterances and saves them for later use
"""
from pathlib import Path

import numpy as np
import hydra
from omegaconf import DictConfig
from tqdm.auto import trange

import core


@hydra.main(config_path="../../config", config_name="config")
def build_dataset_lengths(cfg: DictConfig):
    if core.utils.get_lengths_path(cfg, "waveform").exists() and \
            core.utils.get_lengths_path(cfg, "utterance").exists() and \
            not cfg.lengths.rebuild:
        return
    dataset = core.datasets.get_dataset(name=cfg.dataset.name,
                                        root=hydra.utils.to_absolute_path(cfg.dataset.path))

    waveform_lengths = []
    utterance_lengths = []
    for i in trange(len(dataset)):
        waveform, sample_rate, utterance = dataset[i]
        waveform_lengths.append(waveform.shape[1])
        utterance_lengths.append(len(utterance))
    np.save(core.utils.get_lengths_path(cfg, "waveform"), np.array(waveform_lengths))
    np.save(core.utils.get_lengths_path(cfg, "utterance"), np.array(utterance_lengths))


if __name__ == "__main__":
    build_dataset_lengths()
