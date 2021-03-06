import random
import math
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import numpy as np
import hydra
from omegaconf import DictConfig
import audiomentations as aud
import youtokentome as yttm

import core

def get_bpe_path(cfg, original_path: Optional[Path] = None):
    path = Path(cfg.common.files_path, cfg.bpe.model_path)
    return Path(original_path / path if original_path else hydra.utils.to_absolute_path(path))


def get_lengths_path(cfg, lengths_type: str, original_path: Optional[Path] = None):
    """
    :param lengths_type: either "waveform" or "utterance" for waveform and utterance lengths respectively
    :param original_path: Path object with original working directory. It is used when hydra.utils.to_absolute_path(path)
    is not available (in subprocess)
    """
    path = Path(cfg.common.files_path, cfg.lengths.lengths_filename.format(lengths_type))
    return Path(original_path / path if original_path else hydra.utils.to_absolute_path(path))


def fix_seeds(seed=1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def get_waveform_transforms(transforms: DictConfig, device=None):
    """
    get all necessary transforms from config
    :param transforms: transforms from config
    :param device: device if transforms need to be ported to device
    :return: transforms composed into aud.Compose
    """
    if transforms is None:
        return None
    if device is not None:
        return aud.Compose([
            hydra.utils.instantiate(transform).to(device)
            for transform in transforms
        ])
    else:
        return aud.Compose([
            hydra.utils.instantiate(transform)
            for transform in transforms
        ])


def pad_collate(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor],
                                             torch.Tensor, List[int], torch.Tensor]:
    """
    Collator that transforms batch into tuple of lists and pads spectrograms and utterances
    :param batch: List of tuples of batch components
    :return: Tuple of tensors and lists
    """
    waveforms, spectrograms, sample_rates, utterances = zip(*batch)

    input_lengths = torch.tensor([math.ceil(spectrogram.shape[1] / 2) for spectrogram in spectrograms])
    target_lengths = torch.tensor([utterance.shape[0] for utterance in utterances])

    utterances = nn.utils.rnn.pad_sequence(utterances, True, 0)
    spectrograms = nn.utils.rnn.pad_sequence([s.transpose(0, 1) for s in spectrograms], True, 0)
    spectrograms = spectrograms.transpose(1, 2)

    return input_lengths, target_lengths, waveforms, spectrograms, sample_rates, utterances


def ctc_decode(seq: torch.Tensor, blank_idx: int):
    """
    Simple ctc decoding without beam search
    :param seq: torch tensor with tokens of shape (n_labels, len)
    :param blank_idx: index of blank character
    :return: torch tensor with ctc decoded tokens
    """
    seq = seq.argmax(dim=0)
    seq = torch.cat([seq[[0]], seq[1:][(seq[1:] - seq[:-1]).bool()]])  # Remove consecutive duplicates
    seq = seq[seq != blank_idx]  # Remove blanks
    return seq


def get_texts(utterances, logprobs, bpe: core.transforms.BPETransform, blank_idx) -> Tuple[List[str], List[str]]:
    """
    Get utterances and logprobs, ctc decode, remove padding, bpe decode and return texts
    :param utterances: bpe encoded utterances from dataset
    :param logprobs: output from network
    :param bpe: BPE encoder object for encoding (duh)
    :param blank_idx: index of ctc blank value
    :return: two lists of strings for true and predicted texts
    """
    true_tokens = [np.trim_zeros(utt.tolist()) for utt in utterances]
    pred_tokens = [np.trim_zeros(ctc_decode(prob, blank_idx).tolist()) for prob in logprobs]
    true_texts = [coded[0] if len(coded) > 0 else "" for coded in [bpe.bpe.decode(true) for true in true_tokens]]
    pred_texts = [coded[0] if len(coded) > 0 else "" for coded in [bpe.bpe.decode(pred) for pred in pred_tokens]]
    return true_texts, pred_texts
