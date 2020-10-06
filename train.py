import random
from typing import Tuple, List

import wandb
import torch
import torch.nn as nn
import jiwer
import torch.utils.data as torchdata
import numpy as np
import hydra
from omegaconf import DictConfig
import audiomentations as aud
import youtokentome as yttm
from tqdm.auto import tqdm, trange

import model
import transforms
import datasets


def fix_seeds(seed=1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def get_waveform_transforms(cfg: DictConfig):
    """
    get all necessary transforms from config
    :param cfg: main app config
    :return: transforms composed into aud.Compose
    """
    return aud.Compose([
        hydra.utils.instantiate(transform)
        for transform in cfg.waveform_transforms
    ])


def collate_fn(batch):
    return list(zip(*batch))


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


def process_batch(model: nn.Module, batch: Tuple, criterion: nn.modules.loss._Loss,
                  optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler, train: bool, device) \
        -> Tuple[torch.Tensor, torch.Tensor, List]:
    """
    :param model: model to train
    :param batch: batch with spectrograms, sample_rates, utterances, speaker_ids, chapter_ids, utterance_ids
    :param criterion: criterion to calculate loss
    :param optimizer: optimizer to step
    :param scaler: GradScaler for mixed precision training
    :param train: perform gradient step
    :return: (loss, logprobs, utterances)
    """
    spectrograms, sample_rates, utterances, speaker_ids, chapter_ids, utterance_ids = batch

    input_lengths = torch.tensor([spectrogram.shape[1] for spectrogram in spectrograms])
    target_lengths = torch.tensor([utterance.shape[0] for utterance in utterances])

    utterances = nn.utils.rnn.pad_sequence(utterances, True, 0).to(device)
    spectrograms = nn.utils.rnn.pad_sequence([s.transpose(0, 1) for s in spectrograms], True, 0).to(device)
    spectrograms = spectrograms.transpose(1, 2)

    with torch.cuda.amp.autocast():
        logprobs = model(spectrograms)
        loss = criterion(logprobs.permute(2, 0, 1), utterances, input_lengths, target_lengths)

    if train:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    loss = loss.detach()
    logprobs = logprobs.detach()
    utterances = utterances.detach()

    return loss, logprobs, utterances


def get_texts(utterances, logprobs, bpe: yttm.BPE, blank_idx) -> Tuple[List[str], List[str]]:
    """
    Get utterances and logprobs, ctc decode, remove padding, bpe decode and return texts
    :param utterances: bpe encoded utterances from dataset
    :param logprobs: output from network
    :return: two lists of strings for true and predicted texts
    """
    pred_tokens = [np.trim_zeros(ctc_decode(prob, blank_idx).tolist()) for prob in logprobs]
    true_tokens = [np.trim_zeros(utt.tolist()) for utt in utterances]
    pred_texts = [bpe.decode(pred) for pred in pred_tokens]
    true_texts = [bpe.decode(true) for true in true_tokens]
    return true_texts, pred_texts


@hydra.main(config_name="train_config")
def train(cfg: DictConfig):
    fix_seeds()
    device = torch.device(cfg.training.device)

    # Initialize network
    quartznet = model.QuartzNet(C_in=cfg.preprocessing.n_mels, n_labels=cfg.bpe.vocab_size+1,
                                Cs=cfg.model.channels,
                                Ks=cfg.model.kernels,
                                Rs=cfg.model.repeats, Ss=5).to(device)

    # Load BPE encoder from disk
    bpe = yttm.BPE(model=hydra.utils.to_absolute_path(cfg.bpe.model_path))

    # Create transforms
    waveform_transforms = get_waveform_transforms(cfg)
    utterance_transform = transforms.BPETransform(model=bpe)

    # Create dataset
    dataset = datasets.LIBRISPEECH(root=hydra.utils.to_absolute_path(cfg.data.path),
                                   waveform_transform=waveform_transforms,
                                   utterance_transform=utterance_transform,
                                   url="train-clean-100", download=True)

    # Create optimizer and criterion
    optimizer = torch.optim.Adam(quartznet.parameters(), lr=cfg.training.lr)
    criterion = nn.CTCLoss(blank=cfg.bpe.vocab_size)

    # Split dataset and create dataloaders
    train_idx, val_idx = datasets.get_split(dataset, train_size=cfg.data.train_size, random_state=cfg.common.seed)
    # train_idx = train_idx[:int(train_idx.shape[0] * 0.001)]
    # val_idx = train_idx
    train_dataloader = torchdata.DataLoader(dataset,
                                            num_workers=cfg.training.num_workers,
                                            batch_size=cfg.training.batch_size,
                                            collate_fn=collate_fn,
                                            pin_memory=True,
                                            sampler=torchdata.sampler.SubsetRandomSampler(train_idx))
    val_dataloader = torchdata.DataLoader(dataset,
                                          num_workers=cfg.training.num_workers,
                                          batch_size=cfg.training.batch_size,
                                          collate_fn=collate_fn,
                                          pin_memory=True,
                                          sampler=torchdata.sampler.SubsetRandomSampler(val_idx))

    # Create scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Start training
    wandb.init(project=cfg.wandb.project)
    wandb.watch(quartznet, log="all", log_freq=cfg.wandb.log_interval)
    for epoch_idx in trange(cfg.training.n_epochs, desc="Epoch"):
        # Train part
        quartznet.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training batch", leave=False)):
            loss, logprobs, utterances = process_batch(quartznet, batch, criterion, optimizer,
                                       scaler, train=True, device=device)

            if batch_idx % cfg.wandb.log_interval == 0:
                true_texts, pred_texts = get_texts(utterances, logprobs, bpe, cfg.bpe.vocab_size)
                wer = np.mean([jiwer.wer(true, pred) for true, pred in zip(true_texts, pred_texts)])
                step = epoch_idx * len(train_dataloader) * train_dataloader.batch_size + batch_idx * train_dataloader.batch_size
                wandb.log({
                    "train_loss": loss.item(),
                    "train_wer": wer,
                    "train_examples": wandb.Table(columns=['GT', 'Prediction'], data=zip(true_texts, pred_texts))
                }, step=step)

        # Eval part
        quartznet.eval()
        val_wers = []
        val_losses = []
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Validation batch", leave=False)):
            loss, logprobs, utterances = process_batch(quartznet, batch, criterion, optimizer,
                                                       scaler, train=False, device=device)
            val_losses.append(loss.item())
            true_texts, pred_texts = get_texts(utterances, logprobs, bpe, cfg.bpe.vocab_size)
            val_wers.append(np.mean([jiwer.wer(true, pred) for true, pred in zip(true_texts, pred_texts)]))
        step = (epoch_idx + 1) * len(train_dataloader) * train_dataloader.batch_size
        wandb.log({
            "val_loss": np.mean(val_losses),
            "val_wer": np.mean(val_wers),
            "val_examples": wandb.Table(columns=['GT', 'Prediction'], data=zip(true_texts, pred_texts))
        }, step=step)

        # TODO: Add model saving


if __name__ == "__main__":
    train()
