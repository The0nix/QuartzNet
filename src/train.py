import os
from typing import Tuple, List

import wandb
import torch
import torch.nn as nn
import jiwer
import torch.utils.data as torchdata
import numpy as np
import hydra
from omegaconf import DictConfig
import youtokentome as yttm
from tqdm.auto import tqdm, trange

import core

MODEL_PATH = "model.pth"


def process_batch(model: nn.Module, batch: Tuple, criterion: nn.modules.loss._Loss,
                  optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
                  train: bool, device: torch.cuda.Device) \
        -> Tuple[torch.Tensor, torch.Tensor, List[str], List[torch.Tensor]]:
    """
    :param model: model to train
    :param batch: batch with spectrograms, sample_rates, utterances
    :param criterion: criterion to calculate loss
    :param optimizer: optimizer to step
    :param scaler: GradScaler for mixed precision training
    :param train: perform gradient step
    :param device: cuda device to work on
    :return: (loss, logprobs, utterances)
    """
    input_lengths, target_lengths, waveforms, spectrograms, sample_rates, utterances = batch

    spectrograms = spectrograms.to(device)
    utterances = utterances.to(device)

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

    return loss, logprobs, utterances, waveforms


@hydra.main(config_path="../config", config_name="config")
def train(cfg: DictConfig):
    core.utils.fix_seeds()
    device = torch.device(cfg.training.device)

    # Initialize network
    quartznet = core.model.QuartzNet(C_in=cfg.preprocessing.n_mels, n_labels=cfg.bpe.vocab_size + 1,
                                     Cs=cfg.model.channels,
                                     Ks=cfg.model.kernels,
                                     Rs=cfg.model.repeats,
                                     Ss=cfg.model.block_repeats).to(device)
    if cfg.model.path is not None:
        quartznet.load_state_dict(torch.load(hydra.utils.to_absolute_path(cfg.model.path)))

    # Load BPE encoder from disk
    bpe = yttm.BPE(model=str(core.utils.get_bpe_path(cfg)))

    # Create transforms
    waveform_transforms = core.utils.get_waveform_transforms(cfg.waveform_transforms)
    utterance_transform = core.transforms.BPETransform(model=bpe)

    # Create dataset
    dataset = core.datasets.get_dataset(name=cfg.dataset.name,
                                        root=hydra.utils.to_absolute_path(cfg.dataset.path),
                                        waveform_transform=waveform_transforms,
                                        utterance_transform=utterance_transform)

    # Create optimizer and criterion
    optimizer = torch.optim.Adam(quartznet.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)
    criterion = nn.CTCLoss(blank=cfg.bpe.vocab_size)

    # Split dataset and create dataloaders
    waveform_lengths = np.load(core.utils.get_lengths_path(cfg, "waveform"))
    indices_lt5 = np.where(waveform_lengths < 15 * cfg.dataset.original_sample_rate)
    train_idx, val_idx = core.datasets.get_split(dataset, train_size=cfg.dataset.train_size, random_state=cfg.common.seed)
    train_idx = np.intersect1d(train_idx, indices_lt5)
    val_idx = np.intersect1d(val_idx, indices_lt5)
    # train_idx = train_idx[:int(train_idx.shape[0] * 0.0005)]
    # val_idx = train_idx

    train_dataloader = torchdata.DataLoader(dataset,
                                            num_workers=cfg.training.num_workers,
                                            batch_size=cfg.training.batch_size,
                                            collate_fn=core.utils.pad_collate,
                                            pin_memory=True,
                                            sampler=torchdata.sampler.SubsetRandomSampler(train_idx))
    val_dataloader = torchdata.DataLoader(dataset,
                                          num_workers=cfg.training.num_workers,
                                          batch_size=cfg.training.batch_size,
                                          collate_fn=core.utils.pad_collate,
                                          pin_memory=True,
                                          sampler=torchdata.sampler.SubsetRandomSampler(val_idx))

    # Create scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Start training
    wandb.init(project=cfg.wandb.project, tags=[cfg.dataset.name])
    wandb.watch(quartznet, log="all", log_freq=cfg.wandb.log_interval)
    for epoch_idx in trange(cfg.training.start_epoch or 0, cfg.training.n_epochs, desc="Epoch"):
        # Train part
        quartznet.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training batch", leave=False)):
            loss, logprobs, utterances, waveforms = process_batch(quartznet, batch, criterion, optimizer,
                                                                  scaler, train=True, device=device)

            if batch_idx % cfg.wandb.log_interval == 0:
                true_texts, pred_texts = core.utils.get_texts(utterances, logprobs, bpe, cfg.bpe.vocab_size)
                wer = jiwer.wer(true_texts, pred_texts)
                cer = jiwer.wer([" ".join(t.replace(" ", "*")) for t in true_texts],
                                [" ".join(t.replace(" ", "*")) for t in pred_texts])
                step = (epoch_idx * len(train_dataloader) * train_dataloader.batch_size +
                        batch_idx * train_dataloader.batch_size)
                wandb.log({
                    "epoch": epoch_idx,
                    "lr": optimizer.param_groups[0]['lr'],
                    "train_loss": loss.item(),
                    "train_wer": wer,
                    "train_cer": cer,
                    "train_examples": wandb.Table(columns=['GT', 'Prediction'], data=zip(true_texts, pred_texts)),
                }, step=step)
                # Save the model
                torch.save(quartznet.state_dict(), MODEL_PATH)
        scheduler.step()

        # Eval part
        quartznet.eval()
        val_wers = []
        val_cers = []
        val_losses = []
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Validation batch", leave=False)):
            loss, logprobs, utterances, waveforms = process_batch(quartznet, batch, criterion, optimizer,
                                                                  scaler, train=False, device=device)
            val_losses.append(loss.item())
            true_texts, pred_texts = core.utils.get_texts(utterances, logprobs, bpe, cfg.bpe.vocab_size)
            val_wers.append(jiwer.wer(true_texts, pred_texts))
            val_cers.append(jiwer.wer([" ".join(t.replace(" ", "*")) for t in true_texts],
                                      [" ".join(t.replace(" ", "*")) for t in pred_texts]))
        step = (epoch_idx + 1) * len(train_dataloader) * train_dataloader.batch_size
        wandb.log({
            "val_loss": np.mean(val_losses),
            "val_wer": np.mean(val_wers),
            "val_сer": np.mean(val_wers),
            "val_examples": wandb.Table(columns=['GT', 'Prediction'], data=zip(true_texts, pred_texts)),
            "val_audio": [wandb.Audio(w.numpy().ravel(), sample_rate=cfg.dataset.sample_rate) for w in waveforms],
        }, step=step)

        # TODO: Add model saving


if __name__ == "__main__":
    train()
