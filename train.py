import random

import wandb
import torch
import torch.nn as nn
import torchaudio
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


@hydra.main(config_name="train_config")
def train(cfg: DictConfig):
    fix_seeds()

    device = torch.device(cfg.training.device)
    quartznet = model.QuartzNet(C_in=cfg.preprocessing.n_mels, n_labels=cfg.bpe.vocab_size+1,
                                Cs=cfg.model.channels,
                                Ks=cfg.model.kernels,
                                Rs=cfg.model.repeats, Ss=5).to(device)
    waveform_transforms = get_waveform_transforms(cfg)
    utterance_transform = transforms.BPETransform(model=hydra.utils.to_absolute_path(cfg.bpe.model_path))
    dataset = datasets.LIBRISPEECH(root=hydra.utils.to_absolute_path(cfg.data.path),
                                   waveform_transform=waveform_transforms,
                                   utterance_transform=utterance_transform,
                                   url="train-clean-100", download=True)
    train_idx, test_idx = datasets.get_split(dataset, train_size=cfg.data.train_size, random_state=cfg.common.seed)

    optimizer = torch.optim.Adam(quartznet.parameters(), lr=cfg.training.lr)
    criterion = nn.CTCLoss(blank=cfg.bpe.vocab_size)

    train_dataloader = torchdata.DataLoader(dataset,
                                            num_workers=cfg.training.num_workers,
                                            batch_size=cfg.training.batch_size,
                                            collate_fn=collate_fn,
                                            sampler=torchdata.sampler.SubsetRandomSampler(train_idx))
    test_dataloader = torchdata.DataLoader(dataset,
                                           num_workers=cfg.training.num_workers,
                                           batch_size=cfg.training.batch_size,
                                           collate_fn=collate_fn,
                                           sampler=torchdata.sampler.SubsetRandomSampler(test_idx))

    wandb.init(project=cfg.wandb.project)
    wandb.watch(quartznet, log="all", log_freq=cfg.wandb.log_interval)
    for epoch in trange(cfg.training.n_epochs, desc="Epoch"):
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Batch", leave=False)):
            spectrograms, sample_rates, utterances, speaker_ids, chapter_ids, utterance_ids = batch

            input_lengths = torch.tensor([spectrogram.shape[1] for spectrogram in spectrograms])
            target_lengths = torch.tensor([utterance.shape[0] for utterance in utterances])

            utterances = nn.utils.rnn.pad_sequence(utterances, batch_first=True).to(device)
            spectrograms = nn.utils.rnn.pad_sequence([s.transpose(0, 1) for s in spectrograms], batch_first=True).to(device)
            spectrograms = spectrograms.transpose(1, 2)

            logprobs = quartznet(spectrograms).permute(2, 0, 1)
            loss = criterion(logprobs, utterances, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % cfg.wandb.log_interval == 0:
                wandb.log({"train_loss": loss.item()})

            # TODO: Add model saving
            # TODO: Add validation with WER


if __name__ == "__main__":
    train()
