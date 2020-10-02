import random

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


@hydra.main(config_name="train_config")
def train(cfg: DictConfig):
    fix_seeds()
    quartznet = model.QuartzNet(C_in=cfg.preprocessing.n_mels, n_labels=10,
                                Cs=cfg.model.channels,
                                Ks=cfg.model.kernels,
                                Rs=cfg.model.repeats, Ss=5)
    waveform_transforms = get_waveform_transforms(cfg)
    dataset = datasets.LIBRISPEECH(root=hydra.utils.to_absolute_path(cfg.data.path),
                                   waveform_transform=waveform_transforms,
                                   url="train-clean-100", download=True)
    train_idx, test_idx = datasets.get_split(dataset, train_size=cfg.data.train_size, random_state=cfg.common.seed)
    bpe = yttm.BPE(model=hydra.utils.to_absolute_path(cfg.bpe.model_path))
    print(bpe.encode([dataset.get_utterance(0)]))

    optimizer = torch.optim.Adam(quartznet.parameters(), lr=cfg.training.lr)
    criterion = nn.CTCLoss()

    train_dataloader = torchdata.DataLoader(dataset,
                                            num_workers=cfg.training.num_workers,
                                            batch_size=cfg.trainig.batch_size,
                                            sampler=torchdata.sampler.SubsetRandomSampler(train_idx))
    test_dataloader = torchdata.DataLoader(dataset,
                                           num_workers=cfg.training.num_workers,
                                           batch_size=cfg.trainig.batch_size,
                                           sampler=torchdata.sampler.SubsetRandomSampler(test_idx))

    for epoch in trange(cfg.training.n_epochs, title="Epoch"):
        for data in tqdm(train_dataloader, title="Batch", leave=False):
            spectrogram, sample_rate, utterance, speaker_id, chapter_id, utterance_id = data
            preds = quartznet(spectrogram)
            loss = criterion(preds, utterance, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    import matplotlib.pyplot as plt
    plt.imshow(dataset[0][0])
    plt.show()


if __name__ == "__main__":
    train()
