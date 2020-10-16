from pathlib import Path

import hydra
from omegaconf import DictConfig
import youtokentome as yttm

import core


@hydra.main(config_path="../../config", config_name="config")
def train_bpe(cfg: DictConfig):
    if core.utils.get_bpe_path(cfg).exists() and not cfg.bpe.rebuild:
        return
    dataset = core.datasets.get_dataset(name=cfg.dataset.name,
                                        root=hydra.utils.to_absolute_path(cfg.dataset.path))

    train_idx, test_idx = core.datasets.get_split(dataset, train_size=cfg.dataset.train_size,
                                                  random_state=cfg.common.seed)

    with open(cfg.bpe.train_data_path, "w") as f:
        for idx in train_idx:
            utterance = dataset.get_utterance(idx)
            f.write(f"{utterance}\n")

    yttm.BPE.train(
        data=cfg.bpe.train_data_path,
        model=str(core.utils.get_bpe_path(cfg)),
        vocab_size=cfg.bpe.vocab_size,
        pad_id=0,
    )


if __name__ == "__main__":
    train_bpe()
