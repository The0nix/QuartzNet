import hydra
from omegaconf import DictConfig
import youtokentome as yttm

import datasets


@hydra.main(config_name="train_config")
def train_bpe(cfg: DictConfig):
    print(hydra.utils.to_absolute_path(cfg.data.path))
    dataset = datasets.LIBRISPEECH(root=hydra.utils.to_absolute_path(cfg.data.path),
                                   url="train-clean-100", download=True)

    train_idx, test_idx = datasets.get_split(dataset, train_size=cfg.data.train_size, random_state=cfg.common.seed)

    with open(cfg.bpe.train_data_path, "w") as f:
        for idx in train_idx:
            utterance = dataset.get_utterance(idx)
            f.write(f"{utterance}\n")

    yttm.BPE.train(
        data=cfg.bpe.train_data_path,
        model=hydra.utils.to_absolute_path(cfg.bpe.model_path),
        vocab_size=cfg.bpe.vocab_size
    )


if __name__ == "__main__":
    train_bpe()
