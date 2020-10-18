from pathlib import Path

import numpy as np
import torch
import torchaudio
import hydra
from omegaconf import DictConfig

import core

OUTPUT_DIR = Path("inferenced")


def make_device(device_str: str):
    """
    return device from string that can be either 'cpu', 'cuda' or cuda device number"
    """
    try:
        return torch.device(int(device_str))  # Can fail on conversion to int
    except ValueError:
        return torch.device(device_str)


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    model_path = Path(hydra.utils.to_absolute_path(cfg.inference.model_path))
    file_path = Path(hydra.utils.to_absolute_path(cfg.inference.file))
    device = make_device(cfg.inference.device)
    assert file_path.exists(), f"File path f{file_path} does not exist"
    output_path = Path(hydra.utils.to_absolute_path(OUTPUT_DIR / (file_path.stem + ".txt")))
    waveform, sample_rate = torchaudio.load(file_path)
    quartznet = core.model.QuartzNet(C_in=cfg.preprocessing.n_mels, n_labels=cfg.bpe.vocab_size + 1,
                                     Cs=cfg.model.channels,
                                     Ks=cfg.model.kernels,
                                     Rs=cfg.model.repeats,
                                     Ss=cfg.model.block_repeats).to(device)
    quartznet.load_state_dict(torch.load(model_path, map_location="cpu"))
    utterance_transform = core.transforms.BPETransform(model_path=str(core.utils.get_bpe_path(cfg)))
    waveform_transforms = core.utils.get_waveform_transforms(cfg.waveform_transforms[-2:])  # Get mel and log transforms

    spectrogram = waveform_transforms(samples=waveform, sample_rate=sample_rate)

    logprobs = quartznet(spectrogram)
    ctc_decoded = np.trim_zeros(core.utils.ctc_decode(logprobs[0], cfg.bpe.vocab_size).tolist())
    prediction = utterance_transform.bpe.decode(ctc_decoded)[0]

    print(prediction)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        f.write(prediction)


if __name__ == "__main__":
    main()
