# QuartzNet
Implementation of [QuartzNet](https://arxiv.org/abs/1910.10261) ASR model in PyTorch

## Usage

### Setup
To launch and inference in nvidia-docker container follow these instructions:

0. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
1. Run `./docker-build.sh`

### Training
To launch training follow these instructions:

1. Set preferred configurations in `config/config.yaml`. In particular you might want to set `dataset`: it can be either `numbers` or `librispeech`
2. In `docker-run.sh` change `memory`, `memory-swap`, `shm-size`, `cpuset-cpus`, `gpus`, and data `volume` to desired values
3. Set WANDB_API_KEY environment variable to your wandb key
4. Run `./docker-train.sh`

All outputs including models will be saved to `outputs` dir.

### Inference
To launch inference run the following command:
```
./docker-inference.sh model_path device bpe_path input_path
```
Where:
* `model_path` is a path to .pth model file
* `device` is the device to inference on: either 'cpu', 'cuda' or cuda device number
* `bpe_path` is a path to yttm bpe model .model file
* `input_path` is a path to input audio file to parse text from

Predicted output will be printed to stdout and saved into a file in `inferenced` folder

## Pretrained models
My currently best model trained on librispeech and the respective config can be downloaded [here](https://drive.google.com/drive/folders/1sOEUeHY_KlZY6BNYfJtM6RyKFwNFwFSg?usp=sharing).

It is not very good however because I only trained it to ~59 WER on librispeech 
