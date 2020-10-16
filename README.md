# QuartzNet
Implementation of QuartzNet ASR model in PyTorch

## Usage
### Training
To launch training in nvidia-docker container follow these instructions:

0. Install [nvidia-docker]([https://github.com/NVIDIA/nvidia-docker])
1. Run `./docker-build.sh`
3. Set preferred configurations in `config/config.yaml`. In particular you might want to set `dataset`: it can be either `numbers` or `librispeech`
4. In `docker-run.sh` change `memory`, `memory-swap`, `shm-size`, `cpuset-cpus`, `gpus`, and data `volume` to desired values
5. Set WANDB_API_KEY environment variable to your wandb key
6. Run `./docker-run.sh`

All outputs will be saved to `outputs` dir.
