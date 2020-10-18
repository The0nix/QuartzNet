docker run \
	-it \
	--memory=16g \
	--memory-swap=2g \
	--shm-size=16g \
	--cpuset-cpus=0-11 \
	--gpus '"device=0"' \
	--volume /street/data:/home/user/data \
	--volume $(pwd)/config:/home/user/config \
	--volume $(pwd)/outputs:/home/user/outputs \
	--volume $(pwd)/files:/home/user/files \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	quartznet-tamerlan-tabolov
