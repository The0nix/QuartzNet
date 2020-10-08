docker run \
	-it \
	--memory=16g \
	--memory-swap=2g \
	--cpuset-cpus=0-6 \
	--gpus '"device=0"' \
	--volume ./data:/data \
	--volume ./output:/outputs \
	--workdir /home/user/summer_immersion2020 \
	quartznet
