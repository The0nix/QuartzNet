tmp_dir=$(mktemp -d -t inference-XXXXXXXXXX)
cp $1 $tmp_dir  # Copy model into temp for docker
cp $3 $tmp_dir  # Copy input file

docker run \
	-it \
	--memory=16g \
	--memory-swap=2g \
	--shm-size=16g \
	--cpuset-cpus=0-11 \
	--gpus '"device=0"' \
	--volume $(pwd)/config:/home/user/config \
	--volume $(pwd)/outputs:/home/user/outputs \
	--volume $(pwd)/files:/home/user/files \
	--volume $(pwd)/inferenced:/home/user/inferenced \
	--volume $tmp_dir:/home/user/inference_files \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	quartznet-tamerlan-tabolov \
	bash -c "
	  export PATH=\"/root/anaconda3/bin:'$PATH'\" && \
	  python ./src/scripts/train_bpe.py && \
    python ./src/inference.py \
    +inference.model_path=inference_files/$(basename $1) \
    +inference.device=$2 \
    +inference.file=inference_files/$(basename $3)
	"
rm -rf $tmp_dir
