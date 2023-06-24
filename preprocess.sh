export HF_HOME="/opt/dlami/nvme/hf_cache"
export HF_DATASETS_CACHE="/opt/dlami/nvme/hf_cache"
export CUDA_VISIBLE_DEVICES=0
unset OMPI_COMM_WORLD_LOCAL_RANK
source /fsx/home-bjoern/LMFlow/venv/bin/activate
python examples/preprocess.py $1
