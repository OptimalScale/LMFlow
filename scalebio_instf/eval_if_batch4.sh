MODEL_NAME="google/gemma-3-1b-it"
BASE_OUTPUT_DIR="/mnt/yizhenjia/eval_res/scalebio/gemma1b/raw"
DEVICE_ID=0
mkdir -p "$BASE_OUTPUT_DIR"
nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &

MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
BASE_OUTPUT_DIR="/mnt/yizhenjia/eval_res/scalebio/llama1b/raw"
DEVICE_ID=1
mkdir -p "$BASE_OUTPUT_DIR"
nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &


MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
BASE_OUTPUT_DIR="/mnt/yizhenjia/eval_res/scalebio/llama3b/raw"
DEVICE_ID=2
mkdir -p "$BASE_OUTPUT_DIR"
nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &


MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
BASE_OUTPUT_DIR="/mnt/yizhenjia/eval_res/scalebio/qwen1.5b/raw"
DEVICE_ID=3
mkdir -p "$BASE_OUTPUT_DIR"
nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &


MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
BASE_OUTPUT_DIR="/mnt/yizhenjia/eval_res/scalebio/qwen3b/raw"
DEVICE_ID=4
mkdir -p "$BASE_OUTPUT_DIR"
nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &


MODEL_NAME="/home/yizhenjia/models/scalebio/gemma1b/10-unif-doubled"
BASE_OUTPUT_DIR="/mnt/yizhenjia/eval_res/scalebio/gemma1b/10-unif-doubled"
DEVICE_ID=5
mkdir -p "$BASE_OUTPUT_DIR"
nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &
