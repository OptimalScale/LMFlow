MODEL_NAME="/home/yizhenjia/models/scalebio/llama1b/10-unif"
BASE_OUTPUT_DIR="/mnt/yizhenjia/eval_res/scalebio/llama1b/10-unif"
DEVICE_ID=0
mkdir -p "$BASE_OUTPUT_DIR"
nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &


MODEL_NAME="/home/yizhenjia/models/scalebio/llama1b/10-unif-doubled"
BASE_OUTPUT_DIR="/mnt/yizhenjia/eval_res/scalebio/llama1b/10-unif-doubled"
DEVICE_ID=1
mkdir -p "$BASE_OUTPUT_DIR"
nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &


MODEL_NAME="/home/yizhenjia/models/scalebio/llama1b/15-unif"
BASE_OUTPUT_DIR="/mnt/yizhenjia/eval_res/scalebio/llama1b/15-unif"
DEVICE_ID=2
mkdir -p "$BASE_OUTPUT_DIR"
nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &


MODEL_NAME="/home/yizhenjia/models/scalebio/llama1b/10-weighted"
BASE_OUTPUT_DIR="/mnt/yizhenjia/eval_res/scalebio/llama1b/10-weighted"
DEVICE_ID=3
mkdir -p "$BASE_OUTPUT_DIR"
nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &


MODEL_NAME="/home/yizhenjia/models/scalebio/llama1b/10-weighted-doubled"
BASE_OUTPUT_DIR="/mnt/yizhenjia/eval_res/scalebio/llama1b/10-weighted-doubled"
DEVICE_ID=4
mkdir -p "$BASE_OUTPUT_DIR"
nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &


MODEL_NAME="/home/yizhenjia/models/scalebio/llama1b/15-weighted"
BASE_OUTPUT_DIR="/mnt/yizhenjia/eval_res/scalebio/llama1b/15-weighted"
DEVICE_ID=5
mkdir -p "$BASE_OUTPUT_DIR"
nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &

