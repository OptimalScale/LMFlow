export HF_TOKEN=""
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# BASE_OUTPUT_DIR="/root/autodl-tmp/eval_res/if/llama1b-raw"
# DEVICE_ID=0
# mkdir -p "$BASE_OUTPUT_DIR"
# nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &
# pid0=$!
# echo "waiting eval 0 complete..."
# wait $pid0
# echo "eval 0 complete"

# MODEL_NAME="/root/autodl-tmp/models/llama1b-top10-unif"
# BASE_OUTPUT_DIR="/root/autodl-tmp/eval_res/if/llama1b-top10-unif"
# DEVICE_ID=0
# mkdir -p "$BASE_OUTPUT_DIR"
# nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &
# pid1=$!
# echo "waiting eval 1 complete..."
# wait $pid1
# echo "eval 1 complete"

# MODEL_NAME="/root/autodl-tmp/models/llama1b-all-unif"
# BASE_OUTPUT_DIR="/root/autodl-tmp/eval_res/if/llama1b-all-unif"
# DEVICE_ID=0
# mkdir -p "$BASE_OUTPUT_DIR"
# nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &
# pid2=$!
# echo "waiting eval 2 complete..."
# wait $pid2
# echo "eval 2 complete"

# MODEL_NAME="/root/autodl-tmp/models/llama1b-top10-weighted"
# BASE_OUTPUT_DIR="/root/autodl-tmp/eval_res/if/llama1b-top10-weighted"
# DEVICE_ID=0
# mkdir -p "$BASE_OUTPUT_DIR"
# nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &
# pid3=$!
# echo "waiting eval 3 complete..."
# wait $pid3
# echo "eval 3 complete"

MODEL_NAME="/root/autodl-tmp/models/llama1b-all-weighted"
BASE_OUTPUT_DIR="/root/autodl-tmp/eval_res/if/llama1b-all-weighted"
DEVICE_ID=0
mkdir -p "$BASE_OUTPUT_DIR"
nohup bash scalebio_instf/eval_if.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &
pid4=$!
echo "waiting eval 4 complete..."
wait $pid4
echo "eval 4 complete"