MODEL_NAME="wheresmyhair/dsqw_dstl_unif_default"
BASE_OUTPUT_DIR="/root/autodl-tmp/eval_res/mmlu/dsqw-1b-pruned"
DEVICE_ID=0

mkdir -p "$BASE_OUTPUT_DIR"
nohup bash scalebio_instf/eval_mmlu.sh $MODEL_NAME $BASE_OUTPUT_DIR $DEVICE_ID > $BASE_OUTPUT_DIR/eval.log 2>&1 &
pid2=$!
echo "waiting eval 2 complete..."
wait $pid2
echo "eval 2 complete"