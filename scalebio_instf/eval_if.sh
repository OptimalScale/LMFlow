#!/bin/bash
# args
MODEL_NAME=$1
BASE_OUTPUT_DIR=$2
DEVICE_ID=$3

# cofings
# tasks=(
#     bbh
#     arc_easy
#     arc_challenge
#     hellaswag
#     openbookqa
#     piqa
#     social_iqa
#     winogrande
#     wikitext
#     gpqa
#     minerva_math
#     gsm8k
# )
tasks=(
    # commonsense_qa
    ifeval
    mmlu
    # agieval
    # truthfulqa
)

declare -A task_batch_sizes=(
    [agieval]=16
    [ifeval]=128
    [truthfulqa]=128
    [mmlu]=2
    [arc_easy]=128
    [arc_challenge]=128
    [hellaswag]=128
    [openbookqa]=128
    [piqa]=128
    [social_iqa]=128
    [winogrande]=128
    [wikitext]=4
    [gpqa]=4
    [minerva_math]=128
    [gsm8k]=32
)

declare -A task_few_shots=(
    [bbh]=3
    [truthfulqa]=1
    [mmlu]=5
)

DEFAULT_BATCH_SIZE="auto"
DEFAULT_NUM_FEWSHOT=0

mkdir -p "$BASE_OUTPUT_DIR"

export HF_ALLOW_CODE_EVAL="1"
export CUDA_VISIBLE_DEVICES=$DEVICE_ID

echo "Starting evaluation loop for ${#tasks[@]} tasks..."

for task in "${tasks[@]}"; do
    echo "--------------------------------------------------"
    echo "$(date): Starting task: $task"
    echo "--------------------------------------------------"

    current_batch_size=${task_batch_sizes[$task]:-$DEFAULT_BATCH_SIZE}
    echo "Using Batch Size: $current_batch_size"
    
    current_num_fewshot=${task_few_shots[$task]:-$DEFAULT_NUM_FEWSHOT}
    echo "Using Few-Shot Examples: $current_num_fewshot"

    TASK_OUTPUT_DIR="$BASE_OUTPUT_DIR/$task"
    LOG_FILE="$TASK_OUTPUT_DIR/${task}.log"
    mkdir -p "$TASK_OUTPUT_DIR"

    command="lm_eval --model hf \
        --model_args pretrained=$MODEL_NAME \
        --tasks $task \
        --batch_size $current_batch_size \
        --cache_requests refresh \
        --num_fewshot $current_num_fewshot \
        --output_path $TASK_OUTPUT_DIR \
        --log_samples \
        --confirm_run_unsafe_code \
        --apply_chat_template"
    echo "Running command:"
    echo "$command"
    echo "Logging to: $LOG_FILE"

    $command > "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        echo "$(date): Task $task completed successfully."
    else
        echo "$(date): Task $task FAILED. Check log: $LOG_FILE. Continuing to next task."
    fi
    echo " "

done

echo "=================================================="
echo "$(date): All tasks processed."
echo "Results and logs saved in subdirectories under $BASE_OUTPUT_DIR"
echo "=================================================="