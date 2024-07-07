# Parses arguments
model_name_or_path=stabilityai/stable-diffusion-2-1
model_type="unet"
dataset_path=data/example
output_dir=output
main_port=29500
img_size=768

while [[ $# -ge 1 ]]; do
    key="$1"
    case ${key} in
        -m|--model_name_or_path)
            model_name_or_path="$2"
            shift
            ;;
        -t|--model_type)
            model_type="$2"
            shift
            ;;
        -d|--dataset_path)
            dataset_path="$2"
            shift
            ;;
        -o|--output_dir)
            output_dir="$2"
            shift
            ;;
        -p|--main_port)
            main_port="$2"
            shift
            ;;
        -i|--img_size)
            img_size="$2"
            shift
            ;;
        *)
            echo "error: unknown option \"${key}\"" 1>&2
            exit 1
    esac
    shift
done

echo "model_name_or_path: ${model_name_or_path}"
echo "model_type: ${model_type}"
echo "dataset_path: ${dataset_path}"
echo "output_dir: ${output_dir}"
echo "main_port: ${main_port}"
echo "img_size: ${img_size}"


accelerate launch \
    --config_file=./accelerate_t2i_config.yaml \
    --main_port=${main_port} \
    finetune_t2i.py \
        --model_name_or_path=${model_name_or_path} \
        --model_type=${model_type} \
        --use_lora=True \
        --lora_target_module "to_k" "to_q" "to_v" "to_out.0" "add_k_proj" "add_v_proj" \
        --dataset_path=${dataset_path} \
        --image_folder="img" \
        --image_size=${img_size} \
        --train_file="train.json" \
        --validation_file="valid.json" \
        --test_file="test.json" \
        --output_dir=${output_dir} \
        --logging_dir="logs" \
        --overwrite_output_dir=True \
        --mixed_precision="fp16" \
        --num_train_epochs=100 \
        --train_batch_size=1 \
        --learning_rate=1e-4 \
        --valid_steps=50
