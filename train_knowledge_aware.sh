#!/bin/bash

set -x  # Enable debug mode to print each command

# Activate llama environment
source /home/zqq/anaconda3/bin/activate llama
echo "Activated llama environment"

# Set environment variables for error tracking
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1

# Set CUDA device
export CUDA_VISIBLE_DEVICES=1
echo "Set CUDA device to 1"

# Set NCCL environment variables
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# Training parameters
echo "Starting training..."
CUDA_VISIBLE_DEVICES=1 python -m src.llamafactory.train.train_bash \
    --model_name_or_path /home/zqq/model/Qwen-7B-Chat \
    --train_file /home/zqq/LLaMA-Factory/data/split/train.json \
    --validation_file /home/zqq/LLaMA-Factory/data/split/validation.json \
    --finetuning_type lora \
    --use_dynamic_rank True \
    --rank_pattern gaussian \
    --middle_layer_factor 2.0 \
    --min_rank 4 \
    --max_rank 32 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --use_knowledge_distillation True \
    --teacher_model_name_or_path /home/zqq/model/Qwen-7B-Chat \
    --distillation_temperature 2.0 \
    --distillation_weight 0.5 \
    --output_dir ./output \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --fp16

echo "Training command completed"

# Start web interface
cd /home/zqq/LLaMA-Factory
python src/webui.py 