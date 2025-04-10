#!/bin/bash

# --- 基础配置 ---
MODEL_NAME_OR_PATH="/home/zqq/model/Qwen-7B-Chat" # 你的基础模型路径
DATASET_NAME="traindata"                 # 数据集名称 (需要与 data/ 目录下 yaml 文件名或 Hugging Face Hub 名称对应)
OUTPUT_DIR="./saves/Qwen-7B-Chat/lora/newlora2" # 输出目录

# --- 断点训练配置 ---
# CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoint-latest"  # 断点目录
# RESUME_FROM_CHECKPOINT="--resume_from_checkpoint ${CHECKPOINT_DIR}"  # 断点续训参数

# --- 分布式训练配置 ---
# NUM_GPUS=2  # 使用的GPU数量
# MASTER_PORT=29500  # 主节点端口
# DISTRIBUTED_ARGS="--multi_gpu --num_gpus ${NUM_GPUS} --master_port ${MASTER_PORT}"  # 分布式训练参数

# --- LoRA 和动态秩配置 ---
LORA_RANK=16                     # 基础秩 (可以尝试 8, 16, 32)
LORA_ALPHA=32                    # LoRA Alpha (通常是 rank 的 2 倍)
LORA_TARGET="all"                # 应用 LoRA 的目标模块 ("all" 或 "q_proj,v_proj" 等)
MIN_RANK=8                       # 动态秩最小值
MAX_RANK=48                      # 动态秩最大值
RANK_PATTERN="late_bias"         # 动态秩模式 (尝试: late_bias, early_bias, gaussian, linear)
ATTENTION_BOOST=1.8              # 中层注意力提升因子 (尝试: 1.0, 1.5, 2.0)
RANK_SMOOTH_FACTOR=0.7           # 秩平滑因子 (得分权重)
RANK_BASE_FACTOR=0.3             # 秩平滑基础因子

# --- 训练超参数 ---
NUM_EPOCHS=4.0                   # 训练轮数
BATCH_SIZE=1                     # 降低batch_size
GRAD_ACCUM_STEPS=32             # 增加梯度累积步数
LEARNING_RATE=2e-4               # 学习率 (非常重要，需要调整，可以尝试 1e-4, 5e-5, 2e-5)
LR_SCHEDULER="cosine"            # 学习率调度器
WARMUP_RATIO=0.05                # Warmup 比例
LOGGING_STEPS=10                 # 每 N 步记录一次日志
SAVE_STEPS=100                  # 更频繁保存检查点
# --- 使用 BF16 ---
# FP16=true     # 确保注释掉或删除
BF16=true     # 启用 BF16
# --- 添加验证集分割 ---
VAL_SIZE=0.1
# --- 设置截断长度 ---
CUTOFF_LEN=1024                   # <--- 设置新的截断长度

# --- 其他 ---
TEMPLATE="qwen"                  # 使用的模型模板
STAGE="sft"                      # 训练阶段 (Supervised Fine-tuning)

# --- 只需保留这些环境变量 ---
# NCCL网络通信优化
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_VISIBLE_DEVICES=1    # 指定使用第二块GPU
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# --- 直接使用 python 命令替代 torchrun ---
python src/train.py \
    --stage ${STAGE} \
    --do_train \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dataset ${DATASET_NAME} \
    --template ${TEMPLATE} \
    --finetuning_type lora \
    --lora_target ${LORA_TARGET} \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --lr_scheduler_type ${LR_SCHEDULER} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --plot_loss \
    --bf16 ${BF16} \
    --warmup_ratio ${WARMUP_RATIO} \
    --use_dynamic_rank true \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --min_rank ${MIN_RANK} \
    --max_rank ${MAX_RANK} \
    --rank_pattern ${RANK_PATTERN} \
    --attention_boost_factor ${ATTENTION_BOOST} \
    --rank_smooth_factor ${RANK_SMOOTH_FACTOR} \
    --rank_base_factor ${RANK_BASE_FACTOR} \
    --trust_remote_code true \
    --val_size ${VAL_SIZE} \
    --evaluation_strategy steps \
    --eval_steps ${SAVE_STEPS} \
    --load_best_model_at_end true \
    --cutoff_len ${CUTOFF_LEN} \
    # ${RESUME_FROM_CHECKPOINT} \  # 断点续训参数
    # ${DISTRIBUTED_ARGS} \  # 分布式训练参数
    --gradient_checkpointing true

echo "训练完成！模型保存在: ${OUTPUT_DIR}"