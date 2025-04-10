#!/bin/bash

# 激活虚拟环境
source /home/zqq/anaconda3/bin/activate llama

# 设置CUDA可见设备（使用GPU 0和1）
export CUDA_VISIBLE_DEVICES=0  # 启用多GPU

# 优化PyTorch显存分配策略（减少碎片化）
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# 强制清空CUDA缓存（需在Python中执行）
python -c "import torch; torch.cuda.empty_cache()"

# NCCL网络通信优化
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"


# 进入项目目录并启动Web界面
cd /home/zqq/LLaMA-Factory

# 启动服务（添加多GPU支持参数，假设使用DataParallel）
python src/webui.py  # 根据实际代码调整参数

# 停止监控任务
kill $MONITOR_PID