#!/bin/bash
#SBATCH -A naiss2026-3-261
# -C selects node type (nodes with 4×A100-40GB); --gpus-per-node allocates how many this job uses.
#SBATCH -C 4xA100
#SBATCH --gpus-per-node=A100:4
#SBATCH -t 2-00:00:00
#SBATCH -J sft_trace_3b
#SBATCH -o sft_trace_3b_%j.log

# Pre-requisite: run precompute.py --mode sft --out data/cache/sft_train before submitting.
export PATH=/mimer/NOBACKUP/groups/assert-kth/siruil/conda_envs/CWM/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 --master_port=$((20000 + RANDOM % 10000)) train_sft.py \
  --model model_weights/qwen2.5-coder-3b \
  --output_dir model_weights/sft-coder-3b \
  --cache_dir data/cache/sft_train \
  --epochs 5 \
  --lr 2e-5 \
  --batch_size 1 \
  --grad_accum 8
