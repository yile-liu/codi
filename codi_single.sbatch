#!/bin/bash
#SBATCH -A berzelius-2026-167
#SBATCH -C thin
#SBATCH --gpus=4
#SBATCH -t 1-00:00:00
#SBATCH -J codi_single
#SBATCH -o codi_single_%j.log

# Single-anchor CODI ablation (vanilla CODI): same per-frame latent budget as codi.sbatch,
# but KD only at the last frame. Must share kd_layers with codi.sbatch for a fair comparison.
export PATH=/proj/assert-berzelius/users/x_sirli/conda/envs/CWM/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 --master_port=$((20000 + RANDOM % 10000)) train_codi.py \
  --model model_weights/sft-coder-1.5b \
  --output_dir model_weights/codi-single-coder-1.5b \
  --cache_dir data/cache/codi_train \
  --max_seq_len 2048 \
  --latent_steps 1 \
  --epochs 10 \
  --lr 1e-5 \
  --batch_size 1 \
  --grad_accum 4 \
  --single_anchor
