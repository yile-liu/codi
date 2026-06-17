#!/bin/bash
#SBATCH -A berzelius-2026-167
#SBATCH -C thin
#SBATCH --gpus=2
#SBATCH -t 02:00:00
#SBATCH -J eval_crux
#SBATCH -o eval_crux_%j.log

# Stage-1 baseline eval: CRUXEval-O full-trace pass@1 (greedy). 2x A100 DDP.
# MODEL/N/OUT overridable; defaults = SFT checkpoint-6928 smoke (50 rows).
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
# CRUXEval-O held-out test set (local save_to_disk copy; avoids the offline Hub fetch).
export CRUXEVAL_DIR="${CRUXEVAL_DIR:-/proj/assert-berzelius/users/x_sirli/datasets/cruxeval_test}"

torchrun --nproc_per_node=2 --master_port=$((20000 + RANDOM % 10000)) eval_cruxeval.py \
  --model "${MODEL:-model_weights/sft-coder-1.5b/checkpoint-6928}" \
  --n_samples "${N:--1}" \
  --out "${OUT:-results/cruxeval_ckpt6928.json}"
