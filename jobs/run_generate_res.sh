#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-869
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 6:00:00
#SBATCH -J llm_eval               
#SBATCH -o logs/eval_%j.out       
#SBATCH -e logs/eval_%j.err       
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weiyun.huang.2798@student.uu.se

set -euo pipefail
SHARED_ROOT="/mimer/NOBACKUP/groups/ga_llm_hri"
MY_ROOT="${SHARED_ROOT}/weiyun_zodpo"

# ZODPO repo root — parent of the jobs/ directory this file is in
REPO_ROOT="${MY_ROOT}/ZODPO"

# train.py is in test/, and Hydra expects config/ to be a sibling of train.py
export CODE_DIR="${REPO_ROOT}/src"

export VENV_DIR="${MY_ROOT}/venv_eval"

# ── Reuse the shared HuggingFace cache — no re-downloads needed ───────────────
export HF_HOME="${SHARED_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

export TORCH_HOME="${SHARED_ROOT}/.cache/torch"
export TRITON_CACHE_DIR="${SHARED_ROOT}/.cache/triton"
export TORCH_EXTENSIONS_DIR="${SHARED_ROOT}/.cache/torch_extensions"

echo "=== Starting LLM Evaluation Pipeline ==="
date

module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0
source $VENV_DIR/bin/activate
cd $CODE_DIR/LLM-as-a-judge

OUTPUT_DIR="/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/outputs"

# Rsync models from lab server to local scratch for faster loading
LOCAL_MODEL_DIR="${TMPDIR}/my_models"
mkdir -p $LOCAL_MODEL_DIR

echo "[INFO] Loading models from lab server to local scratch ($LOCAL_MODEL_DIR)..."
SHARED_ROOT="wehu2798@usrl-hal.it.uu.se:/media/tsar_bomba/wehu2798/runs"

rsync -az --info=progress2 $SHARED_ROOT/dpo_bf16_mezo_bs4_gc32_lr2em6_eps5em5_mb0_20260424 $LOCAL_MODEL_DIR/
rsync -az --info=progress2 $SHARED_ROOT/dpo_bf16_agzo_bs4_gc32_lr2em6_eps5em5_mb5_20260424 $LOCAL_MODEL_DIR/
rsync -az --info=progress2 $SHARED_ROOT/trl_baseline_a40/checkpoint-1257 $LOCAL_MODEL_DIR/

echo "[INFO] Rsync complete! Files in $LOCAL_MODEL_DIR:"
ls -lh $LOCAL_MODEL_DIR


echo -e "\n[INFO] Starting response generation..."
python generate_responses.py \
  --model_paths \
    /mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/runs/sft_qwen1.5b-lr2em5/sft/final_model \
    $LOCAL_MODEL_DIR/checkpoint-1257 \
    $LOCAL_MODEL_DIR/dpo_bf16_mezo_bs4_gc32_lr2em6_eps5em5_mb0_20260424/dpo/final_model \
    $LOCAL_MODEL_DIR/dpo_bf16_agzo_bs4_gc32_lr2em6_eps5em5_mb5_20260424/dpo/final_model \
  --model_names sft fodpo mezodpo agzodpo \
  --output_dir $OUTPUT_DIR/responses/ \
  --num_samples 500 \
  --temperature 0.7 \
  --seed 42 \
  --cache_dir $HF_DATASETS_CACHE

echo "=== Pipeline finished successfully ==="
date