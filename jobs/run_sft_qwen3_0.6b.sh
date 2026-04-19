#!/usr/bin/env bash

#SBATCH -A NAISS2025-22-869
#SBATCH -J sft-qwen3-0.6b
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100:1      # 1x A100
#SBATCH -t 6:00:00
#SBATCH -o /mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/logs/sft_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/logs/sft_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weiyun.huang.2798@student.uu.se

set -euo pipefail

mkdir -p /mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/logs

source "/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/ZODPO/jobs/common_header.sh"

# Give each run a unique, readable name
EXP_NAME="sft_qwen0.6b-lr5em5"

python train.py \
    loss=sft \
    exp_name="${EXP_NAME}" \
    model.name_or_path=Qwen/Qwen3-0.6B-Base \
    model.policy_dtype=bfloat16 \
    batch_size=4 \
    gradient_accumulation_steps=16 \
    max_length=2048 \
    max_prompt_length=1024 \
    loss.lr=5e-5 \
    loss.lr_scheduler_type=cosine \
    loss.warmup_ratio=0.03 \
    loss.num_train_epochs=1 \
    hf_cache_dir="${HF_HOME}" \
    hf_dataset_cache_dir="${HF_DATASETS_CACHE}" \
    checkpoint_dir="${CHECKPOINT_DIR}" \
    runs_dir="${RUNS_DIR}" \
    checkpoint_every=50 \
    wandb.enabled=true \
    wandb.project=zo-dpo-0.6b-base \
    hydra.run.dir="${TMPDIR}/hydra/${EXP_NAME}" \
    hydra.output_subdir=null

echo "SFT job ${SLURM_JOB_ID} done."
echo "Model: ${RUNS_DIR}/${EXP_NAME}/sft/final_model"
echo ""
echo "Before submitting DPO jobs, update SFT_EXP_NAME in each job_dpo_*.sh:"
echo "  SFT_EXP_NAME=\"${EXP_NAME}\""
