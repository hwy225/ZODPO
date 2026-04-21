#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-869
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100:4
#SBATCH -t 10:00:00
#SBATCH -o /mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/logs/dpo_baseline_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/logs/dpo_baseline_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weiyun.huang.2798@student.uu.se

set -euo pipefail

source "/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/ZODPO/jobs/common_header.sh"

export EXP_NAME="trl_baseline"
export SFT_MODEL_PATH="/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/runs/sft_qwen1.5b-lr2em5/sft/final_model" # ⚠️ 记得填入你的本地路径
export LR="1e-6"
export BETA="0.1"

export BS=1
export GC=16 # Effective BS = 1 * 16 * 4(GPUs) = 64

torchrun --nproc_per_node=4 train_baseline_trl.py \
    trainer="fo_baseline" \
    loss=dpo \
    exp_name="${EXP_NAME}" \
    model.name_or_path="${SFT_MODEL_PATH}" \
    model.policy_dtype=bfloat16 \
    loss.sft_model_path="${SFT_MODEL_PATH}" \
    loss.beta="${BETA}" \
    loss.num_epochs=1 \
    trainer.lr="${LR}" \
    batch_size="${BS}" \
    gradient_accumulation_steps="${GC}" \
    max_length=2048 \
    max_prompt_length=1024 \
    wandb.enabled=true \
    wandb.project=zo-dpo-1.5b-base \
    hydra.run.dir="${TMPDIR}/hydra/${EXP_NAME}" \
    hydra.output_subdir=null

echo "Job ${SLURM_JOB_ID} done: ${EXP_NAME}"