#!/usr/bin/env bash
# =============================================================================
# sweep_dpo_worker.sh  —  single DPO job, parameterised by env vars
#
# Do NOT submit directly. Called by sweep_dpo.sh via sbatch --export.
# Env vars: EXP_NAME, SFT_MODEL_PATH, LR, EPS, BETA, BS, GC, TRAINER, N_GPU
# =============================================================================

#SBATCH -A NAISS2025-22-869
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:${N_GPU:-1}
#SBATCH -t 4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weiyun.huang.2798@student.uu.se

set -euo pipefail

source "/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/ZODPO/jobs/common_header.sh"

# Default to 1 GPU if N_GPU not set
N_GPU="${N_GPU:-1}"

echo "=== Sweep job ==="
echo "EXP_NAME : ${EXP_NAME}"
echo "TRAINER  : ${TRAINER}"
echo "lr=${LR}  eps=${EPS}  beta=${BETA}  bs=${BS}  gc=${GC}  n_gpu=${N_GPU}"
echo "SFT      : ${SFT_MODEL_PATH}"
echo "================="

EXTRA_ARGS=""
if [[ "${TRAINER}" == "agzo" || "${TRAINER}" == "agzo_plain" ]]; then
    EXTRA_ARGS="trainer.power_iter_steps=5 trainer.rank=1"
fi

# Use torchrun for multi-GPU (DDP); fall back to plain python for single GPU.
if [[ "${N_GPU}" -gt 1 ]]; then
    LAUNCHER="torchrun --nproc_per_node=${N_GPU} --master_port=29500"
else
    LAUNCHER="python"
fi

${LAUNCHER} train_ddp.py \
    trainer="${TRAINER}" \
    loss=dpo \
    compute_logps_fp32=True \
    max_loss_threshold=15.0 \
    exp_name="${EXP_NAME}" \
    model.name_or_path="${SFT_MODEL_PATH}" \
    model.policy_dtype=float32 \
    loss.sft_model_path="${SFT_MODEL_PATH}" \
    loss.beta="${BETA}" \
    loss.num_epochs=1 \
    trainer.lr="${LR}" \
    trainer.eps="${EPS}" \
    ${EXTRA_ARGS} \
    batch_size="${BS}" \
    gradient_accumulation_steps="${GC}" \
    max_length=2048 \
    max_prompt_length=1024 \
    wandb.enabled=true \
    wandb.project=zo-dpo-1.5b-base \
    hydra.run.dir="${TMPDIR}/hydra/${EXP_NAME}" \
    hydra.output_subdir=null

echo "Job ${SLURM_JOB_ID} done: ${EXP_NAME}"