#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-869
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weiyun.huang.2798@student.uu.se

set -euo pipefail

source "/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/ZODPO/jobs/common_header.sh"

echo "=== Sweep job ==="
echo "EXP_NAME : ${EXP_NAME}"
echo "TRAINER  : ${TRAINER}"
echo "lr=${LR} eps=${EPS} beta=${BETA} batch_size=${BS} gradient_accum=${GC} m_beta=${MBETA} rank=${RANK} lazy=${LAZY}"
echo "SFT      : ${SFT_MODEL_PATH}"
echo "================="

N_GPU=1

EXTRA_ARGS=""
if [[ "${TRAINER}" == "agzo" || "${TRAINER}" == "agzo_plain" ]]; then
    # Add AGZO-specific hyperparameters
    EXTRA_ARGS="trainer.power_iter_steps=5 trainer.rank=${RANK}"
fi

RANDOM_PORT=$((20000 + RANDOM % 40000))
# Use torchrun for multi-GPU (DDP); fall back to plain python for single GPU.
if [[ "${N_GPU}" -gt 1 ]]; then
    LAUNCHER="torchrun --nproc_per_node=${N_GPU} --master_port=$RANDOM_PORT"
else
    LAUNCHER="python"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

${LAUNCHER} train.py \
    trainer="${TRAINER}" \
    loss=dpo \
    compute_logps_fp32=True \
    max_loss_threshold=15.0 \
    max_steps=600 \
    eval_every=100 \
    eval_batches=500 \
    exp_name="${EXP_NAME}" \
    model.name_or_path="${SFT_MODEL_PATH}" \
    model.policy_dtype=float32 \
    loss.sft_model_path="${SFT_MODEL_PATH}" \
    loss.beta="${BETA}" \
    loss.num_epochs=1 \
    +trainer.momentum_beta=${MBETA} \
    +trainer.basis_update_freq=${LAZY} \
    trainer.lr="${LR}" \
    trainer.eps="${EPS}" \
    trainer.warmup_ratio=0.1 \
    trainer.lr_scheduler_type="cosine" \
    ${EXTRA_ARGS} \
    batch_size="${BS}" \
    gradient_accumulation_steps="${GC}" \
    max_length=2048 \
    max_prompt_length=1024 \
    checkpoint_every=1000 \
    save_total_limit=5 \
    wandb.enabled=true \
    wandb.project=zo-dpo-1.5b-base-fp32 \
    hydra.run.dir="${TMPDIR}/hydra/${EXP_NAME}" \
    hydra.output_subdir=null

rsync -avP /mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/checkpoints/ wehu2798@usrl-hal.it.uu.se:/media/tsar_bomba/wehu2798/checkpoints/

echo "Job ${SLURM_JOB_ID} done: ${EXP_NAME}"