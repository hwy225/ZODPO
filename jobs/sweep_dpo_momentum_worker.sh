#!/usr/bin/env bash
# =============================================================================
# sweep_momentum_worker.sh
# =============================================================================

#SBATCH -A NAISS2025-22-869
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:2
#SBATCH -t 4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weiyun.huang.2798@student.uu.se

set -euo pipefail

source "/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/ZODPO/jobs/common_header.sh"

echo "=== Sweep job ==="
echo "EXP_NAME : ${EXP_NAME}"
echo "TRAINER  : ${TRAINER}"
echo "lr=${LR} eps=${EPS} beta=${BETA} batch_size=${BS} gradient_accum=${GC} m_beta=${MBETA} rank=${RANK}"
echo "SFT      : ${SFT_MODEL_PATH}"
echo "================="

N_GPU=2

EXTRA_ARGS=""
if [[ "${TRAINER}" == "agzo" || "${TRAINER}" == "agzo_plain" ]]; then
    # 将接收到的 RANK 和 MBETA 参数传入 Hydra 配置
    EXTRA_ARGS="trainer.power_iter_steps=5 trainer.rank=${RANK} +trainer.momentum_beta=${MBETA}"
fi

RANDOM_PORT=$((20000 + RANDOM % 40000))
# Use torchrun for multi-GPU (DDP); fall back to plain python for single GPU.
if [[ "${N_GPU}" -gt 1 ]]; then
    LAUNCHER="torchrun --nproc_per_node=${N_GPU} --master_port=$RANDOM_PORT"
else
    LAUNCHER="python"
fi

# 🚨 警告：这里已经帮你强制改回了 float32！绝对不能用 bfloat16！
${LAUNCHER} train_momentum.py \
    trainer="${TRAINER}" \
    loss=dpo \
    compute_logps_fp32=True \
    max_loss_threshold=15.0 \
    exp_name="${EXP_NAME}" \
    model.name_or_path="${SFT_MODEL_PATH}" \
    model.policy_dtype=bfloat16 \
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