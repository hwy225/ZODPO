#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-869
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100:1
#SBATCH -t 8:00:00
# (output/error files set by sweep_dpo.sh via --output/--error)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weiyun.huang.2798@student.uu.se

set -euo pipefail

source "/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/ZODPO/jobs/common_header.sh"

# All variables injected by sweep_dpo.sh via --export:
#   EXP_NAME, SFT_MODEL_PATH, LR, EPS, BETA, TRAINER

echo "=== Sweep job ==="
echo "EXP_NAME : ${EXP_NAME}"
echo "TRAINER  : ${TRAINER}"
echo "lr=${LR}  eps=${EPS}  beta=${BETA} batch_size=${BS} gradient_accum=${GC}"
echo "SFT      : ${SFT_MODEL_PATH}"
echo "================="

# Build the trainer-specific extra args.
# power_iter_steps and rank only exist in agzo/agzo_plain configs,
# passing them to mezo causes a Hydra struct error.
EXTRA_ARGS=""
if [[ "${TRAINER}" == "agzo" || "${TRAINER}" == "agzo_plain" ]]; then
    EXTRA_ARGS="trainer.power_iter_steps=5 trainer.rank=1"
fi

python train_ddp.py \
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
    wandb.project=zo-dpo-1.7b-base \
    hydra.run.dir="${TMPDIR}/hydra/${EXP_NAME}" \
    hydra.output_subdir=null

echo "Job ${SLURM_JOB_ID} done: ${EXP_NAME}"
