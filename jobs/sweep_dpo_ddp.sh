#!/usr/bin/env bash
# =============================================================================
# sweep_dpo.sh  —  submit a grid of DPO hyperparameter experiments
#
# Run from ZODPO/:
#   bash jobs/sweep_dpo.sh agzo sft_qwen1.7b
#
# Args:
#   $1  trainer name: mezo | agzo | agzo_plain
#   $2  SFT exp name (the directory under $RUNS_DIR)
# =============================================================================

TRAINER="${1:?Usage: bash sweep_dpo.sh <trainer> <sft_exp_name>}"
SFT_EXP="${2:?Usage: bash sweep_dpo.sh <trainer> <sft_exp_name>}"

# ── Hyperparameter grid ───────────────────────────────────────────────────────
# Each row: "lr eps beta bs gc n_gpu"
#   n_gpu=1  → single GPU (python train.py)
#   n_gpu=2  → 2× A100 DDP (torchrun --nproc_per_node=2)
# Effective batch size per optimizer step = bs * gc
CONFIGS=(
    # "1e-7  1e-3  0.1 4 16 1"
    # "1e-7  1e-3  0.1 4 16 2"
    "1e-6  1e-4  0.1 4 16 2"
)

SHARED_ROOT="/mimer/NOBACKUP/groups/ga_llm_hri"
MY_ROOT="${SHARED_ROOT}/weiyun_zodpo"
RUNS_DIR="${MY_ROOT}/runs"
SFT_MODEL_PATH="${RUNS_DIR}/${SFT_EXP}/sft/final_model"

if [ ! -d "${SFT_MODEL_PATH}" ]; then
    echo "ERROR: SFT model not found at ${SFT_MODEL_PATH}"; exit 1
fi

mkdir -p "${MY_ROOT}/logs"

for config in "${CONFIGS[@]}"; do
    read -r lr eps beta bs gc n_gpu <<< "$config"

    lr_tag=$(echo "$lr"  | sed 's/e-/em/;s/\./_/')
    eps_tag=$(echo "$eps" | sed 's/e-/em/;s/\./_/')
    b_tag=$(echo "$beta"  | sed 's/0\./b/;s/\.//')

    EXP_NAME="dpo_${TRAINER}_bs${bs}_gc${gc}_ng${n_gpu}_lr${lr_tag}_eps${eps_tag}_beta${b_tag}_$(date +%Y%m%d)"

    echo "Submitting: $EXP_NAME  (n_gpu=${n_gpu})"

    sbatch \
        --job-name="sweep_${TRAINER}_ng${n_gpu}" \
        --gpus-per-node="A40:${n_gpu}" \
        --output="${MY_ROOT}/logs/${EXP_NAME}_%j.out" \
        --error="${MY_ROOT}/logs/${EXP_NAME}_%j.err" \
        --export=ALL,EXP_NAME="$EXP_NAME",SFT_MODEL_PATH="$SFT_MODEL_PATH",\
LR="$lr",EPS="$eps",BETA="$beta",BS="$bs",GC="$gc",N_GPU="$n_gpu",TRAINER="$TRAINER" \
        "$(dirname "$0")/sweep_dpo_worker_ddp.sh"

    sleep 1
done

echo ""
echo "Submitted ${#CONFIGS[@]} jobs for trainer=${TRAINER}"
echo "Monitor: squeue -u \$USER"
echo "W&B:     https://wandb.ai (project: zo-dpo-1.7b-base)"