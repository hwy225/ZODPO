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
# Start with eps — it has the biggest effect on ZO stability.
# Each row: "lr eps beta bs gc"
CONFIGS=(
    # Sweep eps with fixed lr and beta
    # "1e-6  1e-4  0.1"
    # "1e-6  5e-4  0.1"
    # "1e-6  1e-3  0.1"

    # Sweep lr with best eps candidate
    # "1e-7  1e-4  0.1"
    # "5e-6  1e-4  0.1"

    # Sweep beta with best (lr, eps)
    # "1e-6  1e-4  0.01"
    # "1e-6  1e-4  0.05"

    # "1e-7  1e-3  0.1 4 16"
    # "1e-7  1e-4  0.1 4 16"
    # "2e-7  1e-3  0.1 4 32"
    # "2e-7  1e-4  0.1 4 32"
    # "5e-7  1e-3  0.1 4 64"
    # "5e-7  1e-4  0.1 4 64"
    "1e-6  1e-3  0.1 2 8"
    # "1e-7  1e-3  0.1 4 32"
    # "5e-8  1e-3  0.1 4 16"
    # "5e-8  1e-3  0.1 4 32"
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
    read -r lr eps beta bs gc <<< "$config"

    # Format for exp name: 1e-6 -> 1e6, 1e-4 -> 1e4
    lr_tag=$(echo "$lr"  | sed 's/e-/em/;s/\./_/')
    eps_tag=$(echo "$eps" | sed 's/e-/em/;s/\./_/')
    b_tag=$(echo "$beta"  | sed 's/0\./b/;s/\.//')
    bs_tag=$(echo "$bs")
    gc_tag=$(echo "$gc")

    # EXP_NAME="dpo_${TRAINER}_bs${bs_tag}_gc${gc_tag}_lr${lr_tag}_eps${eps_tag}_beta${b_tag}_$(date +%Y%m%d)"
    EXP_NAME="dpo_${TRAINER}_bs${bs_tag}_gc${gc_tag}_lr${lr_tag}_eps${eps_tag}_$(date +%Y%m%d)"

    echo "Submitting: $EXP_NAME"

    sbatch \
        --job-name="${TRAINER}_bs${bs_tag}_gc${gc_tag}_lr${lr_tag}_eps${eps_tag}" \
        --output="${MY_ROOT}/logs/${EXP_NAME}_%j.out" \
        --error="${MY_ROOT}/logs/${EXP_NAME}_%j.err" \
        --export=ALL,EXP_NAME="$EXP_NAME",SFT_MODEL_PATH="$SFT_MODEL_PATH",\
LR="$lr",EPS="$eps",BETA="$beta",BS="$bs",GC="$gc",TRAINER="$TRAINER" \
        "$(dirname "$0")/run_baseline.sh"

    sleep 1   # avoid SLURM rate limit
done

echo ""
echo "Submitted ${#CONFIGS[@]} jobs for trainer=${TRAINER}"
echo "Monitor: squeue -u \$USER"
echo "W&B:     https://wandb.ai (project: zo-dpo, filter by trainer=${TRAINER})"
