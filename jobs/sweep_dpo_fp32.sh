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
# Each row: "lr eps beta bs gc rank"
CONFIGS=(
    # "1e-7  1e-3  0.1 4 16 4"
    # "1e-7  1e-3  0.1 4 16 8"
    "1e-7  1e-3  0.1 4 32 1"
    "1e-6  1e-4  0.1 4 32 1"
    # "1e-5  1e-3  0.1 8 64 1"
    
    # "5e-6  1e-4  0.1 4 32 4"
    # "5e-6  1e-4  0.1 4 32 8"
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
    read -r lr eps beta bs gc rank <<< "$config"

    # Format for exp name: 1e-6 -> 1e6, 1e-4 -> 1e4
    lr_tag=$(echo "$lr"  | sed 's/e-/em/;s/\./_/')
    eps_tag=$(echo "$eps" | sed 's/e-/em/;s/\./_/')
    b_tag=$(echo "$beta"  | sed 's/0\./b/;s/\.//')
    bs_tag=$(echo "$bs")
    gc_tag=$(echo "$gc")
    rank_tag=$(echo "$rank")

    EXP_NAME="dpo_nspsa_fp32_${TRAINER}_bs${bs_tag}_gc${gc_tag}_lr${lr_tag}_eps${eps_tag}_beta${b_tag}_rank${rank_tag}_$(date +%Y%m%d)"

    echo "Submitting: $EXP_NAME"

    sbatch \
        --job-name="sweep_${TRAINER}" \
        --output="${MY_ROOT}/logs/${EXP_NAME}_%j.out" \
        --error="${MY_ROOT}/logs/${EXP_NAME}_%j.err" \
        --export=ALL,EXP_NAME="$EXP_NAME",SFT_MODEL_PATH="$SFT_MODEL_PATH",\
LR="$lr",EPS="$eps",BETA="$beta",BS="$bs",GC="$gc",RANK="$rank",TRAINER="$TRAINER" \
        "$(dirname "$0")/sweep_dpo_worker_fp32.sh"

    sleep 1   # avoid SLURM rate limit
done

echo ""
echo "Submitted ${#CONFIGS[@]} jobs for trainer=${TRAINER}"
echo "Monitor: squeue -u \$USER"
echo "W&B:     https://wandb.ai (project: zo-dpo-1.7b-base, filter by trainer=${TRAINER})"