#!/usr/bin/env bash
TRAINER="${1:?Usage: bash sweep_momentum.sh <trainer> <sft_exp_name>}"
SFT_EXP="${2:?Usage: bash sweep_momentum.sh <trainer> <sft_exp_name>}"

# ── Hyperparameter grid ───────────────────────────────────────────────────────
# Each row: "lr eps dpo_beta bs gc m_beta rank"
CONFIGS=(
    # "5e-6  5e-5  0.1  4  32  0.0   1" # best for MeZO
    # "5e-6  5e-5  0.1  4  32  0.5   4" # best for AGZO

    # MeZO
    # "5e-6  5e-5  0.1  4  32  0.0   1"
    # "2e-6  5e-5  0.1  4  32  0.0   1"

    # "2e-7  1e-4  0.1  4  32  0.0   1"
    # "2e-7  1e-3  0.1  4  32  0.0   1"
    # "2e-7  5e-5  0.1  4  32  0.0   1"
    # "2e-7  1e-4  0.1  4  32  0.5   1"
    # "2e-7  1e-3  0.1  4  32  0.5   1"
    # "2e-7  5e-5  0.1  4  32  0.5   1"

    # "1e-6  1e-4  0.1  4  32  0.0   1"
    # "1e-6  1e-4  0.1  4  32  0.5   1"
    # "5e-7  1e-4  0.1  4  32  0.0   1"
    # "5e-7  1e-4  0.1  4  32  0.5   1"

    # "2e-7  1e-4  0.05  4  32  0.0   1  1"
    # "1e-6  1e-4  0.05  4  32  0.0   1  1"

    # "2e-7  1e-4  0.05  4  32  0.0   1  16"
    # "1e-6  1e-4  0.05  4  32  0.0   1  16"
    "1e-5  5e-5  0.1  4  32  0.0   1  1"

    # AGZO
    # "5e-6  5e-5  0.1  4  32  0.5   4"
    # "2e-6  5e-5  0.1  4  32  0.5  4"

    # "2e-7  1e-4  0.1  4  32  0.5   1"
    # "2e-7  1e-3  0.1  4  32  0.5   1"
    # "2e-7  5e-5  0.1  4  32  0.5   1"
    # "2e-7  1e-4  0.1  4  32  0.7   1"
    # "2e-7  1e-3  0.1  4  32  0.7   1"
    # "2e-7  5e-5  0.1  4  32  0.7   1"


    # "1e-6  1e-4  0.1  4  32  0.7   1"
    # "1e-6  1e-4  0.1  4  32  0.5   1"
    # "1e-6  1e-4  0.1  4  32  0.0   1"
    # "5e-7  1e-4  0.1  4  32  0.0   1"
    # "5e-7  1e-4  0.1  4  32  0.7   1"
    # "5e-7  1e-4  0.1  4  32  0.5   1"

    # "2e-7  1e-4  0.05  4  32  0.0   1"
    # "1e-6  1e-4  0.05  4  32  0.0   1"
    # "1e-6  1e-4  0.1  4  32  0.7   1"
    # "2e-7  1e-4  0.1  4  32  0.7   1"
    # "1e-6  1e-4  0.05  4  32  0.0   1"
    # "2e-6  5e-5  0.05  4  32  0.8   4  16"
    # "2e-6  5e-5  0.1  4  32  0.8   4  16"
    # "2e-6  1e-4  0.05  4  32  0.8   4  16"

    # "2e-5  1e-4  0.1  4  32  0.8   4  16"
    # "2e-5  1e-4  0.1  4  32  0.8   4  1"

    # "1e-5  5e-5  0.1  4  32  0.8   4  16"
    # "1e-5  5e-5  0.1  4  32  0.8   4  1"
    # "1e-5  5e-5  0.1  4  32  0.0   4  1"

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
    read -r lr eps beta bs gc m_beta rank lazy <<< "$config"

    # Format tags
    lr_tag=$(echo "$lr"  | sed 's/e-/em/;s/\./_/')
    eps_tag=$(echo "$eps" | sed 's/e-/em/;s/\./_/')
    b_tag=$(echo "$beta"  | sed 's/0\./b/;s/\.//')
    mbeta_tag=$(echo "$m_beta" | sed 's/0\./mb/;s/\.//') # e.g. 0.9 -> mb9, 0.95 -> mb95
    bs_tag=$(echo "$bs")
    gc_tag=$(echo "$gc")
    rank_tag=$(echo "$rank")
    lazy_tag=$(echo "$lazy")

    # EXP_NAME="single_dpo_fp32_${TRAINER}_bs${bs_tag}_gc${gc_tag}_lr${lr_tag}_eps${eps_tag}_beta${b_tag}_rank${rank_tag}_${mbeta_tag}_lazy${lazy_tag}_$(date +%Y%m%d)"
    EXP_NAME="dpo_fp32_${TRAINER}_bs${bs_tag}_gc${gc_tag}_lr${lr_tag}_eps${eps_tag}_beta${b_tag}_rank${rank_tag}_${mbeta_tag}_lazy${lazy_tag}_20260512"
    # EXP_NAME="dpo_bf16_mezo_bs4_gc32_lr5em6_eps5em5_mb0_20260428"

    echo "Submitting: $EXP_NAME"

    sbatch \
        --job-name="${TRAINER}_${mbeta_tag}_r${rank_tag}" \
        --output="${MY_ROOT}/logs/${EXP_NAME}_%j.out" \
        --error="${MY_ROOT}/logs/${EXP_NAME}_%j.err" \
        --export=ALL,EXP_NAME="$EXP_NAME",SFT_MODEL_PATH="$SFT_MODEL_PATH",\
LR="$lr",EPS="$eps",BETA="$beta",BS="$bs",GC="$gc",MBETA="$m_beta",RANK="$rank",WR="$wr",TRAINER="$TRAINER",LAZY="$lazy" \
        "$(dirname "$0")/sweep_dpo_worker_fp32.sh"

    sleep 1
done

echo ""
echo "Submitted ${#CONFIGS[@]} jobs for trainer=${TRAINER}"
echo "Monitor: squeue -u \$USER"