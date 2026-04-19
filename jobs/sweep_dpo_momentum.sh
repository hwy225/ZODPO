#!/usr/bin/env bash
# =============================================================================
# sweep_momentum.sh  —  submit a grid of DPO hyperparameter experiments for LOZO-M
#
# Run from ZODPO/:
#   bash jobs/sweep_momentum.sh agzo sft_qwen1.7b
# =============================================================================

TRAINER="${1:?Usage: bash sweep_momentum.sh <trainer> <sft_exp_name>}"
SFT_EXP="${2:?Usage: bash sweep_momentum.sh <trainer> <sft_exp_name>}"

# ── Hyperparameter grid ───────────────────────────────────────────────────────
# 新增了两列: m_beta (动量衰减系数) 和 rank
# Each row: "lr eps dpo_beta bs gc m_beta rank"
CONFIGS=(
    # 推荐的搜索网格 (在确保是 fp32 的前提下，可以尝试较大的 lr)
    "1e-6  1e-4  0.1  4  16  0.9   1"
    # "1e-6  1e-4  0.1  4  16  0.95  1"
    # "1e-6  1e-4  0.1  4  16  0.99  1"
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
    # 解析新增的参数
    read -r lr eps beta bs gc m_beta rank <<< "$config"

    # Format tags
    lr_tag=$(echo "$lr"  | sed 's/e-/em/;s/\./_/')
    eps_tag=$(echo "$eps" | sed 's/e-/em/;s/\./_/')
    b_tag=$(echo "$beta"  | sed 's/0\./b/;s/\.//')
    mbeta_tag=$(echo "$m_beta" | sed 's/0\./mb/;s/\.//') # 例: 0.9 -> mb9, 0.95 -> mb95
    bs_tag=$(echo "$bs")
    gc_tag=$(echo "$gc")
    rank_tag=$(echo "$rank")

    if [[ "${TRAINER}" == "agzo" || "${TRAINER}" == "agzo_plain" ]]; then
        trainer_tag="_mb${mbeta_tag}_r${rank_tag}"
    else
        trainer_tag=""
    fi
    
    EXP_NAME="dpo_fp32_${TRAINER}_bs${bs_tag}_gc${gc_tag}_lr${lr_tag}_eps${eps_tag}${trainer_tag}_$(date +%Y%m%d)"

    echo "Submitting: $EXP_NAME"

    sbatch \
        --job-name="${TRAINER}_${mbeta_tag}_r${rank_tag}" \
        --output="${MY_ROOT}/logs/${EXP_NAME}_%j.out" \
        --error="${MY_ROOT}/logs/${EXP_NAME}_%j.err" \
        --export=ALL,EXP_NAME="$EXP_NAME",SFT_MODEL_PATH="$SFT_MODEL_PATH",\
LR="$lr",EPS="$eps",BETA="$beta",BS="$bs",GC="$gc",MBETA="$m_beta",RANK="$rank",TRAINER="$TRAINER" \
        "$(dirname "$0")/sweep_dpo_momentum_worker.sh"

    sleep 1
done

echo ""
echo "Submitted ${#CONFIGS[@]} jobs for trainer=${TRAINER}"
echo "Monitor: squeue -u \$USER"