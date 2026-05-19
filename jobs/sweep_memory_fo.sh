#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-869
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1  # 依然保持单卡纯净测试
#SBATCH -t 4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weiyun.huang.2798@student.uu.se
#SBATCH --job-name=mem_baseline
#SBATCH --output=/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/logs/mem_baseline_%j.out
#SBATCH --error=/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/logs/mem_baseline_%j.err

set -uo pipefail 

source "/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/ZODPO/jobs/common_header.sh"

SFT_MODEL_PATH="/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/runs/sft_qwen1.5b-lr2em5/sft/final_model" # 请替换为你的实际路径[cite: 1]

LR="5e-6"
BETA="0.1"
GC="1"

run_baseline() {
    local bs=$1
    local seq_len=$2
    local prompt_len=$((seq_len / 2))
    
    local exp_name="mem_test_sgd_trl_bs${bs}_seq${seq_len}"
    echo "=================================================="
    echo "Starting Baseline: ${exp_name}"
    
    # 直接调用 TRL 专属脚本
    python train_baseline_trl.py \
        trainer="fo_baseline" \
        loss=dpo \
        exp_name="${exp_name}" \
        model.name_or_path="${SFT_MODEL_PATH}" \
        model.policy_dtype=bfloat16 \
        loss.sft_model_path="${SFT_MODEL_PATH}" \
        loss.beta="${BETA}" \
        loss.num_epochs=1 \
        trainer.lr="${LR}" \
        batch_size="${bs}" \
        gradient_accumulation_steps="${GC}" \
        max_length="${seq_len}" \
        max_prompt_length="${prompt_len}" \
        max_steps=10 \
        wandb.enabled=true \
        wandb.project=zo-dpo-memory-test \
        hydra.run.dir="${TMPDIR}/hydra/${exp_name}" \
        hydra.output_subdir=null || echo ">>> OOM or Error occurred in ${exp_name}, continuing to next..."
        
    echo "Finished: ${exp_name}"
}

# =========================================================
# 实验 A: 固定 Batch Size = 4, 改变 Sequence Length
# =========================================================
FIXED_BS=4
# SEQ_LENS=(128 256 384 512 768 1024 2048)
SEQ_LENS=(2048)

echo ">>> Starting SGD Baseline Experiment A: Varying Sequence Length"
for seq in "${SEQ_LENS[@]}"; do
    run_baseline "$FIXED_BS" "$seq"
done

# =========================================================
# 实验 B: 固定 Sequence Length = 256, 改变 Batch Size
# =========================================================
# FIXED_SEQ=256
# BATCH_SIZES=(1 2 4 8 16 32)

# echo ">>> Starting SGD Baseline Experiment B: Varying Batch Size"
# for bs in "${BATCH_SIZES[@]}"; do
#     run_baseline "$bs" "$FIXED_SEQ"
# done

echo "=== All Baseline memory sweep jobs completed ==="