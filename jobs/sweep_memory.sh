#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-869
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1  # 强制使用单卡，避免通信显存开销
#SBATCH -t 4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weiyun.huang.2798@student.uu.se
#SBATCH --job-name=mem_sweep
#SBATCH --output=/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/logs/mem_sweep_%j.out
#SBATCH --error=/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/logs/mem_sweep_%j.err

# 设置在遇到 OOM 错误时不退出整个脚本，继续跑下一个参数
set -uo pipefail 

# 引入环境设置
source "/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/ZODPO/jobs/common_header.sh"

SFT_MODEL_PATH="/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/runs/sft_qwen1.5b-lr2em5/sft/final_model" # 请替换为你的实际路径[cite: 1]

# 固定一些不影响显存的超参数
LR="5e-6"
EPS="5e-5"
BETA="0.1"
MBETA="0.0"
RANK="1"
GC="1" # 测显存时，梯度累加通常设为 1

# 统一的运行函数
run_experiment() {
    local trainer=$1
    local bs=$2
    local seq_len=$3
    local prompt_len=$((seq_len / 2)) # 简单假定 prompt 长度为 seq_len 的一半
    
    local exp_name="mem_test_${trainer}_bs${bs}_seq${seq_len}_mbeta${MBETA}"
    echo "=================================================="
    echo "Starting: ${exp_name}"
    
    local EXTRA_ARGS=""
    if [[ "${trainer}" == "agzo" || "${trainer}" == "agzo_plain" ]]; then
        EXTRA_ARGS="trainer.power_iter_steps=5 trainer.rank=${RANK} "
    fi

    # 使用普通 python 运行单卡，并加上 || true 防止 OOM 导致整个脚本中断
    python train.py \
        trainer="${trainer}" \
        loss=dpo \
        compute_logps_fp32=True \
        max_loss_threshold=15.0 \
        max_steps=10 \
        eval_every=100 \
        eval_batches=2 \
        exp_name="${exp_name}" \
        model.name_or_path="${SFT_MODEL_PATH}" \
        model.policy_dtype=bfloat16 \
        loss.sft_model_path="${SFT_MODEL_PATH}" \
        loss.beta="${BETA}" \
        loss.num_epochs=1 \
        trainer.lr="${LR}" \
        +trainer.momentum_beta=${MBETA} \
        ++trainer.eps="${EPS}" \
        trainer.warmup_ratio=0.1 \
        trainer.lr_scheduler_type="cosine" \
        ${EXTRA_ARGS} \
        batch_size="${bs}" \
        gradient_accumulation_steps="${GC}" \
        max_length="${seq_len}" \
        max_prompt_length="${prompt_len}" \
        wandb.enabled=true \
        wandb.project=zo-dpo-memory-test \
        hydra.run.dir="${TMPDIR}/hydra/${exp_name}" \
        hydra.output_subdir=null || echo ">>> OOM or Error occurred in ${exp_name}, continuing..."
        
    echo "Finished: ${exp_name}"
}

# =========================================================
# 实验 A: 固定 Batch Size = 4, 改变 Sequence Length
# =========================================================
FIXED_BS=4
SEQ_LENS=(128 256 384 512 768 1024 2048)
# TRAINERS=("agzo" "mezo" "agzo_plain") # 根据你代码里实际注册的名称修改
TRAINERS=("agzo") # 根据你代码里实际注册的名称修改

echo ">>> Starting Experiment A: Varying Sequence Length"
for seq in "${SEQ_LENS[@]}"; do
    for t in "${TRAINERS[@]}"; do
        run_experiment "$t" "$FIXED_BS" "$seq"
    done
done

# =========================================================
# 实验 B: 固定 Sequence Length = 256, 改变 Batch Size
# =========================================================
FIXED_SEQ=256
BATCH_SIZES=(1 2 8 16 32)

echo ">>> Starting Experiment B: Varying Batch Size"
for bs in "${BATCH_SIZES[@]}"; do
    for t in "${TRAINERS[@]}"; do
        run_experiment "$t" "$bs" "$FIXED_SEQ"
    done
done

echo "=== All memory sweep jobs completed ==="