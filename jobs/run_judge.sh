#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-869
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 6:00:00
#SBATCH -J llm_eval               
#SBATCH -o logs/eval_%j.out       
#SBATCH -e logs/eval_%j.err       
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=weiyun.huang.2798@student.uu.se

set -euo pipefail
SHARED_ROOT="/mimer/NOBACKUP/groups/ga_llm_hri"
MY_ROOT="${SHARED_ROOT}/weiyun_zodpo"

# ZODPO repo root — parent of the jobs/ directory this file is in
REPO_ROOT="${MY_ROOT}/ZODPO"

export CODE_DIR="${REPO_ROOT}/src"

export VENV_DIR="${MY_ROOT}/venv_eval"

# ── Reuse the shared HuggingFace cache — no re-downloads needed ───────────────
export HF_HOME="${MY_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

export TORCH_HOME="${MY_ROOT}/.cache/torch"
export TRITON_CACHE_DIR="${MY_ROOT}/.cache/triton"
export TORCH_EXTENSIONS_DIR="${MY_ROOT}/.cache/torch_extensions"

echo "=== Starting LLM Evaluation Pipeline ==="
date

module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0
source $VENV_DIR/bin/activate
cd $CODE_DIR/LLM-as-a-judge


OUTPUT_DIR="/mimer/NOBACKUP/groups/ga_llm_hri/weiyun_zodpo/outputs"


echo -e "Starting vLLM judge service in the background..."

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8001 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 &

VLLM_PID=$!  # record the PID

echo "Waiting for vLLM judge service to be ready..."
# check every 10 seconds
while ! curl -s http://localhost:8001/v1/models > /dev/null; do
    sleep 10
    echo -n "."
done
echo -e "\n[OK] vLLM judge service is ready! "

echo -e "\nStarting concurrent judgments..."
for pair in "sft fodpo"; do
  read model_a model_b <<< $pair
  echo ">>> Judge: $model_a vs $model_b"
  python judge_winrate.py \
    --response_dir $OUTPUT_DIR/responses/ \
    --model_a $model_a --model_b $model_b \
    --judge_url http://localhost:8001/v1 \
    --judge_model Qwen/Qwen2.5-7B-Instruct \
    --output_dir $OUTPUT_DIR/judgments \
    --concurrency 32
done


echo -e "\nAll judgments completed. Cleaning up vLLM service and generating plots..."
kill $VLLM_PID || true  # kill vllm service

python plot_results.py \
  --judgment_files $OUTPUT_DIR/judgments/sft_vs_mezodpo.jsonl \
                   $OUTPUT_DIR/judgments/sft_vs_agzodpo.jsonl \
                   $OUTPUT_DIR/judgments/sft_vs_fodpo.jsonl \
                   $OUTPUT_DIR/judgments/fodpo_vs_mezodpo.jsonl \
                   $OUTPUT_DIR/judgments/fodpo_vs_agzodpo.jsonl \
                   $OUTPUT_DIR/judgments/mezodpo_vs_agzodpo.jsonl \
  --output_path $OUTPUT_DIR/winrate_plot_final.pdf

echo "=== Pipeline finished successfully ==="
date