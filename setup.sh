#!/usr/bin/env bash

set -euo pipefail

SHARED_ROOT="/mimer/NOBACKUP/groups/ga_llm_hri"
MY_ROOT="${SHARED_ROOT}/weiyun_zodpo"

VENV_DIR="${MY_ROOT}/venv"
CODE_DIR="${MY_ROOT}/ZODPO"

# ── Point at the already-populated shared HF cache ────────────────────────────
export HF_HOME="${SHARED_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

echo "=== Alvis ZO-DPO environment setup ==="
echo "MY_ROOT  : $MY_ROOT"
echo "VENV_DIR : $VENV_DIR"
echo "HF_HOME  : $HF_HOME"
echo ""

# ── Create personal directory structure ───────────────────────────────────────
mkdir -p "${MY_ROOT}/checkpoints"
mkdir -p "${MY_ROOT}/runs"
mkdir -p "${MY_ROOT}/logs"
echo "Created output directories under $MY_ROOT"

# ── Load system modules (CUDA 12.1 + Python 3.11) ────────────────────────────
module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

echo "Python  : $(python3 --version)"
echo "CUDA    : $(nvcc --version | grep release)"

# ── Create venv on /mimer (large quota) ───────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "Venv already exists at $VENV_DIR — skipping creation."
    echo "To force rebuild: rm -rf $VENV_DIR && bash setup.sh"
else
    python3 -m venv "$VENV_DIR"
    echo "Venv created at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

echo ""
echo "Installing torch 2.5.1+cu121 ..."
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --quiet

echo "Installing remaining requirements ..."
grep -v "^torch" "${CODE_DIR}/requirements.txt" \
    | grep -v "^#" | grep -v "^$" \
    | pip install -r /dev/stdin --quiet

echo ""
echo "=== Setup complete ==="
echo ""
