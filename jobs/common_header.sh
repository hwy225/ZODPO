#!/usr/bin/env bash

SHARED_ROOT="/mimer/NOBACKUP/groups/ga_llm_hri"
MY_ROOT="${SHARED_ROOT}/weiyun_zodpo"

# ZODPO repo root — parent of the jobs/ directory this file is in
REPO_ROOT="${MY_ROOT}/ZODPO"

# train.py is in test/, and Hydra expects config/ to be a sibling of train.py
export CODE_DIR="${REPO_ROOT}/src"

export VENV_DIR="${MY_ROOT}/venv"
export CHECKPOINT_DIR="${MY_ROOT}/checkpoints"
export RUNS_DIR="${MY_ROOT}/runs"

# ── Reuse the shared HuggingFace cache — no re-downloads needed ───────────────
export HF_HOME="${SHARED_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

# ── Hydra and wandb write to fast local scratch (auto-cleaned after job) ──────
export TMPDIR="${TMPDIR:-/tmp}"

# ── System modules ────────────────────────────────────────────────────────────
module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

# ── Activate venv ────────────────────────────────────────────────────────────
source "${VENV_DIR}/bin/activate"

# ── cd into test/ so Hydra finds config/ next to train.py ─────────────────────
cd "${CODE_DIR}"

export TOKENIZERS_PARALLELISM=false
