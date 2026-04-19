"""
train_ddp_empty_run.py
======================
Drop-in replacement for train_ddp.py that:

  1. Runs the FULL ZO forward-pass / perturbation loop (same random state).
  2. Skips ALL parameter updates (_apply_update is a no-op).
  3. Logs per-step loss curves AND per-step max parameter drift to W&B /
     stdout so you can tell whether fp16/bf16 perturbations are already
     corrupting θ.
  4. Runs the entire diagnostic in BOTH fp32 and the configured half
     precision (fp16 or bf16), back-to-back, starting from the same
     checkpoint, so the two loss curves are directly comparable.

Launch exactly like train_ddp.py::

    torchrun --nproc_per_node=<N> train_ddp_empty_run.py \\
        trainer.name=mezo loss.name=dpo \\
        model.policy_dtype=float16        # or bfloat16
        empty_run=true                    # activates no-update mode
        [other overrides …]

To compare fp32 vs half-precision in a single run, set::

    empty_run=true
    run_dual_precision_compare=true      # run fp32 first, then half
"""

import copy
import os
import random
import socket

import numpy as np
import torch
import transformers
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig

from trainer_ddp import disable_dropout
from trainer_ddp_empty_run import build_trainer, DualPrecisionEmptyRun

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def _load_policy(config: DictConfig, dtype: torch.dtype):
    """Load a fresh copy of the policy in *dtype*."""
    cache_dir = os.path.expandvars(config.hf_cache_dir)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        device_map={"": local_rank},
    )
    disable_dropout(policy)
    return policy


def _init_wandb(config: DictConfig, run_tag: str, resumed_run_id=None):
    """Initialise (or resume) a W&B run with an extra tag for empty-run."""
    os.makedirs("/tmp/wandb_tmp", exist_ok=True)
    init_kw = dict(
        entity=config.wandb.entity,
        project=config.wandb.project,
        config=OmegaConf.to_container(config, resolve=True),
        name=f"{config.exp_name}__{run_tag}",
        dir="/tmp/wandb_tmp",
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
    )
    if resumed_run_id is not None:
        init_kw["id"]     = resumed_run_id
        init_kw["resume"] = "must"
        print(f"  [wandb] Resuming run {resumed_run_id}")
    wandb.init(**init_kw)


# ============================================================
# Hydra entry point
# ============================================================

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    set_seed(config.seed)

    dual_compare = bool(config.get("run_dual_precision_compare", False))
    empty_run    = bool(config.get("empty_run", True))   # default True here

    print(OmegaConf.to_yaml(config))
    print("=" * 72)
    print(f"Host            : {socket.gethostname()}")
    print(f"Mode            : {'EMPTY RUN (no updates)' if empty_run else 'NORMAL'}")
    print(f"Dual-prec compare: {dual_compare}")
    print(f"Loss/Stage      : {config.loss.name}")
    print(f"Output dir      : {config.runs_dir}")
    print("=" * 72)

    # ── resolve dtypes ────────────────────────────────────────────────
    base_dtype  = getattr(torch, config.model.policy_dtype)     # configured dtype
    # Determine "other" dtype for dual-compare
    if base_dtype == torch.float32:
        # Compare fp32 vs bfloat16 (safer than fp16 on modern hardware)
        other_dtype = torch.bfloat16
    else:
        other_dtype = torch.float32

    # ── checkpoint introspection (for wandb resume) ──────────────────
    stage     = config.loss.name
    ckpt_base = os.path.expandvars(config.checkpoint_dir)
    ckpt_path = os.path.join(ckpt_base, config.exp_name, stage, "checkpoint.pt")

    resumed_run_id = None
    if os.path.exists(ckpt_path):
        try:
            payload        = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            resumed_run_id = payload.get("wandb_run_id", None)
        except Exception:
            pass

    # ================================================================
    # Helper: run one full empty-run pass with a given dtype
    # ================================================================
    def run_one_precision(dtype: torch.dtype, wandb_tag: str):
        dtype_name = str(dtype).split(".")[-1]
        print(f"\n{'='*72}")
        print(f"  PRECISION PASS: {dtype_name}")
        print(f"{'='*72}")

        # Reset global seed so both passes see exactly the same data/ZO order
        set_seed(config.seed)

        policy = _load_policy(config, dtype)

        # Override dtype in config so the trainer uses the right one
        # (we use OmegaConf's read-only workaround)
        import omegaconf
        with omegaconf.open_dict(config):
            config.model.policy_dtype = dtype_name

        trainer = build_trainer(policy, config, empty_run=True)

        if config.wandb.enabled and trainer.accelerator.is_main_process:
            _init_wandb(config, wandb_tag, resumed_run_id)

        try:
            trainer.train()
        finally:
            trainer.cleanup()
            if config.wandb.enabled and trainer.accelerator.is_main_process:
                wandb.finish()

        # Restore original dtype name in config
        with omegaconf.open_dict(config):
            config.model.policy_dtype = str(base_dtype).split(".")[-1]

    # ================================================================
    # Execution paths
    # ================================================================

    if dual_compare:
        # ── Run fp32 first, then the configured half-precision ────────
        # Both use identical seeds, so the loss curves are apples-to-apples.
        run_one_precision(torch.float32, "empty_run_fp32")
        run_one_precision(other_dtype,   f"empty_run_{str(other_dtype).split('.')[-1]}")

    else:
        # ── Single precision pass ─────────────────────────────────────
        dtype_name = str(base_dtype).split(".")[-1]
        policy     = _load_policy(config, base_dtype)
        trainer    = build_trainer(policy, config, empty_run=empty_run)

        if config.wandb.enabled and trainer.accelerator.is_main_process:
            tag = f"empty_run_{dtype_name}" if empty_run else dtype_name
            _init_wandb(config, tag, resumed_run_id)

        try:
            trainer.train()
        finally:
            trainer.cleanup()
            if config.wandb.enabled and trainer.accelerator.is_main_process:
                wandb.finish()


if __name__ == "__main__":
    main()
