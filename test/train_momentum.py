import os
import random
import socket

import numpy as np
import torch
import transformers
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig

from trainer_momentum import build_trainer, disable_dropout

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
    torch.backends.cudnn.benchmark = False


# ============================================================
# Hydra entry point
# ============================================================

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    # ── reproducibility ─────────────────────────────────────────────
    set_seed(config.seed)

    print(OmegaConf.to_yaml(config))
    print("=" * 72)
    print(f"Host      : {socket.gethostname()}")
    if config.loss.name == "sft":
        display_trainer = "TRL SFTTrainer"
    else:
        display_trainer = config.trainer.name

    print(f"Trainer   : {display_trainer}")
    print(f"Loss/Stage: {config.loss.name}")
    print(f"Output dir: {config.runs_dir}")
    print("=" * 72)

    # ── load policy model ────────────────────────────────────────────
    cache_dir = os.path.expandvars(config.hf_cache_dir)
    dtype     = getattr(torch, config.model.policy_dtype)

    print(f"Loading policy: {config.model.name_or_path}")
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        device_map={"": local_rank},   # multi-GPU friendly; single-GPU lands on cuda:0
    )
    disable_dropout(policy)

    # ── build trainer ────────────────────────────────────────────────
    trainer = build_trainer(policy, config)

    # ── wandb  (resume-aware) ────────────────────────────────────────
    stage     = config.loss.name
    ckpt_base = os.path.expandvars(config.checkpoint_dir)
    ckpt_path = os.path.join(ckpt_base, config.exp_name, stage, "checkpoint.pt")
    
    resumed_run_id = None
    if os.path.exists(ckpt_path):
        try:
            # only load metadata, not weights
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            resumed_run_id = payload.get("wandb_run_id", None)
        except Exception:
            pass

    if config.wandb.enabled and trainer.accelerator.is_main_process:
        os.makedirs("/tmp/wandb_tmp", exist_ok=True)
        init_kw = dict(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config, resolve=True),
            name=config.exp_name,
            dir="/tmp/wandb_tmp",
        )
        if resumed_run_id is not None:
            init_kw["id"]     = resumed_run_id
            init_kw["resume"] = "must"
            print(f"  [wandb] Resuming run {resumed_run_id}")
        wandb.init(**init_kw)

    # ── run ──────────────────────────────────────────────────────────
    try:
        trainer.train()
    finally:
        trainer.cleanup()
        if config.wandb.enabled and trainer.accelerator.is_main_process:
            wandb.finish()


if __name__ == "__main__":
    main()