"""
trainer_ddp_empty_run.py
========================
Adds an ``EmptyRunMixin`` that monkey-patches ``_apply_update`` to a no-op,
so you can observe the raw ZO loss signal (and check whether half-precision
perturbations corrupt model parameters) without actually updating θ.

Usage
-----
Replace ``from trainer_ddp import build_trainer`` with
``from trainer_ddp_empty_run import build_trainer`` in train_ddp.py,
OR use the provided ``train_ddp_empty_run.py`` entry-point directly.

The perturbation loop (_perturb +1 / -2 / +1) is intentionally KEPT intact
so the half-precision numeric corruption diagnostic is valid.
"""

import copy
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
import tqdm
from omegaconf import DictConfig

# Re-export everything from the original trainer so callers only need to
# import from this module.
from trainer_ddp import (
    ZOTrainerBase,
    MeZOTrainer,
    AGZOTrainer,
    AGZOPlainTrainer,
    TRAINER_MAP as _BASE_TRAINER_MAP,
    build_trainer as _base_build_trainer,
    get_batch_logps,
    compute_sft_loss,
    compute_dpo_loss,
    disable_dropout,
)


# ============================================================
# Helpers
# ============================================================

def _param_snapshot(policy: nn.Module) -> Dict[str, torch.Tensor]:
    """Capture a CPU copy of every parameter for later diff-ing."""
    return {
        name: param.data.detach().cpu().clone()
        for name, param in policy.named_parameters()
    }


def _param_max_abs_delta(
    policy: nn.Module,
    snapshot: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Return max |Δ| per parameter relative to *snapshot* (cpu tensors)."""
    deltas: Dict[str, float] = {}
    for name, param in policy.named_parameters():
        if name in snapshot:
            delta = (param.data.detach().cpu().float() - snapshot[name].float()).abs().max().item()
            deltas[name] = delta
    return deltas


# ============================================================
# EmptyRunMixin
# ============================================================

class EmptyRunMixin:
    """
    Override ``_apply_update`` with a no-op so the training loop runs
    identically to the real run (same seeds, same perturbations, same
    forward passes) but θ is never changed by the optimiser.

    Additionally, after every ZO step we check whether the ±ε perturbation
    cycle itself has left any residual drift in the parameters (a real risk
    in fp16 due to rounding).  The max drift per-parameter is logged.
    """

    # Will be set by EmptyRunTrainerBase.__init__
    _log_param_drift: bool = True
    _snapshot: Optional[Dict[str, torch.Tensor]] = None

    def _apply_update(self, projected_grad: float):  # type: ignore[override]
        """No-op: intentionally skip the SGD / MeZO update."""
        pass

    def _take_snapshot(self):
        """Record current parameters so we can measure perturbation drift."""
        if self._log_param_drift:
            self._snapshot = _param_snapshot(self.policy)  # type: ignore[attr-defined]

    def _check_and_log_drift(self, step: int):
        """
        Compare current params against the pre-perturbation snapshot.
        In theory, after _perturb(+1) / _perturb(-2) / _perturb(+1) the
        params should be exactly back to θ₀.  In half precision they may not be.
        """
        if not self._log_param_drift or self._snapshot is None:
            return

        deltas = _param_max_abs_delta(self.policy, self._snapshot)  # type: ignore[attr-defined]
        max_drift = max(deltas.values()) if deltas else 0.0
        total_nonzero = sum(1 for v in deltas.values() if v > 0.0)

        accelerator = self.accelerator  # type: ignore[attr-defined]
        if accelerator.is_main_process:
            print(
                f"  [drift] step={step:>6d}  max|Δθ|={max_drift:.3e}  "
                f"params_with_drift={total_nonzero}/{len(deltas)}"
            )
            try:
                wandb.log({
                    "empty_run/max_param_drift":       max_drift,
                    "empty_run/params_with_drift":     total_nonzero,
                    "step": step,
                })
            except Exception:
                pass

        # Reset snapshot for the next step
        self._snapshot = None


# ============================================================
# Concrete empty-run trainer classes
# ============================================================

# ============================================================
# Patch _run_dpo to track global_step_for_drift
# (提前定义，方便直接继承)
# ============================================================
class _StepTrackingMixin:
    """Keep _global_step_for_drift in sync with the real global_step."""

    def _run_dpo(self, beta: float):
        self._global_step_for_drift = 0
        _orig_apply = self._apply_update

        def _counting_noop(g: float):
            self._global_step_for_drift += 1
            _orig_apply(g)

        self._apply_update = _counting_noop  # type: ignore[method-assign]
        try:
            super()._run_dpo(beta)  # type: ignore[misc]
        finally:
            self._apply_update = _orig_apply  # type: ignore[method-assign]


# ============================================================
# Concrete empty-run trainer classes
# ============================================================

class EmptyMeZOTrainer(EmptyRunMixin, _StepTrackingMixin, MeZOTrainer):
    """MeZO with update disabled + per-step drift logging."""

    def __init__(self, policy: nn.Module, config: DictConfig):
        super().__init__(policy, config)
        self._log_param_drift = True

    def _sft_step(self, gpu_batch: dict) -> float:
        self._take_snapshot()
        loss = super()._sft_step(gpu_batch)
        self._check_and_log_drift(getattr(self, "_global_step_for_drift", 0))
        return loss

    def _dpo_loss(self, gpu_batch: dict, beta: float):
        self._take_snapshot()
        result = super()._dpo_loss(gpu_batch, beta)
        self._check_and_log_drift(getattr(self, "_global_step_for_drift", 0))
        return result


class EmptyAGZOTrainer(EmptyRunMixin, _StepTrackingMixin, AGZOTrainer):
    """AGZO with update disabled + per-step drift logging."""

    def __init__(self, policy: nn.Module, config: DictConfig):
        super().__init__(policy, config)
        self._log_param_drift = True

    def _sft_step(self, gpu_batch: dict) -> float:
        self._take_snapshot()
        loss = super()._sft_step(gpu_batch)
        self._check_and_log_drift(getattr(self, "_global_step_for_drift", 0))
        return loss

    def _dpo_loss(self, gpu_batch: dict, beta: float):
        self._take_snapshot()
        result = super()._dpo_loss(gpu_batch, beta)
        self._check_and_log_drift(getattr(self, "_global_step_for_drift", 0))
        return result


class EmptyAGZOPlainTrainer(EmptyRunMixin, _StepTrackingMixin, AGZOPlainTrainer):
    """AGZOPlain with update disabled + per-step drift logging."""

    def __init__(self, policy: nn.Module, config: DictConfig):
        super().__init__(policy, config)
        self._log_param_drift = True

    def _sft_step(self, gpu_batch: dict) -> float:
        self._take_snapshot()
        loss = super()._sft_step(gpu_batch)
        self._check_and_log_drift(getattr(self, "_global_step_for_drift", 0))
        return loss

    def _dpo_loss(self, gpu_batch: dict, beta: float):
        self._take_snapshot()
        result = super()._dpo_loss(gpu_batch, beta)
        self._check_and_log_drift(getattr(self, "_global_step_for_drift", 0))
        return result

# ============================================================
# Precision comparison helper
# ============================================================

class DualPrecisionEmptyRun:
    """
    Run the *same* ZO perturbation sequence in fp32 AND fp16/bf16 and
    compare the resulting loss values and parameter drift side-by-side.

    This is a diagnostic-only utility; it is NOT a trainer.

    Usage::

        runner = DualPrecisionEmptyRun(policy_fp32, config)
        runner.run(n_steps=100)
    """

    def __init__(
        self,
        policy_fp32: nn.Module,
        config: DictConfig,
        half_dtype: torch.dtype = torch.float16,
    ):
        self.config      = config
        self.half_dtype  = half_dtype
        self.policy_fp32 = policy_fp32

        # Build a half-precision copy on the same device
        device = next(policy_fp32.parameters()).device
        self.policy_half = copy.deepcopy(policy_fp32).to(dtype=half_dtype, device=device)
        disable_dropout(self.policy_fp32)
        disable_dropout(self.policy_half)

    # ── internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _perturb_model(
        policy: nn.Module,
        zo_seed: int,
        scaling: float,
        eps: float,
    ):
        torch.manual_seed(zo_seed)
        for p in policy.parameters():
            if p.requires_grad:
                p.data.add_(torch.randn_like(p), alpha=scaling * eps)

    @staticmethod
    def _max_drift(policy: nn.Module, snapshot: Dict[str, torch.Tensor]) -> float:
        diffs = []
        for name, param in policy.named_parameters():
            if name in snapshot:
                diffs.append(
                    (param.data.float() - snapshot[name].float()).abs().max().item()
                )
        return max(diffs) if diffs else 0.0

    def run_step(
        self,
        gpu_batch: dict,
        beta: float,
        zo_seed: int,
        eps: float,
    ) -> dict:
        """
        Execute one ZO step (perturbation only, no update) in both precisions.
        Returns a dict with loss_p/m values and parameter drift for each.
        """
        from trainer_ddp import compute_dpo_loss, get_batch_logps

        results = {}
        for tag, policy in [("fp32", self.policy_fp32), ("half", self.policy_half)]:
            ids_key  = "chosen_input_ids"
            mask_key = "chosen_attention_mask"
            B = gpu_batch["chosen_input_ids"].shape[0]
            ids  = torch.cat([gpu_batch["chosen_input_ids"],      gpu_batch["rejected_input_ids"]],      dim=0)
            mask = torch.cat([gpu_batch["chosen_attention_mask"], gpu_batch["rejected_attention_mask"]], dim=0)

            snap = _param_snapshot(policy)

            with torch.no_grad():
                self._perturb_model(policy, zo_seed, +1, eps)
                lp = compute_dpo_loss(
                    policy(ids, attention_mask=mask).logits[:B],
                    policy(ids, attention_mask=mask).logits[B:],
                    gpu_batch["ref_chosen_logps"],
                    gpu_batch["ref_rejected_logps"],
                    gpu_batch["chosen_labels"],
                    gpu_batch["rejected_labels"],
                    beta,
                    compute_fp32=True,
                ).item()

                self._perturb_model(policy, zo_seed, -2, eps)
                lm = compute_dpo_loss(
                    policy(ids, attention_mask=mask).logits[:B],
                    policy(ids, attention_mask=mask).logits[B:],
                    gpu_batch["ref_chosen_logps"],
                    gpu_batch["ref_rejected_logps"],
                    gpu_batch["chosen_labels"],
                    gpu_batch["rejected_labels"],
                    beta,
                    compute_fp32=True,
                ).item()

                self._perturb_model(policy, zo_seed, +1, eps)  # restore

            drift = self._max_drift(policy, snap)
            results[tag] = {
                "loss_p":    lp,
                "loss_m":    lm,
                "loss_avg":  (lp + lm) / 2,
                "g_hat":     (lp - lm) / (2 * eps),
                "max_drift": drift,
            }
        return results


# ============================================================
# Registry & build_trainer
# ============================================================

EMPTY_TRAINER_MAP: Dict[str, type] = {
    "mezo":       EmptyMeZOTrainer,
    "agzo":       EmptyAGZOTrainer,
    "agzo_plain": EmptyAGZOPlainTrainer,
}

# Merged map (empty-run variants shadow the real ones when empty_run=True)
TRAINER_MAP: Dict[str, type] = {**_BASE_TRAINER_MAP, **EMPTY_TRAINER_MAP}


def build_trainer(
    policy: nn.Module,
    config: DictConfig,
    empty_run: bool = False,
) -> ZOTrainerBase:
    """
    Build a trainer.

    Parameters
    ----------
    policy     : the policy model
    config     : Hydra DictConfig
    empty_run  : if True, use the EmptyRun variant (no parameter updates,
                 but perturbations and forward passes are identical to the
                 real run so loss curves are meaningful diagnostics).
                 Can also be set via ``config.empty_run = true``.
    """
    # Allow config-level override
    if not empty_run:
        empty_run = bool(config.get("empty_run", False))

    name = config.trainer.name

    if empty_run:
        cls = EMPTY_TRAINER_MAP.get(name)
        if cls is None:
            raise ValueError(
                f"No empty-run variant for trainer '{name}'. "
                f"Available: {sorted(EMPTY_TRAINER_MAP.keys())}"
            )
        print(f"  [EmptyRun] Building {cls.__name__} (updates DISABLED)")
    else:
        cls = _BASE_TRAINER_MAP.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown trainer '{name}'. "
                f"Registered trainers: {sorted(_BASE_TRAINER_MAP.keys())}"
            )

    return cls(policy, config)
