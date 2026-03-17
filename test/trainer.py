import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import tqdm
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig
from trl import SFTConfig, SFTTrainer

from preference_datasets_hh import get_chat_template_iterator


# ============================================================
# Loss helpers  (shared by all trainers)
# ============================================================

def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sample sum of token log-probs, ignoring -100 positions."""
    labels    = labels[:, 1:].clone()
    logits    = logits[:, :-1, :]
    loss_mask = labels != -100
    labels[labels == -100] = 0
    per_token = torch.gather(
        logits.log_softmax(-1), 2, labels.unsqueeze(2)
    ).squeeze(2)
    return (per_token * loss_mask).sum(-1)


def compute_sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.CrossEntropyLoss(ignore_index=-100)(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
    )


def compute_dpo_loss(
    pi_chosen_logits:   torch.Tensor,
    pi_rejected_logits: torch.Tensor,
    ref_chosen_logps:   torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    chosen_labels:      torch.Tensor,
    rejected_labels:    torch.Tensor,
    beta: float,
) -> torch.Tensor:
    pi_c   = get_batch_logps(pi_chosen_logits,   chosen_labels)
    pi_r   = get_batch_logps(pi_rejected_logits, rejected_labels)
    logits = (pi_c - pi_r) - (ref_chosen_logps - ref_rejected_logps)
    return -F.logsigmoid(beta * logits).mean()


def disable_dropout(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0


# ============================================================
# ZOTrainerBase
# ============================================================

class ZOTrainerBase:
    """
    Shared training loop for all ZO methods.

    Subclasses MUST implement:
      _perturb(scaling: float)            -- perturb params: theta += scaling*eps*z
      _apply_update(g_hat: float)         -- SGD update (theta is at theta_0 - eps*z on entry)

    Subclasses MAY override:
      _sft_step(gpu_batch) -> float
      _dpo_step(gpu_batch, beta) -> float
      cleanup()                           -- release resources (hooks, etc.)
    """

    def __init__(self, policy: nn.Module, config: DictConfig):
        self.policy        = policy
        self.config        = config
        self.lr            = config.trainer.lr
        self.eps           = config.trainer.eps
        self.total_batches = config.total_batches
        self._zo_seed: int = 0

        # ── Accelerator (single-GPU or multi-GPU transparent) ────────
        # Constructed here so every subclass automatically inherits it.
        # For single-GPU runs Accelerator is a near-zero-overhead wrapper.
        self.accelerator = Accelerator()

        # ── tokenizer ───────────────────────────────────────────────
        hf_model_cache = os.path.expandvars(config.hf_cache_dir)
        self.tokenizer  = transformers.AutoTokenizer.from_pretrained(
            config.model.name_or_path,
            cache_dir=hf_model_cache,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hf_model_cache = hf_model_cache

        # ── dataset cache dir ────────────────────────────────────────
        # Falls back to the HF default (~/.cache/huggingface/datasets)
        # if hf_dataset_cache_dir is not set in config.
        raw_ds_cache = config.get("hf_dataset_cache_dir", None)
        self.dataset_cache_dir: Optional[str] = (
            os.path.expandvars(raw_ds_cache) if raw_ds_cache else None
        )

        # ── checkpoint / run directories ────────────────────────────
        exp   = config.exp_name
        stage = config.loss.name

        ckpt_base = os.path.expandvars(config.checkpoint_dir)
        self.ckpt_dir: str = os.path.join(ckpt_base, exp, stage)

        runs_base = os.path.expandvars(config.runs_dir)
        self.runs_dir: str = os.path.join(runs_base, exp, stage)

        self.checkpoint_every: int = int(config.get("checkpoint_every", 20))

        # wandb run-id is populated by _try_resume() when a checkpoint
        # exists, and stored here so train.py can pass it to wandb.init().
        self.resumed_wandb_run_id: Optional[str] = None

    # ================================================================
    # Abstract interface
    # ================================================================

    def _perturb(self, scaling: float):
        raise NotImplementedError

    def _apply_update(self, projected_grad: float):
        raise NotImplementedError

    # ================================================================
    # Seed management
    # ================================================================

    def _reset_seed(self):
        """Draw a fresh random seed for this ZO step."""
        self._zo_seed = int(torch.randint(0, 2**32, (1,)).item())

    # ================================================================
    # Batch helpers
    # ================================================================

    @staticmethod
    def _concat_cr(gpu_batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stack chosen and rejected along batch dim.
        Sequences are already equal-length (padded by preference_datasets_hh).
        """
        ids  = torch.cat(
            [gpu_batch['chosen_input_ids'],      gpu_batch['rejected_input_ids']],    dim=0
        )
        mask = torch.cat(
            [gpu_batch['chosen_attention_mask'], gpu_batch['rejected_attention_mask']], dim=0
        )
        return ids, mask

    # ================================================================
    # Default step implementations  (subclasses may override)
    # ================================================================

    def _sft_step(self, gpu_batch: dict) -> float:
        """
        Two-sided ZO finite-difference step on the chosen sequence.
        Override to inject pre-step logic (e.g. basis collection in AGZO).
        """
        self._reset_seed()
        with torch.no_grad():
            self._perturb(+1)
            loss_p = compute_sft_loss(
                self.policy(
                    gpu_batch['chosen_input_ids'],
                    attention_mask=gpu_batch['chosen_attention_mask'],
                ).logits,
                gpu_batch['chosen_labels'],
            )
            self._perturb(-2)
            loss_m = compute_sft_loss(
                self.policy(
                    gpu_batch['chosen_input_ids'],
                    attention_mask=gpu_batch['chosen_attention_mask'],
                ).logits,
                gpu_batch['chosen_labels'],
            )
            self._apply_update(((loss_p - loss_m) / (2 * self.eps)).item())
        return ((loss_p + loss_m) / 2).item()

    def _dpo_step(self, gpu_batch: dict, beta: float) -> float:
        """
        Two-sided ZO DPO step; chosen + rejected are concatenated to halve
        the number of forward passes.
        Override to inject pre-step logic (e.g. basis collection).
        """
        self._reset_seed()
        B    = gpu_batch['chosen_input_ids'].shape[0]
        ids, mask = self._concat_cr(gpu_batch)

        def _fwd_split() -> Tuple[torch.Tensor, torch.Tensor]:
            logits = self.policy(ids, attention_mask=mask).logits
            return logits[:B], logits[B:]
        
        with torch.no_grad():
            self._perturb(+1)
            c_p, r_p = _fwd_split()
            loss_p = compute_dpo_loss(
                c_p, r_p,
                gpu_batch['ref_chosen_logps'], gpu_batch['ref_rejected_logps'],
                gpu_batch['chosen_labels'],    gpu_batch['rejected_labels'],
                beta,
            )
            self._perturb(-2)
            c_m, r_m = _fwd_split()
            loss_m = compute_dpo_loss(
                c_m, r_m,
                gpu_batch['ref_chosen_logps'], gpu_batch['ref_rejected_logps'],
                gpu_batch['chosen_labels'],    gpu_batch['rejected_labels'],
                beta,
            )
            self._apply_update(((loss_p - loss_m) / (2 * self.eps)).item())
        return ((loss_p + loss_m) / 2).item()

    # ================================================================
    # Main entry
    # ================================================================

    def train(self):
        stage = self.config.loss.name   # "sft" | "dpo"

        if stage == "sft":
            # Standard gradient-based SFT via TRL SFTTrainer.
            # No ZO loop, no batch caching.
            self._run_sft_standard()
            return

        # DPO (and any future ZO stages): cache batches to CPU RAM first.
        print(f"Caching {self.total_batches} batches ...")
        cached_batches: List[dict] = []
        it = get_chat_template_iterator(
            tokenizer=self.tokenizer,
            split='train',
            batch_size=self.config.batch_size,
            n_examples=self.total_batches * self.config.batch_size,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            shuffle=True,
            cache_dir=self.dataset_cache_dir,
        )
        for _ in tqdm.tqdm(range(self.total_batches), desc="Loading"):
            cached_batches.append(next(it))

        if stage == "dpo":
            self._run_dpo(cached_batches, beta=self.config.loss.beta)
        else:
            raise ValueError(
                f"Unknown loss.name: {stage!r}.  Valid choices: 'sft', 'dpo'."
            )

    # ================================================================
    # Stage runners
    # ================================================================

    def _run_sft_standard(self):
        """
        Standard supervised fine-tuning via trl.SFTTrainer.

        Uses full gradient-based training (not ZO).  The model saved here
        becomes the starting point and frozen reference for the DPO stage.

        Dataset format expected by TRL SFTTrainer (conversational):
            {"messages": [{"role": "user",      "content": "..."},
                          {"role": "assistant", "content": "..."}]}
        or prompt-completion:
            {"prompt": "...", "completion": "..."}

        The hh-rlhf dataset is loaded directly from HuggingFace (not via
        preference_datasets_hh) so TRL can handle tokenisation and masking.
        """
        print("\n--- SFT (TRL SFTTrainer) ---")

        sft_cfg = self.config.loss      # config/loss/sft.yaml
        stage   = "sft"

        # ── build HF dataset for TRL ─────────────────────────────────
        # Load hh-rlhf and convert to the prompt-completion format that
        # TRL understands natively.  Only the "chosen" column is used.
        from datasets import load_dataset
        import re
        raw = load_dataset(
            "Anthropic/hh-rlhf",
            cache_dir=self.dataset_cache_dir,
            split="train",
        )

        tokenizer = self.tokenizer
        def _to_prompt_completion(example):
            text: str = example["chosen"]
            
            parts = re.split(r'\n\n(Human|Assistant):', text)
            msgs = []
            for j in range(1, len(parts), 2):
                role = "user" if parts[j].strip() == "Human" else "assistant"
                content = parts[j+1].strip()
                msgs.append({"role": role, "content": content})
            
            # Locate the final assistant turn
            last_assistant_idx = -1
            for idx in range(len(msgs) - 1, -1, -1):
                if msgs[idx]["role"] == "assistant":
                    last_assistant_idx = idx
                    break
            
            if last_assistant_idx == -1:
                return {"prompt": text, "completion": ""}
                
            prompt_msgs = msgs[:last_assistant_idx]
            chosen_msg = msgs[last_assistant_idx]
            
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            return {
                "prompt": prompt_text,
                "completion": chosen_msg["content"]
            }

        dataset = raw.map(
            _to_prompt_completion,
            remove_columns=raw.column_names,
            desc="Formatting dataset",
        )

        # ── SFTConfig ────────────────────────────────────────────────
        out_dir = os.path.join(self.runs_dir, "final_model")
        sft_config = SFTConfig(
            output_dir=out_dir,
            num_train_epochs=int(sft_cfg.get("num_train_epochs", 1)),

            # max_steps=self.total_batches, # only for test

            per_device_train_batch_size=self.config.batch_size,
            learning_rate=float(sft_cfg.get("lr", 5e-5)),
            lr_scheduler_type=sft_cfg.get("lr_scheduler_type", "cosine"),
            warmup_ratio=float(sft_cfg.get("warmup_ratio", 0.03)),
            max_length=self.config.max_length,
            gradient_checkpointing=True,
            completion_only_loss=True,   # mask prompt tokens, train on completion only
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            bf16=(self.config.model.policy_dtype == "bfloat16"),
            fp16=(self.config.model.policy_dtype == "float16"),
            report_to="wandb" if self.config.wandb.enabled else "none",
        )

        # ── run TRL SFTTrainer ───────────────────────────────────────
        trainer = SFTTrainer(
            model=self.policy,
            args=sft_config,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )
        trainer.train()

        # ── save final model (main process only) ─────────────────────
        self.save_final(stage)

    def _run_sft(self, cached_batches: List[dict]):
        """
        ZO-SFT loop (kept for potential future use / ablations).

        NOTE: this is NOT called when loss=sft.  The standard entry point
        is _run_sft_standard() above.  This method exists so a subclass
        can run a ZO warm-up before DPO without using TRL, e.g.:
            self._run_sft(cached_batches)   # ZO warm-up
            self._run_dpo(...)
        """
        print("\n--- ZO-SFT (zeroth-order) ---")

        resume_step, loss_history = self._try_resume("sft")
        if resume_step > 0:
            print(f"  Resuming ZO-SFT from step {resume_step + 1} / {len(cached_batches)}")

        self.policy.train()
        total = len(cached_batches)

        for step, batch in enumerate(cached_batches, 1):
            if step <= resume_step:
                continue

            gpu_batch = {
                k: batch[k].to(self.accelerator.device)
                for k in ['chosen_input_ids', 'chosen_attention_mask', 'chosen_labels']
            }
            loss = self._sft_step(gpu_batch)
            loss_history.append(loss)

            if self.accelerator.is_main_process:
                print(f"ZO-SFT {step}/{total} | loss={loss:.4f}")
                wandb.log({"train/zo_sft_loss": loss, "step": step})

            if step % self.checkpoint_every == 0:
                self._save_checkpoint("sft", step, loss_history)

        self.save_final("sft")

    def _run_dpo(self, cached_batches: List[dict], beta: float):
        # ── load frozen reference model, compute ref logps ──────────
        ref_path  = self.config.loss.sft_model_path
        dtype     = getattr(torch, self.config.model.policy_dtype)

        print(f"\n--- Pre-computing ref logps  (ref: {ref_path}) ---")
        ref_model = transformers.AutoModelForCausalLM.from_pretrained(
            ref_path,
            cache_dir=self.hf_model_cache,
            torch_dtype=dtype,
            device_map=self.accelerator.device,
        )
        ref_model.eval()
        disable_dropout(ref_model)

        with torch.no_grad():
            for batch in tqdm.tqdm(cached_batches, desc="Ref logps"):
                gb = {
                    k: batch[k].to(self.accelerator.device)
                    for k in [
                        'chosen_input_ids',   'chosen_attention_mask',   'chosen_labels',
                        'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels',
                    ]
                }
                B   = gb['chosen_input_ids'].shape[0]
                ids = torch.cat([gb['chosen_input_ids'],      gb['rejected_input_ids']],      dim=0)
                msk = torch.cat([gb['chosen_attention_mask'], gb['rejected_attention_mask']], dim=0)
                all_logits = ref_model(ids, attention_mask=msk).logits

                batch['ref_chosen_logps']  = get_batch_logps(all_logits[:B], gb['chosen_labels']).cpu()
                batch['ref_rejected_logps'] = get_batch_logps(all_logits[B:], gb['rejected_labels']).cpu()

        del ref_model
        torch.cuda.empty_cache()

        # ── DPO training ─────────────────────────────────────────────
        print("\n--- DPO ---")

        resume_step, loss_history = self._try_resume("dpo")
        if resume_step > 0:
            print(f"  Resuming DPO from step {resume_step + 1} / {len(cached_batches)}")

        self.policy.train()
        total = len(cached_batches)

        for step, batch in enumerate(cached_batches, 1):
            if step <= resume_step:
                continue

            gpu_batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
            loss = self._dpo_step(gpu_batch, beta=beta)
            loss_history.append(loss)

            if self.accelerator.is_main_process:
                print(f"DPO {step}/{total} | loss={loss:.4f}")
                wandb.log({"train/dpo_loss": loss, "step": step})

            if step % self.checkpoint_every == 0:
                self._save_checkpoint("dpo", step, loss_history)

        self.save_final("dpo")

    # ================================================================
    # Checkpoint helpers
    # ================================================================

    @property
    def _ckpt_path(self) -> str:
        """Canonical path of the single checkpoint file."""
        return os.path.join(self.ckpt_dir, "checkpoint.pt")

    @property
    def _ckpt_tmp_path(self) -> str:
        """Temporary write buffer; renamed atomically on success."""
        return os.path.join(self.ckpt_dir, "checkpoint.pt.tmp")

    def _save_checkpoint(self, stage: str, step: int, loss_history: List[float]):
        """
        Save a crash-recovery snapshot (main process only).

        Layout: a single checkpoint.pt containing everything needed to
        resume — model weights, training cursor, RNG states, and the
        W&B run-id so the same run can be continued on the dashboard.

        Atomic write protocol (POSIX rename guarantee):
          1. Serialise into checkpoint.pt.tmp  via torch.save()
          2. os.replace(tmp -> checkpoint.pt)
        If the process dies between steps 1 and 2, the previous
        checkpoint.pt is still intact and will be used on the next start.

        Multi-GPU: only the main process writes to disk; weights are
        gathered from all ranks first via accelerator.get_state_dict().
        """
        if not self.accelerator.is_main_process:
            return

        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Gather full model state from all ranks (no-op on single GPU)
        state_dict = self.accelerator.get_state_dict(self.policy)

        # Collect per-rank RNG states; on multi-GPU each rank will
        # restore its own state independently during _try_resume.
        rng_states = {
            "torch_rng": torch.get_rng_state(),
            "cuda_rng":  (torch.cuda.get_rng_state()
                          if torch.cuda.is_available() else None),
        }

        # W&B run-id: only meaningful when wandb is active
        wandb_run_id: Optional[str] = None
        try:
            if wandb.run is not None:
                wandb_run_id = wandb.run.id
        except Exception:
            pass

        payload = {
            "step":          step,
            "stage":         stage,
            "loss_history":  loss_history,
            "state_dict":    state_dict,
            "rng_states":    rng_states,
            "wandb_run_id":  wandb_run_id,
        }

        # Atomic write: tmp -> final
        torch.save(payload, self._ckpt_tmp_path)
        os.replace(self._ckpt_tmp_path, self._ckpt_path)

        print(f"  [ckpt] Saved checkpoint step={step} --> {self._ckpt_path}")

    def _try_resume(self, stage: str) -> Tuple[int, List[float]]:
        """
        Attempt to resume from checkpoint.pt.

        Returns (resume_step, loss_history).
        resume_step == 0 means start from scratch.

        Side-effects:
          - self.policy weights are replaced in-place on all ranks
          - RNG states are restored on every rank independently
          - self.resumed_wandb_run_id is set if a run-id was found,
            so that train.py can call wandb.init(id=..., resume="must")
            before the first log call
          - accelerator.wait_for_everyone() is called at the end so
            all ranks enter the training loop with identical weights
        """
        if not os.path.exists(self._ckpt_path):
            return 0, []

        try:
            payload = torch.load(self._ckpt_path, map_location="cpu",
                                 weights_only=False)
        except Exception as e:
            print(f"  [ckpt] WARNING: failed to load checkpoint ({e}). "
                  f"Starting from scratch.")
            return 0, []

        saved_stage = payload.get("stage", stage)
        if saved_stage != stage:
            print(f"  [ckpt] WARNING: checkpoint stage '{saved_stage}' != "
                  f"current stage '{stage}'. Ignoring checkpoint.")
            return 0, []

        resume_step  = payload["step"]
        loss_history = payload.get("loss_history", [])

        # ── restore model weights (all ranks) ───────────────────────
        # load_state_dict is called on the unwrapped model so it works
        # regardless of whether Accelerator has wrapped the module.
        unwrapped = self.accelerator.unwrap_model(self.policy)
        missing, unexpected = unwrapped.load_state_dict(
            payload["state_dict"], strict=False
        )
        if missing:
            print(f"  [ckpt] WARNING: missing keys in checkpoint: {missing[:5]}")
        if unexpected:
            print(f"  [ckpt] WARNING: unexpected keys in checkpoint: {unexpected[:5]}")

        # ── restore RNG states (every rank independently) ───────────
        rng = payload.get("rng_states", {})
        if rng.get("torch_rng") is not None:
            torch.set_rng_state(rng["torch_rng"].cpu())
        if rng.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda_rng"].cpu())

        # ── capture W&B run-id for train.py ─────────────────────────
        self.resumed_wandb_run_id = payload.get("wandb_run_id", None)

        # ── barrier: ensure all ranks are ready before training ─────
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            print(f"  [ckpt] Resumed {stage} from step {resume_step}  "
                  f"(wandb_run_id={self.resumed_wandb_run_id})")

        return resume_step, loss_history

    # ================================================================
    # Final model save  (end of stage)
    # ================================================================

    def save_final(self, stage: str):
        """
        Save the finished model to:
            runs/<exp_name>/<stage>/final_model/

        Only the main process writes.  Uses accelerator.get_state_dict()
        to gather weights from all ranks before saving.
        """
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return

        out_dir = os.path.join(self.runs_dir, "final_model")
        os.makedirs(out_dir, exist_ok=True)

        # Gather weights from all ranks, then save via unwrapped model
        state_dict  = self.accelerator.get_state_dict(self.policy)
        unwrapped   = self.accelerator.unwrap_model(self.policy)
        unwrapped.save_pretrained(out_dir, state_dict=state_dict)

        print(f"  [final] Model saved --> {out_dir}")
        self.accelerator.wait_for_everyone()

    # kept for backwards compat with train.py caller
    def save(self, _output_dir: str):
        """Alias: save_final() for the current stage. output_dir arg is ignored."""
        self.save_final(self.config.loss.name)

    # ================================================================
    # Optional teardown
    # ================================================================

    def cleanup(self):
        """Override to release resources (e.g. remove forward hooks)."""
        pass


# ============================================================
# MeZO  --  isotropic Gaussian perturbation
# ============================================================

class MeZOTrainer(ZOTrainerBase):
    """
    Memory-Efficient Zeroth-Order (MeZO) optimizer.
    Perturbation direction z ~ N(0, I) is regenerated from a seed
    so it never has to be stored in memory.
    """

    def __init__(self, policy: nn.Module, config: DictConfig):
        super().__init__(policy, config)
        self._params: List[nn.Parameter] = [
            p for p in policy.parameters() if p.requires_grad
        ]

    def _perturb(self, scaling: float):
        torch.manual_seed(self._zo_seed)
        for p in self._params:
            p.data.add_(torch.randn_like(p), alpha=scaling * self.eps)

    def _apply_update(self, projected_grad: float):
        torch.manual_seed(self._zo_seed)
        for p in self._params:
            z = torch.randn_like(p)
            p.data.add_(z, alpha=self.eps - self.lr * projected_grad)


# ============================================================
# AGZOEngine  --  forward-hook machinery for AGZO
# ============================================================

class AGZOEngine:
    """
    Manages forward hooks and activation subspace computation.

    Public API
    ----------
    collect_sft(fwd_fn)         run fwd_fn() (chosen only), plain activation basis
    collect_plain(fwd_fn)       run fwd_fn() (any batch), plain activation basis
                                 same math as collect_sft; for plain-AGZO DPO steps
    collect_dpo(fwd_fn, B)      run fwd_fn() on [chosen;rejected] concat,
                                 H_diff = H_c - H_r subspace (preference-aware)
    sample_z(name, param)       sample structured perturbation z from current basis
    remove_hooks()              deregister all forward hooks
    """

    def __init__(self, model: nn.Module, power_iter_steps: int, rank: int):
        self.model            = model
        self.power_iter_steps = power_iter_steps
        self.rank             = rank

        self.basis: Dict[str, torch.Tensor] = {}   # param_name -> (r, d_in)
        self._hooks:     List  = []
        self._param_map: Dict[str, nn.Parameter] = {}
        # Modes:
        #   "sft"   -- plain activations from a single (chosen-only) forward
        #   "plain" -- plain activations from any forward (merged or single)
        #   "dpo"   -- H_diff = H_chosen - H_rejected  (preference subspace)
        self._mode: Optional[str] = None
        self._B:    int            = 0     # single-side batch size (dpo mode)

        self._register_hooks()

    # -- hook registration -------------------------------------------

    def _register_hooks(self):
        pd = dict(self.model.named_parameters())
        for mn, m in self.model.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            pname = f"{mn}.weight" if mn else "weight"
            p = pd.get(pname)
            if p is None or not p.requires_grad:
                continue
            self._hooks.append(m.register_forward_hook(self._make_hook(pname)))
            self._param_map[pname] = p

    def _make_hook(self, pname: str):
        def _h(module, inputs, output):
            if self._mode is None:
                return
            act = next((x for x in inputs if isinstance(x, torch.Tensor)), None)
            if act is None:
                return
            act = act.detach()   # (B_total, T, d)

            if self._mode in ("sft", "plain"):
                # Plain activation subspace: use all rows of the activation
                # matrix as-is, regardless of whether this is a single-sequence
                # or a concatenated [chosen; rejected] batch.
                basis = self._make_basis(act.reshape(-1, act.shape[-1]).float(), pname)

            elif self._mode == "dpo":
                B = self._B
                if act.shape[0] != 2 * B:
                    basis = self._make_basis(act.reshape(-1, act.shape[-1]).float(), pname)
                else:
                    h_c = act[:B].reshape(-1, act.shape[-1]).float()  # (B*T, d)
                    h_r = act[B:].reshape(-1, act.shape[-1]).float()  # (B*T, d)
                    h_diff_mean   = h_c.mean(0, keepdim=True) - h_r.mean(0, keepdim=True)
                    h_diff_tokens = h_c - h_r
                    h_diff = torch.cat([h_diff_mean, h_diff_tokens], 0)
                    basis = self._make_basis(h_diff, pname)
            else:
                return

            if basis is not None:
                self.basis[pname] = basis
        return _h

    # -- basis computation -------------------------------------------

    def _make_basis(self, act_2d: torch.Tensor, pname: str) -> Optional[torch.Tensor]:
        if act_2d.numel() == 0 or act_2d.shape[1] == 0:
            return None
        p = self._param_map.get(pname)
        if p is None:
            return None
        d     = act_2d.shape[1]
        max_r = d if p.dim() < 2 else min(d, p.shape[0], p.shape[1])
        r     = max(1, min(self.rank, max_r))
        q     = self._power_iter(act_2d, self.power_iter_steps, r)
        if q is None:
            return None
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = q / (q.norm(2, dim=1, keepdim=True) + 1e-12)
        return q.to(device=p.device, dtype=p.dtype)

    @staticmethod
    def _power_iter(A: torch.Tensor, steps: int, r: int) -> Optional[torch.Tensor]:
        """Block power iteration; returns (r, d) top-r right singular vectors."""
        d = A.shape[1]
        r = max(1, min(r, d))
        q = torch.randn(d, r, device=A.device, dtype=A.dtype)
        q, _ = torch.linalg.qr(q, mode="reduced")
        for _ in range(max(1, steps)):
            y = A.matmul(q)
            z = A.T.matmul(y)
            if not torch.isfinite(z).all():
                break
            q, _ = torch.linalg.qr(z, mode="reduced")
        return q.T   # (r, d)

    # -- collection interface ----------------------------------------

    def collect_sft(self, fwd_fn):
        self.basis.clear()
        self._mode = "sft"
        try:
            with torch.no_grad():
                fwd_fn()
        finally:
            self._mode = None

    def collect_plain(self, fwd_fn):
        """
        Run fwd_fn() and build the activation subspace directly from the
        raw activation matrix -- no chosen/rejected splitting, no H_diff.

        The subspace captures directions of high activation variance in the batch, 
        irrespective of label identity.
        """
        self.basis.clear()
        self._mode = "plain"
        try:
            with torch.no_grad():
                fwd_fn()
        finally:
            self._mode = None

    def collect_dpo(self, fwd_fn, B: int):
        self.basis.clear()
        self._mode = "dpo"
        self._B    = B
        try:
            with torch.no_grad():
                fwd_fn()
        finally:
            self._mode = None
            self._B    = 0

    # -- perturbation sampling ---------------------------------------

    def sample_z(self, name: str, param: nn.Parameter) -> torch.Tensor:
        """
        Structured z:  z = (r @ basis) / sqrt(rank)   when basis is available
        Fallback z:    z ~ N(0, I)                     otherwise
        """
        basis = self.basis.get(name)
        if basis is None or param.dim() < 2:
            return torch.randn_like(param.data)
        basis = basis.to(device=param.device, dtype=param.dtype)
        if basis.shape[1] != param.shape[1]:
            return torch.randn_like(param.data)
        r_vec = torch.randn(param.shape[0], basis.shape[0],
                            device=param.device, dtype=param.dtype)
        return r_vec.matmul(basis) / math.sqrt(basis.shape[0])

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ============================================================
# AGZO  --  activation-guided structured perturbation
# ============================================================

class AGZOTrainer(ZOTrainerBase):
    """
    Activation-Guided Zeroth-Order (AGZO) optimizer.

    Before each pair of +/-eps forward passes, one extra no-grad forward
    computes a per-layer activation subspace (plain for SFT; H_diff for DPO).
    z is then sampled inside that subspace instead of isotropically.

    Forward count per DPO step:
      MeZO : 4  (chosen + rejected) x 2 sides
      AGZO : 3  1 basis (merged) + 2 x 1 (merged +/-eps)
    """

    def __init__(self, policy: nn.Module, config: DictConfig):
        super().__init__(policy, config)
        tcfg = config.trainer
        self._engine = AGZOEngine(
            policy,
            power_iter_steps=int(tcfg.get("power_iter_steps", 5)),
            rank=int(tcfg.get("rank", 1)),
        )
        self._params: List[Tuple[str, nn.Parameter]] = [
            (n, p) for n, p in policy.named_parameters() if p.requires_grad
        ]

    def _sft_step(self, gpu_batch: dict) -> float:
        self._reset_seed()

        self._engine.collect_sft(
            lambda: self.policy(
                gpu_batch['chosen_input_ids'],
                attention_mask=gpu_batch['chosen_attention_mask'],
            )
        )
        with torch.no_grad():
            self._perturb(+1)
            loss_p = compute_sft_loss(
                self.policy(
                    gpu_batch['chosen_input_ids'],
                    attention_mask=gpu_batch['chosen_attention_mask'],
                ).logits,
                gpu_batch['chosen_labels'],
            )
            self._perturb(-2)
            loss_m = compute_sft_loss(
                self.policy(
                    gpu_batch['chosen_input_ids'],
                    attention_mask=gpu_batch['chosen_attention_mask'],
                ).logits,
                gpu_batch['chosen_labels'],
            )
            self._apply_update(((loss_p - loss_m) / (2 * self.eps)).item())
        return ((loss_p + loss_m) / 2).item()

    def _dpo_step(self, gpu_batch: dict, beta: float) -> float:
        self._reset_seed()
        B    = gpu_batch['chosen_input_ids'].shape[0]
        ids, mask = self._concat_cr(gpu_batch)

        self._engine.collect_dpo(
            lambda: self.policy(ids, attention_mask=mask),
            B=B,
        )

        def _fwd_split() -> Tuple[torch.Tensor, torch.Tensor]:
            logits = self.policy(ids, attention_mask=mask).logits
            return logits[:B], logits[B:]
        
        with torch.no_grad():
            self._perturb(+1)
            c_p, r_p = _fwd_split()
            loss_p = compute_dpo_loss(
                c_p, r_p,
                gpu_batch['ref_chosen_logps'], gpu_batch['ref_rejected_logps'],
                gpu_batch['chosen_labels'],    gpu_batch['rejected_labels'],
                beta,
            )
            self._perturb(-2)
            c_m, r_m = _fwd_split()
            loss_m = compute_dpo_loss(
                c_m, r_m,
                gpu_batch['ref_chosen_logps'], gpu_batch['ref_rejected_logps'],
                gpu_batch['chosen_labels'],    gpu_batch['rejected_labels'],
                beta,
            )
            self._apply_update(((loss_p - loss_m) / (2 * self.eps)).item())
        return ((loss_p + loss_m) / 2).item()

    def _perturb(self, scaling: float):
        torch.manual_seed(self._zo_seed)
        for name, param in self._params:
            z = self._engine.sample_z(name, param)
            param.data.add_(z, alpha=scaling * self.eps)

    def _apply_update(self, projected_grad: float):
        torch.manual_seed(self._zo_seed)
        for name, param in self._params:
            z = self._engine.sample_z(name, param)
            param.data.add_(z, alpha=self.eps - self.lr * projected_grad)

    def cleanup(self):
        self._engine.remove_hooks()


# ============================================================
# AGZOPlain  --  plain-activation subspace perturbation
# ============================================================

class AGZOPlainTrainer(AGZOTrainer):
    """
    Plain-AGZO: activation-guided perturbation WITHOUT the H_diff trick.

    Both SFT and DPO steps build the per-layer basis from the raw activation
    matrix of the forward pass (i.e. collect_plain instead of collect_dpo).
    This captures directions of high activation variance in the batch but does
    NOT explicitly align the subspace with the chosen-vs-rejected preference
    direction.

    Use this as an ablation baseline against AGZOTrainer (agzo):
      trainer=agzo_plain  -->  activation guidance, no H_diff
      trainer=agzo        -->  activation guidance + H_diff preference subspace

    Forward count per DPO step (identical to AGZOTrainer):
      3  =  1 plain basis (merged) + 2 x 1 (merged +/-eps)
    """

    # _sft_step is inherited unchanged from AGZOTrainer (already uses collect_sft
    # which is plain-activation, so no override needed).

    def _dpo_step(self, gpu_batch: dict, beta: float) -> float:
        self._reset_seed()
        B    = gpu_batch['chosen_input_ids'].shape[0]
        ids, mask = self._concat_cr(gpu_batch)

        # KEY DIFFERENCE: collect_plain instead of collect_dpo
        # The subspace is built from the raw concatenated activations
        # without splitting or differencing chosen vs rejected.
        self._engine.collect_plain(
            lambda: self.policy(ids, attention_mask=mask)
        )

        def _fwd_split() -> Tuple[torch.Tensor, torch.Tensor]:
            logits = self.policy(ids, attention_mask=mask).logits
            return logits[:B], logits[B:]

        with torch.no_grad():
            self._perturb(+1)
            c_p, r_p = _fwd_split()
            loss_p = compute_dpo_loss(
                c_p, r_p,
                gpu_batch['ref_chosen_logps'], gpu_batch['ref_rejected_logps'],
                gpu_batch['chosen_labels'],    gpu_batch['rejected_labels'],
                beta,
            )
            self._perturb(-2)
            c_m, r_m = _fwd_split()
            loss_m = compute_dpo_loss(
                c_m, r_m,
                gpu_batch['ref_chosen_logps'], gpu_batch['ref_rejected_logps'],
                gpu_batch['chosen_labels'],    gpu_batch['rejected_labels'],
                beta,
            )
            self._apply_update(((loss_p - loss_m) / (2 * self.eps)).item())
        return ((loss_p + loss_m) / 2).item()


# ============================================================
# Registry
# ============================================================

TRAINER_MAP: Dict[str, type] = {
    "mezo":       MeZOTrainer,
    "agzo":       AGZOTrainer,        # activation-guided, H_diff preference subspace (DPO)
    "agzo_plain": AGZOPlainTrainer,   # activation-guided, plain basis (no H_diff)
}


def build_trainer(policy: nn.Module, config: DictConfig) -> ZOTrainerBase:
    """Instantiate the correct trainer from config.trainer.name."""
    name = config.trainer.name
    cls  = TRAINER_MAP.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown trainer '{name}'. "
            f"Registered trainers: {sorted(TRAINER_MAP.keys())}"
        )
    return cls(policy, config)