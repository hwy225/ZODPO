import json
import math
import os
import random
import shutil
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
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

def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor, compute_fp32: bool = False) -> torch.Tensor:
    """Per-sample sum of token log-probs, ignoring -100 positions."""
    labels    = labels[:, 1:].clone()
    logits    = logits[:, :-1, :]
    
    if compute_fp32:
        logits = logits.to(torch.float32)
        
    loss_mask = labels != -100
    labels[labels == -100] = 0
    per_token = torch.gather(
        logits.log_softmax(-1), 2, labels.unsqueeze(2)
    ).squeeze(2)
    return (per_token * loss_mask).sum(-1)


def compute_sft_loss(logits: torch.Tensor, labels: torch.Tensor, compute_fp32: bool = False) -> torch.Tensor:
    if compute_fp32:
        logits = logits.to(torch.float32)
        
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
    compute_fp32: bool = False,
) -> torch.Tensor:
    pi_c   = get_batch_logps(pi_chosen_logits,   chosen_labels, compute_fp32)
    pi_r   = get_batch_logps(pi_rejected_logits, rejected_labels, compute_fp32)
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
        disable_dropout(self.policy)
        self.config        = config
        self.lr            = config.trainer.lr
        self.eps           = config.trainer.eps
        self.total_batches = config.total_batches
        self._zo_seed: int = 0

        # 从 config 中读取配置，提供安全默认值
        self.compute_logps_fp32 = config.get("compute_logps_fp32", True)
        self.max_loss_threshold = float(config.get("max_loss_threshold", 10.0))
        self.max_margin_threshold = float(config.get("max_margin_threshold", float('inf')))

        # ── Accelerator (single-GPU or multi-GPU transparent) ────────
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
        if self.accelerator.is_main_process:
            seed_tensor = torch.randint(0, 2**32, (1,), device=self.accelerator.device)
        else:
            seed_tensor = torch.empty((1,), dtype=torch.int64, device=self.accelerator.device)
        
        if self.accelerator.num_processes > 1:
            dist.broadcast(seed_tensor, src=0)
            
        self._zo_seed = int(seed_tensor.item())

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
        self._reset_seed()

        active_params = [p for p in self.policy.parameters() if p.requires_grad]
        saved_weights = [p.data.clone() for p in active_params]
        with torch.no_grad():
            self._perturb(+1)
            loss_p = compute_sft_loss(
                self.policy(
                    gpu_batch['chosen_input_ids'],
                    attention_mask=gpu_batch['chosen_attention_mask'],
                ).logits,
                gpu_batch['chosen_labels'],
                compute_fp32=self.compute_logps_fp32
            )

            for p, saved_w in zip(active_params, saved_weights):
                p.data.copy_(saved_w)

            self._perturb(-1)
            loss_m = compute_sft_loss(
                self.policy(
                    gpu_batch['chosen_input_ids'],
                    attention_mask=gpu_batch['chosen_attention_mask'],
                ).logits,
                gpu_batch['chosen_labels'],
                compute_fp32=self.compute_logps_fp32
            )

            # self._perturb(+1)
            for p, saved_w in zip(active_params, saved_weights):
                p.data.copy_(saved_w)
            
            self._apply_update(((loss_p - loss_m) / (2 * self.eps)).item())
        return ((loss_p + loss_m) / 2).item()

    def _dpo_loss(self, gpu_batch: dict, beta: float) -> float:
        B    = gpu_batch['chosen_input_ids'].shape[0]
        ids, mask = self._concat_cr(gpu_batch)

        active_params = [p for p in self.policy.parameters() if p.requires_grad]
        saved_weights = [p.data.clone() for p in active_params]

        with torch.no_grad():
            self._perturb(+1)
            logits = self.policy(ids, attention_mask=mask).logits
            loss_p = compute_dpo_loss(
                logits[:B], logits[B:],
                gpu_batch['ref_chosen_logps'], gpu_batch['ref_rejected_logps'],
                gpu_batch['chosen_labels'],    gpu_batch['rejected_labels'],
                beta,
                compute_fp32=self.compute_logps_fp32
            )

            for p, saved_w in zip(active_params, saved_weights):
                p.data.copy_(saved_w)

            self._perturb(-1)

            logits = self.policy(ids, attention_mask=mask).logits
            loss_m = compute_dpo_loss(
                logits[:B], logits[B:],
                gpu_batch['ref_chosen_logps'], gpu_batch['ref_rejected_logps'],
                gpu_batch['chosen_labels'],    gpu_batch['rejected_labels'],
                beta,
                compute_fp32=self.compute_logps_fp32
            )
            # self._perturb(+1)   # restore to θ₀ after ±ε cycle
            for p, saved_w in zip(active_params, saved_weights):
                p.data.copy_(saved_w)

        return loss_p.item(), loss_m.item()

    def _collect_dpo_basis(self, gpu_batch: dict):
        pass

    def _dpo_step(self, gpu_batch: dict, beta: float) -> float:
        self._collect_dpo_basis(gpu_batch)
        self._reset_seed()
        loss_p, loss_m = self._dpo_loss(gpu_batch, beta)
        g_hat = (loss_p - loss_m) / (2 * self.eps)
        
        clip_val = 0.05 / self.eps
        g_hat = max(min(g_hat, clip_val), -clip_val)
        
        self._apply_update(g_hat)
        return (loss_p + loss_m) / 2

    # ================================================================
    # Main entry
    # ================================================================

    def train(self):
        stage = self.config.loss.name   # "sft" | "dpo"

        if stage == "sft":
            self._run_sft_standard()
            return

        if stage == "dpo":
            self._run_dpo(beta=self.config.loss.beta)
        else:
            raise ValueError(
                f"Unknown loss.name: {stage!r}.  Valid choices: 'sft', 'dpo'."
            )

    # ================================================================
    # Stage runners
    # ================================================================

    def _run_sft_standard(self):
        print("\n--- SFT (TRL SFTTrainer) ---")

        sft_cfg = self.config.loss      
        stage   = "sft"

        from datasets import load_dataset
        import re
        raw = load_dataset(
            "Anthropic/hh-rlhf",
            cache_dir=self.dataset_cache_dir,
            split="train",
        )
        eval_raw = load_dataset(
            "Anthropic/hh-rlhf",
            cache_dir=self.dataset_cache_dir,
            split="test",
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
        eval_dataset = eval_raw.map(
            _to_prompt_completion,
            remove_columns=eval_raw.column_names,
            desc="Formatting test dataset",
        )

        out_dir = os.path.join(self.runs_dir, "final_model")
        sft_config = SFTConfig(
            output_dir=out_dir,
            num_train_epochs=int(sft_cfg.get("num_train_epochs", 1)),
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=int(self.config.get("gradient_accumulation_steps", 1)),
            learning_rate=float(sft_cfg.get("lr", 5e-5)),
            lr_scheduler_type=sft_cfg.get("lr_scheduler_type", "cosine"),
            warmup_ratio=float(sft_cfg.get("warmup_ratio", 0.03)),
            max_length=self.config.max_length,
            gradient_checkpointing=True,
            completion_only_loss=True,   
            logging_steps=10,

            eval_strategy="steps",  # 新版 transformers 中参数名变更为 eval_strategy
            eval_steps=500,               # 每训练 500 步执行一次评估 (具体数值根据你的总 step 调整)
            per_device_eval_batch_size=4, # 评估时的批次大小 (尽量设大一点以加快评估速度，不爆显存即可)

            save_strategy="steps",
            save_steps=500,
            save_total_limit=1,
            bf16=(self.config.model.policy_dtype == "bfloat16"),
            fp16=(self.config.model.policy_dtype == "float16"),
            report_to="wandb" if self.config.wandb.enabled else "none",
        )

        trainer = SFTTrainer(
            model=self.policy,
            args=sft_config,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )
        trainer.train()
        self.save_final(stage)

    def _run_sft(self, cached_batches: List[dict]):
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

    def _run_dpo(self, beta: float):
        loss_cfg   = self.config.loss
        ref_path   = loss_cfg.sft_model_path
        dtype      = getattr(torch, self.config.model.policy_dtype)
        n_epochs   = int(loss_cfg.get("num_epochs", 1))
        grad_accum = int(self.config.get("gradient_accumulation_steps", 1))

        eff_bs = self.config.batch_size * grad_accum
        use_full_dataset = (loss_cfg.get("num_epochs", None) is not None)
        n_examples = None if use_full_dataset else self.total_batches * self.config.batch_size

        sft_stage_dir = os.path.dirname(os.path.normpath(ref_path))  
        os.makedirs(sft_stage_dir, exist_ok=True)
        
        fp32_suffix = "_fp32" if self.compute_logps_fp32 else ""
        cache_file = os.path.join(
            sft_stage_dir,
            f"ref_logps_bs{self.config.batch_size}_gc{grad_accum}_bfloat16{fp32_suffix}.pt"
        )

        if os.path.exists(cache_file):
            print(f"\n--- [Cache Hit] Loading ref logps from {cache_file} ---")
            ref_logps: List[Tuple[torch.Tensor, torch.Tensor]] = torch.load(
                cache_file, map_location="cpu", weights_only=True
            )
        else:
            print(f"\n--- [Cache Miss] Computing ref logps (one-time cost) ---")
            # Each rank loads the ref model on its own device.
            # Only rank 0 will write the cache file; others wait at the barrier below.
            ref_model = transformers.AutoModelForCausalLM.from_pretrained(
                ref_path,
                cache_dir=self.hf_model_cache,
                torch_dtype=bfloat16,
                device_map=self.accelerator.device,
            )
            ref_model.eval()
            disable_dropout(ref_model)

            ref_iter = get_chat_template_iterator(
                tokenizer=self.tokenizer,
                split='train',
                batch_size=self.config.batch_size,   
                n_epochs=1,
                n_examples=n_examples,
                max_length=self.config.max_length,
                max_prompt_length=self.config.max_prompt_length,
                shuffle=False,          
                cache_dir=self.dataset_cache_dir,
            )

            ref_logps = []
            n_proc_ref = self.accelerator.num_processes
            rank_ref   = self.accelerator.process_index

            with torch.no_grad():
                all_batches = list(tqdm.tqdm(
                    get_chat_template_iterator(
                        tokenizer=self.tokenizer,
                        split='train',
                        batch_size=self.config.batch_size,
                        n_epochs=1,
                        n_examples=n_examples,
                        max_length=self.config.max_length,
                        max_prompt_length=self.config.max_prompt_length,
                        shuffle=False,
                        cache_dir=self.dataset_cache_dir,
                    ),
                    desc="Loading for ref logps",
                    disable=(rank_ref != 0),
                ))

                # Each rank computes its slice; we assemble in order afterward.
                local_results: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
                for i, batch in enumerate(tqdm.tqdm(
                    all_batches,
                    desc=f"Ref logps (rank {rank_ref})",
                    disable=(rank_ref != 0),
                )):
                    if i % n_proc_ref != rank_ref:
                        continue   # another rank handles this batch
                    gb = {k: batch[k].to(self.accelerator.device)
                          for k in ['chosen_input_ids',   'chosen_attention_mask',   'chosen_labels',
                                    'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels']}
                    B   = gb['chosen_input_ids'].shape[0]
                    ids = torch.cat([gb['chosen_input_ids'],      gb['rejected_input_ids']],      dim=0)
                    msk = torch.cat([gb['chosen_attention_mask'], gb['rejected_attention_mask']], dim=0)
                    all_logits = ref_model(ids, attention_mask=msk).logits
                    local_results[i] = (
                        get_batch_logps(all_logits[:B], gb['chosen_labels'], self.compute_logps_fp32).cpu(),
                        get_batch_logps(all_logits[B:], gb['rejected_labels'], self.compute_logps_fp32).cpu(),
                    )

            # Gather results from all ranks onto rank 0 and assemble in order
            if n_proc_ref > 1:
                gathered = [None] * n_proc_ref
                dist.all_gather_object(gathered, local_results)
                merged: Dict[int, Tuple] = {}
                for d in gathered:
                    merged.update(d)
                ref_logps = [merged[i] for i in sorted(merged.keys())]
            else:
                ref_logps = [local_results[i] for i in sorted(local_results.keys())]

            # Only rank 0 writes the cache (atomic write)
            if self.accelerator.is_main_process:
                tmp_cache_file = f"{cache_file}.tmp.{os.getpid()}"
                torch.save(ref_logps, tmp_cache_file)
                os.replace(tmp_cache_file, cache_file)
                print(f"  [Cache Saved] {len(ref_logps)} mini-batches -> {cache_file}")

            # All ranks must wait before proceeding (so they all load from the same cache)
            self.accelerator.wait_for_everyone()
            del ref_model
            torch.cuda.empty_cache()

        n_batches = len(ref_logps)
        n_proc    = self.accelerator.num_processes
        rank      = self.accelerator.process_index
        print(f"\n--- DPO ({n_epochs} epoch(s), {n_batches} mini-batches/epoch, "
              f"grad_accum={grad_accum}, eff_bs={eff_bs}, n_gpu={n_proc}) ---")

        resume_step, loss_history = self._try_resume("dpo")
        if resume_step > 0:
            print(f"  Resuming DPO from optimizer step {resume_step + 1}")

        self.policy.train()
        global_step = 1
        stop_training = False

        for epoch in range(n_epochs):
            if stop_training:
                break

            print(f"\nEpoch {epoch + 1}/{n_epochs}")

            indices = list(range(n_batches))
            random.shuffle(indices)

            train_iter = get_chat_template_iterator(
                tokenizer=self.tokenizer,
                split='train',
                batch_size=self.config.batch_size,
                n_epochs=1,
                n_examples=n_examples,
                max_length=self.config.max_length,
                max_prompt_length=self.config.max_prompt_length,
                shuffle=False,
                cache_dir=self.dataset_cache_dir,
            )
            epoch_batches = list(tqdm.tqdm(train_iter,
                                           desc=f"Loading epoch {epoch+1}",
                                           leave=False))

            for window_start in tqdm.tqdm(
                range(0, len(indices), grad_accum),
                desc=f"DPO (Epoch {epoch + 1})",
                disable=(rank != 0),   # only rank 0 shows the bar
            ):
                if global_step <= resume_step:
                    global_step += 1
                    continue

                window = indices[window_start : window_start + grad_accum]

                # ── basis collection (AGZO only, from first batch in window) ──
                # Each rank uses the same first batch so the basis is identical.
                first_batch = epoch_batches[window[0]]
                first_ref_c, first_ref_r = ref_logps[window[0]]
                first_gpu = {k: v.to(self.accelerator.device) for k, v in first_batch.items()}
                first_gpu['ref_chosen_logps']  = first_ref_c.to(self.accelerator.device)
                first_gpu['ref_rejected_logps'] = first_ref_r.to(self.accelerator.device)
                self._collect_dpo_basis(first_gpu)

                # ── DDP gradient accumulation ─────────────────────────────────
                # The window of grad_accum mini-batches is split across ranks.
                # rank r processes window[r], window[r + n_proc], window[r + 2*n_proc], ...
                # Each rank accumulates its own local g_hat, then we all_reduce.
                self._reset_seed()
                local_g_hat = 0.0
                local_loss  = 0.0
                local_count = 0

                for sub_idx, idx in enumerate(window):
                    # Assign mini-batch sub_idx to rank (sub_idx % n_proc)
                    if sub_idx % n_proc != rank:
                        continue

                    batch = epoch_batches[idx]
                    ref_chosen_logps, ref_rejected_logps = ref_logps[idx]

                    gpu_batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                    gpu_batch['ref_chosen_logps']  = ref_chosen_logps.to(self.accelerator.device)
                    gpu_batch['ref_rejected_logps'] = ref_rejected_logps.to(self.accelerator.device)

                    loss_p, loss_m = self._dpo_loss(gpu_batch, beta)
                    local_g_hat += (loss_p - loss_m) / (2 * self.eps)
                    local_loss  += (loss_p + loss_m) / 2
                    local_count += 1

                # ── all_reduce: sum g_hat and loss across ranks, then average ──
                if n_proc > 1:
                    # Pack into a single tensor for one communication round
                    stats = torch.tensor(
                        [local_g_hat, local_loss, float(local_count)],
                        dtype=torch.float64,
                        device=self.accelerator.device,
                    )
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    total_g_hat = stats[0].item()
                    total_loss  = stats[1].item()
                    total_count = int(stats[2].item())
                else:
                    total_g_hat = local_g_hat
                    total_loss  = local_loss
                    total_count = local_count

                # Average over all mini-batches in window (equiv to len(window))
                g_hat    = total_g_hat / max(total_count, 1)
                avg_loss = total_loss  / max(total_count, 1)

                clip_val = 0.05 / self.eps
                g_hat = max(min(g_hat, clip_val), -clip_val)

                self._apply_update(g_hat)
                loss_history.append(avg_loss)

                if self.accelerator.is_main_process:
                    print(f"DPO step {global_step} (e{epoch+1}) | "
                          f"loss={avg_loss:.4f} | eff_bs={len(window)*self.config.batch_size}")
                    log_dict = {
                        "train/dpo_loss":        avg_loss,
                        "train/loss_minus_log2": avg_loss - math.log(2),
                        "step":                  global_step,
                        "epoch":                 epoch + 1,
                        "hparams/lr":            self.lr,
                        "hparams/eps":           self.eps,
                        "hparams/beta":          beta,
                        "hparams/eff_bs":        len(window) * self.config.batch_size,
                    }
                    if getattr(self, "_last_margin", None) is not None:
                        log_dict["train/margin"] = self._last_margin
                    wandb.log(log_dict)

                if avg_loss > self.max_loss_threshold:
                    if self.accelerator.is_main_process:
                        print(f"\n[Warning] DPO loss ({avg_loss:.4f}) exceeded the threshold {self.max_loss_threshold} "
                              f"(occurred at step {global_step})。Model has diverged, terminating training...")
                    stop_training = True
                    break

                if global_step % self.checkpoint_every == 0:
                    self._save_checkpoint("dpo", global_step, loss_history)

                global_step += 1

        if stop_training:
            if self.accelerator.is_main_process:
                print("  [Info] Skipping final model save due to divergence.")
                
            # delete checkpoint dir to save space
            if os.path.exists(self.ckpt_dir):
                try:
                    shutil.rmtree(self.ckpt_dir)
                    print(f"  [cleanup] Successfully deleted checkpoint directory --> {self.ckpt_dir}")
                except Exception as e:
                    print(f"  [cleanup] WARNING: Failed to delete checkpoint directory {self.ckpt_dir}. Error: {e}")

        else:
            self.save_final("dpo")

    # ================================================================
    # Checkpoint helpers
    # ================================================================

    @property
    def _ckpt_path(self) -> str:
        return os.path.join(self.ckpt_dir, "checkpoint.pt")

    @property
    def _ckpt_tmp_path(self) -> str:
        return os.path.join(self.ckpt_dir, "checkpoint.pt.tmp")

    def _save_checkpoint(self, stage: str, step: int, loss_history: List[float]):
        if not self.accelerator.is_main_process:
            return

        os.makedirs(self.ckpt_dir, exist_ok=True)

        state_dict = self.accelerator.get_state_dict(self.policy)

        rng_states = {
            "torch_rng": torch.get_rng_state(),
            "cuda_rng":  (torch.cuda.get_rng_state()
                          if torch.cuda.is_available() else None),
        }

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

        torch.save(payload, self._ckpt_tmp_path)
        os.replace(self._ckpt_tmp_path, self._ckpt_path)

        print(f"  [ckpt] Saved checkpoint step={step} --> {self._ckpt_path}")

    def _try_resume(self, stage: str) -> Tuple[int, List[float]]:
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

        unwrapped = self.accelerator.unwrap_model(self.policy)
        missing, unexpected = unwrapped.load_state_dict(
            payload["state_dict"], strict=False
        )
        if missing:
            print(f"  [ckpt] WARNING: missing keys in checkpoint: {missing[:5]}")
        if unexpected:
            print(f"  [ckpt] WARNING: unexpected keys in checkpoint: {unexpected[:5]}")

        rng = payload.get("rng_states", {})
        if rng.get("torch_rng") is not None:
            torch.set_rng_state(rng["torch_rng"].cpu())
        if rng.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda_rng"].cpu())

        self.resumed_wandb_run_id = payload.get("wandb_run_id", None)
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            print(f"  [ckpt] Resumed {stage} from step {resume_step}  "
                  f"(wandb_run_id={self.resumed_wandb_run_id})")

        return resume_step, loss_history

    # ================================================================
    # Final model save  (end of stage)
    # ================================================================

    def save_final(self, stage: str):
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return

        out_dir = os.path.join(self.runs_dir, "final_model")
        os.makedirs(out_dir, exist_ok=True)

        state_dict  = self.accelerator.get_state_dict(self.policy)
        unwrapped   = self.accelerator.unwrap_model(self.policy)
        unwrapped.save_pretrained(out_dir, state_dict=state_dict)

        self.tokenizer.save_pretrained(out_dir)

        print(f"  [final] Model saved --> {out_dir}")

        # delete checkpoint dir to save space
        if os.path.exists(self.ckpt_dir):
            try:
                shutil.rmtree(self.ckpt_dir)
                print(f"  [cleanup] Successfully deleted checkpoint directory --> {self.ckpt_dir}")
            except Exception as e:
                print(f"  [cleanup] WARNING: Failed to delete checkpoint directory {self.ckpt_dir}. Error: {e}")

        self.accelerator.wait_for_everyone()

    def save(self, _output_dir: str):
        self.save_final(self.config.loss.name)

    # ================================================================
    # Optional teardown
    # ================================================================

    def cleanup(self):
        pass


# ============================================================
# MeZO  --  isotropic Gaussian perturbation
# ============================================================

class MeZOTrainer(ZOTrainerBase):
    """
    Memory-Efficient Zeroth-Order (MeZO) optimizer.
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
            p.data.add_(z, alpha= - self.lr * projected_grad)


# ============================================================
# AGZOEngine  --  forward-hook machinery for AGZO
# ============================================================

class AGZOEngine:
    def __init__(self, model: nn.Module, power_iter_steps: int, rank: int):
        self.model            = model
        self.power_iter_steps = power_iter_steps
        self.rank             = rank

        self.basis: Dict[str, torch.Tensor] = {}   
        self._hooks:     List  = []
        self._param_map: Dict[str, nn.Parameter] = {}
        self._mode: Optional[str] = None
        self._B:    int            = 0     

        self._register_hooks()

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
            act = act.detach()   

            if self._mode in ("sft", "plain"):
                basis = self._make_basis(act.reshape(-1, act.shape[-1]).float(), pname)

            elif self._mode == "dpo":
                B = self._B
                if act.shape[0] != 2 * B:
                    basis = self._make_basis(act.reshape(-1, act.shape[-1]).float(), pname)
                else:
                    h_c = act[:B].reshape(-1, act.shape[-1]).float()  
                    h_r = act[B:].reshape(-1, act.shape[-1]).float()  
                    h_diff_mean   = h_c.mean(0, keepdim=True) - h_r.mean(0, keepdim=True)
                    h_diff_tokens = h_c - h_r
                    h_diff = torch.cat([h_diff_mean, h_diff_tokens], 0)
                    basis = self._make_basis(h_diff, pname)
            else:
                return

            if basis is not None:
                self.basis[pname] = basis
        return _h

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
        d = A.shape[1]
        r = max(1, min(r, d))

        gen = torch.Generator(device=A.device)
        gen.manual_seed(1337)

        q = torch.randn(d, r, device=A.device, dtype=A.dtype)
        q, _ = torch.linalg.qr(q, mode="reduced")
        for _ in range(max(1, steps)):
            y = A.matmul(q)
            z = A.T.matmul(y)
            if not torch.isfinite(z).all():
                break
            q, _ = torch.linalg.qr(z, mode="reduced")
        return q.T   

    def collect_sft(self, fwd_fn):
        self.basis.clear()
        self._mode = "sft"
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
                output = fwd_fn()
        finally:
            self._mode = None
            self._B    = 0
        return output   

    def collect_plain(self, fwd_fn):
        self.basis.clear()
        self._mode = "plain"
        try:
            with torch.no_grad():
                output = fwd_fn()
        finally:
            self._mode = None
        return output   

    def sample_z(self, name: str, param: nn.Parameter) -> torch.Tensor:
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

        active_params = [p for p in self.policy.parameters() if p.requires_grad]
        saved_weights = [p.data.clone() for p in active_params]

        with torch.no_grad():
            self._perturb(+1)
            loss_p = compute_sft_loss(
                self.policy(
                    gpu_batch['chosen_input_ids'],
                    attention_mask=gpu_batch['chosen_attention_mask'],
                ).logits,
                gpu_batch['chosen_labels'],
                compute_fp32=self.compute_logps_fp32
            )

            for p, saved_w in zip(active_params, saved_weights):
                p.data.copy_(saved_w)

            self._perturb(-1)
            loss_m = compute_sft_loss(
                self.policy(
                    gpu_batch['chosen_input_ids'],
                    attention_mask=gpu_batch['chosen_attention_mask'],
                ).logits,
                gpu_batch['chosen_labels'],
                compute_fp32=self.compute_logps_fp32
            )

            for p, saved_w in zip(active_params, saved_weights):
                p.data.copy_(saved_w)

            self._apply_update(((loss_p - loss_m) / (2 * self.eps)).item())
        return ((loss_p + loss_m) / 2).item()

    def _collect_dpo_basis(self, gpu_batch: dict):
        B    = gpu_batch['chosen_input_ids'].shape[0]
        ids, mask = self._concat_cr(gpu_batch)
        output = self._engine.collect_dpo(
            lambda: self.policy(ids, attention_mask=mask),
            B=B,
        )
        logits = output.logits
        self._last_margin = (
            get_batch_logps(logits[:B], gpu_batch['chosen_labels'], self.compute_logps_fp32)
            - get_batch_logps(logits[B:], gpu_batch['rejected_labels'], self.compute_logps_fp32)
        ).mean().item()

    def _perturb(self, scaling: float):
        torch.manual_seed(self._zo_seed)
        for name, param in self._params:
            z = self._engine.sample_z(name, param)
            param.data.add_(z, alpha=scaling * self.eps)

    def _apply_update(self, projected_grad: float):
        torch.manual_seed(self._zo_seed)
        for name, param in self._params:
            z = self._engine.sample_z(name, param)
            param.data.add_(z, alpha= - self.lr * projected_grad)

    def cleanup(self):
        self._engine.remove_hooks()


# ============================================================
# AGZOPlain  --  plain-activation subspace perturbation
# ============================================================

class AGZOPlainTrainer(AGZOTrainer):
    def _collect_dpo_basis(self, gpu_batch: dict):
        B    = gpu_batch['chosen_input_ids'].shape[0]
        ids, mask = self._concat_cr(gpu_batch)
        output = self._engine.collect_plain(
            lambda: self.policy(ids, attention_mask=mask)
        )
        logits = output.logits
        self._last_margin = (
            get_batch_logps(logits[:B], gpu_batch['chosen_labels'], self.compute_logps_fp32)
            - get_batch_logps(logits[B:], gpu_batch['rejected_labels'], self.compute_logps_fp32)
        ).mean().item()


# ============================================================
# Registry
# ============================================================

TRAINER_MAP: Dict[str, type] = {
    "mezo":       MeZOTrainer,
    "agzo":       AGZOTrainer,        
    "agzo_plain": AGZOPlainTrainer,   
}


def build_trainer(policy: nn.Module, config: DictConfig) -> ZOTrainerBase:
    name = config.trainer.name
    cls  = TRAINER_MAP.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown trainer '{name}'. "
            f"Registered trainers: {sorted(TRAINER_MAP.keys())}"
        )
    return cls(policy, config)