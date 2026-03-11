import os
import math
import random
import socket

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import hydra
import tqdm
import wandb
from omegaconf import OmegaConf, DictConfig
from typing import Dict, List, Optional, Tuple

from preference_datasets_hh import get_chat_template_iterator

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)


# ===========================================================================
# Helper Function
# ===========================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def disable_dropout(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0


def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute sequence log-probability per sample (summing only over non -100 positions)."""
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)
    labels[labels == -100] = 0
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)


def compute_dpo_loss(
    pi_chosen_logits:   torch.Tensor,
    pi_rejected_logits: torch.Tensor,
    ref_chosen_logps:   torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    chosen_labels:      torch.Tensor,
    rejected_labels:    torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    pi_chosen_logps   = get_batch_logps(pi_chosen_logits,   chosen_labels)
    pi_rejected_logps = get_batch_logps(pi_rejected_logits, rejected_labels)
    logits = (pi_chosen_logps - pi_rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
    return -F.logsigmoid(beta * logits).mean()


def compute_sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels  = labels[..., 1:].contiguous()
    return nn.CrossEntropyLoss(ignore_index=-100)(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )


# ===========================================================================
# AGZO
# ===========================================================================
class AGZOEngine:
    """
    AGZO Optimiser.

    H_diff subspace in DPO mode:
      Prerequisite: preference_datasets_hh.py guarantees chosen and rejected sequences 
                    are padded to the exact same length. This allows concatenating them 
                    into a large batch of shape (2B, T, d) for a single forward pass.
      The hook receives the activation (2B, T, d) and splits it by the batch dimension:
        H_c = activation[:B]   (B, T, d) → reshape → (B*T, d)
        H_r = activation[B:]   (B, T, d) → reshape → (B*T, d)
      Difference matrix construction:
        H_diff_mean   = mean(H_c, 0) - mean(H_r, 0)           shape (1, d)
        H_diff_tokens = H_c - H_r                             shape (B*T, d)
        H_diff        = cat([H_diff_mean, H_diff_tokens], 0)  shape (B*T+1, d)
      A block power iteration is applied to H_diff to obtain the basis (rank, d).
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float,
        lr: float,
        power_iter_steps: int = 5,
        rank: int = 1,
    ):
        self.model = model
        self.eps = eps
        self.lr  = lr
        self.power_iter_steps = power_iter_steps
        self.rank = rank

        # 激活主方向缓存：{param_name: basis (rank, d_in)}
        self.agzo_u: Dict[str, torch.Tensor] = {}

        self._hooks:     List = []
        self._param_map: Dict[str, nn.Parameter] = {}

        # 收集模式：None | "sft" | "dpo"
        # dpo 模式下 hook 假设 batch 前半是 chosen、后半是 rejected
        self._collect_mode: Optional[str] = None
        self._collect_batch_size: int = 0   # 单侧 batch size（chosen 侧的 B）

        self.zo_random_seed: int = 0

        self._register_hooks()

        self.params: List[Tuple[str, nn.Parameter]] = [
            (n, p) for n, p in model.named_parameters() if p.requires_grad
        ]

    # ------------------------------------------------------------------
    # Hook Registration
    # ------------------------------------------------------------------

    def _register_hooks(self):
        params_dict = dict(self.model.named_parameters())
        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            param_name = f"{module_name}.weight" if module_name else "weight"
            param = params_dict.get(param_name)
            if param is None or not param.requires_grad:
                continue
            hook = module.register_forward_hook(self._make_hook(param_name))
            self._hooks.append(hook)
            self._param_map[param_name] = param

    def _make_hook(self, param_name: str):
        def _hook(module, inputs, output):
            if self._collect_mode is None:
                return
            activation = next(
                (inp for inp in inputs if isinstance(inp, torch.Tensor)), None
            )
            if activation is None:
                return

            act = activation.detach()  # (B_total, T, d) 或 (B_total*T, d)

            if self._collect_mode == "sft":
                # SFT：直接全部展平
                act_2d = act.reshape(-1, act.shape[-1]).float()
                basis = self._compute_basis_from_matrix(act_2d, param_name)
                if basis is not None:
                    self.agzo_u[param_name] = basis

            elif self._collect_mode == "dpo":
                # DPO：按 batch 对半分
                B = self._collect_batch_size
                total = act.shape[0]
                if total != 2 * B:
                    # 维度不符预期时 fallback：把全部激活当 chosen 用
                    act_2d = act.reshape(-1, act.shape[-1]).float()
                    basis = self._compute_basis_from_matrix(act_2d, param_name)
                    if basis is not None:
                        self.agzo_u[param_name] = basis
                    return

                # 切分 chosen / rejected，各自展平为 (B*T, d)
                h_c = act[:B].reshape(-1, act.shape[-1]).float()  # (B*T, d)
                h_r = act[B:].reshape(-1, act.shape[-1]).float()  # (B*T, d)

                # H_diff 矩阵
                h_diff_mean   = h_c.mean(0, keepdim=True) - h_r.mean(0, keepdim=True)  # (1, d)
                h_diff_tokens = h_c - h_r                                                # (B*T, d)
                h_diff_full   = torch.cat([h_diff_mean, h_diff_tokens], dim=0)          # (B*T+1, d)

                basis = self._compute_basis_from_matrix(h_diff_full, param_name)
                if basis is not None:
                    self.agzo_u[param_name] = basis

        return _hook

    # ------------------------------------------------------------------
    # Basis Collection
    # ------------------------------------------------------------------

    def collect_basis_sft(self, chosen_fwd_fn):
        """SFT mode: Single forward pass, using only chosen activations."""
        self.agzo_u.clear()
        self._collect_mode = "sft"
        try:
            with torch.no_grad():
                chosen_fwd_fn()
        finally:
            self._collect_mode = None

    def collect_basis_dpo(self, concat_fwd_fn, batch_size: int):
        """
        DPO mode: Single forward pass (chosen + rejected concatenated).
        The hook internally splits the batch row-wise.

        Args:
            concat_fwd_fn: Forward function where inputs are already concatenated.
            batch_size: Single-side batch size B (number of chosen or rejected samples).
        """
        self.agzo_u.clear()
        self._collect_mode = "dpo"
        self._collect_batch_size = batch_size
        try:
            with torch.no_grad():
                concat_fwd_fn()
        finally:
            self._collect_mode = None
            self._collect_batch_size = 0

    # ------------------------------------------------------------------
    # Power Iteration
    # ------------------------------------------------------------------

    def _compute_basis_from_matrix(
        self, act_2d: torch.Tensor, param_name: str
    ) -> Optional[torch.Tensor]:
        if act_2d is None or act_2d.numel() == 0 or act_2d.dim() < 2:
            return None
        param = self._param_map.get(param_name)
        if param is None:
            return None

        hidden_dim = act_2d.shape[1]
        if hidden_dim == 0:
            return None

        max_rank = hidden_dim
        if param.dim() >= 2:
            max_rank = min(max_rank, param.shape[0], param.shape[1])
        rank = max(1, min(self.rank, max_rank))

        basis = self._power_iteration(act_2d, self.power_iter_steps, rank)
        if basis is None:
            return None
        if basis.dim() == 1:
            basis = basis.unsqueeze(0)
        basis = basis / (basis.norm(p=2, dim=1, keepdim=True) + 1e-12)
        return basis.to(device=param.device, dtype=param.dtype)

    @staticmethod
    def _power_iteration(
        act_2d: torch.Tensor, num_steps: int, rank: int
    ) -> Optional[torch.Tensor]:
        hidden_dim = act_2d.shape[1]
        num_steps  = max(1, num_steps)
        rank       = max(1, min(rank, hidden_dim))

        q = torch.randn(hidden_dim, rank, device=act_2d.device, dtype=act_2d.dtype)
        q, _ = torch.linalg.qr(q, mode="reduced")
        for _ in range(num_steps):
            y = act_2d.matmul(q)
            z = act_2d.T.matmul(y)
            if not torch.isfinite(z).all():
                break
            q, _ = torch.linalg.qr(z, mode="reduced")
        return q.T  # (rank, hidden_dim)

    # ------------------------------------------------------------------
    # Sample Perturbation (z)
    # ------------------------------------------------------------------

    def _sample_z(self, name: str, param: nn.Parameter) -> torch.Tensor:
        basis = self.agzo_u.get(name)
        if basis is None or param.dim() < 2:
            return torch.randn_like(param.data)
        basis = basis.to(device=param.device, dtype=param.dtype)
        if basis.shape[1] != param.shape[1]:
            return torch.randn_like(param.data)
        rank = basis.shape[0]
        r = torch.randn(param.shape[0], rank, device=param.device, dtype=param.dtype)
        return torch.matmul(r, basis) / math.sqrt(rank)

    # ------------------------------------------------------------------
    # Apply Perturbation and Update
    # ------------------------------------------------------------------

    def perturb(self, scaling: float):
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.params:
            z = self._sample_z(name, param)
            param.data.add_(z, alpha=scaling * self.eps)

    def apply_update(self, projected_grad: float):
        """θ 当前在 θ_0 - eps*z，净效果：θ_new = θ_0 - lr * g_hat * z"""
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.params:
            z = self._sample_z(name, param)
            param.data.add_(z, alpha=self.eps - self.lr * projected_grad)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ===========================================================================
# Trainer
# ===========================================================================

class AGZOPipelineTrainer:
    """
    SFT(AGZO) → Ref Logp Precomputation → DPO(AGZO, H_diff subspace).

    Forward pass:
      - 1 pass: basis collection (chosen+rejected merged, single forward pass)
      - 2 perturbation passes (±eps, chosen+rejected merged, single forward pass each)
    """

    def __init__(self, policy: nn.Module, config: DictConfig, rank: int):
        self.policy = policy
        self.config = config
        self.rank   = rank

        self.lr               = config.agzo.lr
        self.eps              = config.agzo.eps
        self.beta             = config.agzo.beta
        self.power_iter_steps = config.agzo.get("power_iter_steps", 5)
        self.agzo_rank        = config.agzo.get("rank", 1)
        self.total_batches    = config.total_batches

        cache_dir = os.path.expandvars("/media/tsar_bomba/$USER/huggingface_cache")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.model.name_or_path, cache_dir=cache_dir, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.cache_dir = cache_dir

        self.engine = AGZOEngine(
            model=policy,
            eps=self.eps,
            lr=self.lr,
            power_iter_steps=self.power_iter_steps,
            rank=self.agzo_rank,
        )

    # ------------------------------------------------------------------
    # Helper: Concatenate chosen / rejected along batch dimension
    # ------------------------------------------------------------------

    @staticmethod
    def _concat_chosen_rejected(gpu_batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate aligned chosen and rejected sequences along the batch dim.
        """
        input_ids = torch.cat(
            [gpu_batch['chosen_input_ids'], gpu_batch['rejected_input_ids']], dim=0
        )
        attention_mask = torch.cat(
            [gpu_batch['chosen_attention_mask'], gpu_batch['rejected_attention_mask']], dim=0
        )
        return input_ids, attention_mask

    # ------------------------------------------------------------------
    # SFT 
    # ------------------------------------------------------------------

    def _sft_step(self, gpu_batch: dict) -> float:
        self.engine.zo_random_seed = int(torch.randint(0, 2**32, (1,)).item())

        # Basis collection：only chosen
        self.engine.collect_basis_sft(
            chosen_fwd_fn=lambda: self.policy(
                gpu_batch['chosen_input_ids'],
                attention_mask=gpu_batch['chosen_attention_mask'],
            )
        )

        # +eps
        self.engine.perturb(scaling=+1)
        loss_plus = compute_sft_loss(
            self.policy(
                gpu_batch['chosen_input_ids'],
                attention_mask=gpu_batch['chosen_attention_mask'],
            ).logits,
            gpu_batch['chosen_labels'],
        )

        # -2eps
        self.engine.perturb(scaling=-2)
        loss_minus = compute_sft_loss(
            self.policy(
                gpu_batch['chosen_input_ids'],
                attention_mask=gpu_batch['chosen_attention_mask'],
            ).logits,
            gpu_batch['chosen_labels'],
        )

        proj_grad = ((loss_plus - loss_minus) / (2 * self.eps)).item()
        self.engine.apply_update(proj_grad)
        return ((loss_plus + loss_minus) / 2).item()

    # ------------------------------------------------------------------
    # DPO
    # ------------------------------------------------------------------

    def _dpo_step(self, gpu_batch: dict) -> float:
        self.engine.zo_random_seed = int(torch.randint(0, 2**32, (1,)).item())

        B = gpu_batch['chosen_input_ids'].shape[0]   # 单侧 batch size
        input_ids, attention_mask = self._concat_chosen_rejected(gpu_batch)

        # ── Basis collection：chosen+rejected 单次前向 ──────────────
        self.engine.collect_basis_dpo(
            concat_fwd_fn=lambda: self.policy(input_ids, attention_mask=attention_mask),
            batch_size=B,
        )

        # ── 辅助函数：单次合并前向 → split logits ────────────────────
        def forward_split():
            """前向一次，按 batch 切分返回 (chosen_logits, rejected_logits)。"""
            logits = self.policy(input_ids, attention_mask=attention_mask).logits
            return logits[:B], logits[B:]

        ref_chosen_logps   = gpu_batch['ref_chosen_logps']
        ref_rejected_logps = gpu_batch['ref_rejected_logps']

        # ── +eps perturbation ────────────────────────────────────────────────
        self.engine.perturb(scaling=+1)
        chosen_logits_p, rejected_logits_p = forward_split()
        loss_plus = compute_dpo_loss(
            chosen_logits_p, rejected_logits_p,
            ref_chosen_logps, ref_rejected_logps,
            gpu_batch['chosen_labels'], gpu_batch['rejected_labels'],
            self.beta,
        )

        # ── -2eps perturbation ───────────────────────────────────────────────
        self.engine.perturb(scaling=-2)
        chosen_logits_m, rejected_logits_m = forward_split()
        loss_minus = compute_dpo_loss(
            chosen_logits_m, rejected_logits_m,
            ref_chosen_logps, ref_rejected_logps,
            gpu_batch['chosen_labels'], gpu_batch['rejected_labels'],
            self.beta,
        )

        proj_grad = ((loss_plus - loss_minus) / (2 * self.eps)).item()
        self.engine.apply_update(proj_grad)
        return ((loss_plus + loss_minus) / 2).item()

    # ------------------------------------------------------------------
    # Main training loot
    # ------------------------------------------------------------------

    def train(self):
        print(f"Caching {self.total_batches} batches to CPU RAM...")
        cached_batches = []
        train_iterator = get_chat_template_iterator(
            tokenizer=self.tokenizer,
            split='train',
            batch_size=self.config.batch_size,
            n_examples=self.total_batches * self.config.batch_size,
            shuffle=True,
            cache_dir=self.cache_dir,
        )
        for _ in tqdm.tqdm(range(self.total_batches), desc="Loading batches"):
            cached_batches.append(next(train_iterator))

        # ── Phase 1: SFT ─────────────────────────────────────────
        print("\n--- Phase 1: SFT (AGZO, chosen subspace) ---")
        self.policy.train()
        for step, batch in enumerate(cached_batches, 1):
            gpu_batch = {
                k: batch[k].cuda()
                for k in ['chosen_input_ids', 'chosen_attention_mask', 'chosen_labels']
            }
            avg_loss = self._sft_step(gpu_batch)
            if self.rank == 0:
                print(f"SFT Step {step}/{self.total_batches} | Loss: {avg_loss:.4f}")
                wandb.log({"train/sft_loss": avg_loss, "step_sft": step})

        # ── Phase 2: Precompute Reference Logps ──────────────────────
        # 注：Phase 2 同样可以合并 chosen+rejected 单次前向，节省一倍时间。
        print("\n--- Phase 2: Precomputing Reference Logps (SFT model as ref) ---")
        self.policy.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(cached_batches, desc="Computing Ref Logps"):
                gpu_keys = [
                    'chosen_input_ids',   'chosen_attention_mask',   'chosen_labels',
                    'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels',
                ]
                gpu_batch = {k: batch[k].cuda() for k in gpu_keys}
                B = gpu_batch['chosen_input_ids'].shape[0]

                # 合并前向，一次拿到 chosen + rejected 的 logits
                input_ids, attention_mask = self._concat_chosen_rejected(gpu_batch)
                logits_all = self.policy(input_ids, attention_mask=attention_mask).logits

                batch['ref_chosen_logps']  = get_batch_logps(
                    logits_all[:B], gpu_batch['chosen_labels']
                ).cpu()
                batch['ref_rejected_logps'] = get_batch_logps(
                    logits_all[B:], gpu_batch['rejected_labels']
                ).cpu()

        # ── Phase 3: DPO ─────────────────────────────────────────
        print("\n--- Phase 3: DPO (AGZO, H_diff subspace, merged forward) ---")
        self.policy.train()
        for step, batch in enumerate(cached_batches, 1):
            gpu_batch = {k: v.cuda() for k, v in batch.items()}
            avg_loss = self._dpo_step(gpu_batch)
            if self.rank == 0:
                print(f"DPO Step {step}/{self.total_batches} | Loss: {avg_loss:.4f}")
                wandb.log({"train/dpo_loss": avg_loss, "step_dpo": step})

    def save(self):
        if self.rank == 0:
            os.makedirs(self.config.local_run_dir, exist_ok=True)
            ckpt_path = os.path.join(self.config.local_run_dir, "final_model")
            self.policy.save_pretrained(ckpt_path)
            print(f"Model saved to {ckpt_path}")

    def cleanup(self):
        self.engine.remove_hooks()


# ===========================================================================
# Entrypoint
# ===========================================================================

def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module):
    if rank == 0 and config.wandb.enabled:
        os.makedirs("/tmp/wandb_tmp", exist_ok=True)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            name=config.exp_name,
            dir="/tmp/wandb_tmp",
            settings=wandb.Settings(_disable_stats=True, _disable_meta=True),
        )

    print(f"Creating AGZOPipelineTrainer on process {rank}")
    trainer = AGZOPipelineTrainer(policy, config, rank=rank)
    try:
        trainer.train()
        trainer.save()
    finally:
        trainer.cleanup()
        if rank == 0 and config.wandb.enabled:
            wandb.finish()


@hydra.main(version_base=None, config_path="config", config_name="config_agzo")
def main(config: DictConfig):
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"Global seed: {seed}")
    print(OmegaConf.to_yaml(config))
    print("=" * 80)
    print(f"Host: {socket.gethostname()} | Output dir: {config.local_run_dir}")
    print("=" * 80)

    cache_dir = os.path.expandvars("/media/tsar_bomba/$USER/huggingface_cache")
    print("Building policy model...")
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        cache_dir=cache_dir,
        torch_dtype=policy_dtype,
        device_map='cuda:0',
    )
    disable_dropout(policy)

    print("Starting AGZO+DPO training")
    worker_main(0, 1, config, policy)


if __name__ == "__main__":
    main()