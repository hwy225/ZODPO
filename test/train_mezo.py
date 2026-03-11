import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import transformers
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
import socket
import tqdm
import random 
import numpy as np

from preference_datasets_hh import get_chat_template_iterator

# Global Configurations
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Helper Functions for Loss Computation
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def get_batch_logps(logits, labels):
    """Compute log probabilities for a batch given logits and labels."""
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)
    labels[labels == -100] = 0 
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)

def compute_dpo_loss(pi_chosen_logits, pi_rejected_logits, ref_chosen_logps, ref_rejected_logps, batch, beta=0.1):
    """Compute the DPO loss for a batch."""
    pi_chosen_logps = get_batch_logps(pi_chosen_logits, batch['chosen_labels'])
    pi_rejected_logps = get_batch_logps(pi_rejected_logits, batch['rejected_labels'])
    logits = (pi_chosen_logps - pi_rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
    return -F.logsigmoid(beta * logits).mean()

def compute_sft_loss(logits, labels):
    """Compute the SFT loss for a batch."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

# MeZO Optimizer
class MeZOPipelineTrainer:
    """Trainer class for MeZO pipeline, handling both SFT and DPO phases."""
    def __init__(self, policy, reference_model, config: DictConfig, rank: int):
        self.policy = policy
        self.reference_model = reference_model
        self.config = config
        self.rank = rank
        
        self.lr = config.mezo.lr
        self.eps = config.mezo.eps
        self.beta = config.mezo.beta
        self.total_batches = config.total_batches
        
        # Only include parameters that require gradients in the optimization process
        self.params = [p for p in self.policy.parameters() if p.requires_grad]
        
        # Load Tokenizer
        cache_dir = os.path.expandvars("/media/tsar_bomba/$USER/huggingface_cache")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.model.name_or_path, cache_dir=cache_dir, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.cache_dir = cache_dir

    def _perturb(self, seed, scaling):
        torch.manual_seed(seed)
        for p in self.params:
            p.data.add_(torch.randn_like(p), alpha=scaling * self.eps)

    def _apply_update(self, z_seed, projected_grad):
        torch.manual_seed(z_seed)
        for p in self.params:
            z = torch.randn_like(p)
            update_scale = self.eps - self.lr * projected_grad
            p.data.add_(z, alpha=update_scale)

    def train(self):
        # 
        print("Caching {self.total_batches} Batches to CPU...")
        cached_batches = []
        train_iterator = get_chat_template_iterator(
            tokenizer=self.tokenizer, split='train', batch_size=self.config.batch_size, 
            n_examples=self.total_batches * self.config.batch_size, shuffle=True, cache_dir=self.cache_dir
        )
        for _ in tqdm.tqdm(range(self.total_batches)):
            cached_batches.append(next(train_iterator))

        # Phase 1: SFT
        print("\n--- Phase 1: SFT ---")
        for step, batch in enumerate(cached_batches, 1):
            gpu_batch = {k: batch[k].cuda() for k in ['chosen_input_ids', 'chosen_attention_mask', 'chosen_labels']}
            z_seed = torch.randint(0, 2**32, (1,)).item()
            
            self._perturb(z_seed, scaling=1)
            loss_plus = compute_sft_loss(self.policy(gpu_batch['chosen_input_ids'], attention_mask=gpu_batch['chosen_attention_mask']).logits, gpu_batch['chosen_labels'])

            self._perturb(z_seed, scaling=-2)
            loss_minus = compute_sft_loss(self.policy(gpu_batch['chosen_input_ids'], attention_mask=gpu_batch['chosen_attention_mask']).logits, gpu_batch['chosen_labels'])

            proj_grad = (loss_plus - loss_minus) / (2 * self.eps)
            self._apply_update(z_seed, proj_grad)
            
            avg_loss = (loss_plus + loss_minus) / 2
            if self.rank == 0:
                print(f"SFT Step {step}/{self.total_batches} | Loss: {avg_loss.item():.4f}")
                wandb.log({"train/sft_loss": avg_loss.item(), "step_sft": step})
        
        print("\n--- Phase 2: Precomputing Reference Logps using SFT Model ---")
        self.policy.eval() # evaluation mode
        with torch.no_grad():
            for batch in tqdm.tqdm(cached_batches, desc="Computing Ref Logps"):
                gpu_keys = [
                    'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels',
                    'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels'
                ]
                gpu_batch = {k: batch[k].cuda() for k in gpu_keys}
                
                ref_chosen_logits = self.policy(gpu_batch['chosen_input_ids'], attention_mask=gpu_batch['chosen_attention_mask']).logits
                ref_rejected_logits = self.policy(gpu_batch['rejected_input_ids'], attention_mask=gpu_batch['rejected_attention_mask']).logits
                
                # Save the reference log probabilities back to CPU for later use in DPO phase
                batch['ref_chosen_logps'] = get_batch_logps(ref_chosen_logits, gpu_batch['chosen_labels']).cpu()
                batch['ref_rejected_logps'] = get_batch_logps(ref_rejected_logits, gpu_batch['rejected_labels']).cpu()

        # 4. Phase 3: DPO
        print("\n--- Phase 3: DPO ---")
        self.policy.train() # training mode
        for step, batch in enumerate(cached_batches, 1):
            gpu_batch = {k: v.cuda() for k, v in batch.items()}
            z_seed = torch.randint(0, 2**32, (1,)).item()
            
            self._perturb(z_seed, scaling=1)
            loss_plus = compute_dpo_loss(
                self.policy(gpu_batch['chosen_input_ids'], attention_mask=gpu_batch['chosen_attention_mask']).logits,
                self.policy(gpu_batch['rejected_input_ids'], attention_mask=gpu_batch['rejected_attention_mask']).logits,
                gpu_batch['ref_chosen_logps'], gpu_batch['ref_rejected_logps'], gpu_batch, self.beta
            )

            self._perturb(z_seed, scaling=-2)
            loss_minus = compute_dpo_loss(
                self.policy(gpu_batch['chosen_input_ids'], attention_mask=gpu_batch['chosen_attention_mask']).logits,
                self.policy(gpu_batch['rejected_input_ids'], attention_mask=gpu_batch['rejected_attention_mask']).logits,
                gpu_batch['ref_chosen_logps'], gpu_batch['ref_rejected_logps'], gpu_batch, self.beta
            )

            proj_grad = (loss_plus - loss_minus) / (2 * self.eps)
            self._apply_update(z_seed, proj_grad)
            
            avg_loss = (loss_plus + loss_minus) / 2
            if self.rank == 0:
                print(f"DPO Step {step}/{self.total_batches} | Loss: {avg_loss.item():.4f}")
                wandb.log({"train/dpo_loss": avg_loss.item(), "step_dpo": step})

    def save(self):
        if self.rank == 0:
            os.makedirs(self.config.local_run_dir, exist_ok=True)
            ckpt_path = os.path.join(self.config.local_run_dir, "final_model")
            self.policy.save_pretrained(ckpt_path)
            print(f"Model saved to {ckpt_path}")


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: nn.Module = None):
    """Main function for each worker process. Handles training and logging."""
    if rank == 0 and config.wandb.enabled:
        wandb.init(
            entity=config.wandb.entity, project=config.wandb.project,
            config=OmegaConf.to_container(config), name=config.exp_name
        )

    print(f'Creating trainer on process {rank}')
    trainer = MeZOPipelineTrainer(policy, reference_model, config, rank=rank)

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config_mezo")
def main(config: DictConfig):
    """Main entry point for the training script."""
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"Global seed set to: {seed}")

    print(OmegaConf.to_yaml(config))

    print('=' * 80)
    print(f'Writing outputs to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)
 
    cache_dir = os.path.expandvars("/media/tsar_bomba/$USER/huggingface_cache")
    
    print('Building Policy model...')
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=cache_dir, torch_dtype=policy_dtype, device_map='cuda:0'
    )
    disable_dropout(policy)

    reference_model = None
    
    # Start single-process training
    print('Starting single-process worker')
    worker_main(0, 1, config, policy, reference_model)

if __name__ == '__main__':
    main()