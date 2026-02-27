import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from preference_datasets import get_batch_iterator
import matplotlib.pyplot as plt

torch.set_grad_enabled(False) # inference only, no autograd needed

def print_vram_usage(step, stage=""):
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"[Step {step} - {stage}] VRAM distributed: {allocated:.2f} GB | pre: {reserved:.2f} GB")

def get_batch_logps(logits, labels):
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)
    labels[labels == -100] = 0 
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)

def compute_dpo_loss(pi_chosen_logits, pi_rejected_logits, ref_chosen_logps, ref_rejected_logps, batch, beta=0.1):
    pi_chosen_logps = get_batch_logps(pi_chosen_logits, batch['chosen_labels'])
    pi_rejected_logps = get_batch_logps(pi_rejected_logits, batch['rejected_labels'])
    logits = (pi_chosen_logps - pi_rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
    return -F.logsigmoid(beta * logits).mean()


# MeZO
class MeZOOptimizer:
    def __init__(self, model, lr=5e-5, eps=1e-3):
        self.model = model
        self.lr, self.eps = lr, eps
        self.params = [p for p in model.parameters()]

    def perturb(self, seed, scaling):
        torch.manual_seed(seed)
        for p in self.params:
            p.data.add_(torch.randn_like(p), alpha=scaling * self.eps)

    def step(self, batch, ref_chosen_logps, ref_rejected_logps, beta=0.1):
        z_seed = torch.randint(0, 2**32, (1,)).item()
        
        self.perturb(z_seed, scaling=1)
        loss_plus = compute_dpo_loss(
            self.model(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits,
            self.model(batch['rejected_input_ids'], attention_mask=batch['rejected_attention_mask']).logits,
            ref_chosen_logps, ref_rejected_logps, batch, beta
        )

        self.perturb(z_seed, scaling=-2)
        loss_minus = compute_dpo_loss(
            self.model(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits,
            self.model(batch['rejected_input_ids'], attention_mask=batch['rejected_attention_mask']).logits,
            ref_chosen_logps, ref_rejected_logps, batch, beta
        )

        projected_grad = (loss_plus - loss_minus) / (2 * self.eps)
        
        torch.manual_seed(z_seed)
        for p in self.params:
            z = torch.randn_like(p)
            update_scale = self.eps - self.lr * projected_grad
            p.data.add_(z, alpha=update_scale)

        return (loss_plus + loss_minus) / 2

def plot_loss(loss_history, save_path="mezo_dpo_loss.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='DPO Loss (MeZO)', color='#2c3e50', linewidth=2)
    plt.title('Training Loss over Steps', fontsize=14)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # save plot
    plt.savefig(save_path)
    plt.close() # release memory
    print(f"Loss curve saved to: {save_path}")


def main():
    CACHE_DIR = os.path.expandvars("/media/tsar_bomba/$USER/huggingface_cache")
    MODEL_ID = "Qwen/Qwen2.5-1.5B"

    print("[1/3] Loading dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    train_iterator = get_batch_iterator(
                    names=['hh'], 
                    tokenizer=tokenizer, 
                    split='train', 
                    batch_size=16, 
                    n_examples=16,
                    shuffle=False,
                    # max_length=512,
                    # max_prompt_length=256, 
                    n_epochs=1, 
                    cache_dir=CACHE_DIR
    )
    debug_batch = next(train_iterator)
    for k, v in debug_batch.items():
        if isinstance(v, torch.Tensor): debug_batch[k] = v.cuda()

    print("[2/3] Loading model (bfloat16)...")
    policy_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, 
                torch_dtype=torch.bfloat16, 
                device_map="cuda:0", 
                cache_dir=CACHE_DIR,
                trust_remote_code=True
            ).eval()
    ref_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, 
                torch_dtype=torch.bfloat16, 
                device_map="cuda:0", 
                cache_dir=CACHE_DIR, 
                trust_remote_code=True
            ).eval()
    print_vram_usage(0, "Model loaded")

    print("[3/3] Training...")
    mezo = MeZOOptimizer(policy_model, lr=5e-5, eps=1e-3)
    
    # precompute Reference
    ref_chosen_logps = get_batch_logps(ref_model(debug_batch['chosen_input_ids'], attention_mask=debug_batch['chosen_attention_mask']).logits, debug_batch['chosen_labels'])
    ref_rejected_logps = get_batch_logps(ref_model(debug_batch['rejected_input_ids'], attention_mask=debug_batch['rejected_attention_mask']).logits, debug_batch['rejected_labels'])
    del ref_model # release VRAM

    losses = []
    for step in range(1, 101):
        loss = mezo.step(debug_batch, ref_chosen_logps, ref_rejected_logps)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: step {step} NaN Loss detected, stopping test.")
            break
        losses.append(loss.item())
        print(f"Step {step}/100 | Loss: {loss.item():.4f}")

    if losses:
        plot_loss(losses)

    print("✅ Test completed.")

if __name__ == "__main__":
    main()