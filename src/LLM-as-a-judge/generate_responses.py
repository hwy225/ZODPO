"""
Generate responses for a set of prompts using multiple models and save the results in JSONL format.
Support running multiple models (SFT / FODPO / ZODPO) concurrently, with results saved in JSONL format.

Usage:
  python generate_responses.py \
    --model_paths sft_model fodpo_model zodpo_model \
    --model_names sft fodpo zodpo \
    --output_dir outputs/ \
    --num_samples 500 \
    --seed 42
"""

import argparse
import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def parse_to_messages(text):
    parts = re.split(r'\n\n(Human|Assistant):', text)
    msgs = []
    for j in range(1, len(parts), 2):
        role = "user" if parts[j].strip() == "Human" else "assistant"
        content = parts[j+1].strip()
        msgs.append({"role": role, "content": content})
    return msgs

def load_hh_test_prompts(tokenizer, num_samples: int = 500, seed: int = 42, cache_dir: str = None) -> list[dict]:
    print(f"[INFO] Loading dataset and shuffling with seed {seed}...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="test", cache_dir=cache_dir)
    dataset = dataset.shuffle(seed=seed)
    
    prompts = []
    
    for item in dataset:
        if len(prompts) >= num_samples:
            break
            
        chosen_full_msgs = parse_to_messages(item["chosen"])
        
        last_assistant_idx = -1
        for idx in range(len(chosen_full_msgs) - 1, -1, -1):
            if chosen_full_msgs[idx]["role"] == "assistant":
                last_assistant_idx = idx
                break
        
        if last_assistant_idx == -1: 
            continue
            
        prompt_msgs = chosen_full_msgs[:last_assistant_idx]
        reference_msg = chosen_full_msgs[last_assistant_idx]["content"]
        
        # apply Chat Template
        templated_prompt = tokenizer.apply_chat_template(
            prompt_msgs, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # prepare plain prompt for judge
        plain_context = ""
        for m in prompt_msgs:
            role_name = "Human" if m["role"] == "user" else "Assistant"
            plain_context += f"{role_name}: {m['content']}\n\n"
        
        prompts.append({
            "id": len(prompts),  # 使用当前列表长度作为纯数字 ID (0 到 num_samples-1)
            "prompt_templated": templated_prompt, 
            "prompt_plain": plain_context.strip(), 
            "reference": reference_msg,
        })
        
    return prompts


def generate_with_vllm(
    model_path: str,
    model_name: str,
    tokenizer: AutoTokenizer,
    prompts: list[dict],
    output_path: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    tensor_parallel_size: int = 1,
    seed: int = 42,
):
    """Generate responses for a list of prompts using vLLM and save to JSONL."""
    print(f"\n[INFO] Loading model: {model_name} from {model_path}")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
    )
    
    stop_token_ids = [tokenizer.eos_token_id]

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and isinstance(eot_id, int):
        stop_token_ids.append(eot_id)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        seed=seed,
    )
    
    prompt_texts = [p["prompt_templated"] for p in prompts]
    
    print(f"[INFO] Generating {len(prompt_texts)} responses...")
    outputs = llm.generate(prompt_texts, sampling_params)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for prompt_meta, output in zip(prompts, outputs):
            generated_text = output.outputs[0].text.strip()
            record = {
                "id": prompt_meta["id"],
                "prompt_templated": prompt_meta["prompt_templated"],
                "prompt_plain": prompt_meta["prompt_plain"],
                "reference": prompt_meta["reference"],
                "model": model_name,
                "response": generated_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"[INFO] Saved {len(prompts)} responses to {output_path}")
    
    # release GPU memory immediately after generation
    del llm
    import torch
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs="+", required=True,
                        help="Model checkpoint paths, e.g., /path/to/sft /path/to/fodpo /path/to/zodpo")
    parser.add_argument("--model_names", nargs="+", required=True,
                        help="Model names, e.g., sft fodpo zodpo")
    parser.add_argument("--output_dir", default="outputs/responses",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of samples to take from the test set (recommended: 200-1000)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="TP size for vLLM, adjust based on your GPU memory and model size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset shuffling and vLLM generation")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for Hugging Face datasets")
    args = parser.parse_args()
    
    assert len(args.model_paths) == len(args.model_names), \
        "Error: --model_paths and --model_names must have the same length!"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_paths[0])
    
    prompts = load_hh_test_prompts(tokenizer, args.num_samples, args.seed, args.cache_dir)
    
    for model_path, model_name in zip(args.model_paths, args.model_names):
        output_path = os.path.join(args.output_dir, f"{model_name}_responses.jsonl")
        generate_with_vllm(
            model_path=model_path,
            model_name=model_name,
            tokenizer=tokenizer,
            prompts=prompts,
            output_path=output_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            seed=args.seed,
        )
    
    print("\n[DONE] All responses generated.")


if __name__ == "__main__":
    main()