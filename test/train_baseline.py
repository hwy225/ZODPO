import os
import re
import torch
import transformers
import hydra
from omegaconf import OmegaConf, DictConfig
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

def set_seed(seed: int):
    transformers.set_seed(seed)

# =====================================================================
# 核心：将原文件中的 Regex 解析逻辑无缝平移，保证对比实验数据绝对一致
# =====================================================================
def extract_dpo_messages(example):
    """
    完全复刻 preference_datasets_hh.py 中的 parse_to_messages 逻辑
    将原始文本转化为 TRL 需要的 prompt, chosen, rejected 格式
    """
    def parse_to_messages(text):
        parts = re.split(r'\n\n(Human|Assistant):', text)
        msgs = []
        for j in range(1, len(parts), 2):
            role = "user" if parts[j].strip() == "Human" else "assistant"
            content = parts[j+1].strip()
            msgs.append({"role": role, "content": content})
        return msgs

    try:
        chosen_full_msgs = parse_to_messages(example['chosen'])
        rejected_full_msgs = parse_to_messages(example['rejected'])

        # 找到最后一个 assistant 回复
        last_assistant_idx = -1
        for idx in range(len(chosen_full_msgs) - 1, -1, -1):
            if chosen_full_msgs[idx]["role"] == "assistant":
                last_assistant_idx = idx
                break
        
        last_rej_idx = -1
        for idx in range(len(rejected_full_msgs) - 1, -1, -1):
            if rejected_full_msgs[idx]["role"] == "assistant":
                last_rej_idx = idx
                break

        # 如果解析失败，返回空，后续用 filter 过滤掉
        if last_assistant_idx == -1 or last_rej_idx == -1:
            return {"prompt": [], "chosen": [], "rejected": []}

        # 分离 prompt 和 responses
        prompt_msgs = chosen_full_msgs[:last_assistant_idx]
        
        # TRL DPOTrainer 支持传入 conversational list
        chosen_msg = [chosen_full_msgs[last_assistant_idx]]
        rejected_msg = [rejected_full_msgs[last_rej_idx]]

        return {
            "prompt": prompt_msgs,
            "chosen": chosen_msg,
            "rejected": rejected_msg
        }
    except Exception:
        return {"prompt": [], "chosen": [], "rejected": []}


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    set_seed(config.seed)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("=" * 72)
        print("Trainer   : TRL DPOTrainer Baseline")
        print(f"SFT Path  : {config.loss.sft_model_path}")
        print("=" * 72)

    cache_dir = os.path.expandvars(config.hf_cache_dir)
    dtype = getattr(torch, config.model.policy_dtype)

    # 1. 加载 Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.name_or_path, cache_dir=cache_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载模型
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=cache_dir, torch_dtype=dtype,
    )
    ref_model = transformers.AutoModelForCausalLM.from_pretrained(
        config.loss.sft_model_path, cache_dir=cache_dir, torch_dtype=dtype,
    )
    model.gradient_checkpointing_enable()

    # 3. 加载并处理数据集
    ds_cache = os.path.expandvars(config.get("hf_dataset_cache_dir", ""))
    
    # 加载原始数据
    raw_train = load_dataset('Anthropic/hh-rlhf', split='train', cache_dir=ds_cache)
    raw_eval = load_dataset('Anthropic/hh-rlhf', split='test', cache_dir=ds_cache)
    
    # 映射出 prompt / chosen / rejected
    train_dataset = raw_train.map(
        extract_dpo_messages, 
        num_proc=8, 
        remove_columns=raw_train.column_names
    ).filter(lambda x: len(x["prompt"]) > 0)
    
    eval_dataset = raw_eval.map(
        extract_dpo_messages, 
        num_proc=8, 
        remove_columns=raw_eval.column_names
    ).filter(lambda x: len(x["prompt"]) > 0)

    # 4. 配置 DPO
    training_args = DPOConfig(
        output_dir=os.path.join(os.environ.get("RUNS_DIR", "./runs"), config.exp_name),
        beta=config.loss.beta,
        learning_rate=config.trainer.lr,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        num_train_epochs=config.loss.get("num_epochs", 1),
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        logging_steps=10,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        bf16=(config.model.policy_dtype == "bfloat16"),
        report_to="wandb" if config.wandb.enabled else "none",
        run_name=f"baseline_trl_{config.exp_name}",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        # 显式告知 TRL 自动应用 Chat Template
        dataset_kwargs={"skip_prepare_dataset": False} 
    )

    # 5. 启动 Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

if __name__ == "__main__":
    main()