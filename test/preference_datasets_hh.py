import re
import tqdm
import random
import torch
import torch.nn.functional as F
import datasets
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Iterator, Dict

def get_chat_template_iterator(
    tokenizer,
    split: str = 'train',
    batch_size: int = 1,
    shuffle: bool = True,
    max_length: int = 512,
    max_prompt_length: int = 256,
    sft_mode: bool = False,
    n_epochs: Optional[int] = None,
    n_examples: Optional[int] = None,
    seed: int = 42,
    silent: bool = False,
    cache_dir: Optional[str] = None
) -> Iterator[Dict]:
    """
    Get an iterator over batches of HH-RLHF data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode. In sft mode, we just return chosen_input_ids and chosen_labels.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()
        
    dataset = load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)

    # 1. Extract and format conversation turns (cache for shuffling)
    flat_data = []
    for item in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        try:
            def parse_to_messages(text):
                parts = re.split(r'\n\n(Human|Assistant):', text)
                msgs = []
                for j in range(1, len(parts), 2):
                    role = "user" if parts[j].strip() == "Human" else "assistant"
                    content = parts[j+1].strip()
                    msgs.append({"role": role, "content": content})
                return msgs
            
            chosen_full_msgs = parse_to_messages(item['chosen'])
            rejected_full_msgs = parse_to_messages(item['rejected'])
            
            # Locate the final assistant turn
            last_assistant_idx = -1
            for idx in range(len(chosen_full_msgs) - 1, -1, -1):
                if chosen_full_msgs[idx]["role"] == "assistant":
                    last_assistant_idx = idx
                    break
            if last_assistant_idx == -1: continue 
                
            prompt_msgs = chosen_full_msgs[:last_assistant_idx]
            chosen_msg = chosen_full_msgs[last_assistant_idx]
            
            last_rej_idx = -1
            for idx in range(len(rejected_full_msgs) - 1, -1, -1):
                if rejected_full_msgs[idx]["role"] == "assistant":
                    last_rej_idx = idx
                    break
            if last_rej_idx == -1: continue
                
            rejected_msg = rejected_full_msgs[last_rej_idx]
            
            flat_data.append((prompt_msgs, chosen_msg, rejected_msg))
            
        except Exception:
            continue

    # 2. Epoch loop and tokenization
    rng = random.Random(seed)
    epoch_idx = 0
    example_idx = 0
    done = False

    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent: print(f'Finished generating {n_epochs} epochs on {split} split')
            break

        if shuffle:
            rng.shuffle(flat_data)

        batch_data = []
        for prompt_msgs, chosen_msg, rejected_msg in flat_data:
            if done: break

            # 1. Process Prompt & Chosen
            prompt_ids = tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True)
            
            # Left-truncate prompt if it exceeds max_prompt_length
            if len(prompt_ids) > max_prompt_length:
                prompt_ids = prompt_ids[-max_prompt_length:]

            chosen_msgs = prompt_msgs + [chosen_msg]
            chosen_ids = tokenizer.apply_chat_template(chosen_msgs, tokenize=True)
            
            # Right-truncate full sequence if it exceeds max_length
            if len(chosen_ids) > max_length:
                chosen_ids = chosen_ids[:max_length]

            chosen_labels = chosen_ids.copy()
            # Mask the prompt portion in labels (-100)
            prompt_len = min(len(prompt_ids), len(chosen_labels))
            chosen_labels[:prompt_len] = [-100] * prompt_len

            item_dict = {
                'chosen_input_ids': torch.tensor(chosen_ids),
                'chosen_labels': torch.tensor(chosen_labels)
            }

            # 2. Process Rejected (Skipped in SFT mode)
            if not sft_mode:
                rejected_msgs = prompt_msgs + [rejected_msg]
                rejected_ids = tokenizer.apply_chat_template(rejected_msgs, tokenize=True)
                
                if len(rejected_ids) > max_length:
                    rejected_ids = rejected_ids[:max_length]

                rejected_labels = rejected_ids.copy()
                prompt_len_rej = min(len(prompt_ids), len(rejected_labels))
                rejected_labels[:prompt_len_rej] = [-100] * prompt_len_rej

                item_dict['rejected_input_ids'] = torch.tensor(rejected_ids)
                item_dict['rejected_labels'] = torch.tensor(rejected_labels)

            batch_data.append(item_dict)
            example_idx += 1

            # 3. Yield complete batches
            if len(batch_data) == batch_size:
                yield _pad_and_collate(batch_data, tokenizer)
                
                if n_examples is not None and example_idx >= n_examples:
                    if not silent: print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                    done = True
                
                batch_data = []

        if done: break
        epoch_idx += 1

# --- Helpers: Padding and Masking ---
def _pad_and_collate(batch_data, tokenizer):
    batch = {}
    
    # 1. Standard padding per key
    for k in batch_data[0].keys():
        padding_value = tokenizer.pad_token_id if k.endswith('_input_ids') else -100

        if 'prompt' in k: # Left padding
            to_pad = [ex[k].flip(dims=[0]) for ex in batch_data]
            padded = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
            batch[k] = padded.flip(dims=[1])
        else:             # Right padding
            to_pad = [ex[k] for ex in batch_data]
            batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
        
        # Generate attention_mask
        if k.endswith('_input_ids'):
            mask_key = k.replace('input_ids', 'attention_mask')
            batch[mask_key] = (batch[k] != tokenizer.pad_token_id).long()
            
    # 2. Align chosen and rejected sequence lengths for DPO mode
    if 'chosen_input_ids' in batch and 'rejected_input_ids' in batch:
        max_len = max(batch['chosen_input_ids'].size(1), batch['rejected_input_ids'].size(1))
        
        def pad_tensor_to_max(tensor, pad_value):
            pad_len = max_len - tensor.size(1)
            if pad_len > 0:
                # Pad only on the right
                tensor = F.pad(tensor, (0, pad_len), value=pad_value)
            return tensor

        # Apply secondary padding
        batch['chosen_input_ids'] = pad_tensor_to_max(batch['chosen_input_ids'], tokenizer.pad_token_id)
        batch['chosen_labels'] = pad_tensor_to_max(batch['chosen_labels'], -100)
        batch['chosen_attention_mask'] = pad_tensor_to_max(batch['chosen_attention_mask'], 0)
        
        batch['rejected_input_ids'] = pad_tensor_to_max(batch['rejected_input_ids'], tokenizer.pad_token_id)
        batch['rejected_labels'] = pad_tensor_to_max(batch['rejected_labels'], -100)
        batch['rejected_attention_mask'] = pad_tensor_to_max(batch['rejected_attention_mask'], 0)
            
    return batch