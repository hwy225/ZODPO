#!/usr/bin/env python3
"""
compare_generations.py
======================
Generate responses from multiple trained checkpoints (ZODPO, SFT, FO-DPO,
and optionally the base model) on a shared set of prompts and write a
side-by-side HTML / JSONL report.

Quick start
-----------
python compare_generations.py \\
    --zodpo   runs/my_exp/dpo/final_model \\
    --sft     runs/my_exp/sft/final_model \\
    --fodpo   runs/my_exp_fodpo/dpo/final_model \\
    --base    meta-llama/Llama-3.2-1B-Instruct   \\
    --prompts prompts.json  \\         # optional – list of strings or chat dicts
    --n_hh    50            \\         # pull N random prompts from HH-RLHF test
    --out_dir generation_reports
"""

import argparse
import datetime
import json
import os
import re
import random
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Defaults
# ============================================================

DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE    = 0.7
DEFAULT_TOP_P          = 0.9
DEFAULT_DO_SAMPLE      = True
DEFAULT_N_HH           = 50
DEFAULT_SEED           = 42


# ============================================================
# Prompt utilities
# ============================================================

def _parse_hh_prompt(text: str) -> List[Dict[str, str]]:
    """Convert raw HH-RLHF text into a chat-template message list."""
    parts = re.split(r'\n\n(Human|Assistant):', text)
    msgs  = []
    for j in range(1, len(parts), 2):
        role    = "user" if parts[j].strip() == "Human" else "assistant"
        content = parts[j + 1].strip()
        msgs.append({"role": role, "content": content})
    return msgs


def load_hh_prompts(n: int, seed: int = DEFAULT_SEED, cache_dir: Optional[str] = None) -> List[List[Dict]]:
    """
    Pull *n* random test-set prompts from Anthropic/hh-rlhf.
    Each prompt is the conversation up to (but not including) the final
    assistant turn, formatted as a message list.
    """
    from datasets import load_dataset
    ds = load_dataset("Anthropic/hh-rlhf", split="test", cache_dir=cache_dir)

    rng    = random.Random(seed)
    items  = list(ds)
    rng.shuffle(items)

    prompts: List[List[Dict]] = []
    for item in items:
        if len(prompts) >= n:
            break
        try:
            msgs = _parse_hh_prompt(item["chosen"])
            # Find last assistant turn and drop it (that is what we want to generate)
            last_asst = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=-1)
            if last_asst < 0:
                continue
            prompt_msgs = msgs[:last_asst]
            if not prompt_msgs:
                continue
            prompts.append(prompt_msgs)
        except Exception:
            continue

    print(f"  Loaded {len(prompts)} HH-RLHF prompts.")
    return prompts


def load_custom_prompts(path: str) -> List[List[Dict]]:
    """
    Load prompts from a JSON file.  Each entry may be:
      - a string  -> treated as a single user turn
      - a list of {role, content} dicts  -> used directly
    """
    with open(path) as f:
        raw = json.load(f)
    prompts = []
    for entry in raw:
        if isinstance(entry, str):
            prompts.append([{"role": "user", "content": entry}])
        elif isinstance(entry, list):
            prompts.append(entry)
        else:
            raise ValueError(f"Unexpected prompt format: {type(entry)}")
    print(f"  Loaded {len(prompts)} custom prompts from {path}.")
    return prompts


# ============================================================
# Model loading
# ============================================================

def load_model_and_tokenizer(
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"  Loading: {path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, cache_dir=cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # important for batched generation

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map=device,
        cache_dir=cache_dir,
    )
    model.eval()
    return model, tokenizer


# ============================================================
# Generation
# ============================================================

@torch.inference_mode()
def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[List[Dict]],
    max_new_tokens: int  = DEFAULT_MAX_NEW_TOKENS,
    temperature: float   = DEFAULT_TEMPERATURE,
    top_p: float         = DEFAULT_TOP_P,
    do_sample: bool      = DEFAULT_DO_SAMPLE,
    batch_size: int      = 4,
    seed: int            = DEFAULT_SEED,
) -> List[str]:
    """
    Generate one response per prompt.  Returns a list of response strings
    (no prompt prefix).
    """
    torch.manual_seed(seed)
    device    = next(model.parameters()).device
    responses = []

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]

        # Render each prompt to a string using the model's chat template
        rendered = []
        for msgs in batch_prompts:
            try:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback: naive concatenation
                text = "\n".join(
                    f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                    for m in msgs
                ) + "\nAssistant:"
            rendered.append(text)

        encoding = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        prompt_len = encoding["input_ids"].shape[1]

        gen_ids = model.generate(
            **encoding,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens
        for i, ids in enumerate(gen_ids):
            new_ids  = ids[prompt_len:]
            response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            responses.append(response)

        print(
            f"    Generated {min(batch_start + batch_size, len(prompts))}"
            f"/{len(prompts)} responses",
            end="\r",
        )

    print()
    return responses


# ============================================================
# Report writers
# ============================================================

def write_jsonl(
    prompts: List[List[Dict]],
    model_responses: Dict[str, List[str]],
    out_path: str,
):
    with open(out_path, "w") as f:
        for i, msgs in enumerate(prompts):
            row = {
                "prompt_idx": i,
                "prompt":     msgs,
                "responses":  {tag: resp[i] for tag, resp in model_responses.items()},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  JSONL  --> {out_path}")


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


def write_html(
    prompts: List[List[Dict]],
    model_responses: Dict[str, List[str]],
    out_path: str,
    title: str = "Generation Comparison",
):
    tags = list(model_responses.keys())

    # Colour palette (one per model, cycles if > 8)
    colours = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
        "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    ]
    tag_colour = {t: colours[i % len(colours)] for i, t in enumerate(tags)}

    header_cells = "".join(
        f'<th style="background:{tag_colour[t]};color:#fff">{t}</th>' for t in tags
    )

    rows_html = []
    for i, msgs in enumerate(prompts):
        # Build a readable prompt string
        prompt_lines = []
        for m in msgs:
            prefix = "👤 User" if m["role"] == "user" else "🤖 Assistant"
            prompt_lines.append(f"<b>{prefix}:</b> {_html_escape(m['content'])}")
        prompt_cell = "<br><br>".join(prompt_lines)

        resp_cells = "".join(
            f'<td style="vertical-align:top;padding:8px 12px;border-left:3px solid {tag_colour[t]}">'
            f'{_html_escape(model_responses[t][i])}</td>'
            for t in tags
        )

        rows_html.append(
            f"<tr>"
            f'<td style="font-size:0.85em;color:#555;vertical-align:top;padding:8px 12px">'
            f'<b>#{i+1}</b><br><br>{prompt_cell}</td>'
            f"{resp_cells}"
            f"</tr>"
        )

    table_body = "\n".join(rows_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body   {{ font-family: system-ui, sans-serif; font-size: 14px; margin: 24px; background:#fafafa; }}
  h1     {{ font-size: 1.4em; margin-bottom: 4px; }}
  p.meta {{ color: #888; font-size: 0.85em; margin-bottom: 20px; }}
  table  {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
  th, td {{ border: 1px solid #ddd; padding: 10px 14px; text-align: left; word-wrap: break-word; }}
  th     {{ font-size: 0.9em; letter-spacing: 0.03em; }}
  tr:nth-child(even) td {{ background: #f5f5f5; }}
  tr:hover td {{ background: #fffbe6; }}
  .prompt-col {{ width: 22%; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; {len(prompts)} prompts</p>
<table>
<thead>
  <tr>
    <th class="prompt-col">Prompt</th>
    {header_cells}
  </tr>
</thead>
<tbody>
{table_body}
</tbody>
</table>
</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"  HTML   --> {out_path}")


def write_markdown(
    prompts: List[List[Dict]],
    model_responses: Dict[str, List[str]],
    out_path: str,
):
    """Write a simple Markdown table (truncates long responses for readability)."""
    tags     = list(model_responses.keys())
    MAX_CELL = 200   # characters per cell

    def _trunc(s: str) -> str:
        s = s.replace("|", "\\|").replace("\n", " ")
        return s[:MAX_CELL] + "…" if len(s) > MAX_CELL else s

    lines = ["# Generation Comparison", ""]
    lines.append("| # | Prompt | " + " | ".join(tags) + " |")
    lines.append("|---|--------|" + "|".join(["-----"] * len(tags)) + "|")

    for i, msgs in enumerate(prompts):
        prompt_str = _trunc(msgs[-1]["content"] if msgs else "")
        resp_cells = " | ".join(_trunc(model_responses[t][i]) for t in tags)
        lines.append(f"| {i+1} | {prompt_str} | {resp_cells} |")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Markdown --> {out_path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare generations across ZODPO / SFT / FO-DPO / base models.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--zodpo",   help="Path to ZODPO final_model dir")
    p.add_argument("--sft",     help="Path to SFT  final_model dir")
    p.add_argument("--fodpo",   help="Path to FO-DPO final_model dir  (first-order DPO)")
    p.add_argument("--base",    help="HF model name/path for the base model (optional)")

    p.add_argument("--extra",   nargs="*", metavar="TAG=PATH",
                   help="Any additional models to include.  Format: --extra my_model=/path/to/ckpt")

    p.add_argument("--prompts", help="JSON file with custom prompts (list of strings or chat dicts)")
    p.add_argument("--n_hh",    type=int, default=DEFAULT_N_HH,
                   help=f"Number of HH-RLHF test prompts to include (default: {DEFAULT_N_HH})")
    p.add_argument("--no_hh",   action="store_true",
                   help="Do not include any HH-RLHF prompts (requires --prompts)")

    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--temperature",    type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top_p",          type=float, default=DEFAULT_TOP_P)
    p.add_argument("--greedy",         action="store_true",
                   help="Use greedy decoding (overrides temperature / top_p)")
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--seed",           type=int, default=DEFAULT_SEED)

    p.add_argument("--dtype",  default="bfloat16",
                   choices=["float32", "float16", "bfloat16"],
                   help="Precision for loading models.")
    p.add_argument("--device", default="cuda")

    p.add_argument("--out_dir", default="generation_reports",
                   help="Directory to write reports.")
    p.add_argument("--cache_dir", default=None, help="HF cache dir.")

    p.add_argument("--formats", nargs="+", default=["html", "jsonl", "md"],
                   choices=["html", "jsonl", "md"],
                   help="Output formats (default: html jsonl md).")

    return p.parse_args()


def main():
    args = parse_args()

    # ── collect model paths ──────────────────────────────────────────
    models_to_load: Dict[str, str] = {}
    if args.base:
        models_to_load["base"]  = args.base
    if args.sft:
        models_to_load["SFT"]   = args.sft
    if args.zodpo:
        models_to_load["ZODPO"] = args.zodpo
    if args.fodpo:
        models_to_load["FODPO"] = args.fodpo
    if args.extra:
        for spec in args.extra:
            if "=" not in spec:
                raise ValueError(f"--extra entries must be TAG=PATH, got: {spec!r}")
            tag, path = spec.split("=", 1)
            models_to_load[tag] = path

    if not models_to_load:
        raise SystemExit("Please supply at least one model path (--zodpo, --sft, --fodpo, or --base).")

    # ── build prompt list ────────────────────────────────────────────
    all_prompts: List[List[Dict]] = []

    if args.prompts:
        all_prompts.extend(load_custom_prompts(args.prompts))

    if not args.no_hh and args.n_hh > 0:
        all_prompts.extend(load_hh_prompts(args.n_hh, seed=args.seed, cache_dir=args.cache_dir))

    if not all_prompts:
        raise SystemExit("No prompts to generate!  Use --prompts and/or --n_hh > 0.")

    random.seed(args.seed)
    random.shuffle(all_prompts)
    print(f"\nTotal prompts: {len(all_prompts)}")

    # ── generate ─────────────────────────────────────────────────────
    dtype          = getattr(torch, args.dtype)
    do_sample      = not args.greedy
    model_responses: Dict[str, List[str]] = {}

    for tag, path in models_to_load.items():
        print(f"\n[{tag}] {path}")
        model, tokenizer = load_model_and_tokenizer(
            path, dtype=dtype, device=args.device, cache_dir=args.cache_dir
        )
        responses = generate_responses(
            model, tokenizer, all_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=do_sample,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        model_responses[tag] = responses

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    # ── write reports ─────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.join(args.out_dir, f"gen_compare_{ts}")

    if "jsonl" in args.formats:
        write_jsonl(all_prompts, model_responses, stem + ".jsonl")
    if "html" in args.formats:
        write_html(
            all_prompts, model_responses, stem + ".html",
            title=f"Generation Comparison — {' vs '.join(model_responses)}"
        )
    if "md" in args.formats:
        write_markdown(all_prompts, model_responses, stem + ".md")

    print(f"\nDone.  Reports written to: {args.out_dir}/")


if __name__ == "__main__":
    main()
