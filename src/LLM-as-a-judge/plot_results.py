"""
Plot win rates from LLM-as-a-Judge evaluation results.

Usage:
  python plot_results.py \
    --judgment_files outputs/judgments/sft_vs_zodpo.jsonl \
                     outputs/judgments/sft_vs_fodpo.jsonl \
                     outputs/judgments/fodpo_vs_zodpo.jsonl \
    --output_path outputs/winrate_plot.pdf
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_judgment_file(path: str) -> dict:
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    
    if not records:
        return None
    
    model_a = records[0]["model_a"]
    model_b = records[0]["model_b"]
    total = len(records)
    
    wins_a = sum(1 for r in records if r["winner"] == "model_a")
    wins_b = sum(1 for r in records if r["winner"] == "model_b")
    ties   = sum(1 for r in records if r["winner"] == "tie")
    
    return {
        "label": f"{model_b} vs {model_a}",   # x 轴标签：被比较的模型在前
        "model_a": model_a,
        "model_b": model_b,
        "total": total,
        "win_rate_a": wins_a / total,          # loses for model_b
        "win_rate_b": wins_b / total,          # wins for model_b
        "tie_rate": ties / total,
    }


def plot_winrate(stats_list: list[dict], output_path: str):
    n = len(stats_list)
    labels = [s["label"] for s in stats_list]
    wins   = np.array([s["win_rate_b"] for s in stats_list])
    ties   = np.array([s["tie_rate"]   for s in stats_list])
    loses  = np.array([s["win_rate_a"] for s in stats_list])
    
    fig, ax = plt.subplots(figsize=(8, max(3, 1.0 * n)))
    
    colors = {
        "win":  "#4CAF50",
        "tie":  "#9E9E9E", 
        "lose": "#F44336",
    }
    
    y_pos = np.arange(n)
    bar_height = 0.5
    
    bars_lose = ax.barh(y_pos, loses, bar_height, color=colors["lose"], label="Loses")
    bars_tie  = ax.barh(y_pos, ties,  bar_height, left=loses, color=colors["tie"], label="Ties")
    bars_win  = ax.barh(y_pos, wins,  bar_height, left=loses + ties, color=colors["win"], label="Wins")
    
    for i, (l, t, w) in enumerate(zip(loses, ties, wins)):
        if l > 0.07:
            ax.text(l / 2, i, f"{l*100:.0f}%", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        if t > 0.07:
            ax.text(l + t / 2, i, f"{t*100:.0f}%", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        if w > 0.07:
            ax.text(l + t + w / 2, i, f"{w*100:.0f}%", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    
    for i, s in enumerate(stats_list):
        ax.text(1.01, i, f"n={s['total']}", va="center", fontsize=8,
                color="gray", transform=ax.get_yaxis_transform())
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion", fontsize=11)
    ax.set_title("Win Rate Comparison (LLM-as-a-Judge, position-debiased)", fontsize=12)
    
    ax.axvline(x=0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    
    ax.legend(handles=[
        mpatches.Patch(color=colors["win"],  label="Wins"),
        mpatches.Patch(color=colors["tie"],  label="Ties"),
        mpatches.Patch(color=colors["lose"], label="Loses"),
    ], loc="lower right", fontsize=10)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"[DONE] Plot saved to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgment_files", nargs="+", required=True,
                        help="One or more file paths to JSONL judgment results, e.g., outputs/judgments/sft_vs_zodpo.jsonl")
    parser.add_argument("--output_path", default="outputs/winrate_plot.pdf")
    args = parser.parse_args()
    
    stats_list = []
    for path in args.judgment_files:
        s = load_judgment_file(path)
        if s:
            stats_list.append(s)
            print(f"[INFO] {s['label']}: wins={s['win_rate_b']:.1%}, "
                  f"ties={s['tie_rate']:.1%}, loses={s['win_rate_a']:.1%}")
    
    plot_winrate(stats_list, args.output_path)


if __name__ == "__main__":
    main()