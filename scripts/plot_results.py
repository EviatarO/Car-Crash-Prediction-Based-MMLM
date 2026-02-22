import argparse
import json
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def _load_records(paths: List[Path]) -> List[dict]:
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
    return records


def plot_results(input_paths: List[Path], output_path: Path) -> None:
    # 1. Load data
    records = _load_records(input_paths)

    # 2. Extract unique models and prompts to maintain order
    # Assuming prompt_id "A", "B", "C", "D", "E" and standardized model names
    prompt_map = {} # id -> name
    models = set()
    
    for rec in records:
        pid = rec.get("prompt_id")
        pname = rec.get("prompt_name")
        mid = rec.get("model_id")
        
        if pid and pname:
            prompt_map[pid] = pname
        if mid:
            models.add(mid)
            
    # Sort models and prompts to ensure consistent plotting
    # Order prompts A->E
    sorted_prompt_ids = sorted(prompt_map.keys())
    # Order models: Gemini, GPT-4o, Claude
    model_order_preference = ["google/gemini-2.0-flash-001", "openai/gpt-4o", "anthropic/claude-sonnet-4"]
    sorted_models = [m for m in model_order_preference if m in models]
    # Add any other models not in preference list at the end
    for m in models:
        if m not in sorted_models:
            sorted_models.append(m)

    # 3. Aggregate scores: SUM of Verdicts (YES=1, NO=0)
    # Since we have binary 0.0/1.0 scores in the fixed JSONL, we can just sum them.
    # Group by (prompt_id, model_id)
    grouped: Dict[Tuple[str, str], float] = {}
    
    for rec in records:
        score = rec.get("logprob_score")
        # Ensure we treat None as 0 and threshold probabilities for "YES" count
        # For GPT-4o, score might be 0.999. We want to count this as 1 YES.
        # For fixed Gemini/Claude, score is 1.0 or 0.0.
        val = 0.0
        if score is not None:
            if score >= 0.5:
                val = 1.0
            else:
                val = 0.0
        
        key = (rec["prompt_id"], rec["model_id"])
        grouped[key] = grouped.get(key, 0.0) + val

    # 4. Plot
    prompt_labels = [f"{pid}: {prompt_map[pid]}" for pid in sorted_prompt_ids]
    
    # Colors matching previous style
    model_colors = {
        "google/gemini-2.0-flash-001": "#1f77b4",  # Blue
        "openai/gpt-4o": "#ff7f0e",                # Orange
        "anthropic/claude-sonnet-4": "#2ca02c",    # Green
        "opengvlab/internvl3-78b": "#9467bd",      # Purple
    }
    
    x = list(range(len(sorted_prompt_ids)))
    width = 0.2
    n_models = len(sorted_models)
    
    plt.figure(figsize=(12, 6))
    
    for idx, model in enumerate(sorted_models):
        # Get sum of YES verdicts for this model across all prompts
        ys = [grouped.get((pid, model), 0.0) for pid in sorted_prompt_ids]
        
        # Calculate bar positions dynamically to center groups
        offset = (idx - (n_models - 1) / 2) * width
        xs = [pos + offset for pos in x]
            
        bars = plt.bar(xs, ys, width=width, label=model, color=model_colors.get(model, "gray"))
        
        # Annotate bars with integer counts
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.05,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight='bold'
            )

    plt.xticks(x, prompt_labels, rotation=15, ha="right")
    plt.ylim(0, 3.5) # Max is 3 clips, give some headroom
    plt.yticks([0, 1, 2, 3])
    plt.ylabel("Count of YES Verdicts (Max 3)")
    plt.title("Prompt Experiment: Correct Collision Predictions (YES) by Prompt & Model")
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    print(f"Saving plot to {output_path}")
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot prompt experiment results.")
    parser.add_argument("inputs", nargs="+", help="One or more fixed JSONL files.")
    parser.add_argument("--output", "-o", required=False, help="Output PNG path.")
    args = parser.parse_args()

    input_files = [Path(p) for p in args.inputs]
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_files[0].parent / "prompt_comparison_fixed.png"

    plot_results(input_files, output_file)
