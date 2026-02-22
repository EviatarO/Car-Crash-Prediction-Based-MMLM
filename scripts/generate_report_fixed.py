import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def process_file(jsonl_path: Path):
    print(f"Processing {jsonl_path}...")
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    if not records:
        print(f"No records in {jsonl_path}")
        return

    # Create DataFrame for XLSX
    data = []
    for r in records:
        score = r.get('logprob_score')
        if score is None: continue
        data.append({
            'video_id': r.get('video_id'),
            'model_id': r.get('model_id'),
            'prompt_id': r.get('prompt_id'),
            'prompt_name': r.get('prompt_name'),
            'logprob_score': score,
            'verdict': 'YES' if score >= 0.5 else 'NO',
            'time_to_event': r.get('time_to_event'),
            'requested_tte': r.get('requested_time_to_event')
        })
    
    df = pd.DataFrame(data)
    xlsx_path = jsonl_path.with_suffix('.xlsx')
    df.to_excel(xlsx_path, index=False)
    print(f"Wrote {xlsx_path}")

    # Aggregation for Plot
    models = sorted(df['model_id'].unique())
    prompts = sorted(df['prompt_id'].unique())
    prompt_names = {r['prompt_id']: r['prompt_name'] for r in data}
    
    counts = defaultdict(int)
    for _, row in df.iterrows():
        if row['verdict'] == 'YES':
            counts[(row['model_id'], row['prompt_id'])] += 1
            
    # Plotting
    prompt_ids = prompts
    prompt_labels = [f"{pid}: {prompt_names.get(pid, '')}" for pid in prompt_ids]
    
    model_colors = {
        "google/gemini-2.0-flash-001": "#1f77b4",
        "openai/gpt-4o": "#ff7f0e",
        "anthropic/claude-sonnet-4": "#2ca02c",
        "opengvlab/internvl3-78b": "#9467bd",
    }

    x = range(len(prompt_ids))
    width = 0.2
    n_models = len(models)
    
    plt.figure(figsize=(12, 6))
    
    for idx, model in enumerate(models):
        ys = [counts[(model, pid)] for pid in prompt_ids]
        offset = (idx - (n_models - 1) / 2) * width
        xs = [pos + offset for pos in x]
        
        bars = plt.bar(xs, ys, width=width, label=model, color=model_colors.get(model, "#7f7f7f"))
        
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

    plt.axhline(0, color='black', linewidth=0.5)
    plt.xticks(x, prompt_labels, rotation=20, ha="right")
    plt.ylabel("Count of YES Verdicts (Max 3)")
    
    # Extract TTE for title
    tte_val = df['requested_tte'].iloc[0] if 'requested_tte' in df.columns and not pd.isna(df['requested_tte'].iloc[0]) else "Unknown"
    plt.title(f"Prompt Experiment: YES Verdict Count (Target TTE: {tte_val}s)")
    
    plt.ylim(0, 3.5)
    plt.yticks([0, 1, 2, 3])
    plt.legend()
    plt.tight_layout()
    
    plot_path = jsonl_path.parent / jsonl_path.name.replace('prompt_experiment', 'prompt_comparison').replace('.jsonl', '.png')
    plt.savefig(plot_path)
    print(f"Wrote {plot_path}")
    plt.close()

if __name__ == "__main__":
    files = [
        "MMLM_CursorAI/outputs/prompt_experiment/prompt_experiment_2026-02-10.tte1.0.jsonl",
        "MMLM_CursorAI/outputs/prompt_experiment/prompt_experiment_2026-02-10.tte1.5.jsonl"
    ]
    for p in files:
        if Path(p).exists():
            process_file(Path(p))
        else:
            print(f"File not found: {p}")
