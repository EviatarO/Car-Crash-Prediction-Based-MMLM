import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

TIME_STEP = 0.25
EXPECTED_POINTS = [round(TIME_STEP * i, 2) for i in range(1, 17)]

MODEL_PRICES = {
    "google/gemini-2.0-flash-001": {"input_per_million": 0.10, "output_per_million": 0.40},
    "openai/gpt-4o": {"input_per_million": 2.50, "output_per_million": 10.00},
    "anthropic/claude-sonnet-4-20250514": {"input_per_million": 3.00, "output_per_million": 15.00},
}


def _load_jsonl(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_manifest_index(manifest_path: Path) -> Dict[Tuple[str, int], Dict]:
    index = {}
    for rec in _load_jsonl(manifest_path):
        key = (rec["video_id"], int(rec["end_frame_idx"]))
        index[key] = rec
    return index


def _bucket_time_to_event(time_to_event: float) -> float:
    if time_to_event is None or math.isnan(time_to_event):
        return None
    return round(round(time_to_event / TIME_STEP) * TIME_STEP, 2)


def _score_vs_time(
    records: List[Dict], manifest_index: Dict[Tuple[str, int], Dict]
) -> Dict[float, List[float]]:
    buckets: Dict[float, List[float]] = {t: [] for t in EXPECTED_POINTS}
    for rec in records:
        key = (rec["video_id"], int(rec["end_frame_idx"]))
        manifest = manifest_index.get(key)
        if not manifest:
            continue
        time_of_event = manifest.get("time_of_event")
        if time_of_event is None:
            continue
        time_to_event = time_of_event - float(manifest.get("t_seconds", 0.0))
        bucket = _bucket_time_to_event(time_to_event)
        if bucket in buckets and rec.get("score") is not None:
            buckets[bucket].append(float(rec["score"]))
    return buckets


def _plot_average_scores(avg_scores: Dict[str, Dict[float, float]], out_path: Path):
    plt.figure(figsize=(10, 5))
    for model, scores in avg_scores.items():
        xs = EXPECTED_POINTS
        ys = [scores.get(t, float("nan")) for t in xs]
        plt.plot(xs, ys, marker="o", label=model)
    plt.gca().invert_xaxis()
    plt.xlabel("Time to Event (s)")
    plt.ylabel("Average Score")
    plt.title("Average Collision Score vs Time to Event")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_per_video(records: List[Dict], manifest_index: Dict[Tuple[str, int], Dict], out_path: Path):
    per_video: Dict[str, Dict[float, float]] = {}
    for rec in records:
        key = (rec["video_id"], int(rec["end_frame_idx"]))
        manifest = manifest_index.get(key)
        if not manifest:
            continue
        time_of_event = manifest.get("time_of_event")
        if time_of_event is None:
            continue
        time_to_event = time_of_event - float(manifest.get("t_seconds", 0.0))
        bucket = _bucket_time_to_event(time_to_event)
        if bucket not in EXPECTED_POINTS or rec.get("score") is None:
            continue
        per_video.setdefault(rec["video_id"], {})[bucket] = float(rec["score"])

    plt.figure(figsize=(10, 6))
    for video_id, series in sorted(per_video.items()):
        xs = EXPECTED_POINTS
        ys = [series.get(t, float("nan")) for t in xs]
        plt.plot(xs, ys, marker="o", linewidth=1, label=video_id)
    plt.gca().invert_xaxis()
    plt.xlabel("Time to Event (s)")
    plt.ylabel("Score")
    plt.title("Per-Video Scores vs Time to Event")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _latency_summary(records: List[Dict]) -> Dict[str, float]:
    latencies = [rec["latency_s"] for rec in records if rec.get("latency_s") is not None]
    if not latencies:
        return {}
    series = pd.Series(latencies)
    return {
        "avg": float(series.mean()),
        "p50": float(series.quantile(0.50)),
        "p95": float(series.quantile(0.95)),
    }


def _cost_summary(records: List[Dict], model_name: str) -> Dict[str, float]:
    tokens_in = [rec.get("input_tokens") for rec in records if rec.get("input_tokens") is not None]
    tokens_out = [rec.get("output_tokens") for rec in records if rec.get("output_tokens") is not None]
    total_in = int(sum(tokens_in)) if tokens_in else 0
    total_out = int(sum(tokens_out)) if tokens_out else 0
    pricing = MODEL_PRICES.get(model_name, {"input_per_million": 0.0, "output_per_million": 0.0})
    cost = (total_in / 1_000_000) * pricing["input_per_million"] + (
        total_out / 1_000_000
    ) * pricing["output_per_million"]
    return {
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "estimated_usd": cost,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_index = _build_manifest_index(Path(args.manifest))

    result_files = sorted(results_dir.glob("*.jsonl"))
    if not result_files:
        raise RuntimeError(f"No JSONL files found in {results_dir}")

    avg_scores_by_model: Dict[str, Dict[float, float]] = {}
    latency_rows = []
    cost_rows = []
    excel_writer = pd.ExcelWriter(out_dir / "benchmark_scores.xlsx", engine="openpyxl")

    for path in result_files:
        records = _load_jsonl(path)
        if not records:
            continue
        model_name = records[0].get("model_id", path.stem)

        buckets = _score_vs_time(records, manifest_index)
        avg_scores = {t: (sum(vals) / len(vals) if vals else float("nan")) for t, vals in buckets.items()}
        avg_scores_by_model[model_name] = avg_scores

        per_video = {}
        for rec in records:
            key = (rec["video_id"], int(rec["end_frame_idx"]))
            manifest = manifest_index.get(key)
            if not manifest or manifest.get("time_of_event") is None:
                continue
            time_to_event = manifest["time_of_event"] - float(manifest.get("t_seconds", 0.0))
            bucket = _bucket_time_to_event(time_to_event)
            per_video.setdefault(rec["video_id"], {})[bucket] = rec.get("score")

        rows = []
        for video_id, series in sorted(per_video.items()):
            rows.append(
                {
                    "video_id": video_id,
                    "event-0.5s": series.get(0.5),
                    "event-1.0s": series.get(1.0),
                    "event-1.5s": series.get(1.5),
                }
            )
        df = pd.DataFrame(rows)
        avg_row = {
            "video_id": "AVG",
            "event-0.5s": df["event-0.5s"].mean(),
            "event-1.0s": df["event-1.0s"].mean(),
            "event-1.5s": df["event-1.5s"].mean(),
        }
        df = pd.concat([pd.DataFrame([avg_row]), df], ignore_index=True)
        sheet_name = model_name.split("/")[-1][:31]
        df.to_excel(excel_writer, sheet_name=sheet_name, index=False)

        latency = _latency_summary(records)
        if latency:
            latency_rows.append({"model": model_name, **latency})

        cost = _cost_summary(records, model_name)
        cost_rows.append({"model": model_name, **cost})

        _plot_per_video(records, manifest_index, out_dir / f"{sheet_name}_per_video.png")

    excel_writer.close()

    _plot_average_scores(avg_scores_by_model, out_dir / "score_vs_time_to_event.png")

    if latency_rows:
        pd.DataFrame(latency_rows).to_csv(out_dir / "latency_summary.csv", index=False)
    if cost_rows:
        pd.DataFrame(cost_rows).to_csv(out_dir / "cost_summary.csv", index=False)


if __name__ == "__main__":
    main()
