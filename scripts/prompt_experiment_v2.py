import argparse
import base64
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from prompts.templates import PROMPT_B, PROMPT_D, PROMPT_F


DEFAULT_MODELS = [
    "openai/gpt-4o",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4",
]

PROMPTS = [
    {"id": "B", "name": "Accident Investigator", "text": PROMPT_B},
    {"id": "D", "name": "Emergency Brake Decision", "text": PROMPT_D},
    {"id": "F", "name": "Distillation Specialist", "text": PROMPT_F},
]

MODEL_ORDER = [
    "openai/gpt-4o",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4",
]

MODEL_COLORS = {
    "openai/gpt-4o": "#ff7f0e",
    "google/gemini-3-pro-preview": "#1f77b4",
    "anthropic/claude-sonnet-4": "#2ca02c",
}


def _log_stage(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[STAGE {timestamp}] {message}", file=sys.stderr, flush=True)


def _load_manifest(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _time_to_event(rec: Dict) -> Optional[float]:
    t_seconds = rec.get("t_seconds")
    time_of_event = rec.get("time_of_event")
    if t_seconds is None or time_of_event is None:
        return None
    return float(time_of_event) - float(t_seconds)


def _select_tp_clips(
    manifest: Sequence[Dict],
    n_05: int = 5,
    n_10: int = 5,
    n_15: int = 5,
) -> List[Dict]:
    by_video: Dict[str, List[Dict]] = defaultdict(list)
    for rec in manifest:
        if rec.get("target") == 1:
            by_video[str(rec["video_id"]).zfill(5)].append(rec)

    bucket_specs = [(0.5, n_05), (1.0, n_10), (1.5, n_15)]
    selected: List[Dict] = []
    used_videos = set()

    for target_tte, needed in bucket_specs:
        candidates: List[Tuple[float, float, str, Dict]] = []
        for video_id, recs in by_video.items():
            if video_id in used_videos:
                continue
            best = None
            best_dist = None
            for rec in recs:
                tte = _time_to_event(rec)
                if tte is None:
                    continue
                dist = abs(tte - target_tte)
                # Prefer later clips on ties.
                t_seconds = float(rec.get("t_seconds", 0.0))
                if best is None or dist < best_dist or (dist == best_dist and t_seconds > float(best.get("t_seconds", 0.0))):
                    best = rec
                    best_dist = dist
            if best is not None and best_dist is not None:
                candidates.append((best_dist, -float(best.get("t_seconds", 0.0)), video_id, best))

        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        if len(candidates) < needed:
            raise RuntimeError(
                f"Not enough TP candidates for time_to_event~{target_tte}. Needed={needed}, found={len(candidates)}"
            )

        for _, __, video_id, rec in candidates[:needed]:
            clip = dict(rec)
            clip["requested_time_to_event"] = target_tte
            selected.append(clip)
            used_videos.add(video_id)

    return selected


def _select_tn_clips(manifest: Sequence[Dict], n: int = 15) -> List[Dict]:
    latest_by_video: Dict[str, Dict] = {}
    for rec in manifest:
        if rec.get("target") != 0:
            continue
        video_id = str(rec["video_id"]).zfill(5)
        if video_id not in latest_by_video:
            latest_by_video[video_id] = rec
            continue
        if float(rec.get("t_seconds", -1.0)) > float(latest_by_video[video_id].get("t_seconds", -1.0)):
            latest_by_video[video_id] = rec

    candidates = sorted(
        latest_by_video.values(),
        key=lambda r: (-float(r.get("t_seconds", 0.0)), str(r.get("video_id")).zfill(5)),
    )
    if len(candidates) < n:
        raise RuntimeError(f"Not enough TN candidates. Needed={n}, found={len(candidates)}")
    return [dict(rec) for rec in candidates[:n]]


def _encode_image(path: Path, frame_size: int) -> str:
    if path.exists():
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", (frame_size, frame_size), color=(0, 0, 0))
    if frame_size and img.size != (frame_size, frame_size):
        img = img.resize((frame_size, frame_size))
    from io import BytesIO

    tmp = BytesIO()
    img.save(tmp, format="JPEG")
    buffer = tmp.getvalue()
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _build_messages(prompt: str, image_b64s: Sequence[str], detail: Optional[str]) -> List[Dict]:
    content = [{"type": "text", "text": prompt}]
    for b64 in image_b64s:
        image_url = {"url": b64}
        if detail:
            image_url["detail"] = detail
        content.append({"type": "image_url", "image_url": image_url})
    return [{"role": "user", "content": content}]


def _extract_json_object(raw_response: str) -> Optional[Dict]:
    try:
        return json.loads(raw_response)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            return None

    bracketed = re.search(r"(\{[\s\S]*\})", raw_response)
    if bracketed:
        try:
            return json.loads(bracketed.group(1))
        except Exception:
            return None
    return None


def _extract_reasoning_and_verdict(raw_response: str, prompt_id: str) -> Tuple[Optional[str], Optional[str]]:
    if not raw_response:
        return None, None

    if prompt_id in {"B", "D"}:
        reasoning_match = re.search(
            r"\[Collision_Reasoning\]:\s*(.*?)(?=\[Verdict\]|$)",
            raw_response,
            re.DOTALL,
        )
        verdict_match = re.search(r"\[Verdict\]:\s*(\w+)", raw_response, re.IGNORECASE)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None
        verdict = verdict_match.group(1).strip().upper() if verdict_match else None
        if verdict in {"YES", "NO"}:
            return reasoning, verdict

    if prompt_id == "F":
        obj = _extract_json_object(raw_response)
        if obj:
            verdict = str(obj.get("collision_verdict", "")).strip().upper()
            reasoning = obj.get("causal_reasoning")
            if isinstance(reasoning, str):
                reasoning = reasoning.strip()
            else:
                reasoning = None
            if verdict in {"YES", "NO"}:
                return reasoning, verdict

    fallback_match = re.search(r"\b(YES|NO)\b", raw_response, re.IGNORECASE)
    fallback_verdict = fallback_match.group(1).upper() if fallback_match else None
    return None, fallback_verdict


def _call_model(
    client: OpenAI,
    model: str,
    messages: List[Dict],
    max_retries: int,
    retry_delay: float,
) -> Tuple[str, Dict]:
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            text = response.choices[0].message.content if response.choices else ""
            usage = response.usage.model_dump() if hasattr(response, "usage") and response.usage else {}
            return text or "", usage
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            time.sleep(retry_delay * attempt)
    raise RuntimeError(f"OpenRouter call failed after {max_retries} attempts: {last_exc}")


def _compute_confusion(records: List[Dict], models: Sequence[str], prompts: Sequence[Dict]) -> pd.DataFrame:
    rows = []
    for prompt in prompts:
        for model in models:
            subset = [r for r in records if r["prompt_id"] == prompt["id"] and r["model_id"] == model]
            tp = fp = tn = fn = 0
            for rec in subset:
                target = rec.get("target")
                verdict = rec.get("Verdict")
                if target == 1 and verdict == "YES":
                    tp += 1
                elif target == 1 and verdict == "NO":
                    fn += 1
                elif target == 0 and verdict == "NO":
                    tn += 1
                elif target == 0 and verdict == "YES":
                    fp += 1
            rows.append(
                {
                    "model_id": model,
                    "prompt_id": prompt["id"],
                    "TP": tp,
                    "FP": fp,
                    "TN": tn,
                    "FN": fn,
                }
            )
    return pd.DataFrame(rows)


def _plot_tp_tn_bars(confusion_df: pd.DataFrame, out_path: Path, models: Sequence[str], prompts: Sequence[Dict]) -> None:
    prompt_ids = [p["id"] for p in prompts]
    x = list(range(len(prompt_ids)))
    width = 0.12

    plt.figure(figsize=(12, 6))
    n_bars_per_group = len(models) * 2

    for m_idx, model in enumerate(models):
        tp_vals = []
        tn_vals = []
        for pid in prompt_ids:
            row = confusion_df[(confusion_df["model_id"] == model) & (confusion_df["prompt_id"] == pid)]
            if row.empty:
                tp_vals.append(0)
                tn_vals.append(0)
            else:
                tp_vals.append(int(row.iloc[0]["TP"]))
                tn_vals.append(int(row.iloc[0]["TN"]))

        base_offset = m_idx * 2
        tp_offsets = [(base_offset - (n_bars_per_group - 1) / 2) * width + xi for xi in x]
        tn_offsets = [((base_offset + 1) - (n_bars_per_group - 1) / 2) * width + xi for xi in x]

        color = MODEL_COLORS.get(model, "#7f7f7f")
        plt.bar(tp_offsets, tp_vals, width=width, color=color, label=f"{model} TP")
        plt.bar(tn_offsets, tn_vals, width=width, color=color, hatch="//", alpha=0.55, label=f"{model} TN")

    prompt_labels = [f"{p['id']}: {p['name']}" for p in prompts]
    plt.xticks(x, prompt_labels, rotation=10, ha="right")
    plt.ylabel("Success Count")
    plt.ylim(0, 15.5)
    plt.title("TP and TN Success by Prompt and Model")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    _log_stage("Starting prompt_experiment_v2")
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--detail", default="low")
    parser.add_argument("--frame_size", type=int, default=256)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=2.0)
    parser.add_argument("--n_05", type=int, default=5)
    parser.add_argument("--n_10", type=int, default=5)
    parser.add_argument("--n_15", type=int, default=5)
    parser.add_argument("--n_tn", type=int, default=15)
    parser.add_argument("--output_tag", default="")
    args = parser.parse_args()

    _log_stage("Loading environment variables and OpenRouter credentials")
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    _log_stage("Initializing OpenRouter client")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "MMLM_Prompt_Experiment_V2"),
        },
    )

    _log_stage(f"Loading manifest from {args.manifest}")
    manifest = _load_manifest(Path(args.manifest))
    _log_stage(
        f"Selecting clips: TP buckets n_05={args.n_05}, n_10={args.n_10}, n_15={args.n_15}; TN n={args.n_tn}"
    )
    tp_clips = _select_tp_clips(manifest, n_05=args.n_05, n_10=args.n_10, n_15=args.n_15)
    tn_clips = _select_tn_clips(manifest, n=args.n_tn)
    clips = tp_clips + tn_clips
    _log_stage(f"Selected {len(tp_clips)} TP clips + {len(tn_clips)} TN clips = {len(clips)} total clips")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    raw_tag = args.output_tag.strip()
    safe_tag = "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in raw_tag)
    tag_suffix = f".{safe_tag}" if safe_tag else ""

    out_jsonl = out_dir / f"prompt_experiment_v2_{today}{tag_suffix}.jsonl"
    out_confusion = out_dir / f"confusion_matrix_{today}{tag_suffix}.xlsx"
    out_detailed = out_dir / f"detailed_results_{today}{tag_suffix}.xlsx"
    out_plot = out_dir / f"tp_tn_bars_{today}{tag_suffix}.png"

    requested_models = [m.strip() for m in args.models.split(",") if m.strip()]
    ordered_models = [m for m in MODEL_ORDER if m in requested_models]
    for model in requested_models:
        if model not in ordered_models:
            ordered_models.append(model)

    frames_root = Path(args.frames_root)
    total = len(ordered_models) * len(PROMPTS) * len(clips)
    count = 0
    start_time = time.time()
    _log_stage(
        f"Beginning inference loop across {len(ordered_models)} models x {len(PROMPTS)} prompts x {len(clips)} clips = {total} runs"
    )

    detailed_rows: List[Dict] = []
    all_records: List[Dict] = []

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for model in ordered_models:
            detail = args.detail if model.endswith("gpt-4o") else None
            for prompt in PROMPTS:
                for clip in clips:
                    count += 1
                    clip_start = time.time()
                    frame_indices = clip["frame_indices"]
                    frames_dir = frames_root / str(clip["video_id"]).zfill(5)
                    image_b64s = []
                    for idx in frame_indices:
                        frame_path = frames_dir / f"frame_{int(idx):05d}.jpg"
                        image_b64s.append(_encode_image(frame_path, args.frame_size))

                    messages = _build_messages(prompt["text"], image_b64s, detail)
                    error = None
                    raw_response = ""
                    usage = {}
                    t0 = time.time()
                    try:
                        raw_response, usage = _call_model(
                            client=client,
                            model=model,
                            messages=messages,
                            max_retries=args.max_retries,
                            retry_delay=args.retry_delay,
                        )
                    except Exception as exc:  # pragma: no cover
                        error = str(exc)

                    latency_s = time.time() - t0
                    time_to_event = _time_to_event(clip)
                    reasoning, verdict = _extract_reasoning_and_verdict(raw_response, prompt["id"])

                    record = {
                        **clip,
                        "time_to_event": time_to_event,
                        "model_id": model,
                        "prompt_id": prompt["id"],
                        "prompt_name": prompt["name"],
                        "prompt_text": prompt["text"],
                        "latency_s": latency_s,
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                        "raw_response": raw_response,
                        "Collision_Reasoning": reasoning,
                        "Verdict": verdict,
                        "error": error,
                    }
                    all_records.append(record)
                    fout.write(json.dumps(record) + "\n")

                    detailed_rows.append(
                        {
                            "model_id": model,
                            "prompt_id": prompt["id"],
                            "video_id": clip.get("video_id"),
                            "target": clip.get("target"),
                            "t_seconds": clip.get("t_seconds"),
                            "time_to_event": time_to_event,
                            "time_of_alert": clip.get("time_of_alert"),
                            "time_of_event": clip.get("time_of_event"),
                            "latency_s": latency_s,
                            "Verdict": verdict,
                            "Collision_Reasoning": reasoning,
                        }
                    )

                    clip_total_s = time.time() - clip_start
                    status = "ok" if error is None else "error"
                    print(
                        f"[CLIP {count}/{total}] model={model} prompt={prompt['id']} video_id={clip.get('video_id')} "
                        f"target={clip.get('target')} api_latency_s={latency_s:.2f} clip_total_s={clip_total_s:.2f} status={status}",
                        file=sys.stderr,
                        flush=True,
                    )

                    if count % 5 == 0 or count == total:
                        elapsed = max(time.time() - start_time, 1e-6)
                        items_per_s = count / elapsed
                        eta_s = (total - count) / items_per_s if items_per_s > 0 else None
                        eta_txt = f"{eta_s:.1f}" if eta_s is not None else "unknown"
                        print(
                            f"[{count}/{total}] model={model} prompt={prompt['id']} video_id={clip.get('video_id')} eta_s={eta_txt}",
                            file=sys.stderr,
                            flush=True,
                        )

    _log_stage("Inference loop complete; building confusion matrix")
    confusion_df = _compute_confusion(all_records, ordered_models, PROMPTS)
    confusion_df.to_excel(out_confusion, index=False)

    _log_stage("Writing detailed results Excel")
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df = detailed_df[
        [
            "model_id",
            "prompt_id",
            "video_id",
            "target",
            "t_seconds",
            "time_to_event",
            "time_of_alert",
            "time_of_event",
            "latency_s",
            "Verdict",
            "Collision_Reasoning",
        ]
    ]
    detailed_df.to_excel(out_detailed, index=False)

    _log_stage("Rendering TP/TN bar chart")
    _plot_tp_tn_bars(confusion_df, out_plot, ordered_models, PROMPTS)
    _log_stage("All outputs written successfully")

    print(f"Wrote JSONL results: {out_jsonl}", file=sys.stderr, flush=True)
    print(f"Wrote confusion matrix: {out_confusion}", file=sys.stderr, flush=True)
    print(f"Wrote detailed results: {out_detailed}", file=sys.stderr, flush=True)
    print(f"Wrote TP/TN bars plot: {out_plot}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
