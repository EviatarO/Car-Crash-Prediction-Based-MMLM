import argparse
import base64
import json
import math
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from prompts.templates import PROMPT_A, PROMPT_B, PROMPT_C, PROMPT_D, PROMPT_E


VIDEO_IDS = ["00000", "00007", "00013"]

DEFAULT_MODELS = [
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4",
]

PROMPTS = [
    {"id": "A", "name": "Defensive Driver", "text": PROMPT_A},
    {"id": "B", "name": "Accident Investigator", "text": PROMPT_B},
    {"id": "C", "name": "Trajectory Physics", "text": PROMPT_C},
    {"id": "D", "name": "Emergency Brake Decision", "text": PROMPT_D},
    {"id": "E", "name": "Temporal Delta", "text": PROMPT_E},
]


def _load_manifest(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _select_latest_clips(manifest: Sequence[Dict], video_ids: Sequence[str]) -> List[Dict]:
    video_set = set(video_ids)
    best: Dict[str, Dict] = {}
    for rec in manifest:
        vid = rec.get("video_id")
        if vid not in video_set:
            continue
        if vid not in best or rec.get("t_seconds", -1) > best[vid].get("t_seconds", -1):
            best[vid] = rec
    return [best[vid] for vid in sorted(best.keys())]


def _select_clips_by_time_to_event(
    manifest: Sequence[Dict],
    video_ids: Sequence[str],
    target_time_to_event: float,
) -> List[Dict]:
    video_set = set(video_ids)
    best: Dict[str, Tuple[float, Dict]] = {}
    for rec in manifest:
        vid = rec.get("video_id")
        if vid not in video_set:
            continue
        if rec.get("target") != 1:
            continue
        t_seconds = rec.get("t_seconds")
        time_of_event = rec.get("time_of_event")
        if t_seconds is None or time_of_event is None:
            continue
        time_to_event = time_of_event - t_seconds
        distance = abs(time_to_event - target_time_to_event)

        # Tie-break toward later clips when distance is equal.
        if vid not in best or distance < best[vid][0] or (
            distance == best[vid][0] and t_seconds > best[vid][1].get("t_seconds", -1)
        ):
            clip = dict(rec)
            clip["requested_time_to_event"] = target_time_to_event
            best[vid] = (distance, clip)

    return [best[vid][1] for vid in sorted(best.keys())]


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


def _normalize_token(token: str) -> str:
    return token.strip().lower().strip(".,:;!\"'")


def _extract_logprobs_content(logprobs) -> Optional[List]:
    if not logprobs:
        return None
    if hasattr(logprobs, "content"):
        return logprobs.content
    if isinstance(logprobs, dict) and "content" in logprobs:
        return logprobs["content"]
    return None


def _get_token_field(obj, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_verdict_logprobs(logprobs, response_text: str) -> Tuple[Optional[float], Optional[float], Optional[List[Dict]]]:
    content = _extract_logprobs_content(logprobs)
    if not content:
        return None, None, None

    running_text = ""
    verdict_started = False
    verdict_entry = None

    # First pass: try to find where [Verdict] starts in the tokens
    for entry in content:
        token = _get_token_field(entry, "token", "")
        running_text += token
        if "[Verdict]" in running_text or "[Verdict]:" in running_text:
            verdict_started = True
        
        if verdict_started:
            norm = _normalize_token(token)
            # Check for yes/no tokens
            if norm in {"yes", "no"}:
                verdict_entry = entry
                break
    
    # If not found, it might be the very first token if the model only outputted "YES" (unlikely with CoT but possible)
    # OR the loop logic missed it. 
    
    if verdict_entry is None:
        return None, None, None

    top_logprobs = _get_token_field(verdict_entry, "top_logprobs", None)
    if not top_logprobs:
        return None, None, None

    raw_top = []
    lp_yes = None
    lp_no = None
    
    # Convert top_logprobs to list of dicts for debug
    for candidate in top_logprobs:
        cand_token = _get_token_field(candidate, "token", "")
        cand_lp = _get_token_field(candidate, "logprob", None)
        if cand_lp is None:
            continue
            
        raw_top.append({"token": cand_token, "logprob": cand_lp})
        
        # Robust normalization for matching " YES", "yes", "Yes"
        norm = _normalize_token(cand_token)
        if norm == "yes":
            lp_yes = cand_lp
        elif norm == "no":
            lp_no = cand_lp

    return lp_yes, lp_no, raw_top if raw_top else None


def _compute_p_yes(lp_yes: Optional[float], lp_no: Optional[float]) -> Optional[float]:
    if lp_yes is not None and lp_no is not None:
        p_yes = math.exp(lp_yes)
        p_no = math.exp(lp_no)
        return p_yes / (p_yes + p_no) if (p_yes + p_no) > 0 else None
    if lp_yes is not None:
        return math.exp(lp_yes)
    if lp_no is not None:
        return 1.0 - math.exp(lp_no)
    return None


def _call_model(
    client: OpenAI,
    model: str,
    messages: List[Dict],
    max_retries: int,
    retry_delay: float,
) -> Tuple[str, Dict, Optional[Dict]]:
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                logprobs=True,
                top_logprobs=5,
            )
            text = response.choices[0].message.content if response.choices else ""
            usage = response.usage.model_dump() if hasattr(response, "usage") and response.usage else {}
            logprobs = response.choices[0].logprobs if response.choices else None
            return text or "", usage, logprobs
        except Exception as exc:  # pragma: no cover - network failure path
            last_exc = exc
            time.sleep(retry_delay * attempt)
    raise RuntimeError(f"OpenRouter call failed after {max_retries} attempts: {last_exc}")


def _summarize(records: List[Dict], models: Sequence[str], prompts: Sequence[Dict]) -> None:
    by_prompt_model: Dict[Tuple[str, str], List[float]] = {}
    for rec in records:
        score = rec.get("logprob_score")
        if score is None:
            continue
        key = (rec["prompt_id"], rec["model_id"])
        by_prompt_model.setdefault(key, []).append(score)

    header = ["prompt_id", "prompt_name"] + [m for m in models] + ["count>=0.5"]
    print("\t".join(header), file=sys.stderr)
    for prompt in prompts:
        row = [prompt["id"], prompt["name"]]
        count_ge = 0
        total = 0
        for model in models:
            scores = by_prompt_model.get((prompt["id"], model), [])
            total += len(scores)
            count_ge += sum(1 for s in scores if s >= 0.5)
            avg = sum(scores) / len(scores) if scores else float("nan")
            row.append(f"{avg:.3f}" if scores else "nan")
        row.append(str(count_ge))
        print("\t".join(row), file=sys.stderr)


def _plot_histogram(records: List[Dict], models: Sequence[str], prompts: Sequence[Dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    prompt_ids = [p["id"] for p in prompts]
    prompt_labels = [f"{p['id']}: {p['name']}" for p in prompts]
    model_colors = {
        "google/gemini-2.0-flash-001": "#1f77b4",
        "openai/gpt-4o": "#ff7f0e",
        "anthropic/claude-sonnet-4": "#2ca02c",
        "opengvlab/internvl3-78b": "#9467bd",
    }

    grouped: Dict[Tuple[str, str], List[float]] = {}
    for rec in records:
        score = rec.get("logprob_score")
        if score is None:
            continue
        key = (rec["prompt_id"], rec["model_id"])
        grouped.setdefault(key, []).append(score)

    averages = {
        (pid, model): (sum(scores) / len(scores) if scores else 0.0)
        for (pid, model), scores in grouped.items()
    }

    x = list(range(len(prompt_ids)))
    width = 0.2
    n_models = len(models)

    plt.figure(figsize=(12, 5))
    for idx, model in enumerate(models):
        ys = [averages.get((pid, model), 0.0) for pid in prompt_ids]
        offset = (idx - (n_models - 1) / 2) * width
        xs = [pos + offset for pos in x]
        bars = plt.bar(xs, ys, width=width, label=model, color=model_colors.get(model, "#7f7f7f"))
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xticks(x, prompt_labels, rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Average P(YES)")
    plt.title("Prompt Experiment: Model Confidence P(YES) by Prompt Variant")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--videos", default=",".join(VIDEO_IDS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--detail", default="low")
    parser.add_argument("--frame_size", type=int, default=256)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=2.0)
    parser.add_argument("--target_time_to_event", type=float, default=None)
    parser.add_argument("--output_tag", default="")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "MMLM_Prompt_Experiment"),
        },
    )

    manifest = _load_manifest(Path(args.manifest))
    video_ids = [vid.strip().zfill(5) for vid in args.videos.split(",") if vid.strip()]
    if args.target_time_to_event is None:
        clips = _select_latest_clips(manifest, video_ids)
    else:
        clips = _select_clips_by_time_to_event(manifest, video_ids, args.target_time_to_event)
    if not clips:
        raise RuntimeError("No clips found for the requested video IDs.")
    found_video_ids = {c.get("video_id") for c in clips}
    missing = [vid for vid in video_ids if vid not in found_video_ids]
    if missing:
        raise RuntimeError(f"No clips found for requested video IDs: {', '.join(missing)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    raw_tag = args.output_tag.strip()
    safe_tag = "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in raw_tag)
    tag_suffix = f".{safe_tag}" if safe_tag else ""
    out_path = out_dir / f"prompt_experiment_{today}{tag_suffix}.jsonl"
    plot_path = out_dir / f"prompt_comparison_{today}{tag_suffix}.png"

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    frames_root = Path(args.frames_root)

    total = len(models) * len(PROMPTS) * len(clips)
    count = 0
    start_time = time.time()
    records: List[Dict] = []

    with open(out_path, "w", encoding="utf-8") as fout:
        for model in models:
            detail = args.detail if model.endswith("gpt-4o") else None
            for prompt in PROMPTS:
                for clip in clips:
                    count += 1
                    video_id = clip["video_id"]
                    frame_indices = clip["frame_indices"]
                    frames_dir = frames_root / video_id
                    image_b64s = []
                    
                    # Handle frame limit for InternVL3 (max 4 frames)
                    current_indices = frame_indices
                    if "internvl3" in model.lower() and len(frame_indices) > 4:
                        # Downsample to 4 frames, ensuring we include the last one (current moment)
                        # For 16 frames: 0, 5, 10, 15
                        step = (len(frame_indices) - 1) / 3
                        current_indices = [frame_indices[int(i * step)] for i in range(4)]
                        # Ensure the last one is the actual last one
                        current_indices[-1] = frame_indices[-1]
                        
                    for idx in current_indices:
                        frame_path = frames_dir / f"frame_{int(idx):05d}.jpg"
                        image_b64s.append(_encode_image(frame_path, args.frame_size))

                    prompt_text = prompt["text"]
                    if "internvl3" in model.lower():
                        prompt_text = prompt_text.replace("16 dashcam frames", "4 dashcam frames")

                    messages = _build_messages(prompt_text, image_b64s, detail)

                    t0 = time.time()
                    error = None
                    raw_response = ""
                    usage = {}
                    logprobs = None
                    lp_yes = None
                    lp_no = None
                    logprobs_raw = None
                    score = None
                    try:
                        raw_response, usage, logprobs = _call_model(
                            client,
                            model=model,
                            messages=messages,
                            max_retries=args.max_retries,
                            retry_delay=args.retry_delay,
                        )
                        lp_yes, lp_no, logprobs_raw = _extract_verdict_logprobs(logprobs, raw_response)
                        score = _compute_p_yes(lp_yes, lp_no)
                        
                        # Fallback: if logprob score failed (e.g. Gemini/Claude) OR looks inconsistent
                        # parse the text verdict directly
                        import re
                        verdict_match = re.search(r"\[Verdict\]:\s*(\w+)", raw_response, re.IGNORECASE)
                        text_score = None
                        if verdict_match:
                            v_str = verdict_match.group(1).lower()
                            if v_str in ["yes", "y"]:
                                text_score = 1.0
                            elif v_str in ["no", "n"]:
                                text_score = 0.0
                        
                        if score is None:
                            score = text_score
                        elif text_score is not None:
                            # Sanity check: if logprob says 0 but text says YES, trust text (likely token mismatch bug)
                            if text_score == 1.0 and score < 0.01:
                                score = 1.0
                            elif text_score == 0.0 and score > 0.99:
                                score = 0.0
                    except Exception as exc:  # pragma: no cover
                        error = str(exc)

                    latency_s = time.time() - t0
                    time_of_event = clip.get("time_of_event")
                    t_seconds = clip.get("t_seconds")
                    time_to_event = None
                    if time_of_event is not None and t_seconds is not None:
                        time_to_event = time_of_event - t_seconds

                    record = {
                        **clip,
                        "requested_time_to_event": clip.get("requested_time_to_event"),
                        "time_to_event": time_to_event,
                        "model_id": model,
                        "prompt_id": prompt["id"],
                        "prompt_name": prompt["name"],
                        "prompt_text": prompt["text"],
                        "logprob_score": score,
                        "raw_response": raw_response,
                        "logprobs_raw": logprobs_raw,
                        "latency_s": latency_s,
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                        "error": error,
                    }
                    records.append(record)
                    fout.write(json.dumps(record) + "\n")

                    if count % 5 == 0 or count == total:
                        elapsed = max(time.time() - start_time, 1e-6)
                        clips_per_s = count / elapsed
                        remaining = total - count
                        eta_s = remaining / clips_per_s if clips_per_s > 0 else None
                        print(
                            f"[{count}/{total}] model={model} prompt={prompt['id']} "
                            f"video_id={video_id} elapsed_s={elapsed:.1f} "
                            f"eta_s={eta_s:.1f}" if eta_s else "",
                            file=sys.stderr,
                            flush=True,
                        )

    _summarize(records, models, PROMPTS)
    _plot_histogram(records, models, PROMPTS, plot_path)
    print(f"Wrote prompt experiment results: {out_path}", file=sys.stderr, flush=True)
    print(f"Wrote prompt comparison plot: {plot_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
