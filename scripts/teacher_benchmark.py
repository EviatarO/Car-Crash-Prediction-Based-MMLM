import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# #region agent log
def _debug_log(message: str, data: Dict, hypothesis_id: str) -> None:
    payload = {
        "runId": os.environ.get("SLURM_JOB_ID", "manual"),
        "hypothesisId": hypothesis_id,
        "location": "teacher_benchmark.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open("/home/eprojuser011/.cursor/debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


# #endregion

PILOT_VIDEO_IDS = [
    "00000",
    "00007",
    "00013",
    "00016",
    "00017",
    "00027",
    "00041",
    "00042",
    "00063",
    "00084",
    "00111",
    "00114",
    "00127",
    "00130",
    "00133",
    "00138",
    "00144",
    "00148",
    "00155",
    "00157",
]

DEFAULT_MODELS = [
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4",
]

SCORE_RE = re.compile(r"\[Score\]\s*:\s*([0-9]*\.?[0-9]+)")
REASON_RE = re.compile(r"\[Collision_Reasoning\]\s*:\s*(.*)", re.S)


def _load_manifest(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _iter_pilot_clips(manifest: Sequence[Dict], pilot_ids: Sequence[str]) -> Iterable[Dict]:
    pilot_set = set(pilot_ids)
    for rec in manifest:
        if rec.get("video_id") in pilot_set:
            yield rec


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


def _build_messages(
    prompt: str,
    image_b64s: Sequence[str],
    detail: Optional[str],
) -> List[Dict]:
    content = [{"type": "text", "text": prompt}]
    for b64 in image_b64s:
        image_url = {"url": b64}
        if detail:
            image_url["detail"] = detail
        content.append({"type": "image_url", "image_url": image_url})
    return [{"role": "user", "content": content}]


def _parse_response(text: str) -> Tuple[Optional[str], Optional[float]]:
    reasoning = None
    score = None
    reason_match = REASON_RE.search(text)
    if reason_match:
        reasoning = reason_match.group(1).strip()
    score_match = SCORE_RE.search(text)
    if score_match:
        try:
            score = float(score_match.group(1))
        except ValueError:
            score = None
    return reasoning, score


def _log_progress(log_path: Path, payload: Dict):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


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
        except Exception as exc:  # pragma: no cover - network failure path
            last_exc = exc
            time.sleep(retry_delay * attempt)
    raise RuntimeError(f"OpenRouter call failed after {max_retries} attempts: {last_exc}")


def main():
    _debug_log(
        "startup",
        {
            "cwd": os.getcwd(),
            "sys_path_0": sys.path[0],
            "sys_path": sys.path[:5],
        },
        "H1",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--pilot_videos", default=",".join(PILOT_VIDEO_IDS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--detail", default="low")
    parser.add_argument("--frame_size", type=int, default=256)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=2.0)
    parser.add_argument("--progress_jsonl", default=None)
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
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "MMLM_Teacher_Benchmark"),
        },
    )

    manifest = _load_manifest(Path(args.manifest))
    pilot_ids = [vid.strip().zfill(5) for vid in args.pilot_videos.split(",") if vid.strip()]
    clips = list(_iter_pilot_clips(manifest, pilot_ids))
    if not clips:
        raise RuntimeError("No pilot clips found in manifest.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress_jsonl) if args.progress_jsonl else out_dir / "progress.jsonl"

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    frames_root = Path(args.frames_root)

    total = len(models) * len(clips)
    count = 0
    start_time = time.time()

    try:
        from prompts.templates import TEACHER_BLIND_PROMPT  # pylint: disable=import-outside-toplevel
        _debug_log("prompt_import_ok", {}, "H2")
    except Exception as exc:
        _debug_log("prompt_import_failed", {"error": str(exc)}, "H2")
        raise

    for model in models:
        model_tag = model.split("/")[-1].replace(":", "_")
        out_path = out_dir / f"{model_tag}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fout:
            for clip in clips:
                count += 1
                video_id = clip["video_id"]
                frame_indices = clip["frame_indices"]
                frames_dir = frames_root / video_id
                image_b64s = []
                for idx in frame_indices:
                    frame_path = frames_dir / f"frame_{int(idx):05d}.jpg"
                    image_b64s.append(_encode_image(frame_path, args.frame_size))

                detail = args.detail if model.endswith("gpt-4o") else None
                messages = _build_messages(TEACHER_BLIND_PROMPT, image_b64s, detail)

                t0 = time.time()
                error = None
                raw_response = ""
                usage = {}
                reasoning = None
                score = None
                try:
                    raw_response, usage = _call_model(
                        client,
                        model=model,
                        messages=messages,
                        max_retries=args.max_retries,
                        retry_delay=args.retry_delay,
                    )
                    reasoning, score = _parse_response(raw_response)
                except Exception as exc:  # pragma: no cover
                    error = str(exc)

                latency_s = time.time() - t0
                record = {
                    **clip,
                    "model_id": model,
                    "prompt_text": TEACHER_BLIND_PROMPT,
                    "collision_reasoning": reasoning,
                    "score": score,
                    "raw_response": raw_response,
                    "latency_s": latency_s,
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "error": error,
                }
                fout.write(json.dumps(record) + "\n")

                if count % 5 == 0 or count == total:
                    elapsed = max(time.time() - start_time, 1e-6)
                    clips_per_s = count / elapsed
                    remaining = total - count
                    eta_s = remaining / clips_per_s if clips_per_s > 0 else None
                    _log_progress(
                        progress_path,
                        {
                            "count": count,
                            "total": total,
                            "elapsed_s": elapsed,
                            "clips_per_s": clips_per_s,
                            "eta_s": eta_s,
                            "model": model,
                            "video_id": video_id,
                        },
                    )
                    print(
                        f"[{count}/{total}] model={model} video_id={video_id} "
                        f"elapsed_s={elapsed:.1f} eta_s={eta_s:.1f}" if eta_s else "",
                        file=sys.stderr,
                        flush=True,
                    )

        print(f"Wrote benchmark results: {out_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
