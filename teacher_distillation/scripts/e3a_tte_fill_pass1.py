"""E3a TTE-fill Stage 2: pass-1 v6_balanced on the 178 new variants.

Reads outputs/prompt_bakeoff/e3a_tte_fill/extraction_log.json and runs
PROMPT_G_OPT_v6_balanced at native 1280x720 / detail=high through Gemini 3.1 Pro
on every variant. Resume-safe by composite key (video_id, horizon_label).

Operational discipline (from the approved plan):
  * 10-min wall-clock heartbeat with cum cost, mean cost, last 3 verdicts, ETA
  * per-clip cost anomaly skip: clip_cost > max(3*mu, $0.25) -> write error rec,
    skip (verdict=null so no recovery), 3 anomalies in 10 clips -> stop-and-ask
  * stop-and-ask: 3 same-type API errors, 5 malformed JSON, 402/401, budget
    ceiling $25
"""
from __future__ import annotations

import base64
import concurrent.futures
import importlib.util
import json
import os
import re
import sys
import time
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from PIL import Image

try:
    from openai import (
        APIConnectionError as _OAIConnectionError,
        APITimeoutError as _OAITimeoutError,
        OpenAI,
        RateLimitError as _OAIRateLimitError,
        AuthenticationError as _OAIAuthError,
    )
except ImportError:
    from openai import OpenAI  # type: ignore[no-redef]
    _OAIConnectionError = Exception
    _OAITimeoutError = Exception
    _OAIRateLimitError = Exception
    _OAIAuthError = Exception

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---- Config ----
MODEL_SLUG  = "google/gemini-3.1-pro-preview"
PRICE_IN    = 2.00
PRICE_OUT   = 12.00
TEMPERATURE = 0.1

DEFAULT_TIMEOUT       = 90.0
MAX_RETRIES           = 2
RETRY_DELAY           = 5.0
CLIP_BUDGET_SECS      = 300
INTER_CALL_DELAY      = 1.5
PROMPT_TOKEN_HARD_CAP = 100_000

# Operational thresholds
HEARTBEAT_SECS         = 600           # 10 min
COST_ANOMALY_MULT      = 3.0           # 3x running mean
COST_ANOMALY_ABS       = 0.25          # USD floor for anomaly
ANOMALY_WINDOW         = 10
ANOMALY_WINDOW_LIMIT   = 3
STAGE_BUDGET_CEILING   = 25.00         # USD hard stop
MAX_CONSEC_SAME_ERR    = 3
MAX_CONSEC_MALFORMED   = 5

DEFAULT_FRAMES_ROOT = REPO_ROOT / "dataset" / "train"
PROMPT_FILE         = REPO_ROOT / "prompts" / "PROMPT_G_OPT_v6_balanced.py"
EXTRACTION_LOG      = REPO_ROOT / "outputs" / "prompt_bakeoff" / "e3a_tte_fill" / "extraction_log.json"

OUT_JSONL = REPO_ROOT / "outputs" / "prompt_bakeoff" / "e3a_tte_fill" / "pass1.jsonl"
STOP_FILE = REPO_ROOT / "outputs" / "prompt_bakeoff" / "e3a_tte_fill" / "STOP_REASON.json"


# ---- Inlined utilities (proven in v11_resampled_pass1.py) ----

def _normalize_verdict(value) -> Optional[str]:
    if value is None: return None
    text = str(value).strip().upper()
    return text if text in {"YES", "NO"} else None


def _normalize_confidence(value) -> Optional[str]:
    if value is None: return None
    text = str(value).strip().upper()
    return text if text in {"HIGH", "MEDIUM", "LOW"} else None


def _encode_image(path: Path, frame_size: int) -> str:
    if path.exists():
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", (frame_size or 64, frame_size or 64), color=(0, 0, 0))
    if frame_size and img.size != (frame_size, frame_size):
        img = img.resize((frame_size, frame_size))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _build_messages(prompt: str, image_b64s: Sequence[str], detail: str = "low") -> List[Dict]:
    content: List[Dict] = [{"type": "text", "text": prompt}]
    for b64 in image_b64s:
        image_url: Dict = {"url": b64}
        if detail:
            image_url["detail"] = detail
        content.append({"type": "image_url", "image_url": image_url})
    return [{"role": "user", "content": content}]


def _extract_json_object(raw: str) -> Optional[Dict]:
    if not raw: return None
    try:
        obj = json.loads(raw); return obj if isinstance(obj, dict) else None
    except Exception: pass
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE)
    if fenced:
        try:
            obj = json.loads(fenced.group(1)); return obj if isinstance(obj, dict) else None
        except Exception: pass
    start = raw.find("{"); end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(raw[start: end + 1]); return obj if isinstance(obj, dict) else None
        except Exception: pass
    return None


def _parse_response(raw: str) -> Tuple[Optional[Dict], Optional[str]]:
    parsed = _extract_json_object(raw)
    if parsed is None: return None, None
    parsed["collision_verdict"] = _normalize_verdict(parsed.get("collision_verdict"))
    parsed["confidence"] = _normalize_confidence(parsed.get("confidence"))
    return parsed, parsed["collision_verdict"]


def _call_model(client, model, messages, timeout, max_retries, retry_delay,
                temperature=0.1, max_tokens=8192) -> Tuple[str, Dict, Optional[str]]:
    """Returns (text, usage, error_type_or_None). error_type is one of:
    'timeout', 'rate_limit', 'connection', 'auth', 'other'. Raises only on auth or full failure."""
    last_exc: Optional[Exception] = None
    last_kind: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                timeout=timeout, max_tokens=max_tokens,
            )
            text = response.choices[0].message.content if response.choices else ""
            usage = (response.usage.model_dump() if hasattr(response, "usage") and response.usage else {})
            return text or "", usage, None
        except _OAIAuthError as exc:
            # Hard-stop auth/credit issue. Don't retry.
            raise RuntimeError(f"AUTH_OR_CREDIT_ERROR: {exc}") from exc
        except _OAITimeoutError as exc:
            last_exc, last_kind = exc, "timeout"
        except _OAIRateLimitError as exc:
            last_exc, last_kind = exc, "rate_limit"
        except _OAIConnectionError as exc:
            last_exc, last_kind = exc, "connection"
        except Exception as exc:
            txt = str(exc).lower()
            if "402" in txt or "401" in txt or "credit" in txt or "balance" in txt:
                raise RuntimeError(f"AUTH_OR_CREDIT_ERROR: {exc}") from exc
            last_exc, last_kind = exc, "other"
        wait = retry_delay * (2 ** (attempt - 1))
        print(f"  [retry {attempt}/{max_retries}] {last_kind} -- waiting {wait:.1f}s", flush=True)
        if attempt < max_retries:
            time.sleep(wait)
    return "", {}, last_kind or "other"


def _calc_cost(usage: Dict, price_in: float, price_out: float) -> float:
    in_tok = usage.get("prompt_tokens", 0) or 0
    out_tok = usage.get("completion_tokens", 0) or 0
    return in_tok * price_in / 1_000_000 + out_tok * price_out / 1_000_000


def _load_clip_frames(folder: Path, indices: List[int], frame_size: int) -> List[str]:
    return [_encode_image(folder / f"frame_{i:05d}.jpg", frame_size) for i in indices]


def _load_prompt(path: Path, var_name: str) -> str:
    spec = importlib.util.spec_from_file_location(f"_p_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, var_name)


def _load_jsonl_keyed(path: Path) -> Dict[Tuple[str, str], Dict]:
    out: Dict[Tuple[str, str], Dict] = {}
    if not path.exists(): return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        try:
            rec = json.loads(line)
            out[(rec["video_id"], rec["horizon_label"])] = rec
        except Exception: pass
    return out


def _append(path: Path, rec: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _call_with_budget(client, prompt, frames_folder: Path) -> Tuple[str, Dict, float, Optional[str]]:
    def _inner():
        indices = list(range(1, 17))
        b64s = _load_clip_frames(frames_folder, indices, frame_size=0)
        messages = _build_messages(prompt, b64s, detail="high")
        t0 = time.time()
        raw, usage, err_kind = _call_model(
            client, MODEL_SLUG, messages,
            timeout=DEFAULT_TIMEOUT, max_retries=MAX_RETRIES,
            retry_delay=RETRY_DELAY, temperature=TEMPERATURE,
        )
        latency = time.time() - t0
        prompt_tok = usage.get("prompt_tokens", 0) if usage else 0
        if prompt_tok and prompt_tok > PROMPT_TOKEN_HARD_CAP:
            raise RuntimeError(f"Prompt token count {prompt_tok} > hard cap {PROMPT_TOKEN_HARD_CAP}")
        return raw, usage, latency, err_kind

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_inner)
        try:
            return future.result(timeout=CLIP_BUDGET_SECS)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise RuntimeError(f"Wall-clock budget {CLIP_BUDGET_SECS}s exceeded")


def _write_stop(reason: str, payload: Dict) -> None:
    STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STOP_FILE.write_text(json.dumps({"stage": "pass1", "reason": reason, **payload}, indent=2),
                         encoding="utf-8")


def _fmt_eta(remaining: int, mean_lat: float) -> str:
    secs = int(remaining * mean_lat)
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main() -> None:
    prompt = _load_prompt(PROMPT_FILE, "PROMPT_G_OPT_v6_balanced")
    print(f"Loaded v6_balanced ({len(prompt)} chars)")

    plan = json.loads(EXTRACTION_LOG.read_text(encoding="utf-8"))
    print(f"Loaded extraction_log.json: {len(plan)} variants")

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title":      os.environ.get("OPENROUTER_APP_TITLE", "MMLM_e3a_tte_fill_pass1"),
        },
    )

    existing = _load_jsonl_keyed(OUT_JSONL)
    print(f"Already done: {len(existing)} variants\n")

    # State for ops control
    cum_cost = 0.0
    n_cost_samples = 0
    n_anomalies = 0
    anomaly_window: Deque[bool] = deque(maxlen=ANOMALY_WINDOW)
    consec_err_kind: Optional[str] = None
    consec_err_n = 0
    consec_malformed = 0
    last_heartbeat_t = time.time()
    recent_verdicts: Deque[str] = deque(maxlen=3)
    latencies: List[float] = []
    n_done_call = 0
    n_errors = 0
    n_correct = n_wrong = 0
    t_stage_start = time.time()

    print("=" * 65)
    print(f"Stage 2 START: pass-1 v6_balanced on {len(plan)} variants")
    print(f"  Budget ceiling: ${STAGE_BUDGET_CEILING:.2f}")
    print(f"  Heartbeat:      every {HEARTBEAT_SECS//60} min")
    print("=" * 65)

    for idx, p in enumerate(plan, start=1):
        vid = p["video_id"]
        horizon = p["new_horizon_label"]
        gt = p["gt_verdict"]
        frames_subdir = p["frames_subdir"]
        key = (vid, horizon)

        if key in existing and existing[key].get("verdict") is not None:
            r = existing[key]
            outcome = "MATCH" if r["verdict"] == gt else "MISMATCH"
            print(f"[{idx:3d}/{len(plan)}] {vid} {horizon} -- skip (verdict={r['verdict']} {outcome})")
            cum_cost += r.get("cost_usd", 0.0) or 0.0
            if r["verdict"] == gt: n_correct += 1
            else: n_wrong += 1
            continue

        # Skip records that previously failed cost-anomaly (so a re-run doesn't re-spend)
        if key in existing and existing[key].get("error") == "skipped_cost_anomaly":
            print(f"[{idx:3d}/{len(plan)}] {vid} {horizon} -- skip (prior cost-anomaly)")
            continue

        frames_folder = DEFAULT_FRAMES_ROOT / frames_subdir
        print(f"[{idx:3d}/{len(plan)}] {vid} {horizon} GT={gt} t_new={p['t_new']}", end="  ", flush=True)

        if INTER_CALL_DELAY > 0:
            time.sleep(INTER_CALL_DELAY)

        clip_start = time.time()
        rec: Dict = {
            "video_id": vid, "horizon_label": horizon, "gt_verdict": gt,
            "t_new": p["t_new"], "horizon_s": p["horizon_s"],
            "frames_subdir": frames_subdir,
            "prompt_name": "PROMPT_G_OPT_v6_balanced",
            "resolution": "native_1280x720", "stride": 4, "detail": "high",
        }
        try:
            raw, usage, latency, err_kind = _call_with_budget(client, prompt, frames_folder)
            cost = _calc_cost(usage, PRICE_IN, PRICE_OUT)

            # ---- Cost-anomaly guard ----
            mu = (cum_cost / n_cost_samples) if n_cost_samples > 0 else 0.0
            threshold = max(COST_ANOMALY_MULT * mu, COST_ANOMALY_ABS)
            if n_cost_samples >= 5 and cost > threshold:
                # spent but skip-on-result
                n_anomalies += 1
                anomaly_window.append(True)
                print(f"[ANOMALY] cost=${cost:.4f} > thr=${threshold:.4f} (mu=${mu:.4f}) -- skipping verdict")
                rec.update({
                    "verdict": None, "reasoning": None, "full_json": {}, "raw": raw,
                    "usage": usage, "cost_usd": cost, "latency_s": round(latency, 2),
                    "error": "skipped_cost_anomaly", "anomaly_ratio": (cost / mu) if mu else None,
                })
                _append(OUT_JSONL, rec)
                cum_cost += cost
                if sum(1 for x in anomaly_window if x) >= ANOMALY_WINDOW_LIMIT:
                    _write_stop("anomaly_threshold_hit",
                                {"recent_window": list(anomaly_window), "cum_cost": cum_cost,
                                 "next_idx": idx + 1})
                    print()
                    print(f"  STOP-AND-ASK: {ANOMALY_WINDOW_LIMIT} cost anomalies within last "
                          f"{ANOMALY_WINDOW} clips. Halting Stage 2.")
                    sys.exit(2)
                continue

            anomaly_window.append(False)
            n_cost_samples += 1
            cum_cost += cost
            latencies.append(latency)

            if err_kind is None:
                consec_err_kind = None; consec_err_n = 0

            parsed, verdict = _parse_response(raw)
            if parsed is None:
                consec_malformed += 1
                if consec_malformed >= MAX_CONSEC_MALFORMED:
                    _write_stop("consecutive_malformed", {"count": consec_malformed, "next_idx": idx + 1})
                    print()
                    print(f"  STOP-AND-ASK: {consec_malformed} consecutive malformed JSON responses.")
                    sys.exit(2)
            else:
                consec_malformed = 0

            reasoning = parsed.get("verdict_reasoning") if parsed else None
            outcome = "MATCH" if verdict == gt else ("MISMATCH" if verdict else "NO_VERDICT")
            print(f"verdict={verdict or '??':3s}  {outcome}  cost=${cost:.4f}  "
                  f"in={usage.get('prompt_tokens')}  out={usage.get('completion_tokens')}  {latency:.1f}s")
            if verdict == gt: n_correct += 1
            elif verdict is not None: n_wrong += 1
            recent_verdicts.append(f"{vid}/{horizon}:{verdict or '??'}/{gt}")

            rec.update({
                "verdict": verdict, "reasoning": reasoning,
                "full_json": parsed or {}, "raw": raw, "usage": usage,
                "cost_usd": cost, "latency_s": round(latency, 2),
                "error": None if verdict is not None else "no_verdict_parsed",
            })
        except RuntimeError as e:
            msg = str(e)
            if msg.startswith("AUTH_OR_CREDIT_ERROR"):
                _write_stop("auth_or_credit_error", {"detail": msg, "next_idx": idx})
                print(f"[STOP] {msg}")
                sys.exit(2)
            print(f"[ERR] {msg}")
            n_errors += 1
            # Track consecutive errors (RuntimeError covers timeout-budget too)
            kind = "other"
            if "Wall-clock budget" in msg: kind = "wallclock"
            if consec_err_kind == kind: consec_err_n += 1
            else: consec_err_kind = kind; consec_err_n = 1
            if consec_err_n >= MAX_CONSEC_SAME_ERR:
                _write_stop("consecutive_api_errors",
                            {"kind": kind, "count": consec_err_n, "next_idx": idx + 1})
                print(f"  STOP-AND-ASK: {consec_err_n} consecutive '{kind}' errors. Halting Stage 2.")
                sys.exit(2)
            rec.update({"verdict": None, "reasoning": None, "full_json": {}, "raw": "",
                        "usage": {}, "cost_usd": 0.0,
                        "latency_s": round(time.time() - clip_start, 2), "error": msg})
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[ERR] {msg}")
            n_errors += 1
            rec.update({"verdict": None, "reasoning": None, "full_json": {}, "raw": "",
                        "usage": {}, "cost_usd": 0.0,
                        "latency_s": round(time.time() - clip_start, 2), "error": msg})

        _append(OUT_JSONL, rec)
        n_done_call += 1

        # ---- Budget ceiling ----
        if cum_cost > STAGE_BUDGET_CEILING:
            _write_stop("budget_ceiling_hit",
                        {"cum_cost": cum_cost, "ceiling": STAGE_BUDGET_CEILING, "next_idx": idx + 1})
            print(f"  STOP-AND-ASK: cum cost ${cum_cost:.2f} > ceiling ${STAGE_BUDGET_CEILING:.2f}.")
            sys.exit(2)

        # ---- Heartbeat ----
        now = time.time()
        if now - last_heartbeat_t >= HEARTBEAT_SECS:
            mean_cost = (cum_cost / n_cost_samples) if n_cost_samples else 0.0
            mean_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
            remaining = len(plan) - idx
            print()
            print(f"  >>> HEARTBEAT [{idx}/{len(plan)}]  cum=${cum_cost:.2f}  "
                  f"mean=${mean_cost:.4f}/clip  anomalies={n_anomalies}  errors={n_errors}  "
                  f"recent={list(recent_verdicts)}  ETA={_fmt_eta(remaining, mean_lat)}")
            print()
            last_heartbeat_t = now

    wall = time.time() - t_stage_start
    n_total_correct = n_correct
    n_seen = n_correct + n_wrong
    print()
    print("=" * 65)
    print("Stage 2 COMPLETE")
    print(f"  Rows written: {len(plan)}")
    print(f"  pass-1 verdict accuracy: {n_total_correct}/{n_seen} "
          f"= {(n_total_correct/n_seen):.1%}" if n_seen else "  (no verdicts)")
    print(f"  Errors:       {n_errors}  (skipped_cost_anomaly: {n_anomalies})")
    print(f"  Cum cost:     ${cum_cost:.4f}   "
          f"(stage mean: ${(cum_cost/n_cost_samples):.4f}/clip)" if n_cost_samples else "")
    print(f"  Wall time:    {wall/60:.1f} min")
    print(f"  Output:       {OUT_JSONL}")
    print("=" * 65)


if __name__ == "__main__":
    main()
