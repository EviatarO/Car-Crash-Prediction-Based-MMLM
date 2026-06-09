"""E3a TTE-fill Stage 3: v7.1 recovery debate on pass-1 mispredictions.

Reads pass1.jsonl; for each row where verdict != gt_verdict (and verdict is not
None), routes to the correct recovery prompt:
  - gt='YES', verdict='NO' -> PROMPT_G_OPT_v7_1_TP_RECOVERY
  - gt='NO', verdict='YES' -> PROMPT_G_OPT_v7_1_TN_RECOVERY

Resume-safe by (video_id, horizon_label, recovery_prompt). Same heartbeat /
cost-anomaly / stop-and-ask discipline as Stage 2.
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

HEARTBEAT_SECS         = 600
COST_ANOMALY_MULT      = 3.0
COST_ANOMALY_ABS       = 0.25
ANOMALY_WINDOW         = 10
ANOMALY_WINDOW_LIMIT   = 3
STAGE_BUDGET_CEILING   = 10.00
MAX_CONSEC_SAME_ERR    = 3
MAX_CONSEC_MALFORMED   = 5

DEFAULT_FRAMES_ROOT = REPO_ROOT / "dataset" / "train"
PROMPT_DIR          = REPO_ROOT / "prompts"
PASS1_JSONL         = REPO_ROOT / "outputs" / "prompt_bakeoff" / "e3a_tte_fill" / "pass1.jsonl"
OUT_JSONL           = REPO_ROOT / "outputs" / "prompt_bakeoff" / "e3a_tte_fill" / "recovery.jsonl"
STOP_FILE           = REPO_ROOT / "outputs" / "prompt_bakeoff" / "e3a_tte_fill" / "STOP_REASON_pass2.json"


# ---- Inlined utilities (same as pass1) ----

def _normalize_verdict(value):
    if value is None: return None
    text = str(value).strip().upper()
    return text if text in {"YES", "NO"} else None


def _encode_image(path: Path, frame_size: int) -> str:
    if path.exists():
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", (frame_size or 64, frame_size or 64), color=(0, 0, 0))
    if frame_size and img.size != (frame_size, frame_size):
        img = img.resize((frame_size, frame_size))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _build_messages(prompt, image_b64s, detail="low"):
    content = [{"type": "text", "text": prompt}]
    for b64 in image_b64s:
        d = {"url": b64}
        if detail: d["detail"] = detail
        content.append({"type": "image_url", "image_url": d})
    return [{"role": "user", "content": content}]


def _extract_json_object(raw):
    if not raw: return None
    try:
        obj = json.loads(raw); return obj if isinstance(obj, dict) else None
    except Exception: pass
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE)
    if fenced:
        try:
            obj = json.loads(fenced.group(1)); return obj if isinstance(obj, dict) else None
        except Exception: pass
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            obj = json.loads(raw[s:e+1]); return obj if isinstance(obj, dict) else None
        except Exception: pass
    return None


def _parse_response(raw):
    parsed = _extract_json_object(raw)
    if parsed is None: return None, None
    parsed["collision_verdict"] = _normalize_verdict(parsed.get("collision_verdict"))
    return parsed, parsed["collision_verdict"]


def _call_model(client, model, messages, timeout, max_retries, retry_delay, temperature=0.1, max_tokens=8192):
    last_kind = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                timeout=timeout, max_tokens=max_tokens,
            )
            text = response.choices[0].message.content if response.choices else ""
            usage = response.usage.model_dump() if hasattr(response, "usage") and response.usage else {}
            return text or "", usage, None
        except _OAIAuthError as exc:
            raise RuntimeError(f"AUTH_OR_CREDIT_ERROR: {exc}") from exc
        except _OAITimeoutError: last_kind = "timeout"
        except _OAIRateLimitError: last_kind = "rate_limit"
        except _OAIConnectionError: last_kind = "connection"
        except Exception as exc:
            txt = str(exc).lower()
            if "402" in txt or "401" in txt or "credit" in txt or "balance" in txt:
                raise RuntimeError(f"AUTH_OR_CREDIT_ERROR: {exc}") from exc
            last_kind = "other"
        wait = retry_delay * (2 ** (attempt - 1))
        print(f"  [retry {attempt}/{max_retries}] {last_kind} -- waiting {wait:.1f}s", flush=True)
        if attempt < max_retries:
            time.sleep(wait)
    return "", {}, last_kind or "other"


def _calc_cost(usage, p_in, p_out):
    return (usage.get("prompt_tokens", 0) or 0) * p_in / 1_000_000 + \
           (usage.get("completion_tokens", 0) or 0) * p_out / 1_000_000


def _load_clip_frames(folder, indices, frame_size):
    return [_encode_image(folder / f"frame_{i:05d}.jpg", frame_size) for i in indices]


def _load_prompt(path, var_name):
    spec = importlib.util.spec_from_file_location(f"_p_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return getattr(mod, var_name)


def _load_jsonl(path):
    if not path.exists(): return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_recovery_keyed(path) -> Dict[Tuple[str, str, str], Dict]:
    if not path.exists(): return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        try:
            r = json.loads(line)
            out[(r["video_id"], r["horizon_label"], r["recovery_prompt"])] = r
        except Exception: pass
    return out


def _append(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _call_with_budget(client, prompt, folder):
    def _inner():
        indices = list(range(1, 17))
        b64s = _load_clip_frames(folder, indices, frame_size=0)
        messages = _build_messages(prompt, b64s, detail="high")
        t0 = time.time()
        raw, usage, err_kind = _call_model(
            client, MODEL_SLUG, messages,
            timeout=DEFAULT_TIMEOUT, max_retries=MAX_RETRIES,
            retry_delay=RETRY_DELAY, temperature=TEMPERATURE,
        )
        return raw, usage, time.time() - t0, err_kind
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        f = ex.submit(_inner)
        try: return f.result(timeout=CLIP_BUDGET_SECS)
        except concurrent.futures.TimeoutError:
            f.cancel(); raise RuntimeError(f"Wall-clock budget {CLIP_BUDGET_SECS}s exceeded")


def _write_stop(reason, payload):
    STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STOP_FILE.write_text(json.dumps({"stage": "pass2", "reason": reason, **payload}, indent=2),
                         encoding="utf-8")


def _fmt_eta(remaining, mean_lat):
    secs = int(remaining * mean_lat); h, r = divmod(secs, 3600); m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main():
    pass1 = _load_jsonl(PASS1_JSONL)
    if not pass1:
        raise SystemExit(f"No pass-1 records at {PASS1_JSONL}")

    # Decide who needs recovery
    queue: List[Tuple[Dict, str, str]] = []  # (pass1_rec, recovery_prompt_name, prompt_text)
    tn_prompt = _load_prompt(PROMPT_DIR / "PROMPT_G_OPT_v7_1_TN_RECOVERY.py", "PROMPT_G_OPT_v7_1_TN_RECOVERY")
    tp_prompt = _load_prompt(PROMPT_DIR / "PROMPT_G_OPT_v7_1_TP_RECOVERY.py", "PROMPT_G_OPT_v7_1_TP_RECOVERY")

    for r in pass1:
        v = r.get("verdict"); gt = r.get("gt_verdict")
        if v is None: continue
        if v == gt: continue
        if v == "YES" and gt == "NO":
            queue.append((r, "PROMPT_G_OPT_v7_1_TN_RECOVERY", tn_prompt))
        elif v == "NO" and gt == "YES":
            queue.append((r, "PROMPT_G_OPT_v7_1_TP_RECOVERY", tp_prompt))

    print(f"Pass-1 rows:                       {len(pass1)}")
    print(f"Need recovery (verdict != gt):    {len(queue)}")
    tp_n = sum(1 for _, n, _ in queue if "TP_" in n)
    tn_n = sum(1 for _, n, _ in queue if "TN_" in n)
    print(f"  TP_RECOVERY (FN clips):  {tp_n}")
    print(f"  TN_RECOVERY (FP clips):  {tn_n}")

    if not queue:
        print("\nNothing to do.")
        return

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title":      os.environ.get("OPENROUTER_APP_TITLE", "MMLM_e3a_tte_fill_pass2"),
        },
    )

    existing = _load_recovery_keyed(OUT_JSONL)
    print(f"Already done: {len(existing)} recovery records\n")

    cum_cost = 0.0; n_cost_samples = 0; n_anomalies = 0
    anomaly_window: Deque[bool] = deque(maxlen=ANOMALY_WINDOW)
    consec_err_kind = None; consec_err_n = 0; consec_malformed = 0
    last_heartbeat_t = time.time()
    recent: Deque[str] = deque(maxlen=3)
    latencies = []; n_errors = 0; n_fixed = n_stillwrong = 0
    t_start = time.time()

    print("=" * 65)
    print(f"Stage 3 START: v7.1 recovery on {len(queue)} clips")
    print(f"  Budget ceiling: ${STAGE_BUDGET_CEILING:.2f}")
    print("=" * 65)

    for idx, (p1, prompt_name, prompt_text) in enumerate(queue, start=1):
        vid = p1["video_id"]; horizon = p1["horizon_label"]; gt = p1["gt_verdict"]
        frames_subdir = p1["frames_subdir"]
        key = (vid, horizon, prompt_name)

        if key in existing and existing[key].get("recovery_verdict") is not None:
            r = existing[key]
            outcome = "FIXED" if r["recovery_verdict"] == gt else "still-wrong"
            print(f"[{idx:3d}/{len(queue)}] {vid} {horizon} -- skip (rv={r['recovery_verdict']} {outcome})")
            cum_cost += r.get("cost_usd", 0.0) or 0.0
            if r["recovery_verdict"] == gt: n_fixed += 1
            else: n_stillwrong += 1
            continue
        if key in existing and existing[key].get("error") == "skipped_cost_anomaly":
            print(f"[{idx:3d}/{len(queue)}] {vid} {horizon} -- skip (prior cost-anomaly)")
            continue

        folder = DEFAULT_FRAMES_ROOT / frames_subdir
        print(f"[{idx:3d}/{len(queue)}] {vid} {horizon} GT={gt} pass1={p1['verdict']} -> {prompt_name}",
              end="  ", flush=True)
        if INTER_CALL_DELAY > 0: time.sleep(INTER_CALL_DELAY)

        clip_start = time.time()
        rec: Dict = {
            "video_id": vid, "horizon_label": horizon, "recovery_prompt": prompt_name,
            "gt_verdict": gt, "pass1_verdict": p1["verdict"], "t_new": p1.get("t_new"),
            "frames_subdir": frames_subdir, "resolution": "native_1280x720", "detail": "high",
        }

        try:
            raw, usage, latency, err_kind = _call_with_budget(client, prompt_text, folder)
            cost = _calc_cost(usage, PRICE_IN, PRICE_OUT)

            mu = (cum_cost / n_cost_samples) if n_cost_samples > 0 else 0.0
            threshold = max(COST_ANOMALY_MULT * mu, COST_ANOMALY_ABS)
            if n_cost_samples >= 5 and cost > threshold:
                n_anomalies += 1; anomaly_window.append(True)
                print(f"[ANOMALY] cost=${cost:.4f} > thr=${threshold:.4f} -- skipping verdict")
                rec.update({"recovery_verdict": None, "recovery_reasoning": None, "full_json": {},
                            "raw": raw, "usage": usage, "cost_usd": cost,
                            "latency_s": round(latency, 2),
                            "error": "skipped_cost_anomaly",
                            "anomaly_ratio": (cost / mu) if mu else None})
                _append(OUT_JSONL, rec); cum_cost += cost
                if sum(1 for x in anomaly_window if x) >= ANOMALY_WINDOW_LIMIT:
                    _write_stop("anomaly_threshold_hit",
                                {"recent": list(anomaly_window), "cum_cost": cum_cost,
                                 "next_idx": idx + 1})
                    sys.exit(2)
                continue

            anomaly_window.append(False); n_cost_samples += 1
            cum_cost += cost; latencies.append(latency)
            if err_kind is None:
                consec_err_kind, consec_err_n = None, 0

            parsed, verdict = _parse_response(raw)
            if parsed is None:
                consec_malformed += 1
                if consec_malformed >= MAX_CONSEC_MALFORMED:
                    _write_stop("consecutive_malformed", {"count": consec_malformed, "next_idx": idx + 1})
                    sys.exit(2)
            else: consec_malformed = 0

            reasoning = parsed.get("verdict_reasoning") if parsed else None
            outcome = "FIXED" if verdict == gt else ("still-wrong" if verdict else "NO_VERDICT")
            print(f"rv={verdict or '??':3s}  {outcome}  cost=${cost:.4f}  "
                  f"in={usage.get('prompt_tokens')}  out={usage.get('completion_tokens')}  {latency:.1f}s")
            if verdict == gt: n_fixed += 1
            elif verdict is not None: n_stillwrong += 1
            recent.append(f"{vid}/{horizon}:{verdict or '??'}/{gt}")

            rec.update({"recovery_verdict": verdict, "recovery_reasoning": reasoning,
                        "full_json": parsed or {}, "raw": raw, "usage": usage,
                        "cost_usd": cost, "latency_s": round(latency, 2),
                        "error": None if verdict is not None else "no_verdict_parsed"})
        except RuntimeError as e:
            msg = str(e)
            if msg.startswith("AUTH_OR_CREDIT_ERROR"):
                _write_stop("auth_or_credit_error", {"detail": msg, "next_idx": idx})
                sys.exit(2)
            print(f"[ERR] {msg}"); n_errors += 1
            kind = "wallclock" if "Wall-clock budget" in msg else "other"
            if consec_err_kind == kind: consec_err_n += 1
            else: consec_err_kind, consec_err_n = kind, 1
            if consec_err_n >= MAX_CONSEC_SAME_ERR:
                _write_stop("consecutive_api_errors",
                            {"kind": kind, "count": consec_err_n, "next_idx": idx + 1})
                sys.exit(2)
            rec.update({"recovery_verdict": None, "recovery_reasoning": None, "full_json": {},
                        "raw": "", "usage": {}, "cost_usd": 0.0,
                        "latency_s": round(time.time() - clip_start, 2), "error": msg})
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[ERR] {msg}"); n_errors += 1
            rec.update({"recovery_verdict": None, "recovery_reasoning": None, "full_json": {},
                        "raw": "", "usage": {}, "cost_usd": 0.0,
                        "latency_s": round(time.time() - clip_start, 2), "error": msg})

        _append(OUT_JSONL, rec)

        if cum_cost > STAGE_BUDGET_CEILING:
            _write_stop("budget_ceiling_hit",
                        {"cum_cost": cum_cost, "ceiling": STAGE_BUDGET_CEILING, "next_idx": idx + 1})
            print(f"  STOP-AND-ASK: cum cost ${cum_cost:.2f} > ceiling ${STAGE_BUDGET_CEILING:.2f}.")
            sys.exit(2)

        now = time.time()
        if now - last_heartbeat_t >= HEARTBEAT_SECS:
            mean_cost = (cum_cost / n_cost_samples) if n_cost_samples else 0.0
            mean_lat = (sum(latencies) / len(latencies)) if latencies else 0.0
            remaining = len(queue) - idx
            print()
            print(f"  >>> HEARTBEAT [{idx}/{len(queue)}]  cum=${cum_cost:.2f}  "
                  f"mean=${mean_cost:.4f}/clip  anomalies={n_anomalies}  errors={n_errors}  "
                  f"recent={list(recent)}  ETA={_fmt_eta(remaining, mean_lat)}")
            print()
            last_heartbeat_t = now

    wall = time.time() - t_start
    n_seen = n_fixed + n_stillwrong
    print()
    print("=" * 65)
    print("Stage 3 COMPLETE")
    print(f"  Rows written: {len(queue)}")
    print(f"  FIXED:        {n_fixed}/{n_seen}" if n_seen else "  (no recoveries)")
    print(f"  still-wrong:  {n_stillwrong}")
    print(f"  Errors:       {n_errors}  (skipped_cost_anomaly: {n_anomalies})")
    print(f"  Cum cost:     ${cum_cost:.4f}" + (f"  (mean: ${cum_cost/n_cost_samples:.4f}/clip)"
                                                  if n_cost_samples else ""))
    print(f"  Wall time:    {wall/60:.1f} min")
    print(f"  Output:       {OUT_JSONL}")
    print("=" * 65)


if __name__ == "__main__":
    main()
