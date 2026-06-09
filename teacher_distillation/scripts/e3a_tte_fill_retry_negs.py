"""E3a TTE-fill — retry the 7 still-wrong negatives by re-sampling a clean window.

All 7 reds are NO clips where the teacher emitted a false-positive YES at the
MID-4 / MID-8 horizon. For negatives the exact timing is non-critical, so we
re-sample a different collision-free window until the teacher confidently says NO.

Three of the four MID-8 reds (01704, 01768, 01995) were FLOORED at t=2.0s, so a
literal MID-9 shift re-uses identical frames -> identical wrong verdict. For those
we sweep a ladder of alternate clean windows (incl. LATER windows -- any
collision-free segment is a valid negative). The 4 non-floored reds get the simple
-5 / -9 shift the user asked for.

The horizon bucket label (MID-4 / MID-8) is preserved so every video keeps exactly
3 variants; the actual offset used is recorded in horizon_s / t_new.

Per target, walk the candidate ladder:
  * extract 16 frames stride-4 (native 1280x720) into <vid>_hires_retry_<int(t*10)>/
  * run v6_balanced. verdict NO -> accept (append pass-1 row to pass1.jsonl), stop.
  * verdict YES -> run TN_RECOVERY. recovery NO -> accept (append pass-1 row
    verdict=YES + recovery row to recovery.jsonl), stop.
  * still YES -> next candidate. Ladder exhausted -> keep best attempt, report red.

build_combined() keys pass1/recovery by (video_id, horizon_label) taking the LAST
occurrence, so appended corrected rows automatically supersede the originals.

Ops discipline: cost-anomaly skip max(3*mu, $0.25), budget ceiling $5, auth/credit
hard-stop. (Short run -- a few dozen calls at most.)
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
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
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

# ---- Config (matches pass1/pass2) ----
MODEL_SLUG  = "google/gemini-3.1-pro-preview"
PRICE_IN    = 2.00
PRICE_OUT   = 12.00
TEMPERATURE = 0.1

DEFAULT_TIMEOUT = 90.0
MAX_RETRIES     = 2
RETRY_DELAY     = 5.0
CLIP_BUDGET_SECS = 300
INTER_CALL_DELAY = 1.5

COST_ANOMALY_MULT    = 3.0
COST_ANOMALY_ABS     = 0.25
STAGE_BUDGET_CEILING = 5.00

WINDOW = 16
STRIDE = 4
T_FLOOR = 2.0

SRC_VIDEOS = Path(
    r"C:\Users\eviatar.ohayon\Ramon Space\PycharmProjects\Thesis"
    r"\Data-Centric-Crash-Prediction-Using-3LC-and-MViT\src\Nexar_DataSet\train"
)
DST_ROOT     = REPO_ROOT / "dataset" / "train"
PROMPT_DIR   = REPO_ROOT / "prompts"
OUT_DIR      = REPO_ROOT / "outputs" / "prompt_bakeoff" / "e3a_tte_fill"
PASS1_JSONL  = OUT_DIR / "pass1.jsonl"
RECOVERY_JSONL = OUT_DIR / "recovery.jsonl"
RETRY_LOG    = OUT_DIR / "retry_log.json"

# ---- Targets ----
# anchor = midpoint (t_seconds). For non-floored reds: single candidate at the -5/-9 shift.
# For floored reds: ladder of alternate clean windows (offsets from anchor), tried in order.
TARGETS: List[Dict] = [
    {"vid": "01136", "horizon": "MID-4", "anchor": 28.033, "offsets": [-5.0]},
    {"vid": "01583", "horizon": "MID-4", "anchor": 9.600,  "offsets": [-5.0]},
    {"vid": "01806", "horizon": "MID-4", "anchor": 10.333, "offsets": [-5.0]},
    {"vid": "02064", "horizon": "MID-8", "anchor": 19.233, "offsets": [-9.0]},
    # floored -> sweep clean windows (later first, since the FP content sits early)
    {"vid": "01704", "horizon": "MID-8", "anchor": 4.167,  "offsets": [+4.0, +8.0, +2.0, 0.0]},
    {"vid": "01768", "horizon": "MID-8", "anchor": 8.967,  "offsets": [+4.0, +8.0, -2.0, +2.0]},
    {"vid": "01995", "horizon": "MID-8", "anchor": 7.367,  "offsets": [+4.0, +8.0, -2.0, +2.0]},
]


# ---- Inlined utilities (proven in pass1/pass2) ----
def _normalize_verdict(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in {"YES", "NO"} else None


def _encode_image(path: Path) -> str:
    img = Image.open(path).convert("RGB") if path.exists() else Image.new("RGB", (64, 64), (0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _build_messages(prompt: str, image_b64s: Sequence[str], detail: str = "high") -> List[Dict]:
    content: List[Dict] = [{"type": "text", "text": prompt}]
    for b64 in image_b64s:
        content.append({"type": "image_url", "image_url": {"url": b64, "detail": detail}})
    return [{"role": "user", "content": content}]


def _extract_json_object(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw, flags=re.IGNORECASE)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            obj = json.loads(raw[s:e + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    return None


def _parse_response(raw: str) -> Tuple[Optional[Dict], Optional[str]]:
    parsed = _extract_json_object(raw)
    if parsed is None:
        return None, None
    parsed["collision_verdict"] = _normalize_verdict(parsed.get("collision_verdict"))
    return parsed, parsed["collision_verdict"]


def _call_model(client, messages) -> Tuple[str, Dict, Optional[str]]:
    last_kind = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_SLUG, messages=messages, temperature=TEMPERATURE,
                timeout=DEFAULT_TIMEOUT, max_tokens=8192,
            )
            text = response.choices[0].message.content if response.choices else ""
            usage = response.usage.model_dump() if hasattr(response, "usage") and response.usage else {}
            return text or "", usage, None
        except _OAIAuthError as exc:
            raise RuntimeError(f"AUTH_OR_CREDIT_ERROR: {exc}") from exc
        except _OAITimeoutError:
            last_kind = "timeout"
        except _OAIRateLimitError:
            last_kind = "rate_limit"
        except _OAIConnectionError:
            last_kind = "connection"
        except Exception as exc:
            txt = str(exc).lower()
            if any(k in txt for k in ("402", "401", "credit", "balance")):
                raise RuntimeError(f"AUTH_OR_CREDIT_ERROR: {exc}") from exc
            last_kind = "other"
        wait = RETRY_DELAY * (2 ** (attempt - 1))
        print(f"    [retry {attempt}/{MAX_RETRIES}] {last_kind} -- waiting {wait:.1f}s", flush=True)
        if attempt < MAX_RETRIES:
            time.sleep(wait)
    return "", {}, last_kind or "other"


def _calc_cost(usage: Dict) -> float:
    return (usage.get("prompt_tokens", 0) or 0) * PRICE_IN / 1_000_000 + \
           (usage.get("completion_tokens", 0) or 0) * PRICE_OUT / 1_000_000


def _call_with_budget(client, prompt: str, folder: Path) -> Tuple[str, Dict, float, Optional[str]]:
    def _inner():
        b64s = [_encode_image(folder / f"frame_{i:05d}.jpg") for i in range(1, WINDOW + 1)]
        messages = _build_messages(prompt, b64s, detail="high")
        t0 = time.time()
        raw, usage, err = _call_model(client, messages)
        return raw, usage, time.time() - t0, err
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_inner)
        try:
            return fut.result(timeout=CLIP_BUDGET_SECS)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise RuntimeError(f"Wall-clock budget {CLIP_BUDGET_SECS}s exceeded")


def _load_prompt(path: Path, var_name: str) -> str:
    spec = importlib.util.spec_from_file_location(f"_p_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, var_name)


def _append(path: Path, rec: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _extract_window(vid: str, t_new: float, suffix: str) -> Tuple[Path, Dict]:
    """Extract 16 stride-4 frames ending at t_new. Returns (folder, meta). Idempotent."""
    mp4 = SRC_VIDEOS / f"{vid}.mp4"
    if not mp4.exists():
        raise FileNotFoundError(f"MP4 not found: {mp4}")
    out_dir = DST_ROOT / f"{vid}_hires_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = round(t_new * fps)
    indices = [max(0, min(total - 1, end - (WINDOW - 1 - i) * STRIDE)) for i in range(WINDOW)]

    n_written = 0
    for i, fr_idx in enumerate(indices, start=1):
        dst = out_dir / f"frame_{i:05d}.jpg"
        if dst.exists():
            n_written += 1
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr_idx)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read frame {fr_idx} from {mp4}")
        cv2.imwrite(str(dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n_written += 1
    cap.release()
    return out_dir, {"fps": round(fps, 3), "total_frames": total,
                     "duration_s": round(total / fps, 2) if fps else 0.0,
                     "n_frames": n_written}


def _video_duration(vid: str) -> float:
    cap = cv2.VideoCapture(str(SRC_VIDEOS / f"{vid}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total / fps if fps else 0.0


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing")

    v6 = _load_prompt(PROMPT_DIR / "PROMPT_G_OPT_v6_balanced.py", "PROMPT_G_OPT_v6_balanced")
    tn = _load_prompt(PROMPT_DIR / "PROMPT_G_OPT_v7_1_TN_RECOVERY.py", "PROMPT_G_OPT_v7_1_TN_RECOVERY")
    print(f"Loaded v6_balanced ({len(v6)} chars), TN_RECOVERY ({len(tn)} chars)")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title":      os.environ.get("OPENROUTER_APP_TITLE", "MMLM_e3a_tte_fill_retry"),
        },
    )

    cum_cost = 0.0
    n_cost_samples = 0
    retry_log: List[Dict] = []
    t_start = time.time()

    print("=" * 70)
    print(f"RETRY START: {len(TARGETS)} failed negatives  (budget ceiling ${STAGE_BUDGET_CEILING:.2f})")
    print("=" * 70)

    def _anomaly(cost: float) -> bool:
        if n_cost_samples < 5:
            return False
        mu = cum_cost / n_cost_samples
        return cost > max(COST_ANOMALY_MULT * mu, COST_ANOMALY_ABS)

    for ti, tgt in enumerate(TARGETS, start=1):
        vid, horizon, anchor = tgt["vid"], tgt["horizon"], tgt["anchor"]
        dur = _video_duration(vid)
        print(f"\n[{ti}/{len(TARGETS)}] {vid} {horizon}  anchor={anchor:.3f}s  dur={dur:.2f}s")

        resolved = False
        for offset in tgt["offsets"]:
            t_raw = anchor + offset
            t_new = round(max(T_FLOOR, min(t_raw, dur - 0.2)), 3)
            suffix = f"retry_{int(round(t_new * 10))}"

            try:
                folder, meta = _extract_window(vid, t_new, suffix)
            except Exception as e:
                print(f"   offset={offset:+.1f} t_new={t_new}  EXTRACT-FAIL {type(e).__name__}: {e}")
                continue

            # ---- pass-1 (v6_balanced) ----
            time.sleep(INTER_CALL_DELAY)
            raw, usage, lat, err = _call_with_budget(client, v6, folder)
            cost = _calc_cost(usage)
            if _anomaly(cost):
                print(f"   offset={offset:+.1f} t_new={t_new}  [ANOMALY] cost=${cost:.4f} -- skip candidate")
                retry_log.append({"vid": vid, "horizon": horizon, "offset": offset, "t_new": t_new,
                                  "stage": "pass1", "verdict": None, "error": "skipped_cost_anomaly",
                                  "cost_usd": cost})
                continue
            cum_cost += cost
            n_cost_samples += 1
            parsed, verdict = _parse_response(raw)
            reasoning = parsed.get("verdict_reasoning") if parsed else None
            print(f"   offset={offset:+.1f} t_new={t_new}  pass1 verdict={verdict or '??'}  "
                  f"cost=${cost:.4f}  {lat:.1f}s")

            base_rec = {
                "video_id": vid, "horizon_label": horizon, "gt_verdict": "NO",
                "t_new": t_new, "horizon_s": offset, "frames_subdir": folder.name,
                "prompt_name": "PROMPT_G_OPT_v6_balanced", "resolution": "native_1280x720",
                "stride": 4, "detail": "high", "retry": True,
            }

            if verdict == "NO":
                rec = {**base_rec, "verdict": "NO", "reasoning": reasoning,
                       "full_json": parsed or {}, "raw": raw, "usage": usage,
                       "cost_usd": cost, "latency_s": round(lat, 2), "error": None}
                _append(PASS1_JSONL, rec)
                retry_log.append({"vid": vid, "horizon": horizon, "offset": offset, "t_new": t_new,
                                  "resolved_at": "pass1", "final_verdict": "NO",
                                  "frames_subdir": folder.name, "cost_usd": cost})
                print(f"   -> ACCEPT (pass1 NO) via {folder.name}")
                resolved = True
                break

            # ---- pass-1 said YES -> append the pass1 row, then TN_RECOVERY ----
            rec = {**base_rec, "verdict": verdict, "reasoning": reasoning,
                   "full_json": parsed or {}, "raw": raw, "usage": usage,
                   "cost_usd": cost, "latency_s": round(lat, 2),
                   "error": None if verdict is not None else "no_verdict_parsed"}
            _append(PASS1_JSONL, rec)

            if cum_cost > STAGE_BUDGET_CEILING:
                print(f"  STOP: cum cost ${cum_cost:.2f} > ceiling ${STAGE_BUDGET_CEILING:.2f}")
                sys.exit(2)

            time.sleep(INTER_CALL_DELAY)
            raw2, usage2, lat2, err2 = _call_with_budget(client, tn, folder)
            cost2 = _calc_cost(usage2)
            cum_cost += cost2
            n_cost_samples += 1
            parsed2, verdict2 = _parse_response(raw2)
            reasoning2 = parsed2.get("verdict_reasoning") if parsed2 else None
            print(f"                    TN_RECOVERY rv={verdict2 or '??'}  cost=${cost2:.4f}  {lat2:.1f}s")

            rec2 = {
                "video_id": vid, "horizon_label": horizon,
                "recovery_prompt": "PROMPT_G_OPT_v7_1_TN_RECOVERY", "gt_verdict": "NO",
                "pass1_verdict": verdict, "t_new": t_new, "frames_subdir": folder.name,
                "resolution": "native_1280x720", "detail": "high", "retry": True,
                "recovery_verdict": verdict2, "recovery_reasoning": reasoning2,
                "full_json": parsed2 or {}, "raw": raw2, "usage": usage2,
                "cost_usd": cost2, "latency_s": round(lat2, 2),
                "error": None if verdict2 is not None else "no_verdict_parsed",
            }
            _append(RECOVERY_JSONL, rec2)

            if verdict2 == "NO":
                retry_log.append({"vid": vid, "horizon": horizon, "offset": offset, "t_new": t_new,
                                  "resolved_at": "recovery", "final_verdict": "NO",
                                  "frames_subdir": folder.name, "cost_usd": cost + cost2})
                print(f"   -> ACCEPT (recovery NO) via {folder.name}")
                resolved = True
                break

            print(f"   offset={offset:+.1f} still YES after recovery -- next candidate")
            if cum_cost > STAGE_BUDGET_CEILING:
                print(f"  STOP: cum cost ${cum_cost:.2f} > ceiling ${STAGE_BUDGET_CEILING:.2f}")
                sys.exit(2)

        if not resolved:
            print(f"   !! UNRESOLVED: {vid} {horizon} exhausted ladder, still YES")
            retry_log.append({"vid": vid, "horizon": horizon, "resolved_at": None,
                              "final_verdict": "YES", "note": "ladder_exhausted"})

    RETRY_LOG.write_text(json.dumps(retry_log, indent=2), encoding="utf-8")
    wall = time.time() - t_start
    n_ok = sum(1 for r in retry_log if r.get("final_verdict") == "NO")
    print("\n" + "=" * 70)
    print("RETRY COMPLETE")
    print(f"  Resolved to NO: {n_ok}/{len(TARGETS)}")
    print(f"  Cum cost:       ${cum_cost:.4f}")
    print(f"  Wall time:      {wall/60:.1f} min")
    print(f"  Retry log:      {RETRY_LOG}")
    print("=" * 70)


if __name__ == "__main__":
    main()
