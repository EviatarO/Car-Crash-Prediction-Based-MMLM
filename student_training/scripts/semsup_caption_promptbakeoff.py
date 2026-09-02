"""
semsup_caption_promptbakeoff.py
=================================
Calls a vision model over OpenRouter to caption a clip manifest (default:
dataset/manifests/semsup_promptbakeoff.jsonl, 498 rows from
semsup_sample_clips.py; also used against dataset/manifests/val_e3a.jsonl for
the 18-clip teacher bake-offs) using one of three prompts (--prompt):
  v2 = PROMPT_SEMSUP_V2.py (direct caption, the default)
  v3 = PROMPT_SEMSUP_V3_COT.py (v6-style chain-of-thought pipeline, then distill)
  v4 = PROMPT_SEMSUP_V4_QWEN.py (Qwen-native structure + worked examples +
       explicit anti-under-calling instruction)
Produces the raw JSONL that semsup_caption_qa.py expects as --input (one row:
video_id, requested_time_to_event, caption_neutral, risk_clause, verdict,
confidence, plus V3's optional scene_context/dynamic_objects/temporal_analysis
when present).

Reuses this repo's established OpenRouter conventions rather than
reimplementing them from scratch: the retry/backoff logic, JSON-extraction
strategies, and image-encoding approach below are copied (not imported) from
teacher_distillation/scripts/Teacher_dataset_distill_v11.py, because that
module's own top-level import (`from prompts.templates import ...`) is
currently broken - `prompts/templates.py` no longer exists (the prompts it
held now live under `prompts/old prompts/`), a pre-existing issue unrelated
to this script, left as-is rather than fixed here. Model default and API
mechanics (OPENROUTER_API_KEY env var via python-dotenv, OpenAI SDK pointed
at OpenRouter's base_url) match that script; frame_size/timeout/max_retries/
detail/temperature defaults match its production CLI defaults
(256 / 90s / 3 / low / 0.1) for consistency.

Resumable: skips any video_id already present in --out (append mode), so a
crashed or rate-limited run can just be re-invoked with the same --out.

Cost note (see chat): ~4,800 input tokens/clip at the "low" detail tier x 498
clips =~ 2.4M input tokens. Check OpenRouter's current per-token rate for
whichever --model you pick before a full run - rates and per-image
tokenization are not something to trust from memory.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

try:
    from openai import (
        APIConnectionError as _OAIConnectionError,
        APITimeoutError as _OAITimeoutError,
        RateLimitError as _OAIRateLimitError,
    )
except ImportError:  # pragma: no cover - older openai SDKs
    _OAIConnectionError = Exception  # type: ignore[assignment,misc]
    _OAITimeoutError = Exception      # type: ignore[assignment,misc]
    _OAIRateLimitError = Exception    # type: ignore[assignment,misc]

try:
    import httpx as _httpx
except ImportError:
    _httpx = None  # type: ignore[assignment]

from prompts.PROMPT_SEMSUP_V2 import PROMPT_SEMSUP_V2  # noqa: E402
from prompts.PROMPT_SEMSUP_V3_COT import PROMPT_SEMSUP_V3_COT  # noqa: E402
from prompts.PROMPT_SEMSUP_V4_QWEN import PROMPT_SEMSUP_V4_QWEN  # noqa: E402
from prompts.PROMPT_SEMSUP_V5_BALANCED import PROMPT_SEMSUP_V5_BALANCED  # noqa: E402
from prompts.PROMPT_SEMSUP_V6_KINEMATIC import PROMPT_SEMSUP_V6_KINEMATIC  # noqa: E402
from prompts.PROMPT_SEMSUP_V7_EGOFRAME import PROMPT_SEMSUP_V7_EGOFRAME  # noqa: E402
from prompts.PROMPT_SEMSUP_V8_NARRATIVE import PROMPT_SEMSUP_V8_NARRATIVE  # noqa: E402
from prompts.PROMPT_SEMSUP_V9_MINIMAL import PROMPT_SEMSUP_V9_MINIMAL  # noqa: E402
from prompts.PROMPT_SEMSUP_V10_GT import build_prompt as _build_v10_prompt  # noqa: E402
from prompts.PROMPT_SEMSUP_V10Q_GT import build_prompt as _build_v10q_prompt  # noqa: E402
from prompts.PROMPT_SEMSUP_V12_NEUTRAL import build_prompt as _build_v12_prompt  # noqa: E402
from prompts.PROMPT_SEMSUP_V13_CAUSAL import build_prompt as _build_v13_prompt  # noqa: E402

DEFAULT_MODEL = "google/gemini-3.7-flash"
# Fixed 2026-08-27: was a stale "google/gemini-3.1-pro-preview" - already flagged as a known
# bug in docs_agents/PROJECT_STATE.md ("a re-run that omits the flag silently uses the WRONG
# teacher"). Confirmed hit live: an omitted --model on 2026-08-27 silently captioned 36 clips
# with 3.1-pro-preview instead of the project's actual teacher (gemini-3.6-flash, used for the
# full 1,761-pool corpus). google/gemini-3.7-flash is the newer flash release the user has
# selected as the current teacher going forward - confirmed to exist on OpenRouter with real
# pricing (not gemini-3.6-flash) as of this fix.
DEFAULT_MANIFEST = PROJECT_ROOT / "dataset" / "manifests" / "semsup_promptbakeoff.jsonl"
DEFAULT_FRAMES_ROOT = PROJECT_ROOT / "dataset" / "train"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "semantic_captions" / "promptbakeoff" / "raw_captions.jsonl"

PROMPTS = {
    "v2": PROMPT_SEMSUP_V2, "v3": PROMPT_SEMSUP_V3_COT,
    "v4": PROMPT_SEMSUP_V4_QWEN, "v5": PROMPT_SEMSUP_V5_BALANCED,
    "v6": PROMPT_SEMSUP_V6_KINEMATIC, "v7": PROMPT_SEMSUP_V7_EGOFRAME,
    "v8": PROMPT_SEMSUP_V8_NARRATIVE, "v9": PROMPT_SEMSUP_V9_MINIMAL,
}
# v10/v10q are TEMPLATES (per-clip GT injection), not static strings - see
# build_prompt_for_row() below. Kept in a separate dict so `--prompt` argparse
# choices still list them without PROMPTS[key] being called on a plain string.
#
# NOTE (2026-08-04): a V11 dual-prompt variant (separate TP/TN prompts with
# neutral field names) was built and then reverted here - see
# outputs/prompt_bakeoff/semsup_val18_gt/summary.md's correction notice. The
# fabrication V11 was built to fix (01643) turned out to be caused by the GT
# block itself, not by the hazard_* field names, and V10 in --gt-mode blind
# already produced the correct agent:'None' on that clip. V11 measured worse
# on negatives (recall 0.435->0.38, MATCH 6->4, fabrications 1->2) and was
# deleted. Do not re-propose a field-name-only prompt split without first
# checking whether --gt-mode blind alone already resolves the failure.
def _v12_builder(gt_mode=None, is_positive=None):
    """Adapter matching build_prompt_for_row()'s calling convention
    (builder(gt_mode, is_positive=...) or builder('blind')). V12's own
    build_prompt() takes no arguments BY DESIGN - there is no per-class
    branch, so gt_mode/is_positive are accepted here and ignored, not
    threaded through. --gt-mode has no effect on v12; the flag is still
    accepted on the CLI (default 'blind') so existing runbooks that always
    pass it don't need a special case."""
    return _build_v12_prompt()


def _v13_builder(gt_mode=None, is_positive=None):
    """Same no-argument, no-per-class-branch contract as _v12_builder - see
    that function's docstring. --gt-mode has no effect on v13 either."""
    return _build_v13_prompt()


TEMPLATE_BUILDERS = {"v10": _build_v10_prompt, "v10q": _build_v10q_prompt,
                     "v12": _v12_builder, "v13": _v13_builder}
# V12_REQUIRED: no verdict/risk_score/confidence/risk_clause in ANY mode - V12
# never asks the model to judge the scene, only describe it (see the prompt's
# module docstring for why removing the decision layer is the point).
V12_REQUIRED = ("caption_neutral", "primary_agent", "agent_motion", "agent_position",
                 "gap_trend", "evidence_frames", "agent_visible")
V12_GAP_TREND_VALUES = ("decreasing", "increasing", "constant", "none_visible")

# V13_REQUIRED: V12's fields plus the 5 new closed-vocabulary causal-cue fields
# (see PROMPT_SEMSUP_V13_CAUSAL.py's module docstring for why these target
# non-inferable cues rather than restating what the vision encoder already sees).
V13_REQUIRED = V12_REQUIRED + ("lead_vehicle_lighting", "ego_maneuver",
                                "road_geometry", "signal_state",
                                "occluded_or_peripheral")
# 'flashers_on', not 'hazards_on': the caption is required to verbalize this field
# and 'hazard' is on the banned outcome-word list - an enum the caption may not utter
# is an enum whose information silently never reaches the SigLIP target.
V13_LIGHTING_VALUES = ("brake_lights_on", "indicator_left", "indicator_right",
                        "flashers_on", "none_visible")
V13_MANEUVER_VALUES = ("straight", "braking", "accelerating", "turning_left",
                        "turning_right", "lane_change", "stopped")
V13_GEOMETRY_VALUES = ("straight_road", "intersection", "merge", "curve",
                        "roundabout", "parking_area")
V13_SIGNAL_VALUES = ("green", "amber", "red", "stop_sign", "uncontrolled",
                      "none_visible")
# "red"/"green" deliberately excluded: both are legitimate signal_state vocabulary
# ("green signal", "red light") the prompt explicitly asks for - banning them here
# would flag correct traffic-light reporting as a false leak. Vehicle-colour leakage
# via literal "red"/"green" car mentions is covered by the PROMPT's own instruction;
# this python-side list is a soft diagnostic NOTE, not the enforcement mechanism.
V13_COLOR_WORDS = ("blue", "white", "black", "silver", "grey", "gray",
                    "yellow", "orange", "brown", "purple", "tan", "beige")
REQUIRED_KEYS = ("caption_neutral", "risk_clause", "verdict", "confidence")
# v10/v10q common keys, required in BOTH gt and blind modes - these are what let
# the scorer compute a "rationalization rate" (agents the GT arm names that the
# blind arm, on the identical clip, never mentioned).
V10_REQUIRED_COMMON = ("caption_neutral", "risk_clause", "hazard_agent", "hazard_motion",
                        "hazard_position", "closing_dynamic", "evidence_frames",
                        "mechanism_visible")
# blind-mode-only: needed to keep AP/AUC computable on the blind control arm.
# Omitted in gt mode on purpose - the model already has the label, so asking it
# to also emit a verdict would just have it parrot the label back.
V10_REQUIRED_BLIND = ("risk_score", "verdict", "confidence")
# V3-only fields: carried through to the output row when present, never required
# (a V3 response missing them is a real finding - the model ignored part of the
# schema - not silently treated as a validation failure of the whole row).
OPTIONAL_COT_KEYS = ("scene_context", "dynamic_objects", "temporal_analysis")
# V5-only fields: risk_score is required for v5 (it's what verdict is derived
# from); counter_evidence is carried through when present but not hard-required,
# same rationale as OPTIONAL_COT_KEYS above.
V5_REQUIRED_KEYS = ("risk_score",)
V5_OPTIONAL_KEYS = ("counter_evidence",)
# V6-only fields: risk_score is required (sum of the 4 sub-scores, verdict is
# derived from it, same contract as V5). The 4 sub-scores and the 3 observation
# fields are required too - V6's whole design rests on them being populated,
# so a response missing them is a real finding, not something to paper over.
V6_REQUIRED_KEYS = ("risk_score", "closing_risk", "lateral_risk", "intrusion_risk",
                     "unreacted_risk", "ego_motion", "lateral_watch", "final_delta")
V6_OPTIONAL_KEYS = ("counter_evidence",)
# V7-only fields: same 4 sub-scores as V6, but the observation fields are the
# ego-frame decomposition (static_reference -> ego_path -> apparent_vs_true) plus
# an explicit conflict_source enum. All required - the whole prompt rests on them.
V7_REQUIRED_KEYS = ("risk_score", "closing_risk", "lateral_risk", "intrusion_risk",
                     "unreacted_risk", "static_reference", "ego_path",
                     "apparent_vs_true", "conflict_source", "final_delta")
V7_OPTIONAL_KEYS = ("counter_evidence",)
V7_CONFLICT_SOURCES = ("ego_into_other", "other_into_ego", "longitudinal", "none")
# V8: same 4 sub-scores as V6/V7 (deliberately unchanged so the caption grammar is
# the isolated variable). Observation fields are the narrative decomposition
# delta -> true_movers -> cause -> ego_response; all required.
V8_REQUIRED_KEYS = ("risk_score", "closing_risk", "lateral_risk", "intrusion_risk",
                     "unreacted_risk", "delta", "true_movers", "cause", "ego_response")


# ---------------------------------------------------------------------------
# Copied from teacher_distillation/scripts/Teacher_dataset_distill_v11.py
# (see module docstring for why this is a copy, not an import). Logic
# unchanged - image encoding, message building, JSON extraction, retry/backoff.
# ---------------------------------------------------------------------------

def _encode_image(path: Path, frame_size: int) -> str:
    if path.exists():
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", (frame_size, frame_size), color=(0, 0, 0))
    if frame_size and img.size != (frame_size, frame_size):
        img = img.resize((frame_size, frame_size))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _build_messages(prompt: str, image_b64s: Sequence[str], detail: Optional[str]) -> List[Dict]:
    content: List[Dict] = [{"type": "text", "text": prompt}]
    for b64 in image_b64s:
        image_url: Dict = {"url": b64}
        if detail:
            image_url["detail"] = detail
        content.append({"type": "image_url", "image_url": image_url})
    return [{"role": "user", "content": content}]


def _extract_json_object(raw: str) -> Optional[Dict]:
    """Try three strategies to extract a JSON object from a model response."""
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
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    return None


def _call_model(
    client: OpenAI, model: str, messages: List[Dict], timeout: float,
    max_retries: int, retry_delay: float, temperature: float = 0.1,
    max_tokens: Optional[int] = None, provider_order: Optional[List[str]] = None,
) -> Tuple[str, Dict]:
    """Call the chat completions endpoint with robust retry logic (timeout /
    rate-limit / connection-error / generic, each with exponential back-off;
    rate-limit gets a 2x multiplier to respect rate windows).

    max_tokens=None omits the parameter entirely, preserving prior behavior
    for models where the provider default is sufficient. Thinking models can
    exhaust a small budget on internal reasoning before emitting the JSON
    output - if that happens (empty/truncated response), pass an explicit,
    generous --max-tokens (e.g. 20000, see semsup_v6_control_rerun.py's
    Qwen3.7 Flash experience).

    provider_order: OpenRouter provider slug(s) to pin (e.g. ["google-vertex"]),
    sent as extra_body={"provider": {"order": [...], "allow_fallbacks": False}}.
    allow_fallbacks=False is deliberate - see --provider-order's CLI help for why
    a silent fallback to a differently-priced provider is worse than a loud
    failure here (measured 2026-08-27: Vertex 75%-off vs AI Studio 50%-off on
    the SAME model, gemini-3.7-flash)."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {"model": model, "messages": messages, "temperature": temperature,
                      "timeout": timeout}
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if provider_order:
                kwargs["extra_body"] = {"provider": {"order": provider_order,
                                                       "allow_fallbacks": False}}
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content if response.choices else ""
            usage = (response.usage.model_dump()
                     if hasattr(response, "usage") and response.usage else {})
            return text or "", usage
        except _OAITimeoutError as exc:
            last_exc = exc
            wait = retry_delay * (2 ** (attempt - 1))
            print(f"  [retry {attempt}/{max_retries}] timeout after {timeout}s -- "
                  f"waiting {wait:.1f}s before retry", flush=True)
            if attempt < max_retries:
                time.sleep(wait)
        except _OAIRateLimitError as exc:
            last_exc = exc
            wait = retry_delay * 2 * (2 ** (attempt - 1))
            print(f"  [retry {attempt}/{max_retries}] rate-limit -- "
                  f"waiting {wait:.1f}s before retry", flush=True)
            if attempt < max_retries:
                time.sleep(wait)
        except _OAIConnectionError as exc:
            last_exc = exc
            wait = retry_delay * (2 ** (attempt - 1))
            print(f"  [retry {attempt}/{max_retries}] connection error -- "
                  f"waiting {wait:.1f}s before retry", flush=True)
            if attempt < max_retries:
                time.sleep(wait)
        except Exception as exc:
            is_http_timeout = (_httpx is not None and isinstance(exc, _httpx.TimeoutException)) \
                or "timeout" in str(exc).lower()
            last_exc = exc
            wait = retry_delay * (2 ** (attempt - 1))
            label = "timeout" if is_http_timeout else "error"
            print(f"  [retry {attempt}/{max_retries}] {label}: {exc!r} -- "
                  f"waiting {wait:.1f}s before retry", flush=True)
            if attempt < max_retries:
                time.sleep(wait)
    raise RuntimeError(f"OpenRouter call failed after {max_retries} attempts: {last_exc}") from last_exc


def resolve_gt_label(row: dict) -> bool:
    """True/False ground-truth label for --gt-mode gt. Accepts either manifest
    schema seen in this repo: val_e3a.jsonl uses 'target' (0/1),
    train4500_hires.jsonl uses 'event_occurs' (0/1). Fails LOUDLY if neither
    key is present rather than defaulting to negative - a silent default
    would corrupt every positive clip's GT block into a negative one."""
    if "target" in row:
        return bool(int(row["target"]))
    if "event_occurs" in row:
        return bool(int(row["event_occurs"]))
    raise KeyError(
        f"--gt-mode gt requires a ground-truth label but row for video_id="
        f"{row.get('video_id')!r} has neither 'target' nor 'event_occurs'. "
        f"Refusing to silently assume negative."
    )


def build_prompt_for_row(prompt_key: str, gt_mode: str, row: dict) -> str:
    """Builds the per-clip prompt for a TEMPLATE_BUILDERS entry (v10/v10q).
    gt_mode='blind' never looks at the row's label at all - resolve_gt_label
    is only called in 'gt' mode, so a manifest without labels still works for
    the blind control arm."""
    builder = TEMPLATE_BUILDERS[prompt_key]
    if gt_mode == "gt":
        return builder("gt", is_positive=resolve_gt_label(row))
    return builder("blind")


def load_manifest(path: Path) -> list:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def row_resume_key(row: dict) -> str:
    """The identity a manifest row (or an already-written output row) is
    resumed/deduped on. Prefers frames_dir - the one field that is unique
    PER WINDOW across every manifest schema in this repo (val_e3a.jsonl,
    semsup_promptbakeoff.jsonl, train4500-derived manifests all carry it) -
    and falls back to video_id only when frames_dir is absent (legacy output
    rows written before this field was added; safe ONLY when video_id
    happens to be unique in that particular manifest).

    WHY THIS MATTERS (found 2026-08-04, train4500_failures_pos_269.jsonl /
    neg_318.jsonl): unlike every manifest this runner had seen before (which
    were built one-row-per-video_id by construction, e.g.
    semsup_sample_clips.py's explicit "no sibling-TTE reuse"), a manifest
    reconstructed from the train4500 failure set can and does have the SAME
    video_id appear multiple times at different TTE/MID buckets (76/170
    positive video_ids and 70/226 negative video_ids repeat). The old
    video_id-only resume key would silently treat a video's second failing
    window as "already captioned" the moment its first window was written -
    dropping ~146/587 windows from a real run with no error or warning."""
    return row.get("frames_dir") or row["video_id"]


def load_done_ids(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    for l in open(out_path, encoding="utf-8"):
        l = l.strip()
        if l:
            done.add(row_resume_key(json.loads(l)))
    return done


V6_SUBSCORE_KEYS = ("closing_risk", "lateral_risk", "intrusion_risk", "unreacted_risk")


def _stamp_token_len(out_row: dict, siglip_tok, cap: int) -> bool:
    """Tokenizes out_row['caption_neutral'] with the SigLIP tokenizer (WITHOUT
    truncation, so the true length is measured, not clamped to 64 by padding
    options) and stamps caption_token_len onto the row in place. Returns True
    if the true length exceeds `cap`. Reported by the caller, not enforced
    here - a rejected row would waste a paid API call; see --token-cap's help."""
    n = len(siglip_tok(out_row["caption_neutral"], truncation=False)["input_ids"])
    out_row["caption_token_len"] = n
    return n > cap


def validate_parsed(parsed: dict, prompt_key: str = "v2", gt_mode: str = "blind") -> tuple:
    """Returns (ok, error_message_or_None). Enforces the schema PROMPT_SEMSUP_V2
    asks for - a response missing a required key is a hard failure, not a
    row silently written with a missing field the rest of the pipeline
    would choke on later. For v5/v6, also requires risk_score (verdict is
    supposed to be derived from it) and checks the derivation held. For v6,
    also checks risk_score == sum of the 4 sub-scores (a soft NOTE, not a
    hard failure - same rationale as the v5 verdict/score check).

    v10/v10q have a different contract: `verdict`/`confidence`/`risk_score`
    are required ONLY in blind mode (gt mode never asks for them - see
    PROMPT_SEMSUP_V10_GT.py's docstring), and hazard_*/mechanism_visible are
    required in BOTH modes."""
    if parsed is None:
        return False, "could not extract a JSON object from the response"

    if prompt_key in ("v10", "v10q"):
        missing = [k for k in V10_REQUIRED_COMMON if k not in parsed]
        if gt_mode == "blind":
            missing += [k for k in V10_REQUIRED_BLIND if k not in parsed]
        if missing:
            return False, f"missing keys: {missing}"
        if parsed["mechanism_visible"] not in (True, False, "true", "false", "True", "False"):
            return False, f"mechanism_visible not boolean: {parsed['mechanism_visible']!r}"
        notes = []
        if gt_mode == "blind":
            if parsed["verdict"] not in (0, 1, "0", "1"):
                return False, f"verdict not 0/1: {parsed['verdict']!r}"
            try:
                score = float(parsed["risk_score"])
            except (TypeError, ValueError):
                return False, f"risk_score not numeric: {parsed['risk_score']!r}"
            if not (0 <= score <= 100):
                return False, f"risk_score out of [0,100]: {score!r}"
            expected_verdict = 1 if score >= 50 else 0
            if int(parsed["verdict"]) != expected_verdict:
                notes.append(f"verdict/risk_score mismatch (score={score}, "
                              f"verdict={parsed['verdict']}, expected={expected_verdict})")
        return (True, "NOTE: " + "; ".join(notes)) if notes else (True, None)

    if prompt_key == "v12":
        missing = [k for k in V12_REQUIRED if k not in parsed]
        if missing:
            return False, f"missing keys: {missing}"
        if parsed["agent_visible"] not in (True, False, "true", "false", "True", "False"):
            return False, f"agent_visible not boolean: {parsed['agent_visible']!r}"
        if parsed["gap_trend"] not in V12_GAP_TREND_VALUES:
            return False, (f"gap_trend not one of {V12_GAP_TREND_VALUES}: "
                            f"{parsed['gap_trend']!r}")
        # Soft check, not a hard failure (same rationale as v5's verdict/score
        # note): the prompt requires the gap_trend word IN the caption text so
        # the SigLIP target actually carries the standardized token, not just
        # the structured field. A response missing it is a real finding worth
        # surfacing, not silently dropped.
        gt_word = parsed["gap_trend"]
        if gt_word != "none_visible" and gt_word not in str(parsed["caption_neutral"]).lower():
            return True, f"NOTE: gap_trend {gt_word!r} not found verbatim in caption_neutral"
        return True, None

    if prompt_key == "v13":
        missing = [k for k in V13_REQUIRED if k not in parsed]
        if missing:
            return False, f"missing keys: {missing}"
        if parsed["agent_visible"] not in (True, False, "true", "false", "True", "False"):
            return False, f"agent_visible not boolean: {parsed['agent_visible']!r}"
        if parsed["gap_trend"] not in V12_GAP_TREND_VALUES:
            return False, (f"gap_trend not one of {V12_GAP_TREND_VALUES}: "
                            f"{parsed['gap_trend']!r}")
        notes = []
        # Closed-vocab membership: soft NOTEs (same rationale as v12's gap_trend-
        # in-caption check and v7's conflict_source check) - an off-enum value is
        # a real instruction-following finding, not grounds to discard a paid call.
        for field, allowed in [("lead_vehicle_lighting", V13_LIGHTING_VALUES),
                                ("ego_maneuver", V13_MANEUVER_VALUES),
                                ("road_geometry", V13_GEOMETRY_VALUES),
                                ("signal_state", V13_SIGNAL_VALUES)]:
            val = str(parsed[field]).strip().lower()
            if val not in allowed:
                notes.append(f"{field} not in {allowed}: {parsed[field]!r}")
        gt_word = parsed["gap_trend"]
        cap_lower = str(parsed["caption_neutral"]).lower()
        if gt_word != "none_visible" and gt_word not in cap_lower:
            notes.append(f"gap_trend {gt_word!r} not found verbatim in caption_neutral")
        # word-boundary match, not raw substring - a naive `in` scan wrongly flagged
        # "tan" inside "distance"/"constant" on 15/15 gate-run captions with zero
        # real colour leaks (verified by inspection 2026-08-27).
        color_hits = [w for w in V13_COLOR_WORDS if re.search(rf"\b{w}\b", cap_lower)]
        if color_hits:
            notes.append(f"caption_neutral contains banned colour word(s): {color_hits}")

        # Word FLOOR (2026-08-27): the first gate run averaged 26.7 words against a
        # ceiling-only "<=45 words" rule - barely half the encoder budget. The floor
        # is now the operative constraint, so an under-length caption is a real
        # finding, not silently accepted.
        n_words = len(str(parsed["caption_neutral"]).split())
        if n_words < 42:
            notes.append(f"caption_neutral is {n_words} words, below the 42-word floor")
        elif n_words > 52:
            notes.append(f"caption_neutral is {n_words} words, above the 52-word ceiling")

        # Field-coverage: every POPULATED field must actually reach the caption, since
        # the caption is the only thing the SigLIP target consumes. Checked by a
        # representative keyword per field rather than the raw enum token (the prompt
        # asks for natural prose - "brake lights", not "brake_lights_on").
        _COVERAGE = {
            "lead_vehicle_lighting": {"brake_lights_on": ("brake light",),
                                       "indicator_left": ("indicator", "signal"),
                                       "indicator_right": ("indicator", "signal"),
                                       "flashers_on": ("flasher",)},
            "signal_state": {"green": ("green",), "amber": ("amber",), "red": ("red",),
                              "stop_sign": ("stop sign",),
                              "uncontrolled": ("uncontrolled",)},
            "road_geometry": {"straight_road": ("straight",), "intersection": ("intersection",),
                               "merge": ("merg",), "curve": ("curve", "curving"),
                               "roundabout": ("roundabout",), "parking_area": ("parking",)},
            "ego_maneuver": {"straight": ("straight",), "braking": ("brak",),
                              "accelerating": ("accelerat",), "turning_left": ("turn",),
                              "turning_right": ("turn",), "lane_change": ("lane change",),
                              "stopped": ("stop",)},
        }
        for field, vocab in _COVERAGE.items():
            val = str(parsed[field]).strip().lower()
            expected = vocab.get(val)
            if expected and not any(kw in cap_lower for kw in expected):
                notes.append(f"{field}={val!r} not verbalized in caption_neutral "
                              f"(expected one of {expected})")
        occ = str(parsed["occluded_or_peripheral"]).strip()
        if occ and not any(kw in cap_lower for kw in
                            ("occlud", "peripher", "partly hidden", "behind", "edge")):
            notes.append("occluded_or_peripheral is populated but not verbalized "
                          "in caption_neutral")
        return (True, "NOTE: " + "; ".join(notes)) if notes else (True, None)

    missing = [k for k in REQUIRED_KEYS if k not in parsed]
    if prompt_key in ("v5", "v9"):
        missing += [k for k in V5_REQUIRED_KEYS if k not in parsed]
    elif prompt_key == "v6":
        missing += [k for k in V6_REQUIRED_KEYS if k not in parsed]
    elif prompt_key == "v7":
        missing += [k for k in V7_REQUIRED_KEYS if k not in parsed]
    elif prompt_key == "v8":
        missing += [k for k in V8_REQUIRED_KEYS if k not in parsed]
    if missing:
        return False, f"missing keys: {missing}"
    if parsed["verdict"] not in (0, 1, "0", "1"):
        return False, f"verdict not 0/1: {parsed['verdict']!r}"

    notes = []
    if prompt_key in ("v5", "v6", "v7", "v8", "v9"):
        try:
            score = float(parsed["risk_score"])
        except (TypeError, ValueError):
            return False, f"risk_score not numeric: {parsed['risk_score']!r}"
        if not (0 <= score <= 100):
            return False, f"risk_score out of [0,100]: {score!r}"
        expected_verdict = 1 if score >= 50 else 0
        if int(parsed["verdict"]) != expected_verdict:
            # Not a hard failure - a real finding worth keeping (the model
            # broke the mechanical derivation instruction) - flagged via the
            # returned note so it isn't silently lost.
            notes.append(f"verdict/risk_score mismatch (score={score}, "
                          f"verdict={parsed['verdict']}, expected={expected_verdict})")
    if prompt_key in ("v6", "v7", "v8"):
        try:
            subscores = [float(parsed[k]) for k in V6_SUBSCORE_KEYS]
        except (TypeError, ValueError):
            return False, f"a sub-score is not numeric: " \
                          f"{ {k: parsed.get(k) for k in V6_SUBSCORE_KEYS} }"
        oob = [k for k, v in zip(V6_SUBSCORE_KEYS, subscores) if not (0 <= v <= 25)]
        if oob:
            return False, f"sub-score(s) out of [0,25]: {oob}"
        subtotal = sum(subscores)
        if abs(subtotal - score) > 0.5:
            notes.append(f"risk_score ({score}) != sum of sub-scores ({subtotal})")
    if prompt_key == "v7":
        cs = str(parsed["conflict_source"]).strip().lower()
        if cs not in V7_CONFLICT_SOURCES:
            # Soft note, not a hard failure: an off-enum value is a real finding
            # about instruction-following, and the row is still usable.
            notes.append(f"conflict_source not in enum: {parsed['conflict_source']!r}")

    if notes:
        return True, "NOTE: " + "; ".join(notes)
    return True, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--frames-root", default=str(DEFAULT_FRAMES_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default="v2",
                     choices=list(PROMPTS.keys()) + list(TEMPLATE_BUILDERS.keys()),
                     help="v2 = PROMPT_SEMSUP_V2 (direct caption), "
                          "v3 = PROMPT_SEMSUP_V3_COT (v6-style CoT pipeline, then distill), "
                          "v4 = PROMPT_SEMSUP_V4_QWEN (Qwen-native structure + worked examples), "
                          "v5 = PROMPT_SEMSUP_V5_BALANCED (0-100 risk_score + pre-mortem, "
                          "verdict derived mechanically), "
                          "v6 = PROMPT_SEMSUP_V6_KINEMATIC (ego-motion/lateral-drift "
                          "observation fields + 4 decomposed 0-25 sub-scores summed to "
                          "risk_score, verdict derived mechanically), "
                          "v7 = PROMPT_SEMSUP_V7_EGOFRAME (static-frame ego-rotation "
                          "estimate + apparent-vs-true motion test + conflict_source enum), "
                          "v8 = PROMPT_SEMSUP_V8_NARRATIVE (delta/cause/ego-response caption "
                          "grammar, path-relative motion vocabulary), "
                          "v9 = PROMPT_SEMSUP_V9_MINIMAL (the less-is-more arm: V2 length "
                          "plus only the evidence-backed insights, no scaffolding fields), "
                          "v10 = PROMPT_SEMSUP_V10_GT (v6-CoT envelope, GT-informed mechanism "
                          "captioning, per-clip template - see --gt-mode), "
                          "v10q = PROMPT_SEMSUP_V10Q_GT (same schema, Qwen-native envelope), "
                          "v12 = PROMPT_SEMSUP_V12_NEUTRAL (register-neutral: no GT block, no "
                          "verdict, closed-vocabulary gap_trend instead of free-text "
                          "closing_dynamic - fixes the register leak V10 has between classes; "
                          "--gt-mode is accepted but has no effect), "
                          "v13 = PROMPT_SEMSUP_V13_CAUSAL (V12's anti-leak machinery + 5 new "
                          "closed-vocab causal-cue fields - lead vehicle lighting, ego "
                          "maneuver, road geometry, signal state, occlusion note - targeting "
                          "information NOT trivially inferable from raw pixels, <=45-word "
                          "caption budget, colour banned; --gt-mode accepted but has no "
                          "effect)")
    ap.add_argument("--gt-mode", default="blind", choices=["gt", "blind"],
                     help="v10/v10q only. 'gt' injects the manifest row's ground-truth label "
                          "(from 'target' or 'event_occurs') into the prompt and asks the "
                          "model to explain the mechanism rather than predict it. 'blind' "
                          "(default) is the control arm - identical schema minus the GT "
                          "statement, still asks for risk_score/verdict/confidence. Ignored "
                          "for v2-v9.")
    ap.add_argument("--label-filter", default="all", choices=["all", "pos", "neg"],
                     help="Filter the manifest to only positive ('pos', GT label=1) or only "
                          "negative ('neg', GT label=0) rows before captioning, resolved via "
                          "resolve_gt_label() ('target' or 'event_occurs'). Useful for running "
                          "--gt-mode gt on only the positive half of a mixed manifest and "
                          "--gt-mode blind on only the negative half (the V10-hybrid config). "
                          "'all' (default) is unchanged behavior.")
    ap.add_argument("--frame-size", type=int, default=256,
                     help="0 = native resolution, no resize (matches v6's setting)")
    ap.add_argument("--detail", default="low", choices=["low", "high", "auto"])
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=None,
                     help="omitted by default (provider default applies); set explicitly "
                          "and generously (e.g. 20000) for thinking models, which can "
                          "exhaust a small budget on internal reasoning before the JSON output")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--retry-delay", type=float, default=3.0)
    ap.add_argument("--inter-clip-delay", type=float, default=0.0,
                     help="seconds to sleep between clips, for rate-limit avoidance. NO-OP "
                          "under --concurrency > 1 (all requests are submitted up front to "
                          "the thread pool; this only paced the old strictly-serial loop).")
    ap.add_argument("--concurrency", type=int, default=4,
                     help="OpenRouter requests in flight at once. This call is dominated by "
                          "network round-trip + generation time, not local CPU (measured "
                          "~11.8s/clip end-to-end, effectively all wait) - same latency-bound "
                          "shape as the GPU trainer's frame-read bottleneck (see "
                          "semsup_common.py's prefetch_clips()), so concurrency is the actual "
                          "speed knob here too. Cost is UNCHANGED by concurrency - OpenRouter "
                          "bills by tokens processed, not wall-clock time or request count. "
                          "Start conservative: this account's rate-limit tier is unknown, and "
                          "too-high concurrency just means more 429s hitting the existing "
                          "retry/backoff in _call_model rather than a hard failure.")
    ap.add_argument("--limit", type=int, default=0, help="debug: caption only the first N pending rows")
    ap.add_argument("--dry-run", action="store_true",
                     help="print the plan (n pending, estimated tokens) without calling the API")
    ap.add_argument("--provider-order", default=None,
                     help="comma-separated OpenRouter provider slug(s) to pin, e.g. "
                          "'google-vertex' (passed as extra_body={'provider': {'order': [...], "
                          "'allow_fallbacks': False}}). allow_fallbacks=False is deliberate: "
                          "without it OpenRouter may silently route to a different provider at "
                          "a different price (e.g. Google AI Studio's 50%-off vs Vertex's "
                          "75%-off launch pricing on gemini-3.7-flash, measured 2026-08-27) - "
                          "a routing fallback would silently overpay, not fail loudly. Default "
                          "None = OpenRouter's normal auto-routing, unchanged behavior.")
    ap.add_argument("--token-cap", type=int, default=None,
                     help="if set, tokenize caption_neutral with the SigLIP tokenizer "
                          "(google/siglip-base-patch16-224) after parsing and stamp "
                          "caption_token_len onto the output row. Reported, not a hard "
                          "failure - a rejected row wastes a paid call; over-cap rows are "
                          "counted in the run summary for a targeted second pass instead.")
    args = ap.parse_args()

    load_dotenv()

    manifest = load_manifest(Path(args.manifest))
    if args.label_filter != "all":
        want_positive = args.label_filter == "pos"
        before = len(manifest)
        manifest = [r for r in manifest if resolve_gt_label(r) == want_positive]
        print(f"--label-filter {args.label_filter!r}: {before} rows -> {len(manifest)} rows.")
    done_ids = load_done_ids(Path(args.out))
    pending = [r for r in manifest if row_resume_key(r) not in done_ids]
    print(f"Manifest: {len(manifest)} rows. Already captioned: {len(done_ids)}. Pending: {len(pending)}.")
    if args.limit:
        pending = pending[:args.limit]
        print(f"--limit {args.limit}: capping this run to {len(pending)} rows.")

    if args.dry_run:
        # rough per-image token estimates: 'low' detail is a fixed ~258 tokens/image
        # regardless of resolution; 'high'/'auto' tile the image, roughly 1.5-2k
        # tokens/image at native 1280x720 (v6's setting) - both are estimates, not
        # a guarantee, check OpenRouter's current rate for the chosen model.
        per_image = 258 if args.detail == "low" else 1750
        prompt_tokens = {"v2": 650, "v3": 950, "v4": 1100, "v5": 1300, "v6": 1900, "v7": 2400,
                          "v8": 2300, "v9": 800, "v10": 1200, "v10q": 1900,
                          "v12": 1050, "v13": 1350}[args.prompt]  # v12 + the 5 causal-cue fields
        est_tokens = len(pending) * (16 * per_image + prompt_tokens)
        effective_gt_mode = args.gt_mode
        gt_note = f" gt-mode={effective_gt_mode!r}" if args.prompt in TEMPLATE_BUILDERS else ""
        print(f"[dry-run] would call model={args.model!r} prompt={args.prompt!r}{gt_note} "
              f"detail={args.detail!r} frame_size={args.frame_size} on {len(pending)} clips "
              f"(~{est_tokens/1e6:.2f}M input tokens, estimated - "
              f"check OpenRouter's current rate for {args.model!r} before running for real).")
        if args.prompt in TEMPLATE_BUILDERS:
            # Verification: eyeball that the GT block actually differs per clip
            # (gt mode), and that blind mode contains zero "GROUND TRUTH" text
            # (a silently-constant or leaking GT block would invalidate the run).
            print()
            print("=" * 70)
            print(f"[dry-run] built prompt preview (effective gt-mode {effective_gt_mode!r}):")
            shown = 0
            seen_labels = set()
            for row in pending:
                if effective_gt_mode == "gt":
                    label = resolve_gt_label(row)
                    if label in seen_labels:
                        continue
                    seen_labels.add(label)
                built = build_prompt_for_row(args.prompt, effective_gt_mode, row)
                tag = ("POSITIVE" if effective_gt_mode == "gt" and label else
                       "NEGATIVE" if effective_gt_mode == "gt" else "BLIND")
                has_gt_text = "GROUND TRUTH" in built
                print(f"--- video_id={row['video_id']} [{tag}] "
                      f"contains 'GROUND TRUTH': {has_gt_text} ---")
                print(built)
                print()
                shown += 1
                if effective_gt_mode == "gt" and len(seen_labels) >= 2:
                    break
                if effective_gt_mode == "blind" and shown >= 1:
                    break
            print("=" * 70)
        return

    if not pending:
        print("Nothing pending.")
        return

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_require_api_key(),
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "MMLM_Semsup_PromptBakeoff",
        },
    )

    frames_root = Path(args.frames_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_template = args.prompt in TEMPLATE_BUILDERS
    prompt_text = None if is_template else PROMPTS[args.prompt]

    provider_order = ([s.strip() for s in args.provider_order.split(",") if s.strip()]
                       if args.provider_order else None)
    if provider_order:
        print(f"[cfg] pinned OpenRouter provider order: {provider_order} "
              f"(allow_fallbacks=False - a routing fallback would silently bill at a "
              f"different rate rather than fail loudly)")

    # Lazy: only load the SigLIP tokenizer (a real HF download/cache hit) when
    # --token-cap is actually requested - most invocations of this script don't
    # need it and shouldn't pay the import/load cost.
    siglip_tok = None
    n_over_cap = 0
    if args.token_cap:
        from transformers import AutoTokenizer
        siglip_tok = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        print(f"[cfg] --token-cap {args.token_cap}: caption_neutral will be tokenized "
              f"with the SigLIP tokenizer and stamped as caption_token_len (reported, "
              f"not a hard failure)")

    n_ok = n_failed = 0
    t0 = time.time()

    def _fetch_one(row):
        """Runs on a worker thread: missing-frame check + prompt build + image
        encode + the OpenRouter API call ONLY. Returns (row, raw_text_or_None,
        error_message_or_None, usage_dict_or_None). Parsing/validation/
        out-row-building/writing all stay in the main thread below, unchanged
        from the old serial code - so file writes are never concurrent and
        every per-prompt branch (v10/v10q/v12/v2-v9) is exercised exactly as
        before.

        `usage` is the real per-call token count OpenRouter returns - real
        billing data, not an estimate. It was silently discarded before this
        fix; now it's logged to <out>.usage.jsonl so cost is verifiable after
        the fact instead of guessed from a stale doc figure (see the
        2026-08-11 cost-discrepancy discussion: a documented "~$81" estimate
        for this exact 1,761-window job was explicitly marked
        "not re-verified against actual billing" and should never have been
        repeated as if it were confirmed).

        WHY CONCURRENCY HELPS: this call is dominated by network round-trip +
        the model's generation time, not local CPU (measured ~11.8s/clip
        end-to-end, effectively all wait) - the same latency-bound shape as
        the GPU trainer's frame-read bottleneck (semsup_common.py's
        prefetch_clips()), just against OpenRouter instead of a network disk
        volume. Concurrent requests overlap that wait instead of paying it
        once per clip, serially."""
        vid = row["video_id"]
        frame_dir = frames_root / row["frames_dir"]
        frame_paths = [frame_dir / f"frame_{i:05d}.jpg" for i in range(1, 17)]
        missing = [p for p in frame_paths if not p.exists()]
        if missing:
            return row, None, f"{len(missing)} frame(s) missing (e.g. {missing[0]})", None

        if is_template:
            try:
                row_prompt_text = build_prompt_for_row(args.prompt, args.gt_mode, row)
            except KeyError as e:
                return row, None, str(e), None
        else:
            row_prompt_text = prompt_text

        image_b64s = [_encode_image(p, frame_size=args.frame_size) for p in frame_paths]
        messages = _build_messages(row_prompt_text, image_b64s, detail=args.detail)

        try:
            raw_text, usage = _call_model(
                client, args.model, messages,
                timeout=args.timeout, max_retries=args.max_retries,
                retry_delay=args.retry_delay, temperature=args.temperature,
                max_tokens=args.max_tokens, provider_order=provider_order,
            )
        except RuntimeError as e:
            return row, None, str(e), None
        return row, raw_text, None, usage

    # Real per-call token usage (and $ cost, when OpenRouter includes it in the
    # response) logged here - verifiable ground truth, not a pre-run estimate.
    # See the 2026-08-11 discussion: a documented cost figure for this exact
    # job was never checked against actual billing and shouldn't have been
    # trusted as one. Sidecar file, not mixed into the caption schema.
    usage_path = out_path.with_suffix(out_path.suffix + ".usage.jsonl")
    total_prompt_tok = total_completion_tok = 0
    total_cost = 0.0
    have_cost_field = False

    def _cost_str():
        """Real running totals from OpenRouter's own usage field - shared by
        every progress print below so the cost story stays consistent instead
        of being recomputed (and potentially drifting) at each call site."""
        tok_part = f"tok={total_prompt_tok + total_completion_tok:,}"
        if have_cost_field:
            return f"{tok_part} cost=${total_cost:.3f}"
        return tok_part  # OpenRouter didn't include a cost field on this account/model

    with open(out_path, "a", encoding="utf-8") as out_f, \
            open(usage_path, "a", encoding="utf-8") as usage_f, \
            ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        # All jobs submitted up front - unlike the GPU prefetch pipeline, no
        # bounded queue is needed here (no GPU memory to exhaust by having
        # many results in flight at once); the executor's internal work queue
        # naturally throttles execution to max_workers concurrent requests.
        futures = [pool.submit(_fetch_one, row) for row in pending]
        for idx, fut in enumerate(as_completed(futures), start=1):
            row, raw_text, fetch_err, usage = fut.result()
            vid = row["video_id"]
            if usage:
                usage_f.write(json.dumps({"video_id": vid, **usage}) + "\n")
                usage_f.flush()
                total_prompt_tok += usage.get("prompt_tokens", 0) or 0
                total_completion_tok += usage.get("completion_tokens", 0) or 0
                if usage.get("cost") is not None:
                    have_cost_field = True
                    total_cost += float(usage["cost"])
            if fetch_err is not None:
                print(f"  [{idx:4d}/{len(pending)}] [FAIL] {vid}: {fetch_err}")
                n_failed += 1
                continue

            parsed = _extract_json_object(raw_text)
            ok, err = validate_parsed(parsed, prompt_key=args.prompt, gt_mode=args.gt_mode)
            if not ok:
                print(f"  [{idx:4d}/{len(pending)}] [BAD-JSON] {vid}: {err}. Raw (first 200 chars): "
                      f"{raw_text[:200]!r}")
                n_failed += 1
                continue
            if err:  # ok=True but a note was returned (e.g. v5 verdict/score mismatch)
                print(f"  [{idx:4d}/{len(pending)}] [NOTE] {vid}: {err}")

            if is_template and args.prompt in ("v10", "v10q"):
                effective_gt_mode = args.gt_mode
                out_row = {
                    "video_id": vid,
                    # frames_dir is the per-WINDOW unique key (see
                    # row_resume_key() docstring) - written here so resuming
                    # this file works correctly on manifests where video_id
                    # repeats across buckets, and so downstream tooling can
                    # join back to the manifest unambiguously. horizon_label/
                    # event_occurs/t_seconds are carried through when present
                    # (train4500-derived manifests) for the same reason -
                    # val_e3a-schema manifests don't have them, hence .get().
                    "frames_dir": row.get("frames_dir"),
                    "horizon_label": row.get("horizon_label"),
                    "event_occurs": row.get("event_occurs"),
                    "t_seconds": row.get("t_seconds"),
                    "requested_time_to_event": row.get("requested_time_to_event"),
                    "gt_mode": effective_gt_mode,
                    "caption_neutral": parsed["caption_neutral"],
                    "risk_clause": parsed["risk_clause"],
                    "hazard_agent": parsed["hazard_agent"],
                    "hazard_motion": parsed["hazard_motion"],
                    "hazard_position": parsed["hazard_position"],
                    "closing_dynamic": parsed["closing_dynamic"],
                    "evidence_frames": parsed["evidence_frames"],
                    "mechanism_visible": parsed["mechanism_visible"],
                }
                if effective_gt_mode == "gt":
                    out_row["gt_label"] = int(resolve_gt_label(row))
                else:
                    out_row["risk_score"] = float(parsed["risk_score"])
                    out_row["verdict"] = int(parsed["verdict"])
                    out_row["confidence"] = float(parsed["confidence"])
                for k in OPTIONAL_COT_KEYS:  # scene_context/dynamic_objects/temporal_analysis
                    if k in parsed:
                        out_row[k] = parsed[k]
                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                out_f.flush()
                n_ok += 1
                if idx % 25 == 0 or idx == len(pending):
                    print(f"  [{idx:4d}/{len(pending)}] ok={n_ok} failed={n_failed} "
                          f"{_cost_str()} ({time.time()-t0:.0f}s)")
                if args.inter_clip_delay:
                    time.sleep(args.inter_clip_delay)
                continue

            if is_template and args.prompt == "v12":
                # Separate branch from v10/v10q above: field names differ
                # (primary_agent/agent_motion/agent_position/gap_trend/
                # agent_visible, not hazard_*/closing_dynamic/
                # mechanism_visible) and there is no verdict/risk_score/
                # confidence/risk_clause/gt_label at all - falling through to
                # the v10 branch or the v2-v9 writer below would KeyError or
                # silently drop these fields.
                out_row = {
                    "video_id": vid,
                    "frames_dir": row.get("frames_dir"),  # see row_resume_key()
                    "horizon_label": row.get("horizon_label"),
                    "event_occurs": row.get("event_occurs"),
                    "t_seconds": row.get("t_seconds"),
                    "requested_time_to_event": row.get("requested_time_to_event"),
                    "caption_neutral": parsed["caption_neutral"],
                    "primary_agent": parsed["primary_agent"],
                    "agent_motion": parsed["agent_motion"],
                    "agent_position": parsed["agent_position"],
                    "gap_trend": parsed["gap_trend"],
                    "evidence_frames": parsed["evidence_frames"],
                    "agent_visible": parsed["agent_visible"],
                }
                for k in OPTIONAL_COT_KEYS:  # scene_context/dynamic_objects/temporal_analysis
                    if k in parsed:
                        out_row[k] = parsed[k]
                if siglip_tok is not None:
                    if _stamp_token_len(out_row, siglip_tok, args.token_cap):
                        n_over_cap += 1
                        print(f"  [{idx:4d}/{len(pending)}] [OVER-CAP] {vid}: "
                              f"caption_token_len={out_row['caption_token_len']} > {args.token_cap}")
                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                out_f.flush()
                n_ok += 1
                if idx % 25 == 0 or idx == len(pending):
                    print(f"  [{idx:4d}/{len(pending)}] ok={n_ok} failed={n_failed} "
                          f"{_cost_str()} ({time.time()-t0:.0f}s)")
                if args.inter_clip_delay:
                    time.sleep(args.inter_clip_delay)
                continue

            if is_template and args.prompt == "v13":
                # Separate branch from v12 above: adds the 5 closed-vocab causal-cue
                # fields (see PROMPT_SEMSUP_V13_CAUSAL.py's docstring) - falling
                # through to v12's branch would silently drop them.
                out_row = {
                    "video_id": vid,
                    "frames_dir": row.get("frames_dir"),  # see row_resume_key()
                    "horizon_label": row.get("horizon_label"),
                    "event_occurs": row.get("event_occurs"),
                    "t_seconds": row.get("t_seconds"),
                    "requested_time_to_event": row.get("requested_time_to_event"),
                    "caption_neutral": parsed["caption_neutral"],
                    "primary_agent": parsed["primary_agent"],
                    "agent_motion": parsed["agent_motion"],
                    "agent_position": parsed["agent_position"],
                    "gap_trend": parsed["gap_trend"],
                    "lead_vehicle_lighting": parsed["lead_vehicle_lighting"],
                    "ego_maneuver": parsed["ego_maneuver"],
                    "road_geometry": parsed["road_geometry"],
                    "signal_state": parsed["signal_state"],
                    "occluded_or_peripheral": parsed["occluded_or_peripheral"],
                    "evidence_frames": parsed["evidence_frames"],
                    "agent_visible": parsed["agent_visible"],
                }
                for k in OPTIONAL_COT_KEYS:  # scene_context/dynamic_objects/temporal_analysis
                    if k in parsed:
                        out_row[k] = parsed[k]
                if siglip_tok is not None:
                    if _stamp_token_len(out_row, siglip_tok, args.token_cap):
                        n_over_cap += 1
                        print(f"  [{idx:4d}/{len(pending)}] [OVER-CAP] {vid}: "
                              f"caption_token_len={out_row['caption_token_len']} > {args.token_cap}")
                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                out_f.flush()
                n_ok += 1
                if idx % 25 == 0 or idx == len(pending):
                    print(f"  [{idx:4d}/{len(pending)}] ok={n_ok} failed={n_failed} "
                          f"{_cost_str()} ({time.time()-t0:.0f}s)")
                if args.inter_clip_delay:
                    time.sleep(args.inter_clip_delay)
                continue

            out_row = {
                "video_id": vid,
                "frames_dir": row.get("frames_dir"),  # see row_resume_key()
                "requested_time_to_event": row["requested_time_to_event"],
                "caption_neutral": parsed["caption_neutral"],
                "risk_clause": parsed["risk_clause"],
                "verdict": int(parsed["verdict"]),
                "confidence": float(parsed["confidence"]),
            }
            if args.prompt == "v3":
                missing_cot = [k for k in OPTIONAL_COT_KEYS if k not in parsed]
                if missing_cot:
                    print(f"  [{idx:4d}/{len(pending)}] [NOTE] {vid}: V3 response missing "
                          f"CoT field(s) {missing_cot} - schema partially ignored")
                for k in OPTIONAL_COT_KEYS:
                    if k in parsed:
                        out_row[k] = parsed[k]
            if args.prompt == "v9":
                out_row["risk_score"] = float(parsed["risk_score"])
            if args.prompt == "v5":
                out_row["risk_score"] = float(parsed["risk_score"])
                missing_v5_opt = [k for k in V5_OPTIONAL_KEYS if k not in parsed]
                if missing_v5_opt:
                    print(f"  [{idx:4d}/{len(pending)}] [NOTE] {vid}: V5 response missing "
                          f"optional field(s) {missing_v5_opt}")
                for k in V5_OPTIONAL_KEYS:
                    if k in parsed:
                        out_row[k] = parsed[k]
            if args.prompt == "v6":
                out_row["risk_score"] = float(parsed["risk_score"])
                for k in V6_SUBSCORE_KEYS:
                    out_row[k] = float(parsed[k])
                for k in ("ego_motion", "lateral_watch", "final_delta"):
                    out_row[k] = parsed[k]
                missing_v6_opt = [k for k in V6_OPTIONAL_KEYS if k not in parsed]
                if missing_v6_opt:
                    print(f"  [{idx:4d}/{len(pending)}] [NOTE] {vid}: V6 response missing "
                          f"optional field(s) {missing_v6_opt}")
                for k in V6_OPTIONAL_KEYS:
                    if k in parsed:
                        out_row[k] = parsed[k]
            if args.prompt == "v8":
                out_row["risk_score"] = float(parsed["risk_score"])
                for k in V6_SUBSCORE_KEYS:
                    out_row[k] = float(parsed[k])
                for k in ("delta", "true_movers", "cause", "ego_response"):
                    out_row[k] = parsed[k]
            if args.prompt == "v7":
                out_row["risk_score"] = float(parsed["risk_score"])
                for k in V6_SUBSCORE_KEYS:
                    out_row[k] = float(parsed[k])
                for k in ("static_reference", "ego_path", "apparent_vs_true",
                          "conflict_source", "final_delta"):
                    out_row[k] = parsed[k]
                for k in V7_OPTIONAL_KEYS:
                    if k in parsed:
                        out_row[k] = parsed[k]
            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            out_f.flush()
            n_ok += 1
            if idx % 25 == 0 or idx == len(pending):
                print(f"  [{idx:4d}/{len(pending)}] ok={n_ok} failed={n_failed} "
                      f"{_cost_str()} ({time.time()-t0:.0f}s)")

            if args.inter_clip_delay:
                time.sleep(args.inter_clip_delay)

    print()
    print("=" * 70)
    print(f"DONE. ok={n_ok} failed={n_failed} wall={time.time()-t0:.0f}s")
    if siglip_tok is not None:
        print(f"Token-cap ({args.token_cap}): {n_over_cap}/{n_ok} rows exceed cap "
              f"(reported only - not auto-regenerated; re-run the specific "
              f"frames_dir values against a fresh --out if a targeted fix is needed)")
    print(f"Output: {out_path}")
    print(f"Usage this run: {_cost_str()}  ->  {usage_path}")
    if have_cost_field and n_ok:
        print(f"  (real avg cost/clip this run: ${total_cost/n_ok:.4f} - "
              f"this is billing data from OpenRouter, not an estimate)")
    elif n_ok:
        print(f"  (OpenRouter did not include a 'cost' field for this model/account - "
              f"only token counts are verifiable from the API response; check the "
              f"OpenRouter dashboard directly for exact $ spent)")
    if n_failed:
        print(f"NOTE: {n_failed} rows failed/skipped - re-run this exact command to retry them "
              f"(resumable: already-captioned video_ids are skipped).")
    print("=" * 70)


def _require_api_key() -> str:
    import os
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set (check .env)")
    return key


if __name__ == "__main__":
    main()
