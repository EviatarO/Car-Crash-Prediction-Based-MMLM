"""
PROMPT_SEMSUP_V10Q_GT -- GT-informed mechanism-captioning prompt (Qwen envelope).

Same design and same output schema as PROMPT_SEMSUP_V10_GT (see that module's
docstring for the full rationale) -- this file differs only in ENVELOPE:
Qwen3-VL's own recommended prompt structure (Role -> Task -> Context ->
Instructions-with-forced-step-by-step-thinking -> worked examples -> Do NOT ->
Priority), copied from PROMPT_SEMSUP_V4_QWEN.py, which produced the highest
caption-fidelity mean of any prompt tested against Qwen3-VL-235B-Thinking.
Using the identical schema across both envelopes is what makes the V10 vs
V10Q comparison a fair A-vs-B test of MODEL, not of prompt structure.

Motivation for testing Qwen here specifically: with the original v6_balanced
prompt (predict-the-verdict), Qwen3-VL-235B-Thinking collapsed to 0/18 YES
predictions -- a total recall failure. That collapse was in the PREDICTION
task. This prompt removes prediction from the GT arm entirely (the label is
given), so it tests whether the collapse was specific to verdict-prediction
or a more general failure on this model family -- at a meaningfully lower
price than Gemini 3.6 Flash if it works.

See PROMPT_SEMSUP_V10_GT.py for the full schema docstring (GT mode vs blind
mode field lists, the mechanism_visible escape hatch, why hazard_* fields
are emitted in both modes, why risk_score/verdict/confidence are blind-only).
"""

ROLE_TASK = (
    "ROLE: You are a calibrated autonomous-driving safety analyst who also writes "
    "precise scene captions for a computer vision training pipeline.\n\n"

    "TASK: Given 16 sequential dashcam frames (Frame 1 = earliest, Frame 16 = "
    "latest, ~2 seconds of forward-facing ego-vehicle footage), analyze the scene "
    "and produce a short, literal caption of the physical mechanism at play.\n\n"
)

CONTEXT_BLOCK = (
    "CONTEXT:\n"
    "- caption_neutral is NOT for a human reader. It will be encoded by a SigLIP "
    "text encoder and used as a training target for a vision model, so it must be "
    "dense, literal, alt-text-style language -- not narrative prose.\n"
    "- The other fields (scene_context, dynamic_objects, temporal_analysis, "
    "hazard_*) are your working analysis and may be written more naturally.\n\n"
)

GT_CONTEXT_POSITIVE = (
    "- GROUND TRUTH (provided, train-time only -- this label is never available "
    "at inference and this caption is never used to test the model): this clip "
    "DOES end in a collision within 0-3 seconds after the final frame. Your task "
    "is NOT to decide whether -- it is to identify WHICH agent and WHICH motion "
    "causes it.\n\n"
)

GT_CONTEXT_NEGATIVE = (
    "- GROUND TRUTH (provided, train-time only -- this label is never available "
    "at inference and this caption is never used to test the model): this clip "
    "does NOT end in a collision. Identify the dominant benign dynamic instead.\n\n"
)

BLIND_CONTEXT = (
    "- Most dashcam clips do NOT end in collision, but a substantial fraction DO. "
    "Do not assume either outcome by default -- judge each clip strictly on its "
    "own visual evidence.\n\n"
)

INSTRUCTIONS_GT = (
    "INSTRUCTIONS:\n"
    "1. First, think step-by-step through the following before writing any output:\n"
    "   STEP 1 -- SCENE: road type, lane structure, traffic density, ego vehicle "
    "motion.\n"
    "   STEP 2 -- AGENTS: for each relevant road user, its relative position, "
    "direction of motion, and lane relation to ego (stable / diverging / "
    "parallel / crossing / converging).\n"
    "   STEP 3 -- TEMPORAL COMPARISON: compare early frames (1-5), middle frames "
    "(6-11), and recent frames (12-16). Does spacing stay consistent, or does a "
    "trajectory conflict emerge or escalate toward the final frames?\n"
    "   STEP 4 -- MECHANISM: given the ground truth above, identify the specific "
    "hazard_agent, its hazard_motion, its hazard_position relative to ego, and "
    "the closing_dynamic across frames. Ground every claim in visible evidence "
    "and cite the specific evidence_frames. If you genuinely cannot identify a "
    "plausible mechanism (the cause may act after the final frame or be "
    "off-screen), set mechanism_visible=false and describe the most safety-"
    "relevant agent you DO see instead -- do NOT invent a mechanism to satisfy "
    "the label.\n"
    "   STEP 5 -- DISTILL: reduce the above into one dense caption sentence (see "
    "caption_neutral rules below), stating the most important relation FIRST.\n"
    "2. Then output ONLY the final JSON (schema below) -- do not include your "
    "step-by-step reasoning in the output.\n\n"
)

INSTRUCTIONS_BLIND = (
    "INSTRUCTIONS:\n"
    "1. First, think step-by-step through the following before writing any output:\n"
    "   STEP 1 -- SCENE: road type, lane structure, traffic density, ego vehicle "
    "motion.\n"
    "   STEP 2 -- AGENTS: for each relevant road user, its relative position, "
    "direction of motion, and lane relation to ego (stable / diverging / "
    "parallel / crossing / converging).\n"
    "   STEP 3 -- TEMPORAL COMPARISON: compare early frames (1-5), middle frames "
    "(6-11), and recent frames (12-16). Does spacing stay consistent, or does a "
    "trajectory conflict emerge or escalate toward the final frames?\n"
    "   STEP 4 -- HAZARD IDENTIFICATION: identify the most safety-relevant agent "
    "(hazard_agent/hazard_motion/hazard_position/closing_dynamic), whether or not "
    "you judge it becomes a collision. Ground every claim in visible evidence and "
    "cite the specific evidence_frames.\n"
    "   STEP 5 -- DECISION: predict collision (verdict=1) ONLY if at least one "
    "holds -- (A) an object has a clear closing trajectory toward ego with "
    "projected path intersection within ~3 seconds; (B) an agent is crossing "
    "into ego's trajectory with insufficient time or space to avoid conflict; "
    "(C) ego is rapidly approaching a stationary or slow obstacle with "
    "insufficient stopping space. If, and only if, NONE of these clearly hold, "
    "predict verdict=0. Do not default to 0 for safety, and do not default to 1 "
    "to avoid missing danger -- weigh both errors equally.\n"
    "   STEP 6 -- DISTILL: reduce the above into one dense caption sentence (see "
    "caption_neutral rules below), stating the most important relation FIRST.\n"
    "2. Then output ONLY the final JSON (schema below) -- do not include your "
    "step-by-step reasoning in the output.\n\n"
)

CAPTION_RULES = (
    "caption_neutral (STRICT: at most 40 words): Describe ONLY the observable "
    "physical situation -- the hazard/dominant agent's position and motion "
    "relative to ego, and the closing dynamic. State the most important relation "
    "FIRST. Two clips must never produce the same sentence -- always name the "
    "specific actor, its direction, and its proximity. Use these exact terms "
    "whenever they apply, so vocabulary stays consistent across clips: braking, "
    "closing distance, following distance, lane change, merging, yielding, "
    "right-of-way, crosswalk, intersection, drifting, crossing -- these are the "
    "only words that should repeat. caption_neutral MUST NOT contain any word "
    "implying danger, risk, safety, or outcome (risk, danger, collision, crash, "
    "imminent, safe, avoid, impact, hazard, accident) and MUST NOT mention "
    "time-to-event or seconds. Observable risk DYNAMICS (closing distance, gap "
    "narrowing, braking, drifting) ARE required where present -- only the "
    "outcome word is banned.\n\n"

    "risk_clause (STRICT: at most 8 words): a short evaluative judgment of the "
    "collision risk implied by caption_neutral. This is the ONLY field where "
    "risk/outcome language belongs.\n\n"
)

EXAMPLE_GT = (
    "EXAMPLE (GT positive, mechanism visible):\n"
    "GROUND TRUTH: collision. Frames show a grey SUV in the right adjacent lane "
    "drifting left across the lane marking toward ego, gap narrowing across "
    "frames 11-16.\n"
    'Output: {"scene_context": "urban intersection, clear day", "dynamic_objects": '
    '"grey SUV right adjacent lane, black sedan ahead", "temporal_analysis": '
    '"SUV lateral position shifts toward ego lane from frame 10 onward", '
    '"hazard_agent": "grey SUV", "hazard_motion": "drifting left across lane '
    'marking", "hazard_position": "right adjacent lane", "closing_dynamic": '
    '"lateral gap narrowing frames 11-16", "evidence_frames": [11,12,13,14,15,16], '
    '"mechanism_visible": true, "caption_neutral": "grey SUV in right adjacent '
    'lane drifts left across the lane marking toward ego, lateral gap narrowing '
    'through the final frames", "risk_clause": "high risk, lateral drift into ego '
    'path"}\n\n'
)

EXAMPLE_BLIND = (
    "EXAMPLE (blind, verdict=1):\n"
    "Frames show a box truck directly ahead in the ego lane with brake lights on; "
    "the gap between ego and the truck shrinks sharply across frames 12-16 with "
    "no evasive lane change available.\n"
    'Output: {"scene_context": "urban, ego lane single-file traffic", '
    '"dynamic_objects": "box truck directly ahead", "temporal_analysis": "gap to '
    'truck shrinks sharply frames 12-16", "hazard_agent": "box truck", '
    '"hazard_motion": "braking ahead in ego lane", "hazard_position": "directly '
    'ahead", "closing_dynamic": "gap closing rapidly, no adjacent lane available", '
    '"evidence_frames": [12,13,14,15,16], "mechanism_visible": true, '
    '"caption_neutral": "Box truck ahead in ego lane braking hard, ego closing '
    'distance rapidly with no adjacent lane available", "risk_clause": "high '
    'collision risk, rear-end impact likely", "risk_score": 82, "verdict": 1, '
    '"confidence": 0.85}\n\n'
)

DO_NOT = (
    "DO NOT:\n"
    "- Do NOT hallucinate objects, agents, or motion not visible in the frames.\n"
    "- Do NOT include your step-by-step reasoning, markdown fences, or any text "
    "outside the JSON object in your output.\n"
    "- Do NOT use risk/outcome/time vocabulary inside caption_neutral.\n"
)
DO_NOT_GT = DO_NOT + (
    "- Do NOT invent a mechanism just to match the ground-truth label -- use "
    "mechanism_visible=false when you genuinely cannot find one.\n\n"
)
DO_NOT_BLIND = DO_NOT + (
    "- Do NOT default to verdict=0 as a safe choice, and do NOT default to "
    "verdict=1 to avoid missing danger -- decide from evidence only.\n\n"
)

PRIORITY = (
    "PRIORITY: Accuracy to the visible evidence over caution, and over stylistic "
    "variety. A confident, evidence-grounded description is preferred to a "
    "hedged or generic one.\n\n"
)

_SCHEMA_GT = (
    "OUTPUT -- return ONLY this JSON, no markdown fences, no extra text:\n"
    '{"scene_context": "", "dynamic_objects": "", "temporal_analysis": "", '
    '"hazard_agent": "", "hazard_motion": "", "hazard_position": "", '
    '"closing_dynamic": "", "evidence_frames": [], "mechanism_visible": true, '
    '"caption_neutral": "<=40 words, no outcome/time language", '
    '"risk_clause": "<=8 words"}'
)

_SCHEMA_BLIND = (
    "OUTPUT -- return ONLY this JSON, no markdown fences, no extra text:\n"
    '{"scene_context": "", "dynamic_objects": "", "temporal_analysis": "", '
    '"hazard_agent": "", "hazard_motion": "", "hazard_position": "", '
    '"closing_dynamic": "", "evidence_frames": [], "mechanism_visible": true, '
    '"caption_neutral": "<=40 words, no outcome/time language", '
    '"risk_clause": "<=8 words", "risk_score": "0-100", "verdict": "1 or 0", '
    '"confidence": "0.0-1.0"}'
)


def build_prompt(gt_mode: str, is_positive: bool = None) -> str:
    """gt_mode: 'gt' or 'blind'.
    is_positive: required when gt_mode == 'gt' (True/False from the manifest's
    GT label). Ignored when gt_mode == 'blind'.
    """
    if gt_mode not in ("gt", "blind"):
        raise ValueError(f"gt_mode must be 'gt' or 'blind', got {gt_mode!r}")

    parts = [ROLE_TASK, CONTEXT_BLOCK]

    if gt_mode == "gt":
        if is_positive is None:
            raise ValueError("is_positive is required when gt_mode == 'gt'")
        parts.append(GT_CONTEXT_POSITIVE if is_positive else GT_CONTEXT_NEGATIVE)
        parts.append(INSTRUCTIONS_GT)
        parts.append(CAPTION_RULES)
        parts.append(EXAMPLE_GT)
        parts.append(DO_NOT_GT)
        parts.append(PRIORITY)
        parts.append(_SCHEMA_GT)
    else:
        parts.append(BLIND_CONTEXT)
        parts.append(INSTRUCTIONS_BLIND)
        parts.append(CAPTION_RULES)
        parts.append(EXAMPLE_BLIND)
        parts.append(DO_NOT_BLIND)
        parts.append(PRIORITY)
        parts.append(_SCHEMA_BLIND)

    return "".join(parts)
