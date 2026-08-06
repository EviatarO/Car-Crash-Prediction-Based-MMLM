"""
PROMPT_SEMSUP_V10_GT -- GT-informed mechanism-captioning prompt (Gemini envelope).

WHY THIS EXISTS
----------------
Nine prior rounds (V2 -> V9) optimized the wrong axis: verdict accuracy. For
semantic supervision the teacher's verdict is irrelevant -- GT labels come
from dataset/train.csv. The teacher only needs to describe the MECHANISM. On
clip 00687 the model's own caption named the correct hazard ("grey SUV
merging into ego lane") while its verdict said NO -- perception was right,
the decision layer discounted it. Handing the teacher the GT label removes
that decision layer from the loop and converts an open-ended search ("is
this dangerous?") into a constrained one ("GT says collision -- find the
mechanism"), which is a strictly easier task.

This is a TEMPLATE, not a static prompt: build_prompt() injects a per-clip
ground-truth statement (positive/negative) or leaves it empty (blind
control arm). See the 2026-08-02 plan
(~/.claude/plans/CCP based BADAS/2026-08-02_Plan-V10-GT-informed-...).

Keeps PROMPT_G_OPT_v6_balanced's 7-step CoT-gated structure (the only
structure in this project that has produced strong caption fidelity --
9/18 MATCH on the 18-clip val set, the best of any round tested), and
carries over V2/V9's caption discipline unchanged:
  - caption_neutral <=40 words (SigLIP hard-truncates at 64 tokens --
    student_training/scripts/semsup_common.py, siglip_text_embed(),
    max_length=64).
  - Banned from caption_neutral: any OUTCOME word (risk, danger, collision,
    crash, imminent, safe, avoid, impact, hazard, accident) and any
    time-to-event/seconds reference. This is what stops the caption from
    degenerating into a verbalized label -- a label-collinear caption makes
    the semantic branch a redundant copy of the crash loss (that is exactly
    arm C, the label-only control, from the original prompt-bakeoff plan).
  - Observable risk DYNAMICS (closing distance, gap narrowing, braking) are
    REQUIRED, not banned -- they are the discriminative SigLIP signal. Only
    the outcome word is forbidden.
  - Canonical relational vocabulary; no two clips may produce the same
    sentence.

ANTI-HALLUCINATION ESCAPE HATCH (load-bearing): a required boolean
`mechanism_visible`. A model told "there IS a collision" that cannot
actually see one must be able to say so rather than invent an agent to
justify the label -- without this, the GT block would manufacture
confident fiction. `mechanism_visible=false` is a legitimate, expected
answer on a non-trivial fraction of clips, not a failure.

hazard_agent / hazard_motion / hazard_position / closing_dynamic /
evidence_frames are emitted in BOTH modes (gt and blind) -- this is what
lets the scorer compute a "rationalization rate": agents the GT arm names
that the blind arm, on the identical clip, never mentioned. That
comparison is the direct replacement for a two-pass debate prompt.

risk_score / verdict / confidence (the V9 contract) are emitted ONLY in
blind mode. In GT mode the model already has the label -- asking it to
also emit a verdict would just have it parrot the label back, and it
keeps AP/AUC computable on the blind arm without adding a redundant field
to the GT arm's schema.

Output schema, GT mode (strict JSON, no markdown fences, no extra text):
    {
      "scene_context": "...", "dynamic_objects": "...", "temporal_analysis": "...",
      "hazard_agent": "grey SUV",
      "hazard_motion": "drifting across lane marking into ego path",
      "hazard_position": "right adjacent lane",
      "closing_dynamic": "lateral gap narrowing frames 11-16",
      "evidence_frames": [11, 12, 13, 14, 15, 16],
      "mechanism_visible": true,
      "caption_neutral": "<=40 words, no outcome/time language",
      "risk_clause": "<=8 words"
    }

Output schema, blind mode: same as above PLUS
    "risk_score": 0-100, "verdict": 1 or 0 (derived, >=50 -> 1), "confidence": 0.0-1.0
"""

ROLE_AUDIENCE = (
    "ROLE: You are a calibrated autonomous-driving safety analyst who also writes "
    "precise scene captions for a computer vision training pipeline.\n\n"

    "AUDIENCE: The caption_neutral field is NOT for a human reader. It will be "
    "encoded by a SigLIP text encoder and used as a training target for a vision "
    "model. Write dense, literal, alt-text-style language there -- not narrative "
    "prose. The other fields (scene_context, dynamic_objects, temporal_analysis, "
    "hazard_*) are your working analysis and may be written more naturally.\n\n"
)

INPUT_BLOCK = (
    "INPUT:\n"
    "- 16 chronologically ordered dashcam frames\n"
    "- Frame 1 = oldest, Frame 16 = current moment\n"
    "- Sequence duration ~2 seconds\n"
    "- Forward-facing ego vehicle camera\n\n"
)

STEP123_BLOCK = (
    "STEP 1 -- SCENE CONTEXT:\n"
    "Describe road type, lane structure, traffic density, weather/visibility, ego "
    "vehicle motion.\n\n"

    "STEP 2 -- DYNAMIC OBJECTS:\n"
    "Identify relevant road agents. For each: relative position, motion direction, "
    "lane relation to ego, and whether motion is stable / diverging / parallel / "
    "crossing / converging.\n\n"

    "STEP 3 -- TEMPORAL ANALYSIS:\n"
    "Compare early frames (1-5), middle frames (6-11), and recent frames (12-16). "
    "Note whether spacing stays consistent, whether any agent's trajectory "
    "converges toward ego's path, and whether that convergence escalates, "
    "stabilizes, or resolves across the sequence.\n\n"
)

GT_BLOCK_POSITIVE = (
    "GROUND TRUTH (provided, train-time only -- this label is never available at "
    "inference and this caption is never used to test the model): this clip DOES "
    "end in a collision within 0-3 seconds after the final frame.\n\n"

    "STEP 4 -- MECHANISM IDENTIFICATION: Your task is NOT to decide whether a "
    "collision occurs -- it already does. Using STEP 1-3, identify WHICH agent and "
    "WHICH motion causes it. Ground every claim in specific frames. If, after "
    "careful review, you genuinely cannot identify a plausible hazard mechanism in "
    "the visible frames (the causal agent may act after the final frame, or be "
    "off-screen), set mechanism_visible=false and describe the most safety-"
    "relevant agent you DO see instead -- do NOT invent a mechanism to satisfy "
    "the label.\n\n"
)

GT_BLOCK_NEGATIVE = (
    "GROUND TRUTH (provided, train-time only -- this label is never available at "
    "inference and this caption is never used to test the model): this clip does "
    "NOT end in a collision.\n\n"

    "STEP 4 -- MECHANISM IDENTIFICATION: Using STEP 1-3, identify the dominant "
    "agent and dynamic in the scene -- the most safety-relevant road user, even "
    "though it does not lead to a collision (e.g. it maintains a stable gap, "
    "diverges, or ego brakes in time). Ground every claim in specific frames. Set "
    "mechanism_visible=true if the scene is unambiguous, false if it is genuinely "
    "empty of any relevant agent (e.g. an empty road).\n\n"
)

BLIND_DECISION_BLOCK = (
    "STEP 4 -- HAZARD IDENTIFICATION: Using STEP 1-3, identify the most safety-"
    "relevant agent in the scene, whether or not you judge it becomes a "
    "collision, and ground every claim in specific frames.\n\n"

    "STEP 5 -- FINAL DECISION GATES: Predict collision (verdict=1) ONLY if at "
    "least one holds: (A) an object has a clear closing trajectory toward ego "
    "with projected path intersection within ~3 seconds; (B) an agent is "
    "crossing into ego's trajectory with insufficient time or space to avoid "
    "conflict; (C) ego is rapidly approaching a stationary or slow obstacle with "
    "insufficient stopping space. If NONE clearly hold, predict verdict=0. Most "
    "traffic interactions do NOT result in collisions -- object presence alone "
    "does not imply danger; prefer 0 when evidence is ambiguous.\n\n"
)

CAPTION_RULES = (
    "CAPTION RULES:\n\n"

    "caption_neutral (STRICT: at most 40 words): Describe ONLY the observable "
    "physical situation -- the hazard/dominant agent's position and motion "
    "relative to ego, and the closing dynamic. State the most important relation "
    "FIRST. Two clips must never produce the same sentence -- always name the "
    "specific actor, its direction, and its proximity.\n"
    "Use these exact terms whenever they apply, so vocabulary stays consistent "
    "across clips: braking, closing distance, following distance, lane change, "
    "merging, yielding, right-of-way, crosswalk, intersection, drifting, "
    "crossing. These are the only words that should repeat -- the actor, "
    "direction, and proximity around them must be specific to this clip.\n"
    "caption_neutral MUST NOT contain any word implying danger, risk, safety, or "
    "outcome (risk, danger, collision, crash, imminent, safe, avoid, impact, "
    "hazard, accident), and MUST NOT mention time-to-event or seconds. Observable "
    "risk DYNAMICS (closing distance, gap narrowing, braking, drifting) ARE "
    "required where present -- only the outcome word is banned.\n\n"

    "risk_clause (STRICT: at most 8 words): a short evaluative judgment of the "
    "collision risk. This is the ONLY field where risk/outcome language belongs.\n\n"

    "Never invent objects, agents, or motion not visible in the frames.\n\n"
)

_SCHEMA_GT = (
    "OUTPUT -- return ONLY this JSON, no markdown fences, no extra text:\n"
    "{\n"
    '  "scene_context": "", "dynamic_objects": "", "temporal_analysis": "",\n'
    '  "hazard_agent": "", "hazard_motion": "", "hazard_position": "", '
    '"closing_dynamic": "",\n'
    '  "evidence_frames": [], "mechanism_visible": true,\n'
    '  "caption_neutral": "<=40 words, no outcome/time language", '
    '"risk_clause": "<=8 words"\n'
    "}"
)

_SCHEMA_BLIND = (
    "OUTPUT -- return ONLY this JSON, no markdown fences, no extra text:\n"
    "{\n"
    '  "scene_context": "", "dynamic_objects": "", "temporal_analysis": "",\n'
    '  "hazard_agent": "", "hazard_motion": "", "hazard_position": "", '
    '"closing_dynamic": "",\n'
    '  "evidence_frames": [], "mechanism_visible": true,\n'
    '  "caption_neutral": "<=40 words, no outcome/time language", '
    '"risk_clause": "<=8 words",\n'
    '  "risk_score": 0-100, "verdict": 1 or 0, "confidence": 0.0-1.0\n'
    "}"
)


def build_prompt(gt_mode: str, is_positive: bool = None) -> str:
    """gt_mode: 'gt' or 'blind'.
    is_positive: required when gt_mode == 'gt' (True/False from the manifest's
    GT label). Ignored when gt_mode == 'blind'.
    """
    if gt_mode not in ("gt", "blind"):
        raise ValueError(f"gt_mode must be 'gt' or 'blind', got {gt_mode!r}")

    parts = [ROLE_AUDIENCE, INPUT_BLOCK, STEP123_BLOCK]

    if gt_mode == "gt":
        if is_positive is None:
            raise ValueError("is_positive is required when gt_mode == 'gt'")
        parts.append(GT_BLOCK_POSITIVE if is_positive else GT_BLOCK_NEGATIVE)
        parts.append(CAPTION_RULES)
        parts.append(_SCHEMA_GT)
    else:
        parts.append(BLIND_DECISION_BLOCK)
        parts.append(CAPTION_RULES)
        parts.append(_SCHEMA_BLIND)

    return "".join(parts)
