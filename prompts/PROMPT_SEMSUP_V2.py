"""
PROMPT_SEMSUP_V2 -- vision-grounded captioning prompt for the semantic-supervision
prompt-bakeoff (see PLAN: prompt-bakeoff-harness, 2026-07-27, and
docs_agents/DECISIONS.md's "Which scale-up path?" entry).

Replaces the original two-prompt design (Driving-Semantic / Risk-Aware-Causal).
ONE prompt, ONE teacher call per clip, THREE arms built locally from its output
(see the plan for why: half the teacher cost, no description-drift confound
between arms, and a paired A-vs-B comparison instead of an unpaired one).

Hard constraint this prompt is built around: the trained SigLIP text tower this
caption feeds truncates at 64 tokens (student_training/scripts/semsup_common.py,
siglip_text_embed(), max_length=64) -- confirmed via tokenizer.model_max_length.
A representative caption in the ORIGINAL two-prompt spec (70-120 words) measured
at 128 SigLIP tokens: 50% discarded, and the discarded half was always the
outcome clause. caption_neutral is capped at <=40 words here (measured ~<=55
SigLIP tokens on the existing 267-caption set's register) and states the
outcome/motion FIRST, never last, so nothing load-bearing sits past the cutoff.

Output schema (strict JSON, no markdown fences, no extra text):
    {
      "caption_neutral": "<=40 words, no risk/outcome language, see rules below>",
      "risk_clause": "<=8 words, the ONLY field allowed to state risk/outcome>",
      "verdict": 1 or 0,
      "confidence": 0.0-1.0
    }

caption_neutral + ", " + risk_clause = Arm B. caption_neutral alone = Arm A.
verdict/confidence are QA-only and NEVER enter either caption arm's text --
embedding them would make Arm A's "neutral" claim false and collapse the A/B
contrast the whole experiment depends on.
"""

PROMPT_SEMSUP_V2 = (
    "ROLE: You are an autonomous-driving scene analyst producing a short semantic "
    "description of a dashcam video clip (16 sequential frames, Frame 1 = earliest, "
    "Frame 16 = latest).\n\n"

    "AUDIENCE: This description is NOT for a human reader. It will be encoded by a "
    "SigLIP text encoder and used as a training target for a vision model. Write "
    "dense, literal, alt-text-style language -- not narrative prose.\n\n"

    "TASK: Produce two separate pieces of text about this clip.\n\n"

    "PART 1 -- caption_neutral (STRICT: at most 40 words):\n"
    "Describe ONLY the observable physical situation: which road user(s) are "
    "present, their relative position and direction of motion with respect to the "
    "ego vehicle, and their motion (approaching, braking, merging, turning, "
    "crossing, yielding, stopped). State the most important relation FIRST, not "
    "last. Every clip must produce a DIFFERENT sentence -- always name the "
    "specific actor, its direction of approach, and its proximity; never fall "
    "back to a generic template.\n\n"
    "Use these exact terms whenever they apply, so vocabulary stays consistent "
    "across clips: braking, closing distance, following distance, lane change, "
    "merging, yielding, right-of-way, crosswalk, intersection. These are the only "
    "words that should repeat -- the actor, direction, and proximity around them "
    "must be specific to this clip.\n\n"
    "caption_neutral MUST NOT contain: any word implying danger, risk, safety, or "
    "outcome (e.g. risk, danger, collision, crash, imminent, safe, avoid, impact, "
    "hazard, accident) -- those belong ONLY in risk_clause below. MUST NOT mention "
    "time-to-event, seconds, or that 'an event' is about to happen -- you do not "
    "know whether one will. Describe strictly what is visible in the 16 frames.\n\n"

    "PART 2 -- risk_clause (STRICT: at most 8 words):\n"
    "A short evaluative judgment of the collision risk implied by the situation "
    "in caption_neutral (e.g. 'high collision risk, impact likely' or "
    "'low risk, normal driving'). This is the ONLY field where risk/outcome "
    "language is allowed.\n\n"

    "PART 3 -- verdict and confidence (QA fields, not description):\n"
    "verdict: 1 if you judge a collision is about to occur, else 0. "
    "confidence: your confidence in that verdict, 0.0-1.0.\n\n"

    "Use objective observations only. Never invent unseen objects. Never "
    "hallucinate events not visible in the frames.\n\n"

    "OUTPUT -- return ONLY this JSON, no markdown fences, no extra text:\n"
    '{"caption_neutral": "<=40 words, no risk/outcome/time language>", '
    '"risk_clause": "<=8 words>", "verdict": 1 or 0, "confidence": 0.0-1.0}'
)
