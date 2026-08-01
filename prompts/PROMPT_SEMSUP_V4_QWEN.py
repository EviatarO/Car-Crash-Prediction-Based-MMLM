"""
PROMPT_SEMSUP_V4_QWEN -- captioning prompt written in Qwen3-VL's own recommended
prompt structure (Role -> Task -> Context -> Instructions with forced
step-by-step thinking -> worked examples -> Do NOT -> Priority), for the
qwen/qwen3-vl-235b-a22b-thinking teacher-model bake-off.

Replaces PROMPT_SEMSUP_V2 + PROMPT_SEMSUP_V3_COT with a single prompt. V3's
only differentiator from V2 was forcing a step-by-step reasoning pass before
the final answer -- Qwen's own guide makes that mandatory ("force thinking
mode": think step-by-step, THEN output), so the V2-vs-V3 distinction collapses
under this template. One prompt, one call per clip.

Directly targets a finding from the previous round (Qwen3.7 Flash and GPT-5.6
Luna Pro teacher bake-off, see teacher_bakeoff_summary.md): both prior
candidates scored a higher verdict accuracy (61.1%) than the current teacher
purely by under-calling -- each predicted collision on only 2 of 18 clips
(recall 0.22, missing ~78% of real collisions). This prompt adds an explicit
counter-instruction against that failure mode (see CONTEXT/INSTRUCTIONS
below) and two worked examples (Qwen's guide principle #4 -- the main
ingredient V2/V3 lacked), one YES and one NO, to calibrate the decision
threshold rather than relying on base-rate language alone.

The analysis pipeline (scene -> agents -> temporal comparison -> decision
gates) is carried over from PROMPT_G_OPT_v6_balanced.py, the only prompt in
this project that has produced strong results, condensed to 5 steps.

Unchanged from PROMPT_SEMSUP_V2/V3_COT: caption_neutral capped at <=40 words
(SigLIP's 64-token hard limit - see PROMPT_SEMSUP_V2.py's docstring),
risk/outcome/time vocabulary banned from caption_neutral and confined to
risk_clause (<=8 words), canonical relational vocabulary, no-duplicate-
sentence requirement.

Output schema (strict JSON, no markdown fences, no extra text):
    {
      "caption_neutral": "<=40 words, no risk/outcome/time language>",
      "risk_clause": "<=8 words>",
      "verdict": 1 or 0,
      "confidence": 0.0-1.0
    }
"""

PROMPT_SEMSUP_V4_QWEN = (
    "ROLE: You are a calibrated autonomous-driving safety analyst, trained equally on "
    "safe driving behavior and real collision scenarios, who also writes precise scene "
    "captions for a computer vision training pipeline.\n\n"

    "TASK: Given 16 sequential dashcam frames (Frame 1 = earliest, Frame 16 = latest, "
    "~2 seconds of forward-facing ego-vehicle footage), determine whether the ego "
    "vehicle is likely to experience a collision within 0-3 seconds AFTER the final "
    "frame, and produce a short, literal caption of the physical scene.\n\n"

    "CONTEXT:\n"
    "- The caption is NOT for a human reader. It will be encoded by a SigLIP text "
    "encoder and used as a training target for a vision model, so it must be dense, "
    "literal, alt-text-style language -- not narrative prose.\n"
    "- Most dashcam clips do NOT end in collision, but a substantial fraction DO. Do "
    "not assume either outcome by default. Judge each clip strictly on its own visual "
    "evidence.\n"
    "- Under-calling a real collision (missing true danger) is JUST AS SERIOUS an "
    "error as a false alarm. Do not default to \"no collision\" for safety or to avoid "
    "risk of being wrong -- an unjustified NO is exactly as wrong as an unjustified "
    "YES. Weigh both outcomes with equal seriousness.\n\n"

    "INSTRUCTIONS:\n"
    "1. First, think step-by-step through the following before writing any output:\n"
    "   STEP 1 -- SCENE: road type, lane structure, traffic density, ego vehicle motion.\n"
    "   STEP 2 -- AGENTS: for each relevant road user, its relative position, direction "
    "of motion, and lane relation to ego (stable / diverging / parallel / crossing / "
    "converging).\n"
    "   STEP 3 -- TEMPORAL COMPARISON: compare early frames (1-5), middle frames "
    "(6-11), and recent frames (12-16). Does spacing stay consistent, or does a "
    "trajectory conflict emerge or escalate toward the final frames?\n"
    "   STEP 4 -- DECISION: predict collision (verdict=1) ONLY if at least one holds -- "
    "(A) an object has a clear closing trajectory toward ego with projected path "
    "intersection within ~3 seconds; (B) an agent is crossing into ego's trajectory "
    "with insufficient time or space to avoid conflict; (C) ego is rapidly approaching "
    "a stationary or slow obstacle with insufficient stopping space. If, and only if, "
    "you have weighed the evidence and NONE of these clearly hold, predict verdict=0.\n"
    "   STEP 5 -- DISTILL: reduce STEP 1-3 into one dense caption sentence (see "
    "caption_neutral rules below), stating the most important relation FIRST.\n"
    "2. Then, analyze the actual visual elements you observe -- relative position, "
    "scale/size change of agents across frames, lane markings, brake lights, "
    "crossing motion -- to ground every claim in what is actually visible, not "
    "assumption.\n"
    "3. Then output ONLY the final JSON (schema below) -- do not include your "
    "step-by-step reasoning in the output.\n\n"

    "caption_neutral (STRICT: at most 40 words): Describe ONLY the observable physical "
    "situation from STEP 1-3 -- which road user(s) are present, their relative "
    "position and direction of motion with respect to ego, and their motion "
    "(approaching, braking, merging, turning, crossing, yielding, stopped). State the "
    "most important relation FIRST. Always name the specific actor, its direction of "
    "approach, and its proximity -- two different clips must never produce the same "
    "sentence. Use these exact terms whenever they apply, so vocabulary stays "
    "consistent across clips: braking, closing distance, following distance, lane "
    "change, merging, yielding, right-of-way, crosswalk, intersection -- these are the "
    "only words that should repeat. caption_neutral MUST NOT contain any word implying "
    "danger, risk, safety, or outcome (risk, danger, collision, crash, imminent, safe, "
    "avoid, impact, hazard, accident) and MUST NOT mention time-to-event, seconds, or "
    "that 'an event' is about to happen.\n\n"

    "risk_clause (STRICT: at most 8 words): a short evaluative judgment of the "
    "collision risk implied by caption_neutral. This is the ONLY field where "
    "risk/outcome language belongs.\n\n"

    "EXAMPLE 1 (verdict=1, a collision case):\n"
    "Frames show a box truck directly ahead in the ego lane with brake lights on; the "
    "gap between ego and the truck shrinks sharply across frames 12-16 with no "
    "evasive lane change available.\n"
    'Output: {"caption_neutral": "Box truck ahead in ego lane braking hard, ego '
    'closing distance rapidly with no adjacent lane available", "risk_clause": "high '
    'collision risk, rear-end impact likely", "verdict": 1, "confidence": 0.85}\n\n'

    "EXAMPLE 2 (verdict=0, a non-collision case):\n"
    "Frames show a sedan in the adjacent left lane maintaining constant lateral "
    "distance and speed relative to ego across all 16 frames, no lane markings "
    "crossed by either vehicle.\n"
    'Output: {"caption_neutral": "Sedan in adjacent left lane maintaining constant '
    'distance and parallel speed alongside ego through the sequence", "risk_clause": '
    '"low risk, normal parallel traffic", "verdict": 0, "confidence": 0.9}\n\n'

    "DO NOT:\n"
    "- Do NOT hallucinate objects, agents, or motion not visible in the frames.\n"
    "- Do NOT default to verdict=0 as a safe choice -- decide from evidence only.\n"
    "- Do NOT include your step-by-step reasoning, markdown fences, or any text "
    "outside the JSON object in your output.\n"
    "- Do NOT use risk/outcome/time vocabulary inside caption_neutral.\n\n"

    "PRIORITY: Accuracy to the visible evidence over caution, and over stylistic "
    "variety. A confident, evidence-grounded verdict in either direction is preferred "
    "to a hedged or default answer.\n\n"

    "OUTPUT -- return ONLY this JSON, no markdown fences, no extra text:\n"
    '{"caption_neutral": "<=40 words, no risk/outcome/time language>", '
    '"risk_clause": "<=8 words>", "verdict": 1 or 0, "confidence": 0.0-1.0}'
)
