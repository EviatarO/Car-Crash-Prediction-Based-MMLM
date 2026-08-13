"""
PROMPT_SEMSUP_V12_NEUTRAL -- register-neutral scene captioning (Gemini envelope).

WHY THIS EXISTS
----------------
V10's captions leak the crash label through PROSE STYLE, not through content. V10
runs one of two mutually exclusive instruction blocks depending on the manifest's
GT label: GT_BLOCK_POSITIVE tells the model "this clip DOES end in a collision --
identify WHICH agent and WHICH motion causes it", GT_BLOCK_NEGATIVE tells it
"identify the dominant agent... even though it does not lead to a collision", and
the blind arm's decision gates say "prefer 0 when evidence is ambiguous". Three
different instructions produce three different registers. Measured on the real
1,761-window corpus (2026-08-08): "closing distance" appears in 88.0% of positive
captions vs 20.7% of negatives; "maintain" appears in 5.1% of positives vs 65.6%
of negatives. A bag-of-words classifier recovers the crash label from caption
text ALONE at AUC=0.964 -- higher than the vision model's own test AP (0.900).
Whatever a semantic-supervision arm measures on that corpus, it cannot be
distinguished from "a second, noisier copy of the label with different loss
geometry" (see docs_agents/DECISIONS.md, 2026-08-08).

V12 removes the per-class branch entirely: build_prompt() takes NO arguments.
There is no GT block, no decision gate, no verdict, no risk_clause -- the model
is never told, and never asked to guess, whether this clip precedes a collision.

THREE CHANGES THAT DO THE ACTUAL WORK (not just deleting the GT block)
------------------------------------------------------------------------
1. NEUTRALITY_BLOCK -- an explicit instruction that the model will caption both
   outcomes and must not let its language imply which one this is. No base-rate
   hint is given ("most clips are...") -- the real pools are near-balanced
   (1,761-pool: 856 YES / 905 NO; 4,446-pool: 2223/2223), so a base-rate
   statement would be both false and itself a prior nudging the description.

2. gap_trend -- a CLOSED four-way vocabulary (decreasing / increasing / constant
   / none_visible) that REPLACES V10's free-text closing_dynamic. This is the
   actual fix for the "maintain"/"closing" asymmetry: the descriptor is forced
   to a token selected by the frames' physics, not by what the model believes
   comes next. On a genuinely balanced corpus, "decreasing" must appear on
   negatives whenever the gap in fact decreases, and "constant" on positives
   whenever it in fact holds.

3. Symmetric register bans. caption_neutral MUST NOT contain: (a) V10's
   outcome-noun list (risk, danger, collision, crash, imminent, avoid, impact,
   hazard, accident); (b) an ALARM register (about to, fails to, unable to,
   inevitably, will strike, too late, no time) that V10 never banned because it
   never appeared under V10's own asymmetric instructions; (c) a REASSURANCE
   register (safe, safely, no risk, uneventful, normal, routine, poses no,
   without incident) -- this is the list that directly targets "maintain /
   stable / consistent"'s 65.6%-of-negatives share. Note explicitly: ordinary
   descriptive words like "maintains" or "stable" are NOT banned outright (they
   are frequently the correct description) -- only the words that assert an
   outcome or a state of safety/danger are banned, symmetrically, regardless of
   which class the clip belongs to.

WHAT IS DELIBERATELY NOT CLAIMED
-----------------------------------
Field renames (hazard_agent -> primary_agent, hazard_motion -> agent_motion,
hazard_position -> agent_position, mechanism_visible -> agent_visible) are for
register hygiene only. The V11 post-mortem (2026-08-04, see DECISIONS.md)
established that field-NAME changes alone do not fix fabrication or register
issues -- the earlier V11 attempt (separate TP/TN prompts, neutral field names)
measured WORSE than V10-blind and was reverted. The renames here ride along with
(1)-(3) above; they are not themselves claimed to do anything.

RETAINED FROM V10 (unchanged, already proven)
-------------------------------------------------
STEP 1-3 chain-of-thought (scene/objects/temporal) -- the structure with the best
measured caption fidelity of any round tested (18-clip val screen). caption_neutral
<=40 words (SigLIP hard-truncates at 64 tokens -- semsup_common.py,
siglip_text_embed(), max_length=64). Canonical relational vocabulary requirement
and "two clips must never produce the same sentence". evidence_frames grounding.
"Never invent objects, agents, or motion not visible in the frames."

Output schema (strict JSON, no markdown fences, no extra text):
    {
      "scene_context": "...", "dynamic_objects": "...", "temporal_analysis": "...",
      "primary_agent": "grey SUV",
      "agent_motion": "drifting across lane marking toward ego path",
      "agent_position": "right adjacent lane",
      "gap_trend": "decreasing",
      "evidence_frames": [11, 12, 13, 14, 15, 16],
      "agent_visible": true,
      "caption_neutral": "<=40 words, must contain the gap_trend word, no outcome/alarm/reassurance/time language"
    }

No verdict, no risk_score, no confidence, no risk_clause. This prompt does not
ask the model to judge the scene at all -- only to describe it.
"""

NEUTRALITY_BLOCK = (
    "NEUTRALITY REQUIREMENT (the single most important instruction):\n"
    "You will caption many clips. Some are followed by a collision, some are not. "
    "You are NOT told which, and must NOT try to infer it. Your description must "
    "read IDENTICALLY in register whether or not anything happens after the final "
    "frame.\n"
    "If two clips show the same physical situation -- a vehicle entering ego's "
    "lane with the gap decreasing -- they must receive the same kind of "
    "description, in the same vocabulary, regardless of the outcome.\n"
    "Do NOT signal. Do NOT hedge. Do NOT reassure. Do NOT dramatize. Geometry and "
    "motion only.\n\n"
)

ROLE_AUDIENCE = (
    "ROLE: You are a calibrated scene-description analyst who writes precise "
    "captions for a computer vision training pipeline. You do not judge outcomes; "
    "you report geometry and motion.\n\n"

    "AUDIENCE: The caption_neutral field is NOT for a human reader. It will be "
    "encoded by a SigLIP text encoder and used as a training target for a vision "
    "model. Write dense, literal, alt-text-style language there -- not narrative "
    "prose. The other fields (scene_context, dynamic_objects, temporal_analysis, "
    "agent_*) are your working analysis and may be written more naturally.\n\n"
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

STEP4_BLOCK = (
    "STEP 4 -- PRIMARY AGENT: Using STEP 1-3, identify the single most kinematically "
    "relevant road agent in the scene relative to ego -- the one whose motion most "
    "affects ego's path, regardless of whether that motion leads anywhere in "
    "particular. Ground every claim in specific frames. Set agent_visible=true if "
    "such an agent is present; set agent_visible=false if the scene has no agent "
    "worth reporting (e.g. an empty road) -- do NOT invent one to fill the field.\n\n"
)

GAP_TREND_BLOCK = (
    "STEP 5 -- GAP TREND: report how the distance between ego and the primary "
    "agent changes across frames 1->16. Choose EXACTLY ONE of these four values "
    "for gap_trend:\n"
    "  decreasing   -- the gap closes\n"
    "  increasing   -- the gap opens\n"
    "  constant     -- the gap holds within roughly its own width\n"
    "  none_visible -- no primary agent present (matches agent_visible=false)\n"
    "This is a MEASUREMENT, not an assessment. A decreasing gap is an ordinary, "
    "frequent event in traffic and implies nothing on its own about what follows. "
    "Report exactly what the frames show, independent of any judgment about "
    "danger.\n\n"
)

CAPTION_RULES = (
    "CAPTION RULES:\n\n"

    "caption_neutral (STRICT: at most 40 words): Describe ONLY the observable "
    "physical situation -- the primary agent's position and motion relative to "
    "ego, and the gap_trend value. State the most important relation FIRST. Two "
    "clips must never produce the same sentence -- always name the specific "
    "actor, its direction, and its proximity. The caption MUST contain the "
    "gap_trend word (decreasing / increasing / constant) whenever gap_trend is "
    "not none_visible.\n"
    "Use these exact terms whenever they apply, so vocabulary stays consistent "
    "across clips: braking, closing distance, following distance, lane change, "
    "merging, yielding, right-of-way, crosswalk, intersection, drifting, "
    "crossing. These are the only words that should repeat -- the actor, "
    "direction, and proximity around them must be specific to this clip.\n\n"

    "caption_neutral MUST NOT contain any word from these lists, for ANY clip "
    "regardless of what you believe happens next:\n"
    "  OUTCOME words: risk, danger, collision, crash, imminent, avoid, impact, "
    "hazard, accident\n"
    "  ALARM words/phrases: about to, fails to, unable to, inevitably, will "
    "strike, too late, no time\n"
    "  REASSURANCE words/phrases: safe, safely, no risk, uneventful, normal, "
    "routine, poses no, without incident\n"
    "  TIME references: any time-to-event or seconds-until statement\n"
    "Ordinary descriptive words (e.g. 'maintains', 'stable') are allowed when "
    "factually accurate -- only words that assert an outcome, an alarm, or a "
    "reassurance are banned, and this ban applies identically to every clip.\n\n"

    "Never invent objects, agents, or motion not visible in the frames.\n\n"
)

_SCHEMA = (
    "OUTPUT -- return ONLY this JSON, no markdown fences, no extra text:\n"
    "{\n"
    '  "scene_context": "", "dynamic_objects": "", "temporal_analysis": "",\n'
    '  "primary_agent": "", "agent_motion": "", "agent_position": "",\n'
    '  "gap_trend": "decreasing|increasing|constant|none_visible",\n'
    '  "evidence_frames": [], "agent_visible": true,\n'
    '  "caption_neutral": "<=40 words, must contain the gap_trend word, '
    'no outcome/alarm/reassurance/time language"\n'
    "}"
)


def build_prompt() -> str:
    """No arguments -- that is the point. There is no per-class branch, so
    there is no per-class register for the label to leak through."""
    return "".join([
        NEUTRALITY_BLOCK, ROLE_AUDIENCE, INPUT_BLOCK, STEP123_BLOCK,
        STEP4_BLOCK, GAP_TREND_BLOCK, CAPTION_RULES, _SCHEMA,
    ])
