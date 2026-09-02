"""
PROMPT_SEMSUP_V13_CAUSAL -- register-neutral, information-dense scene captioning.

WHY THIS EXISTS
----------------
Two measured findings (2026-08-27) motivate this prompt over V12:

1. The SigLIP 64-token cap was never the binding constraint. Measured across the full
   V12 corpus (1,761 + 200 windows): median 22 tokens, p90 29, max 43 -- 0% truncated.
   The real limit was V12's OWN <=40-word rule, and the teacher never even reached that
   (median caption is half the allowed length). There is ~3x unused token headroom.

2. V12 captions describe what the V-JEPA2 encoder can plausibly already see from raw
   pixels (colour, lane position, gap trend) -- the open "Caption content redesign" item
   in docs_agents/PROJECT_STATE.md. Measured: correlation between caption token length
   and SigLIP embedding distinctiveness across 1,761 V12 captions is -0.0017 (zero) --
   longer captions of the SAME KIND of content do not separate better in embedding
   space. The fix is not more words, it is DIFFERENT information: cues that are
   genuinely hard to recover from pixels alone (a small brake light, an occluded agent,
   what the EGO vehicle itself is doing) rather than restating things the vision
   encoder's own patches already encode.

V13 = V12's proven anti-leak machinery (no GT block, closed vocabularies, symmetric
register bans, evidence_frames grounding) + five new closed-vocabulary observation
fields targeting causally-relevant, non-inferable cues + a widened caption budget
(<=45 words, still well inside the measured token headroom) + colour is dropped from
caption_neutral (colour was never causally relevant and consumed word budget).

NEW OBSERVATION FIELDS (all closed-vocabulary except occluded_or_peripheral, so class
cannot leak through free-text register -- the same anti-leak principle as V12's
gap_trend):
  lead_vehicle_lighting -- brake_lights_on / indicator_left / indicator_right /
                           flashers_on / none_visible
                           ('flashers_on', not 'hazards_on': the caption must be able
                           to verbalize this field, and 'hazard' is on the banned
                           outcome-word list -- an enum value the caption is forbidden
                           to say is an enum value that silently drops information)
  ego_maneuver          -- straight / braking / accelerating / turning_left /
                           turning_right / lane_change / stopped
  road_geometry         -- straight_road / intersection / merge / curve /
                           roundabout / parking_area
  signal_state          -- green / amber / red / stop_sign / uncontrolled /
                           none_visible
  occluded_or_peripheral -- free text, factual only: an agent partly hidden, at the
                           frame edge, or emerging from between other vehicles. Empty
                           string if nothing occluded/peripheral is relevant.

These are asked in EVERY clip regardless of class (a crash clip and a calm clip
produce the identical field set, differing only in values) -- the same structural
anti-bias mechanism that keeps V12 leak-free, applied to the new fields too.

RETAINED FROM V12 (unchanged, already proven)
-------------------------------------------------
NEUTRALITY_BLOCK, no GT block ever, gap_trend closed vocabulary, the symmetric
outcome/alarm/reassurance/time-reference bans, evidence_frames grounding, "never
invent objects not visible", STEP 1-3 chain-of-thought structure.

WHAT CHANGED FROM V12, EXPLICITLY
-------------------------------------------------
- caption_neutral: 40 -> a 42-52 word BAND (floor AND ceiling). The floor is the
  operative change: a ceiling-only instruction ("<=45 words") produced a measured
  mean of 26.7 words / 30.4 SigLIP tokens on the first 15-clip gate run -- barely
  half the available budget, because nothing penalized brevity. At this corpus's
  measured ~1.14 tokens/word, 42-52 words lands at ~48-59 tokens, i.e. it actually
  uses the --token-cap 58 budget while staying under SigLIP's hard 64.
- caption_neutral must now verbalize EVERY populated STEP 4/5/6 field, not "at least
  one causal cue". The first gate run showed the weaker instruction let the model
  record a fact in a structured field (e.g. an occluded yield sign) and then omit it
  from the caption -- but the caption is the only thing the SigLIP target sees, so a
  field-only fact is a discarded fact.
- Colour is explicitly BANNED from caption_neutral (drop colour -- it was never a
  causal cue and only consumed word budget V13 needs for the new fields).
- caption_neutral MUST incorporate at least one of: lighting, ego_maneuver, signal
  state, or the occlusion note, in addition to the gap_trend word -- this is what
  forces the new information into the actual SigLIP training target, not just into
  a side field the loss never sees.

Output schema (strict JSON, no markdown fences, no extra text):
    {
      "scene_context": "...", "dynamic_objects": "...", "temporal_analysis": "...",
      "primary_agent": "SUV", "agent_motion": "drifting toward ego lane",
      "agent_position": "right adjacent lane",
      "gap_trend": "decreasing",
      "lead_vehicle_lighting": "brake_lights_on",
      "ego_maneuver": "straight",
      "road_geometry": "straight_road",
      "signal_state": "none_visible",
      "occluded_or_peripheral": "a pedestrian is partly hidden behind the parked van",
      "evidence_frames": [11, 12, 13, 14, 15, 16],
      "agent_visible": true,
      "caption_neutral": "<=45 words, no colour, must contain gap_trend word + at "
                          "least one causal-cue word, no outcome/alarm/reassurance/"
                          "time language"
    }
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
    "you report geometry, motion, and causally-relevant scene facts.\n\n"

    "AUDIENCE: The caption_neutral field is NOT for a human reader. It will be "
    "encoded by a SigLIP text encoder and used as a training target for a vision "
    "model that ALREADY sees raw pixels -- so caption_neutral must prioritize facts "
    "the model cannot trivially re-derive from colour/shape alone: what the LEAD "
    "VEHICLE is signaling (brake lights, indicators), what EGO itself is doing, "
    "what the traffic control state is, and any agent that is occluded or at the "
    "frame's periphery and easy to miss. Write dense, literal, alt-text-style "
    "language there -- not narrative prose. The other fields (scene_context, "
    "dynamic_objects, temporal_analysis, agent_*) are your working analysis and may "
    "be written more naturally.\n\n"
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
    "crossing / converging. Explicitly check for any agent that is PARTLY OCCLUDED "
    "(behind another vehicle, a pole, signage) or at the FRAME EDGE / periphery -- "
    "these are easy to miss and are exactly what STEP 6 asks about.\n\n"

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

CAUSAL_CUES_BLOCK = (
    "STEP 6 -- CAUSAL/OBSERVATION CUES: report four additional facts, each from a "
    "CLOSED list. Pick the option that best matches the LAST frame where it is "
    "determinable; use none_visible/uncontrolled/stopped-appropriate defaults if the "
    "cue genuinely is not visible -- do NOT guess.\n\n"

    "lead_vehicle_lighting (the primary agent's or nearest lead vehicle's visible "
    "lighting -- look specifically at the rear of any vehicle ahead):\n"
    "  brake_lights_on | indicator_left | indicator_right | flashers_on | "
    "none_visible\n"
    "  (use 'flashers_on' for four-way emergency flashers -- the word 'hazard' is "
    "banned from caption_neutral, so describe them as 'flashers' there)\n\n"

    "ego_maneuver (what the EGO (camera) vehicle itself is doing across the "
    "sequence, independent of any other agent):\n"
    "  straight | braking | accelerating | turning_left | turning_right | "
    "lane_change | stopped\n\n"

    "road_geometry (the road structure at/around the current position):\n"
    "  straight_road | intersection | merge | curve | roundabout | parking_area\n\n"

    "signal_state (traffic control visible to ego, if any):\n"
    "  green | amber | red | stop_sign | uncontrolled | none_visible\n\n"

    "occluded_or_peripheral (free text, ONE short factual clause, empty string if "
    "not applicable): note any agent that is partly hidden by another object/vehicle, "
    "or sits at the frame's edge, that a viewer could plausibly miss. State only "
    "what is visible -- do not speculate about what an occluded agent might do "
    "next.\n\n"
)

CAPTION_RULES = (
    "CAPTION RULES:\n\n"

    "caption_neutral (STRICT LENGTH: between 42 and 52 words -- this is a FLOOR as "
    "well as a ceiling. A 25-word caption is a FAILURE: it wastes the encoder budget "
    "this prompt exists to use. Aim for ~48 words.)\n\n"

    "caption_neutral MUST verbalize EVERY ONE of the STEP 4/5/6 findings below -- "
    "the structured fields are working notes; this caption is the ONLY thing the "
    "training pipeline actually consumes, so a fact recorded in a field but missing "
    "from the caption is a fact thrown away. Include, in natural prose:\n"
    "  1. the primary agent: what it is, where it is relative to ego, how it moves\n"
    "  2. the gap_trend word verbatim (decreasing / increasing / constant), unless "
    "gap_trend is none_visible\n"
    "  3. the ego_maneuver (what ego itself is doing) -- ALWAYS, every caption\n"
    "  4. the road_geometry (intersection / merge / curve / straight road / "
    "roundabout / parking area) -- ALWAYS, every caption\n"
    "  5. the lead_vehicle_lighting, unless it is none_visible -- say 'brake lights', "
    "'left indicator', 'right indicator', or 'flashers'\n"
    "  6. the signal_state, unless it is none_visible -- say 'green signal', 'red "
    "signal', 'amber signal', 'stop sign', or 'uncontrolled intersection'\n"
    "  7. the occluded_or_peripheral observation, unless that field is empty -- name "
    "the occluded/peripheral agent and where it is\n"
    "State the most important relation FIRST, then the remaining facts. Two clips "
    "must never produce the same sentence -- always name the specific actor, its "
    "direction, and its proximity.\n\n"

    "WORKED EXAMPLE of the required density (48 words -- note that ego maneuver, "
    "road geometry, signal state, lead lighting, gap trend AND the peripheral agent "
    "all appear):\n"
    "  \"Ego moves straight through an uncontrolled intersection on a wet roadway, "
    "closing on a lead van directly ahead that is displaying brake lights, with the "
    "following distance decreasing across the sequence, while a cyclist waits partly "
    "occluded behind parked vehicles on the right periphery near the crosswalk.\"\n\n"
    "Do NOT mention colour of any vehicle, object, or surface -- describe shape, "
    "type, position, and motion instead. Colour is not causally relevant and this "
    "prompt needs the word budget for STEP 6 facts instead.\n"
    "Use these exact terms whenever they apply, so vocabulary stays consistent "
    "across clips: braking, closing distance, following distance, lane change, "
    "merging, yielding, right-of-way, crosswalk, intersection, drifting, "
    "crossing, brake lights, indicator, occluded. These are the only words that "
    "should repeat -- the actor, direction, and proximity around them must be "
    "specific to this clip.\n\n"

    "caption_neutral MUST NOT contain any word from these lists, for ANY clip "
    "regardless of what you believe happens next:\n"
    "  OUTCOME words: risk, danger, collision, crash, imminent, avoid, impact, "
    "hazard, accident\n"
    "  ALARM words/phrases: about to, fails to, unable to, inevitably, will "
    "strike, too late, no time\n"
    "  REASSURANCE words/phrases: safe, safely, no risk, uneventful, normal, "
    "routine, poses no, without incident\n"
    "  TIME references: any time-to-event or seconds-until statement\n"
    "  COLOUR words: any colour name (red, blue, white, black, silver, grey, "
    "etc.)\n"
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
    '  "lead_vehicle_lighting": '
    '"brake_lights_on|indicator_left|indicator_right|flashers_on|none_visible",\n'
    '  "ego_maneuver": '
    '"straight|braking|accelerating|turning_left|turning_right|lane_change|stopped",\n'
    '  "road_geometry": '
    '"straight_road|intersection|merge|curve|roundabout|parking_area",\n'
    '  "signal_state": "green|amber|red|stop_sign|uncontrolled|none_visible",\n'
    '  "occluded_or_peripheral": "",\n'
    '  "evidence_frames": [], "agent_visible": true,\n'
    '  "caption_neutral": "42-52 words (FLOOR and ceiling), no colour, must verbalize '
    'ALL of: primary agent, gap_trend word, ego_maneuver, road_geometry, plus '
    'lighting/signal_state/occlusion whenever those are not their none_visible/empty '
    'defaults. No outcome/alarm/reassurance/time language."\n'
    "}"
)


def build_prompt() -> str:
    """No arguments -- same structural anti-leak property as V12: there is no
    per-class branch, so there is no per-class register for the label to leak
    through."""
    return "".join([
        NEUTRALITY_BLOCK, ROLE_AUDIENCE, INPUT_BLOCK, STEP123_BLOCK,
        STEP4_BLOCK, GAP_TREND_BLOCK, CAUSAL_CUES_BLOCK, CAPTION_RULES, _SCHEMA,
    ])
