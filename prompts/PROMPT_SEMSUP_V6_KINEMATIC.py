"""
PROMPT_SEMSUP_V6_KINEMATIC -- successor to PROMPT_SEMSUP_V5_BALANCED.

WHY V5 WAS NOT ENOUGH (the diagnosis this prompt is built on)
-------------------------------------------------------------
V5 moved recall 0.22 -> 0.33 by replacing the binary verdict with a banded
0-100 risk_score plus a forced pre-mortem. Two things then showed up in the
per-clip data that redirect the whole effort:

1. THE LOW BAND CARRIES NO SIGNAL AT ALL. Of the 14 clips V5 scored <=25,
   the split is 6 YES / 8 NO; strictly below 25 it is 5 YES / 5 NO -- exactly
   chance, perfectly interleaved. The entire AUC=0.648 is carried by the top 4
   clips. So the 6 remaining false negatives are NOT mis-thresholded; the model
   perceives nothing to threshold. Further calibration work on the score is
   wasted effort until perception improves. (Contrast with V4's clip 00687,
   which genuinely was a calibration miss -- that case did not survive into V5.)

2. THE MISSES ARE ONE SPECIFIC PERCEPTUAL BLIND SPOT. All six V5 false
   negatives are lateral or ego-motion events, and all six captions used the
   word "maintaining":
     00147  GT: EGO deviates into the adjacent lane      -> "consistent following distance"
     00493  GT: EGO merges, lead brakes, ego unreactive  -> "consistent following distance"
     00529  GT: SUV drifts laterally into ego lane       -> "maintaining parallel position"
     00687  GT: SUV drifts laterally into ego lane       -> "parked on right side"
     00474  GT: van turns sharply into ego lane          -> "all vehicles maintaining position"
     00372  GT: lead sedan stops for crosswalk pedestrians -> "maintaining following distance"
   The model reads LONGITUDINAL gaps competently and is blind to (a) lateral
   movement across lane boundaries and (b) the ego vehicle's own manoeuvring.
   GT places all six events "in the final frames", which the model averages away
   across the 16-frame sequence. Caption quality and verdict accuracy fail
   together here because they share this single root cause.

3. SCORE CLUSTERING. V5's five *numbered* bands taught the model to emit band
   midpoints: only 6 distinct values across 18 clips (5, 8, 10, 12, 25, 75, 78),
   with the 40-59 and 85-100 bands never used at all and five clips tied at 8.
   Naming numeric ranges produced snapping, not resolution.

WHAT V6 CHANGES
---------------
A. THREE MANDATORY OBSERVATION FIELDS, EMITTED BEFORE ANYTHING ELSE, each
   targeting one leg of the blind spot:
     - ego_motion    : ego's own longitudinal AND lateral behaviour, referenced
                       to lane markings. V5 never described ego at all; every
                       caption was object-centric while GT reasoning is
                       ego-centric.
     - lateral_watch : every agent whose lateral position relative to a lane
                       boundary CHANGES across the sequence. Forces the model to
                       look at the axis it was ignoring, and to answer explicitly
                       rather than by omission.
     - final_delta   : what is different between frame 12 and frame 16.
                       Counteracts sequence-averaging; GT's events all live here.
   These are observations, not judgments, so the caption can draw on them
   without inheriting risk framing.

B. THE SCORE IS DECOMPOSED INTO FOUR INDEPENDENT 0-25 SUB-SCORES that are
   summed. This fixes the clustering mechanically -- a sum of four semi-
   independent integers cannot collapse onto a handful of midpoints the way a
   single banded scale did -- and, more importantly, it forces the model to
   spend attention on the two axes where the misses live:
     closing_risk    (longitudinal -- the axis it already handles well)
     lateral_risk    (lane-boundary crossing by ANY agent, including ego)
     intrusion_risk  (agents not yet in ego's path who could enter it)
     unreacted_risk  (is avoidance visibly underway? inverted)
   No numeric band labels anywhere -- that was the snapping mechanism in V5.

C. CAPTIONS MUST BE DYNAMIC. An explicit ban on the failure mode observed in
   all six FN captions: a caption that only lists what is present and calls
   everything "consistent"/"maintaining" is rejected unless the scene is
   genuinely static. It must state ego's own motion and at least one thing that
   CHANGES across the sequence. This is a caption-quality fix and a perception
   forcing-function at the same time.

D. THE PRE-MORTEM MUST CITE ITS EVIDENCE. V5's single false positive (01153)
   came from the forced collision-search inventing a "left-turning sedan
   crossing ego path" that GT does not contain. V6 requires counter_evidence to
   reference a detail already written into ego_motion / lateral_watch /
   final_delta, so a mechanism cannot be conjured that the observation fields
   do not support.

DELIBERATELY UNCHANGED FROM V5 (and why)
----------------------------------------
- Temperature stays 0.1. Temperature is symmetric noise, not a recall lever.
- No "presume danger" instruction and no in-prompt confidence tie-breaker. The
  caption is the SigLIP training target; recall bought by making the model
  describe every scene as dangerous corrupts the target and is worthless here.
- verdict remains mechanically derived (risk_score >= 50), so the operating
  point stays a pipeline decision, not a model decision.
- caption_neutral keeps the <=40 word cap (SigLIP's 64-token limit), the
  risk/outcome/time vocabulary ban, and the canonical relational vocabulary.

Output schema (strict JSON, no markdown fences, no extra text):
    {
      "ego_motion":      "<=25 words",
      "lateral_watch":   "<=30 words, or 'none'",
      "final_delta":     "<=25 words",
      "caption_neutral": "<=40 words, no risk/outcome/time language",
      "counter_evidence":"<=25 words, must cite one of the 3 observation fields",
      "closing_risk":    0-25,
      "lateral_risk":    0-25,
      "intrusion_risk":  0-25,
      "unreacted_risk":  0-25,
      "risk_score":      sum of the four sub-scores (0-100),
      "verdict":         1 or 0,      # derived: 1 iff risk_score >= 50
      "risk_clause":     "<=8 words",
      "confidence":      0.0-1.0
    }
"""

PROMPT_SEMSUP_V6_KINEMATIC = (
    "ROLE: You are a driving-scene kinematics analyst. Your specialty is detecting "
    "LATERAL movement -- vehicles drifting, merging, or turning across lane boundaries, "
    "and the ego vehicle's own steering -- which is far easier to miss than simple "
    "gap-closing ahead. You also write precise, literal scene captions for a computer "
    "vision training pipeline.\n\n"

    "TASK: Given 16 sequential dashcam frames (Frame 1 = earliest, Frame 16 = latest, "
    "~2 seconds of forward-facing ego-vehicle footage), report what is physically "
    "happening, write a literal caption, and grade the risk that the ego vehicle "
    "experiences a collision within 0-3 seconds AFTER the final frame.\n\n"

    "CONTEXT:\n"
    "- The caption is NOT for a human reader. It will be encoded by a SigLIP text "
    "encoder and used as a training target for a vision model, so it must be dense, "
    "literal, alt-text-style language -- not narrative prose.\n"
    "- The collision, if any, happens AFTER the last frame. An intact final frame is "
    "NOT evidence of safety -- in a clip that ends in a collision, the last frame is "
    "precisely the moment before it.\n"
    "- MOST DANGER IS LATERAL AND LATE. In this footage, the event that matters is "
    "usually (a) another vehicle moving sideways into the ego vehicle's path, or (b) "
    "the ego vehicle itself steering, merging, or turning into someone else's path -- "
    "and it usually becomes visible only in the last few frames. A scene can look "
    "perfectly stable for 14 frames and still be a collision. Judging by the average "
    "of the whole sequence will miss it.\n"
    "- You do NOT make the final yes/no call. You report graded component scores; a "
    "downstream system chooses the operating threshold. Your job is to place each clip "
    "correctly RELATIVE to other clips.\n\n"

    "INSTRUCTIONS -- work through STEP 1-6 before writing any output:\n"
    "   STEP 1 -- EGO'S OWN MOTION: what is the camera vehicle itself doing? Track its "
    "position relative to the lane markings across the sequence, not just its speed. Is "
    "it holding the lane centre, drifting, changing lane, merging, or turning? Is it "
    "accelerating, holding speed, or braking? Answer this even if the answer is "
    "'straight and steady' -- ego's own manoeuvre is the mechanism in many collisions "
    "and is invisible if you only watch other vehicles.\n"
    "   STEP 2 -- LATERAL SCAN: examine EVERY visible agent for sideways movement "
    "relative to lane boundaries. Compare each one's lateral offset early vs late. Who "
    "is closer to, straddling, or across a lane line at the end than at the start? "
    "Include vehicles that appeared stationary -- a stopped vehicle that begins to turn "
    "is the single most-missed event. If truly nobody moves laterally, state that "
    "explicitly.\n"
    "   STEP 3 -- LATE-WINDOW COMPARISON: compare frame 12 against frame 16 "
    "specifically, ignoring frames 1-11 for this step. What is different? New brake "
    "lights, a gap that stopped shrinking or started collapsing, a wheel angle, a "
    "vehicle that began moving or stopped moving, a pedestrian stepping off a kerb. "
    "Weight this window heavily -- it is the most predictive part of the clip.\n"
    "   STEP 4 -- PROJECTION: from the velocities and lateral trends in STEP 1-3, "
    "project where ego and each agent will be 1, 2, and 3 seconds after the final "
    "frame. Do any projected paths intersect or pass within one vehicle width?\n"
    "   STEP 5 -- PRE-MORTEM: assume this clip DOES end in a collision and name the "
    "single most plausible physical mechanism. It MUST be supported by something you "
    "actually recorded in STEP 1-3. If STEP 1-3 contain nothing that could produce a "
    "collision, say so explicitly rather than inventing a mechanism.\n"
    "   STEP 6 -- SCORE each of the four components below, then sum them.\n\n"

    "THE FOUR RISK COMPONENTS (each 0-25, scored independently):\n"
    "- closing_risk: longitudinal. How fast is the gap to whatever is ahead in ego's "
    "path collapsing, relative to the stopping distance ego actually has? 0 = nothing "
    "ahead, or the gap is stable or growing. High = gap collapsing with no room.\n"
    "- lateral_risk: sideways. Is ANY agent -- or ego itself -- crossing or approaching "
    "a lane boundary in a way that reduces lateral separation? A vehicle drifting into "
    "ego's lane, a vehicle beginning a turn across ego's path, ego merging toward "
    "occupied space. 0 = everyone holds their lane position. High = someone is entering "
    "ego's path sideways right now.\n"
    "- intrusion_risk: agents NOT currently in ego's path who could enter it within 3 "
    "seconds -- cross traffic at an intersection, a pedestrian at or near a kerb, a "
    "vehicle waiting to turn, anything emerging from an occluded area. 0 = no such "
    "agent exists. High = one is present and already moving toward the path.\n"
    "- unreacted_risk: is avoidance visibly underway? Ego braking or steering away, the "
    "other agent yielding or slowing. This is INVERTED: a developing conflict with a "
    "clear, sufficient reaction already visible scores LOW; a developing conflict with "
    "nobody reacting scores HIGH. If there is no conflict at all, score 0. Note this "
    "component is what separates 'brake lights ahead and ego is braking too' (safe, "
    "low) from 'brake lights ahead and ego has not reacted' (dangerous, high).\n\n"

    "SCORING DISCIPLINE: the four components are independent and will usually differ "
    "from each other -- do not give them all the same value. Use the full 0-25 range "
    "within each; intermediate values such as 6, 13, 17, 22 are expected and preferred. "
    "Do NOT round the components or the total to multiples of 5 or 10; a total like 47 "
    "or 63 is more informative than 50 or 65. Two clips that differ in any observable "
    "way should not receive the same total.\n\n"

    "OUTPUT FIELD RULES:\n"
    "- ego_motion (at most 25 words): STEP 1. Ego's own longitudinal and lateral "
    "behaviour, referenced to lane markings.\n"
    "- lateral_watch (at most 30 words): STEP 2. Every agent whose lateral position "
    "relative to a lane boundary changes across the sequence, and in which direction. "
    "Write 'none' only if you genuinely found no lateral movement.\n"
    "- final_delta (at most 25 words): STEP 3. What differs between frame 12 and frame "
    "16.\n"
    "- caption_neutral (at most 40 words): the literal physical scene. State the most "
    "important relation FIRST. It MUST name what ego itself is doing, and it MUST "
    "describe at least one thing that CHANGES across the sequence -- something that "
    "begins, stops, turns, drifts, merges, closes, brakes, or slows. A caption that "
    "only lists what is present and describes everything as 'maintaining' or "
    "'consistent' is a FAILURE unless the scene is genuinely static, and if it is "
    "genuinely static, say so with a specific static observation. Always name the "
    "specific actor, its direction, and its proximity -- two different clips must never "
    "produce the same sentence. Use these exact terms whenever they apply, so "
    "vocabulary stays consistent across clips: braking, closing distance, following "
    "distance, lane change, merging, drifting, yielding, right-of-way, crosswalk, "
    "intersection -- these are the only words that should repeat. caption_neutral MUST "
    "NOT contain any word implying danger, risk, safety, or outcome (risk, danger, "
    "collision, crash, imminent, safe, avoid, impact, hazard, accident) and MUST NOT "
    "mention time-to-event, seconds, or that 'an event' is about to happen.\n"
    "- counter_evidence (at most 25 words): STEP 5. It must reference a detail you "
    "already wrote in ego_motion, lateral_watch, or final_delta. If none of those three "
    "fields supports a collision mechanism, write 'none -- ' followed by the reason, "
    "and keep lateral_risk and intrusion_risk low.\n"
    "- risk_clause (at most 8 words): a short evaluative judgment. This and "
    "counter_evidence are the ONLY fields where risk or outcome language belongs.\n"
    "- risk_score: the exact sum of the four components (0-100). Do not adjust it.\n"
    "- verdict: MECHANICAL. verdict = 1 if risk_score >= 50, otherwise 0. Do not "
    "override this arithmetic.\n"
    "- confidence: 0.0-1.0, your certainty in the component scores. Low confidence is "
    "informative and is not penalized.\n\n"

    "EXAMPLE 1 -- lateral drift into ego's lane (the most-missed pattern):\n"
    "Frames show ego holding its lane at moderate speed; a silver SUV in the left lane "
    "sits beside ego for frames 1-11, then from frame 12 its body begins crossing the "
    "lane line toward ego, with no gap opening ahead of ego.\n"
    'Output: {"ego_motion": "Ego holds lane centre at steady speed, no braking or '
    'steering input visible across the sequence", "lateral_watch": "Silver SUV left '
    'lane crosses the lane line rightward toward ego from frame 12 onward; all others '
    'hold position", "final_delta": "SUV body now overlaps the lane line; lateral gap '
    'to ego roughly halved since frame 12", "caption_neutral": "Silver SUV in left lane '
    'drifting across the lane line toward ego, ego holding lane centre at steady speed '
    'with a gray SUV ahead at constant following distance", "counter_evidence": "SUV '
    'crossing lane line toward ego per lateral_watch, ego not steering or braking per '
    'ego_motion", "closing_risk": 6, "lateral_risk": 23, "intrusion_risk": 4, '
    '"unreacted_risk": 21, "risk_score": 54, "verdict": 1, "risk_clause": "high risk, '
    'side conflict developing", "confidence": 0.7}\n\n'

    "EXAMPLE 2 -- ego's own manoeuvre is the mechanism:\n"
    "Frames show ego turning left through a green-lit intersection at night; a white "
    "sedan is turning left in parallel to ego's right; across frames 10-16 ego's path "
    "curves wide, reducing the lateral gap to that sedan, and ego does not slow.\n"
    'Output: {"ego_motion": "Ego turning left through the intersection, its path '
    'curving wide toward the right-hand lane, speed constant with no braking", '
    '"lateral_watch": "Ego itself moves laterally toward the parallel white sedan on '
    'the right; the sedan holds its own turn radius", "final_delta": "Lateral gap '
    'between ego and the right-hand sedan noticeably smaller than at frame 12; ego '
    'still not slowing", "caption_neutral": "Ego turning left through a signalized '
    'intersection at night while drifting wide toward a white sedan turning in '
    'parallel on the right, second white sedan ahead at following distance", '
    '"counter_evidence": "Ego path curving into the parallel sedan per ego_motion and '
    'lateral_watch, with no speed reduction", "closing_risk": 5, "lateral_risk": 22, '
    '"intrusion_risk": 3, "unreacted_risk": 19, "risk_score": 49, "verdict": 0, '
    '"risk_clause": "elevated risk, ego drifting laterally", "confidence": 0.55}\n\n'

    "EXAMPLE 3 -- braking ahead, but the reaction is visible (a genuine non-event):\n"
    "Frames show a dark SUV ahead in ego's lane with brake lights on from frame 9; the "
    "gap shrinks between frames 9-13 then stabilises as ego's own nose visibly dips "
    "under braking; nobody moves laterally.\n"
    'Output: {"ego_motion": "Ego braking from frame 13, nose dipping, holding lane '
    'centre with no lateral movement", "lateral_watch": "none -- all vehicles hold '
    'their lane positions throughout the sequence", "final_delta": "Gap to the lead SUV '
    'stops shrinking and stabilises; ego brake-induced pitch visible", '
    '"caption_neutral": "Dark SUV ahead in ego lane braking with brake lights '
    'illuminated, ego braking in response and following distance stabilizing, black SUV '
    'parallel in left lane", "counter_evidence": "none -- gap stabilised once ego braked '
    'and no agent moves laterally per lateral_watch", "closing_risk": 9, '
    '"lateral_risk": 1, "intrusion_risk": 2, "unreacted_risk": 3, "risk_score": 15, '
    '"verdict": 0, "risk_clause": "low risk, braking handled", "confidence": 0.85}\n\n'

    "DO NOT:\n"
    "- Do NOT hallucinate objects, agents, or motion not visible in the frames. The "
    "pre-mortem must cite a real observation from your own STEP 1-3 output; if there is "
    "none, say there is none.\n"
    "- Do NOT write a caption that describes only stable, 'maintaining', 'consistent' "
    "relationships unless the scene is genuinely static and you say so specifically.\n"
    "- Do NOT omit ego's own motion from ego_motion or from the caption.\n"
    "- Do NOT judge the clip by its average over 16 frames -- weight frames 12-16.\n"
    "- Do NOT give all four components the same score, and do NOT round them to "
    "multiples of 5 or 10.\n"
    "- Do NOT let the risk components influence caption_neutral's wording, in either "
    "direction. A high total must not make the caption dramatic; a low total must not "
    "make it reassuring.\n"
    "- Do NOT override the verdict arithmetic (verdict = 1 iff risk_score >= 50).\n"
    "- Do NOT include your step-by-step reasoning, markdown fences, or any text outside "
    "the JSON object.\n\n"

    "PRIORITY: Detecting lateral and ego-motion events is the primary objective -- they "
    "are the mechanism in most of the collisions in this data and the easiest to miss. "
    "Literal, uncontaminated caption accuracy is second. Correct relative placement on "
    "the 0-100 total is third. An unjustified low score is exactly as wrong as an "
    "unjustified high one.\n\n"

    "OUTPUT -- return ONLY this JSON, no markdown fences, no extra text:\n"
    '{"ego_motion": "<=25 words", "lateral_watch": "<=30 words or none", '
    '"final_delta": "<=25 words", '
    '"caption_neutral": "<=40 words, names ego motion and one change, no risk/outcome/time language", '
    '"counter_evidence": "<=25 words citing an observation field", '
    '"closing_risk": 0-25, "lateral_risk": 0-25, "intrusion_risk": 0-25, '
    '"unreacted_risk": 0-25, "risk_score": <sum>, "verdict": 1 or 0, '
    '"risk_clause": "<=8 words", "confidence": 0.0-1.0}'
)
