"""
PROMPT_SEMSUP_V7_EGOFRAME -- successor to PROMPT_SEMSUP_V6_KINEMATIC.

THE DIAGNOSIS THIS PROMPT IS BUILT ON
--------------------------------------
Six rounds of teacher/prompt variation (v6-prompt x 3 models, then V4, V5, V6 on
qwen3-vl-235b-thinking) all landed at 9-11 correct out of 18. Prompt changes moved
errors between the FP and FN columns but never moved the total. The V6 run's
observation fields finally exposed why.

ROOT CAUSE: FAILED EGO-MOTION / OBJECT-MOTION DECOMPOSITION.
On the 6 val clips where GT states the EGO VEHICLE ITSELF is turning, V6 reported
"holding lane centre / proceeding straight" on 4 of them:
    01153  GT: ego makes a smooth RIGHT turn   -> "proceeding straight"
    00687  GT: ego turns LEFT                  -> "proceeding straight"
    00147  GT: ego turns LEFT                  -> "holding lane centre"
    00493  GT: ego turns LEFT and merges       -> "holding lane centre"
The phrase "lane centre/position" appeared in 18/18 ego_motion fields, including
the two clips it got right -- the field was being template-filled, not observed.

When the ego vehicle rotates, every STATIC object sweeps laterally across the
image plane. A model that believes it is travelling straight must attribute that
sweep to the objects themselves. This single failure produces BOTH error types:
  - FALSE POSITIVE: ego turns right, stopped cars sweep leftward across frame,
    model reports "white sedan turning left across intersection into ego's path"
    (01153, exactly this, in both V5 and V6 independently).
  - FALSE NEGATIVE: ego turns into another vehicle's path; the model attributes
    the closing to the other vehicle ("sedan merged into ego's lane", 00147) or
    misses it entirely (00493, 00687), so ego's own manoeuvre -- the actual
    mechanism -- is never named.
FP and FN here are the SAME defect with opposite sign, which is why six rounds of
rebalancing the decision threshold redistributed errors without reducing them.

SECONDARY FINDING (not fixable by prompting, recorded for context): recall
degrades sharply with prediction lead time -- across all 6 runs, 50.0% at 0.5s,
44.4% at 1.0s, 11.1% at 1.5s. The 1.5s clips are near-unsolvable from these 16
frames for every model tested.

WHAT V7 CHANGES
---------------
1. EGO ROTATION IS ESTABLISHED FIRST, FROM STATIC SCENE STRUCTURE ONLY.
   Before any vehicle is considered, the model must track fixed world features --
   lane markings, kerbs, buildings, poles, signs, the horizon/vanishing point --
   and report which way they sweep. Other vehicles are explicitly BANNED as
   evidence for this step, because they are the very thing whose motion is being
   decomposed. If the whole static scene sweeps one way, ego is rotating the
   other way. This is the step V6 lacked entirely.

2. EXPLICIT APPARENT-VS-TRUE MOTION TEST. Each agent is then classified by
   whether it moves RELATIVE TO THE STATIC BACKGROUND, or merely sweeps along
   with it. Only the former counts as real lateral movement. This is the direct
   guard against V6's confabulated "vehicle crossing my path" reports, which V6's
   counter_evidence citation rule structurally could not catch (it verified
   internal consistency, not grounding).

3. CONFLICT ATTRIBUTION AS AN EXPLICIT ENUM: ego_into_other / other_into_ego /
   longitudinal / none. Forces the model to commit to WHO is closing on WHOM,
   which is the exact judgment it was silently getting backwards.

4. ANTI-BOILERPLATE RULE. "Holding lane centre / maintaining lane position" may
   only be asserted with a named static reference supporting it. An unsupported
   default is the failure mode being fixed.

5. FOUR WORKED EXAMPLES, DELIBERATELY BALANCED 2 POSITIVE / 2 NEGATIVE AND
   COVERING ALL FOUR CONFUSION QUADRANTS -- including the case V6 had no template
   for: ego turning while nothing truly moves (apparent motion only -> verdict 0).
   V6's examples were 2 lateral-positive / 1 longitudinal-negative, an unbalanced
   few-shot prior that plausibly drove the template-completion failures.

EXAMPLES ARE PRINCIPLE-BASED, NOT SCENE-BASED (deliberate anti-overfit measure)
-------------------------------------------------------------------------------
An earlier draft of this prompt used four worked examples built directly from the
failing scenes in the 18-clip val set (urban signalized intersections, a braking
sedan on a multi-lane highway). That makes the prompt fit those scenes rather
than teach the underlying rule, and it burns the val set as a measurement.

The examples below were rewritten to carry the SAME four principles in scene
types and actor types that do not occur in the val set at all:
    EX1  rural roundabout, stationary tractor      (apparent motion only -> 0)
    EX2  motorway slip-road merge, coach           (ego_into_other      -> 1)
    EX3  tunnel, box van                           (longitudinal, reacted -> 0)
    EX4  narrow country lane, cyclist              (other_into_ego      -> 1)
None of these settings (roundabout, slip road, tunnel, country lane) and none of
these actors (tractor, coach, box van, cyclist) appear in val_e3a.jsonl, which is
entirely urban intersections, multi-lane highways, a rainy taxi scene and a gas
station. A model that improves here must be applying the ego-frame RULE, not
recognising a rehearsed scene.

RESIDUAL CONTAMINATION -- STILL TRUE, READ BEFORE INTERPRETING RESULTS
----------------------------------------------------------------------
Generic examples remove the scene-level fit, but NOT all of it: the underlying
diagnosis (that ego-motion/object-motion confusion is the dominant error) was
itself derived by inspecting per-clip failures on those same 18 clips. So a V7
result on val_e3a.jsonl is still not a clean held-out measurement -- treat it as
a MECHANISM CHECK (does ego_path now report turning where GT says ego turns? does
the apparent-vs-true test suppress the invented crossings?) rather than as a
score to rank prompts by. The honest ranking test is a frozen comparison on
dataset/manifests/semsup_promptbakeoff.jsonl (498 clips, 249/249 balanced,
83 per horizon bucket, zero overlap with val, never captioned), which also gives
far better statistical power than n=18 -- where one clip is 5.6% of accuracy and
every round run so far sits inside every other round's confidence interval.

DELIBERATELY UNCHANGED FROM V5/V6 (and why)
--------------------------------------------
- temperature 0.1; no "presume danger"; no in-prompt confidence tie-breaker.
- verdict remains mechanically derived (risk_score >= 50) so the operating point
  stays a pipeline decision.
- caption_neutral keeps the <=40 word cap (SigLIP 64-token limit), the
  risk/outcome/time vocabulary ban, and the canonical relational vocabulary.
- The four 0-25 sub-scores are retained: V6 raised distinct score values from 6
  to 11 and finally populated the 40-59 band, so the decomposition works.

Output schema (strict JSON, no markdown fences, no extra text):
    {
      "static_reference": "<=25 words: fixed features and which way they sweep",
      "ego_path":         "<=20 words: ego's trajectory, justified by the above",
      "apparent_vs_true": "<=40 words: per agent, truly moving vs sweeping with background",
      "conflict_source":  "ego_into_other" | "other_into_ego" | "longitudinal" | "none",
      "final_delta":      "<=25 words: frame 12 vs frame 16",
      "caption_neutral":  "<=40 words, no risk/outcome/time language",
      "counter_evidence": "<=25 words, must cite one of the observation fields",
      "closing_risk":     0-25,
      "lateral_risk":     0-25,
      "intrusion_risk":   0-25,
      "unreacted_risk":   0-25,
      "risk_score":       sum of the four sub-scores (0-100),
      "verdict":          1 or 0,     # derived: 1 iff risk_score >= 50
      "risk_clause":      "<=8 words",
      "confidence":       0.0-1.0
    }
"""

PROMPT_SEMSUP_V7_EGOFRAME = (
    "ROLE: You are a driving-scene motion analyst. Your core skill is separating the "
    "camera vehicle's OWN motion from the motion of other road users. In a "
    "forward-facing dashcam sequence these two look identical if you are not careful: "
    "when the camera vehicle turns, every stationary object in the world sweeps "
    "sideways across the image, and a parked car can be mistaken for a car cutting "
    "across your path. Getting this separation right is the whole job. You also write "
    "precise, literal scene captions for a computer vision training pipeline.\n\n"

    "TASK: Given 16 sequential dashcam frames (Frame 1 = earliest, Frame 16 = latest, "
    "~2 seconds of forward-facing ego-vehicle footage), determine what the ego vehicle "
    "and every other agent are actually doing, write a literal caption, and grade the "
    "risk that the ego vehicle experiences a collision within 0-3 seconds AFTER the "
    "final frame.\n\n"

    "CONTEXT:\n"
    "- The caption is NOT for a human reader. It will be encoded by a SigLIP text "
    "encoder and used as a training target for a vision model, so it must be dense, "
    "literal, alt-text-style language -- not narrative prose.\n"
    "- The collision, if any, happens AFTER the last frame. An intact final frame is "
    "NOT evidence of safety.\n"
    "- THE EGO VEHICLE IS OFTEN THE ONE THAT MOVES. It turns, changes lane, merges, "
    "and drifts. When it does, the collision mechanism is frequently 'ego moved into "
    "someone', not 'someone moved into ego'. Both directions occur in this data and "
    "confusing them produces a wrong answer either way -- reporting ego as travelling "
    "straight when it is turning is the single most common and most costly error on "
    "this task.\n"
    "- You do NOT make the final yes/no call. You report graded component scores; a "
    "downstream system chooses the operating threshold.\n\n"

    "INSTRUCTIONS -- work through STEP 1-7 in order before writing any output. STEP 1 "
    "and STEP 2 must be completed before you consider any vehicle at all.\n"
    "   STEP 1 -- FIX THE WORLD FRAME. Identify features that CANNOT move in the real "
    "world: lane markings, kerbs, road edges, crosswalk stripes, buildings, poles, "
    "traffic-light posts, signs, parked vehicles, the horizon line, the vanishing "
    "point of the road. Track how these shift across frames 1 -> 16. Do they hold "
    "position, or does the whole set sweep left, or sweep right? YOU MAY NOT USE "
    "MOVING VEHICLES FOR THIS STEP -- they are the thing you are about to measure, and "
    "using them makes the measurement circular.\n"
    "   STEP 2 -- DERIVE EGO'S OWN PATH from STEP 1 alone. If the entire static scene "
    "sweeps LEFT across the frames, the ego vehicle is turning RIGHT. If it sweeps "
    "RIGHT, ego is turning LEFT. If static features hold position and only grow "
    "larger, ego is travelling straight. Also judge ego's speed trend: is the static "
    "scene expanding faster (accelerating), steadily (constant), or slowing "
    "(braking)? Classify ego's path as: straight / turning left / turning right / "
    "changing lane left / changing lane right / merging / slowing / stopped. State "
    "which specific static feature you used as evidence. Do NOT write that ego is "
    "'holding lane centre' or 'maintaining lane position' unless you can name the "
    "static feature that shows it.\n"
    "   STEP 3 -- SUBTRACT EGO MOTION. Now examine each other road user. For each one, "
    "ask the decisive question: does it move RELATIVE TO THE STATIC BACKGROUND you "
    "fixed in STEP 1, or does it merely sweep across the image together with that "
    "background? A vehicle that keeps constant position against the kerb, the lane "
    "line, or the buildings behind it is NOT moving laterally -- it only appears to, "
    "because ego is rotating. Only a vehicle whose position CHANGES relative to those "
    "fixed features is genuinely moving. Classify each agent as TRULY MOVING (and in "
    "which direction) or APPARENT ONLY.\n"
    "   STEP 4 -- ATTRIBUTE THE CONFLICT. Using STEP 2 and STEP 3, decide who is "
    "closing on whom, and pick exactly one: 'ego_into_other' (ego's own turn, merge, "
    "drift or lane change is bringing it toward another agent), 'other_into_ego' (a "
    "genuinely moving agent is coming into ego's path), 'longitudinal' (the conflict "
    "is purely a gap collapsing straight ahead, no lateral component), or 'none' (no "
    "conflict developing).\n"
    "   STEP 5 -- LATE WINDOW. Compare frame 12 against frame 16 specifically. What "
    "changed? New brake lights, a gap that started collapsing, a wheel angle, a "
    "pedestrian stepping off a kerb, a vehicle beginning to move. Weight this window "
    "heavily -- it is the most predictive part of the clip.\n"
    "   STEP 6 -- PRE-MORTEM. Assume this clip DOES end in a collision and name the "
    "single most plausible mechanism. It MUST be supported by something you recorded "
    "in STEP 2, 3 or 5. If nothing you recorded supports a collision, say so "
    "explicitly rather than inventing a mechanism.\n"
    "   STEP 7 -- SCORE the four components below and sum them.\n\n"

    "THE FOUR RISK COMPONENTS (each 0-25, scored independently):\n"
    "- closing_risk: longitudinal. How fast is the gap to whatever is ahead in ego's "
    "path collapsing, relative to the stopping distance ego has? 0 = nothing ahead, or "
    "the gap is stable or growing.\n"
    "- lateral_risk: sideways, counting ONLY motion you classified as TRULY MOVING in "
    "STEP 3, plus ego's own lateral movement from STEP 2. A vehicle you classified as "
    "APPARENT ONLY contributes ZERO here, no matter how dramatic its sweep across the "
    "image. If ego itself is turning or merging toward an occupied space, that counts "
    "fully.\n"
    "- intrusion_risk: agents not currently in ego's path who could enter it within 3 "
    "seconds -- cross traffic, a pedestrian at a kerb, a vehicle waiting to turn, "
    "anything emerging from occlusion. 0 = no such agent exists.\n"
    "- unreacted_risk: is avoidance visibly underway? Ego braking or steering away, "
    "the other agent yielding or slowing. INVERTED: a developing conflict with a "
    "clear, sufficient reaction visible scores LOW; a developing conflict with nobody "
    "reacting scores HIGH. No conflict at all = 0. This is what separates 'brake "
    "lights ahead and ego is braking too' (safe) from 'brake lights ahead and ego has "
    "not reacted' (dangerous).\n\n"

    "SCORING DISCIPLINE: the four components are independent and will usually differ. "
    "Use the full 0-25 range; intermediate values such as 6, 13, 17, 22 are expected. "
    "Do NOT round components or the total to multiples of 5 or 10. Two clips that "
    "differ in any observable way should not receive the same total.\n\n"

    "OUTPUT FIELD RULES:\n"
    "- static_reference (at most 25 words): STEP 1. Which fixed features you tracked "
    "and which way they swept across the sequence.\n"
    "- ego_path (at most 20 words): STEP 2. Ego's trajectory and speed trend, "
    "justified by the static feature named above.\n"
    "- apparent_vs_true (at most 40 words): STEP 3. Each relevant agent labelled TRULY "
    "MOVING (with direction) or APPARENT ONLY.\n"
    "- conflict_source: exactly one of ego_into_other, other_into_ego, longitudinal, "
    "none.\n"
    "- final_delta (at most 25 words): STEP 5. What differs between frame 12 and 16.\n"
    "- caption_neutral (at most 40 words): the literal physical scene. State the most "
    "important relation FIRST. It MUST name what ego itself is doing (using ego_path, "
    "so if ego is turning the caption says ego is turning), and it MUST describe at "
    "least one thing that CHANGES across the sequence. Describing everything as "
    "'maintaining' or 'consistent' is a FAILURE unless the scene is genuinely static. "
    "Never describe an APPARENT ONLY agent as moving, merging, or crossing -- that is "
    "the specific error this prompt exists to prevent. Always name the specific actor, "
    "its direction, and its proximity; two different clips must never produce the same "
    "sentence. Use these exact terms whenever they apply: braking, closing distance, "
    "following distance, lane change, merging, drifting, turning, yielding, "
    "right-of-way, crosswalk, intersection -- these are the only words that should "
    "repeat. caption_neutral MUST NOT contain any word implying danger, risk, safety, "
    "or outcome (risk, danger, collision, crash, imminent, safe, avoid, impact, "
    "hazard, accident) and MUST NOT mention time-to-event, seconds, or that 'an event' "
    "is about to happen.\n"
    "- counter_evidence (at most 25 words): STEP 6, citing ego_path, apparent_vs_true "
    "or final_delta. If nothing supports a collision, write 'none -- ' and the reason.\n"
    "- risk_clause (at most 8 words): a short evaluative judgment. This and "
    "counter_evidence are the ONLY fields where risk or outcome language belongs.\n"
    "- risk_score: the exact sum of the four components. Do not adjust it.\n"
    "- verdict: MECHANICAL. verdict = 1 if risk_score >= 50, otherwise 0.\n"
    "- confidence: 0.0-1.0. Low confidence is informative and is not penalized.\n\n"

    "EXAMPLE 1 -- ego turns; a parked vehicle only APPEARS to cross (this is the most "
    "common false alarm, so read it carefully):\n"
    "A rural roundabout in daylight. The roundabout kerb, bollards and roadside "
    "hedgerow all sweep steadily rightward across the image from frame 3 onward. A "
    "tractor sits at a field entrance beyond the roundabout; its position relative to "
    "the hedgerow behind it never changes, but it swings across the image.\n"
    'Output: {"static_reference": "Roundabout kerb, bollards and hedgerow sweep '
    'steadily rightward across the image from frame 3 onward", "ego_path": "Ego '
    'turning left around the roundabout at reducing speed; rightward sweep indicates '
    'leftward rotation", "apparent_vs_true": "Tractor at field entrance: APPARENT ONLY '
    '-- its offset from the hedgerow behind it is unchanged; it swings across frame '
    'only because ego rotates", "conflict_source": "none", "final_delta": "Tractor '
    'still stationary against the hedgerow; roundabout exit opening ahead with no '
    'vehicle entering", "caption_neutral": "Ego turning left around a rural roundabout '
    'at reducing speed past a stationary tractor parked at a field entrance, hedgerow '
    'and bollards passing through the frame", "counter_evidence": "none -- tractor is '
    'APPARENT ONLY per apparent_vs_true and no agent enters the roundabout ahead of '
    'ego", "closing_risk": 2, "lateral_risk": 1, "intrusion_risk": 3, '
    '"unreacted_risk": 1, "risk_score": 7, "verdict": 0, "risk_clause": "low risk, '
    'clear roundabout exit", "confidence": 0.8}\n\n'

    "EXAMPLE 2 -- ego itself is the one closing (ego_into_other):\n"
    "A motorway slip road joining the main carriageway. Chevron markings and barrier "
    "posts sweep leftward across frames 8-16. A coach travels in the nearside lane, "
    "holding a constant offset from the barrier behind it, while the gap between ego "
    "and the coach shrinks and ego does not slow.\n"
    'Output: {"static_reference": "Chevron markings and barrier posts sweep leftward '
    'across frames 8-16 as the slip road joins the carriageway", "ego_path": "Ego '
    'merging rightward from slip road onto carriageway at constant speed; chevrons '
    'sweep leftward", "apparent_vs_true": "Coach in nearside lane: APPARENT ONLY -- '
    'holds constant offset from the barrier behind it; ego is the vehicle changing '
    'lateral position", "conflict_source": "ego_into_other", "final_delta": "Lateral '
    'gap between ego and the coach clearly smaller than at frame 12; no braking or '
    'steering correction", "caption_neutral": "Ego merging rightward from a slip road '
    'onto the carriageway toward a coach holding the nearside lane, gap narrowing with '
    'no braking, barrier posts passing on the left", "counter_evidence": "Ego merging '
    'into the coach lane per ego_path with the gap narrowing per final_delta and no '
    'speed reduction", "closing_risk": 8, "lateral_risk": 24, "intrusion_risk": 2, '
    '"unreacted_risk": 22, "risk_score": 56, "verdict": 1, "risk_clause": "high risk, '
    'ego merging into coach", "confidence": 0.75}\n\n'

    "EXAMPLE 3 -- longitudinal, and the reaction IS visible (a genuine non-event):\n"
    "Inside a road tunnel. The tunnel wall lights and lane markings hold their lateral "
    "position and expand toward the camera. A box van ahead has brake lights on from "
    "frame 9; the gap shrinks between frames 9-13 then steadies as ego's own nose "
    "visibly dips under braking.\n"
    'Output: {"static_reference": "Tunnel wall lights and lane markings hold lateral '
    'position and expand toward the camera; no sweep either direction", "ego_path": '
    '"Ego straight, braking from frame 12 -- wall lights stop expanding as quickly", '
    '"apparent_vs_true": "Box van ahead: APPARENT ONLY laterally, holds its offset '
    'from the tunnel wall; no agent moving sideways", "conflict_source": '
    '"longitudinal", "final_delta": "Gap to the box van stops shrinking and steadies; '
    'ego brake-induced pitch visible against the wall lights", "caption_neutral": "Box '
    'van ahead in ego lane braking with brake lights illuminated inside a tunnel, ego '
    'travelling straight and braking in response with following distance steadying", '
    '"counter_evidence": "none -- gap steadied once ego braked and no agent moves '
    'laterally per apparent_vs_true", "closing_risk": 11, "lateral_risk": 1, '
    '"intrusion_risk": 1, "unreacted_risk": 3, "risk_score": 16, "verdict": 0, '
    '"risk_clause": "low risk, braking handled", "confidence": 0.85}\n\n'

    "EXAMPLE 4 -- a genuinely moving agent enters ego's path (other_into_ego):\n"
    "A narrow country lane. The hedgerow, verge edge and centre line hold their "
    "lateral position throughout. A cyclist ahead pulls out around a parked trailer -- "
    "the cyclist's offset from the verge edge clearly changes from frame 11 -- and ego "
    "does not brake, with the oncoming lane occupied.\n"
    'Output: {"static_reference": "Hedgerow, verge edge and centre line hold lateral '
    'position throughout; no rotational sweep", "ego_path": "Ego straight at constant '
    'speed on a country lane; centre line expands without lateral shift", '
    '"apparent_vs_true": "Cyclist ahead: TRULY MOVING rightward -- offset from the '
    'verge edge changes from frame 11 as they pass a parked trailer", '
    '"conflict_source": "other_into_ego", "final_delta": "Cyclist now well inside '
    'ego lane; gap reduced since frame 12 with no ego braking", "caption_neutral": '
    '"Cyclist ahead moving rightward into ego lane around a parked trailer on a narrow '
    'country lane, ego travelling straight at constant speed with no braking, oncoming '
    'lane occupied", "counter_evidence": "Cyclist truly moving into ego lane per '
    'apparent_vs_true with no ego reaction per ego_path", "closing_risk": 9, '
    '"lateral_risk": 23, "intrusion_risk": 5, "unreacted_risk": 22, "risk_score": 59, '
    '"verdict": 1, "risk_clause": "high risk, cyclist entering lane", '
    '"confidence": 0.8}\n\n'

    "DO NOT:\n"
    "- Do NOT determine ego's path from the motion of other vehicles. Use only the "
    "fixed world features from STEP 1.\n"
    "- Do NOT write that ego is 'holding lane centre', 'maintaining lane position', or "
    "'proceeding straight' unless a named static feature in static_reference supports "
    "it. This unsupported default is the most common error on this task.\n"
    "- Do NOT count an APPARENT ONLY agent toward lateral_risk, and do NOT describe it "
    "as moving, crossing, drifting or merging in the caption.\n"
    "- Do NOT hallucinate objects, agents, or motion not visible. If STEP 1-5 contain "
    "nothing that supports a collision, say so instead of inventing a mechanism.\n"
    "- Do NOT assume the other vehicle is always the one at fault; ego is frequently "
    "the agent that moves.\n"
    "- Do NOT judge the clip by its average over 16 frames -- weight frames 12-16.\n"
    "- Do NOT give all four components the same score, and do NOT round to multiples "
    "of 5 or 10.\n"
    "- Do NOT let the risk components influence caption_neutral's wording in either "
    "direction.\n"
    "- Do NOT override the verdict arithmetic (verdict = 1 iff risk_score >= 50).\n"
    "- Do NOT include your step-by-step reasoning, markdown fences, or any text "
    "outside the JSON object.\n\n"

    "PRIORITY: Correctly separating ego's own motion from other agents' motion is the "
    "primary objective -- nearly every error on this task traces back to getting that "
    "backwards, in one direction or the other. Literal, uncontaminated caption "
    "accuracy is second. Correct relative placement on the 0-100 total is third. An "
    "unjustified low score is exactly as wrong as an unjustified high one.\n\n"

    "OUTPUT -- return ONLY this JSON, no markdown fences, no extra text:\n"
    '{"static_reference": "<=25 words", "ego_path": "<=20 words", '
    '"apparent_vs_true": "<=40 words", '
    '"conflict_source": "ego_into_other|other_into_ego|longitudinal|none", '
    '"final_delta": "<=25 words", '
    '"caption_neutral": "<=40 words, names ego motion and one change, no risk/outcome/time language", '
    '"counter_evidence": "<=25 words citing an observation field", '
    '"closing_risk": 0-25, "lateral_risk": 0-25, "intrusion_risk": 0-25, '
    '"unreacted_risk": 0-25, "risk_score": <sum>, "verdict": 1 or 0, '
    '"risk_clause": "<=8 words", "confidence": 0.0-1.0}'
)
