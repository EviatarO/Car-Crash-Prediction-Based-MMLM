"""
PROMPT_SEMSUP_V9_MINIMAL -- the "less is more" arm.

WHY THIS EXISTS
---------------
Seven prompt versions (V2 -> V8) grew from 619 to 4,072 tokens and produced
verdict accuracies of 9, 9, 11, 11, 10, 12, 9 out of 18 -- every one inside every
other one's confidence interval. Every version ADDED constraints; none ever
subtracted. This is the untested hypothesis: that the scaffolding itself is the
problem, and the model is spending capacity satisfying instructions instead of
looking at the frames.

The evidence that motivated it:

  version  tokens  verdict acc  caption score
  V2          619        9/18           4.61
  V3        1,244        9/18           4.78
  V4        1,386       11/18           5.11   <- best captions
  V5        2,273       11/18           4.50
  V6        3,257       10/18           4.33
  V7        4,072       12/18           4.83
  V8        3,357        9/18           4.22   <- worst captions

  r(tokens, verdict accuracy) = +0.46      r(tokens, caption score) = -0.39

Neither correlation is significant at n=7, but the signs split: longer prompts
trend slightly better on verdicts and WORSE on captions -- and the caption is the
deliverable (it is the SigLIP training target; the verdict is QA-only). The
sharpest single data point: V2 at 619 tokens, running on the drifted Gemini,
wrote better captions (4.61) than V5, V6 and V8 -- all 3.7-5.4x longer and running
on the stronger Qwen3-VL-235B.

Direct mechanistic evidence that constraint-satisfaction crowds out perception:
  - V6: 18/18 ego_motion fields contained identical boilerplate. Field filled,
    not observed.
  - V7: ego_path emitted 18/18, correct 9/18 -- a coin flip presented as an
    observation.
  - V8: change-vocabulary compliance rose 3->8/18 and banned direction-words fell
    to 0/18 (perfect rule-following) while reasoning alignment got WORSE
    (4 CONTRADICT, the worst of any round). The model satisfied a stylistic rule
    exactly while its perception did not change at all.

WHAT V9 KEEPS -- only insights with evidence behind them
--------------------------------------------------------
1. V2's caption constraints (<=40 words, banned risk/outcome/time vocabulary,
   canonical relational terms, no duplicate sentences). Hard SigLIP requirement:
   the text tower truncates at 64 tokens (semsup_common.py, siglip_text_embed,
   max_length=64).
2. "Describe the change, not the state" -- one line. Across V5/V6/V7/V8, static
   vocabulary ran 12-17/18 while change vocabulary ran 1-3/18; V8 showed the
   instruction does move the vocabulary (3->8/18).
3. "End with what ego did" -- one line. 7 of 18 GT reasonings turn explicitly on
   whether ego braked, and it was the deciding cue between 00493 (YES: "ego does
   not slow down") and 01504 (NO: "ego noticed this and also braked in time").
4. Path-relative motion, left/right banned for motion -- one line. V8 proved the
   ban works mechanically (0/18 direction words) and it removes the systematic
   mirroring seen in V5/V6/V7 (e.g. 00283: GT has a right-lane vehicle turning
   left; all three reported a left-lane vehicle moving right).
5. "Ignore agents holding position against the static background" -- one line.
   This is V7's apparent-vs-true test compressed to a sentence; in V7 it took
   false positives from 3 to 0.
6. A single continuous risk_score 0-100 with verdict derived at >=50. Continuous
   scoring enables AP/AUC (this project's headline metric) and V5's failure mode
   -- NAMED numeric bands causing midpoint snapping to 6 distinct values -- is
   avoided by giving no band labels at all.

WHAT V9 DROPS
-------------
Every observation/scaffolding field (V5's counter_evidence; V6's ego_motion,
lateral_watch, final_delta; V7's static_reference, ego_path, apparent_vs_true,
conflict_source; V8's delta, true_movers, cause, ego_response), the four 0-25
sub-scores, the anchored score bands, and three of four worked examples. Output
is 5 fields: caption_neutral, risk_clause, risk_score, verdict, confidence.

This is deliberate and is the point of the experiment. The target model
(qwen3-vl-235b-a22b-thinking) is a reasoning-native model that performs an
internal chain of thought before emitting output -- the explicit scaffolding may
simply be redundant with capability it already has, while still consuming the
attention budget. If V9 matches or beats the heavier prompts, the seven rounds of
added structure were net-negative and the production prompt should be short.

One worked example is retained rather than zero: V4 (two examples) produced the
best captions of any round, so examples are not obviously part of the problem,
and one anchors JSON format compliance cheaply.

Output schema (strict JSON, no markdown fences, no extra text):
    {
      "caption_neutral": "<=40 words, no risk/outcome/time language",
      "risk_clause":     "<=8 words",
      "risk_score":      0-100,
      "verdict":         1 or 0,     # derived: 1 iff risk_score >= 50
      "confidence":      0.0-1.0
    }
"""

PROMPT_SEMSUP_V9_MINIMAL = (
    "ROLE: You are an autonomous-driving scene analyst describing a dashcam clip "
    "(16 sequential frames, Frame 1 = earliest, Frame 16 = latest, ~2 seconds).\n\n"

    "AUDIENCE: This description is NOT for a human reader. It will be encoded by a "
    "SigLIP text encoder and used as a training target for a vision model. Write "
    "dense, literal, alt-text-style language -- not narrative prose.\n\n"

    "WHAT MATTERS: a list of vehicles and their positions is not useful -- two clips "
    "with identical vehicles can have opposite outcomes. What separates them is what "
    "CHANGED across the frames, and whether the ego vehicle reacted to it.\n\n"

    "caption_neutral (STRICT: at most 40 words):\n"
    "- Describe what CHANGES between the early frames (1-6) and the late frames "
    "(11-16): a gap closing or opening, an agent entering or leaving ego's path, "
    "brake lights coming on, a vehicle starting or stopping moving, a pedestrian "
    "stepping off a kerb. Scan the whole frame, including the edges. If nothing "
    "changes, say so directly. Do not write a static inventory.\n"
    "- END with what ego did about it -- 'with ego braking in response', 'with ego "
    "not slowing'. To see it, look for the gap ceasing to shrink, the image pitching "
    "down as the nose dips, or the lead vehicle's growth in frame slowing. This is "
    "the single most decisive detail in the clip; never omit it.\n"
    "- Describe MOTION relative to ego's path: into / out of / across / along ego's "
    "path, gap closing / opening / steady. Do NOT write 'moving left', 'turning "
    "right' or 'drifting leftward' -- which way something turned is unreliable to "
    "judge and is not what determines a collision. Left/right is fine for static "
    "position ('a van in the right lane').\n"
    "- An agent that keeps constant position against the lane markings, kerb or "
    "buildings behind it is NOT moving -- it only appears to because ego is turning. "
    "Do not describe it as entering ego's path.\n"
    "- Name the specific actor and its proximity; two clips must never produce the "
    "same sentence. Use these terms whenever they apply: braking, closing distance, "
    "following distance, lane change, merging, crossing, yielding, right-of-way, "
    "crosswalk, intersection.\n"
    "- MUST NOT contain any word implying danger, risk, safety or outcome (risk, "
    "danger, collision, crash, imminent, safe, avoid, impact, hazard, accident), and "
    "MUST NOT mention time-to-event or seconds.\n\n"

    "risk_clause (STRICT: at most 8 words): a short evaluative judgment of the "
    "collision risk. This is the ONLY field where risk/outcome language is allowed.\n\n"

    "risk_score: an integer 0-100 -- how likely a collision is within 0-3 seconds "
    "AFTER the final frame. Use the whole range and avoid multiples of 5 or 10; two "
    "clips that differ in any observable way should not get the same number. "
    "verdict = 1 if risk_score >= 50, otherwise 0 -- this is arithmetic, do not "
    "override it. confidence: 0.0-1.0, your certainty in the score.\n\n"

    "EXAMPLE:\n"
    'Output: {"caption_neutral": "Box van ahead in ego lane braking behind stopped '
    'traffic, gap closing steadily with ego not slowing", "risk_clause": "high risk, '
    'closing on stopped traffic", "risk_score": 71, "verdict": 1, "confidence": 0.8}\n\n'

    "Never invent objects, agents or events not visible in the frames. The collision, "
    "if any, happens after the last frame -- an intact final frame is not evidence of "
    "safety.\n\n"

    "OUTPUT -- return ONLY this JSON, no markdown fences, no extra text:\n"
    '{"caption_neutral": "<=40 words, no risk/outcome/time language", '
    '"risk_clause": "<=8 words", "risk_score": 0-100, "verdict": 1 or 0, '
    '"confidence": 0.0-1.0}'
)
