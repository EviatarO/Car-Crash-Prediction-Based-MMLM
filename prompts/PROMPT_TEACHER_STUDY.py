"""
PROMPT_TEACHER_STUDY -- prompts of the teacher prompt study (plan 2026-10-08_Plan-Teacher-Prompt-Study.md).

All variants are BLIND (no label), contain no closed lists and no time-to-impact field, write a free-text explanation, and END
with risk_score (0-100) then collision yes/no (yes if risk >= 50).

  e1    : V12 neutral caption prompt (its own blocks), raw frames, + the decision at the end
  e2    : e1 + the BOX paragraph (boxed frames) + box_ok           -> e1 vs e2 isolates the box
  e3    : PROMPT_G_OPT_v6_balanced re-balanced toward TP (no "prefer NO" / default-safe blocks, symmetric gates), STEP 4
          centred on the boxed agent, + BOX + WRITING + DECISION
  e4_tp : PROMPT_G_OPT_v7_1_TP_RECOVERY + BOX + WRITING + DECISION   (debate: Exp #3 false negatives, routed by the label)
  e4_tn : PROMPT_G_OPT_v7_1_TN_RECOVERY + BOX + WRITING + DECISION   (debate: Exp #3 false positives, routed by the label)

The source prompts are imported and edited with exact string replacements (asserted), so every kept part is byte-identical to the
original. `python -m prompts.PROMPT_TEACHER_STUDY` prints every prompt, the e1->e2 diff and an audit of forbidden content.
"""
import re

from prompts.PROMPT_G_OPT_v6_balanced import PROMPT_G_OPT_v6_balanced
from prompts.PROMPT_G_OPT_v7_1_TN_RECOVERY import PROMPT_G_OPT_v7_1_TN_RECOVERY
from prompts.PROMPT_G_OPT_v7_1_TP_RECOVERY import PROMPT_G_OPT_v7_1_TP_RECOVERY
from prompts.PROMPT_SEMSUP_V12_NEUTRAL import INPUT_BLOCK as V12_INPUT, STEP123_BLOCK as V12_STEP123, STEP4_BLOCK as V12_STEP4

VARIANTS = ("e1", "e2", "e3", "e4_tp", "e4_tn")
BOXED = {"e1": False, "e2": True, "e3": True, "e4_tp": True, "e4_tn": True}

# ------------------------------------------------------------------ shared blocks
BOX = (
    "BOXED OBJECT:\n"
    "One road user is marked with a thin red box in the frames where it is visible. It is the object an automatic "
    "driving-risk model focused on. The box may be tiny when the object is far away and may be missing in early frames. "
    "Treat the boxed object as the primary agent and write about it as a road user -- never mention the box. If the box is "
    "not on a real road user (a parked object, a reflection, empty road), set box_ok = false and use the most relevant road "
    "user instead.\n\n"
)

WRITING = (
    "WRITING THE EXPLANATION (25-45 words, complete sentences):\n"
    "Write in your own words, in natural, specific sentences about THIS clip: what the agent is (car, truck, bus, "
    "motorcycle, bicycle, pedestrian...), where it is relative to the ego lane, what it does, and how the distance to the "
    "ego vehicle changes over the frames. Do not repeat stock phrases, do not use colours, and do not give times or seconds. "
    "Do not use outcome words in the explanation (risk, danger, collision, crash, imminent, avoid, impact, hazard, accident); "
    "the judgement goes only into risk_score and collision.\n\n"
)

DECISION = (
    "DECISION (last):\n"
    "Finally, and only after the analysis above, judge whether the ego vehicle will collide with something within about 3 "
    "seconds after frame 16. Give risk_score from 0 (no risk) to 100 (collision certain), then collision = \"yes\" if "
    "risk_score >= 50, else \"no\".\n\n"
)


def _json(keys):
    body = ",\n".join(f'  "{k}": {v}' for k, v in keys)
    return "OUTPUT -- return ONLY this JSON, no markdown fences, no extra text, keys in this order:\n{\n" + body + "\n}"


def _rep(text, old, new):
    assert old in text, f"expected text not found: {old[:60]!r}"
    return text.replace(old, new, 1)


# ------------------------------------------------------------------ e1 / e2 (V12)
E1_ROLE = (
    "ROLE: You are a calibrated scene-description analyst who writes precise, literal descriptions of dashcam clips for a "
    "computer-vision training pipeline. Describe first; you will be asked for one judgement at the end.\n\n"
)
E1_NEUTRAL = (
    "NEUTRAL DESCRIPTION FIRST: describe the scene before you judge it. Your description must read the same way whether or "
    "not anything happens after the last frame (no outcome, alarm or reassurance words in \"explanation\": risk, danger, "
    "collision, crash, imminent, avoid, impact, hazard, accident, about to, too late, safe, safely, normal, routine).\n\n"
)
E1_GAP = (
    "STEP 5 -- GAP: in one sentence, say in your own words whether the distance between the ego vehicle and the primary "
    "agent closes, holds or opens across frames 1->16.\n\n"
)


def _e12(box: bool) -> str:
    keys = [("scene_context", '""'), ("dynamic_objects", '""'), ("temporal_analysis", '""'), ("primary_agent", '""'),
            ("agent_motion", '""'), ("agent_position", '""'), ("gap_description", '""'), ("agent_visible", "true")]
    if box:
        keys.append(("box_ok", "true"))
    keys += [("explanation", '"25-45 words"'), ("risk_score", "0"), ("collision", '"yes|no"')]
    return "".join([E1_ROLE, E1_NEUTRAL, V12_INPUT, BOX if box else "", V12_STEP123, V12_STEP4, E1_GAP, WRITING, DECISION]) + _json(keys)


# ------------------------------------------------------------------ e3 (v6_balanced, re-balanced)
V6_OUTPUT_HEAD = "OUTPUT FORMAT -- STRICT JSON ONLY:\n"
V6_INPUT_END = "- Forward-facing ego vehicle camera\n\n"


def _e3_head(box: bool = True) -> str:
    """E3's text up to (not including) the output format. box=False (E7, raw frames) drops BOX and the 'boxed agent' wording."""
    t = PROMPT_G_OPT_v6_balanced
    t = _rep(t,
             "IMPORTANT BASE-RATE PRINCIPLES:\n"
             "- Most traffic interactions do NOT result in collisions.\n"
             "- Object presence alone does NOT imply danger.\n"
             "- Object growth in the image alone does NOT imply collision risk.\n"
             "- Nearby vehicles often maintain safe parallel or diverging trajectories.\n"
             "- Predict collision ONLY when clear future trajectory conflict exists.\n"
             "- If evidence is ambiguous or insufficient, prefer NO.\n\n",
             "BASE RATE:\n"
             "About half of the clips you review are followed by a collision, so neither answer is the safe default. Object "
             "presence or growth alone is not evidence; judge the motion.\n\n")
    t = _rep(t, V6_INPUT_END, V6_INPUT_END + (BOX if box else ""))
    t = _rep(t, "IMPORTANT:\nParallel motion and stable spacing usually indicate safe traffic flow.\n\n", "")
    t = _rep(t, "IMPORTANT:\nNormal lane following, stable spacing, and parallel motion are evidence for NO.\n\n", "")
    t = _rep(t, "STEP 4 -- SAFETY vs CONFLICT ANALYSIS:\nCheck for BOTH:\n\n",
             ("STEP 4 -- SAFETY vs CONFLICT ANALYSIS (BOXED AGENT FIRST):\n"
              "Focus on the boxed agent in frames 12-16: is it in or moving toward the ego lane, and is the gap to the ego "
              "vehicle shrinking faster than in frames 1-11?\n" if box else
              "STEP 4 -- SAFETY vs CONFLICT ANALYSIS (MOST RELEVANT AGENT FIRST):\n"
              "Focus on the most relevant road user in frames 12-16: is it in or moving toward the ego lane, and is the gap to "
              "the ego vehicle shrinking faster than in frames 1-11?\n") +
             "Check for BOTH:\n\n")
    t = _rep(t, "- pedestrian entering ego trajectory\n",
             "- pedestrian entering ego trajectory\n- vehicle pulling out from parking or the roadside into ego path\n")
    t = _rep(t,
             "STEP 7 -- FINAL DECISION GATES:\n"
             "Predict YES ONLY if at least ONE holds:\n\n"
             "(A) An object has a clear closing trajectory toward ego vehicle AND projected "
             "path intersection within ~3 seconds.\n\n"
             "(B) An agent is crossing into ego trajectory with insufficient time or space "
             "to avoid conflict.\n\n"
             "(C) Ego vehicle is rapidly approaching a stationary or slow obstacle with "
             "insufficient stopping space.\n\n"
             "If NONE of (A), (B), or (C) clearly hold, predict NO.\n\n"
             "DEFAULT ASSUMPTION:\n"
             "Traffic continues safely unless clear trajectory conflict evidence exists.\n\n",
             "STEP 7 -- DECISION GATES (symmetric):\n"
             "Predict YES if ANY plausibly holds: (A) the agent's gap to the ego vehicle is shrinking in frames 12-16 and its "
             "path meets the ego path; (B) a crossing, merge or pull-out into the ego path is unresolved at frame 16; (C) the "
             "ego vehicle closes on a stopped or slow object without slowing; (D) a pedestrian or cyclist moves toward the ego "
             "path.\n"
             "Predict NO if the late frames show a stable or growing gap, a completed merge, or diverging paths.\n\n")
    head, _ = t.split(V6_OUTPUT_HEAD)
    return head


def _e3() -> str:
    keys = [("scene_context", '""'), ("dynamic_objects", '""'), ("temporal_analysis", '""'), ("safe_interpretation", '""'),
            ("collision_interpretation", '""'), ("box_ok", "true"), ("explanation", '"25-45 words"'), ("risk_score", "0"),
            ("collision", '"yes|no"')]
    return _e3_head(True) + WRITING + DECISION + _json(keys)


# ------------------------------------------------------------------ e5 / e6 (anchored) and e7 (hindsight)
E56_VARIANTS = ("e5", "e6")
PREDICTION_V1 = (                                   # used for the 2026-10-09 E5-E7 runs; repeated the explanation (median ~21 words)
    "PREDICTION (last):\n"
    "One sentence in your own words that ties what you saw to what happens next, for example in the style of "
    "\"..., so a crash is about to happen.\" or \"..., so the road stays clear and no hazard appears.\" Outcome words are "
    "allowed here only. Do not copy these examples.\n\n"
)
PREDICTION = (                                      # 2026-10-10: outcome only
    "PREDICTION (last):\n"
    "Only the outcome, in one short clause of 4-15 words. Do not repeat or summarise the explanation -- it is already written "
    "above. Name the road user involved if there is one. Examples of the form: \"A collision with the truck follows.\" / "
    "\"No collision; the road stays clear.\" Outcome words are allowed here. Do not copy these examples.\n\n"
)
PRED_KEY = '"4-15 words, outcome only"'
PRE_ALERT_NOTE = (
    "TIMING (verified annotation): the collision happens at the end of clip C, but by the annotation the danger becomes "
    "recognisable only AFTER clip B ends. Check this against clip B: is the involved road user, or any warning cue, already "
    "visible in clip B? Answer hazard_visible honestly; if nothing in clip B announces the collision yet, a low risk_score is "
    "the correct judgement for clip B's moment.\n\n"
)
CLIP_RULE = ("Never mention clip A, clip C, the reference, the reviewer or the hindsight in any field.")

E56_INPUT = (
    "INPUT:\n"
    "You receive TWO clips from the SAME drive, each 16 chronologically ordered dashcam frames (about 2 seconds, forward-facing "
    "ego camera). CLIP A (reference) is the first 16 images; CLIP B (target) is the last 16 images and ends {delta} s {dir} than "
    "clip A. In each clip frame 1 = oldest and frame 16 = current moment. All analysis steps below refer to CLIP B.\n\n"
)
BOX2 = BOX.replace("in the frames where it is visible.", "in the frames of both clips where it is visible.")
assert BOX2 != BOX
REFERENCE = (
    "REFERENCE ANALYSIS OF CLIP A (verified correct):\n"
    "A reviewer analysed clip A with the same steps you will follow, and their final judgement, collision = \"{verdict}\", was "
    "CORRECT: the drive {followed}. Their reasoning:\n"
    "- Temporal analysis: \"{temporal}\"\n"
    "- Why this outcome: \"{why}\"\n"
    "- Summary: \"{summary}\" (risk_score {risk})\n"
    "HOW TO USE IT: this reasoning was right for clip A. Use it as your starting point for clip B: find the same road user and "
    "the same cues in clip B, and follow how they look {delta} s {dir} than in clip A. Then decide whether the same reasoning "
    "still holds at clip B's moment. Describe only what is visible in clip B. If a cue from clip A is not yet visible (or no "
    "longer visible) in clip B, say so instead of copying it. " + CLIP_RULE + "\n\n"
)
TIMING = (
    "TIMING CHECK:\n"
    "Clip A and clip B show the same drive at different moments. The road user or cue that decided clip A may not be in clip B "
    "yet (it can enter the frame later), or may already have changed. First find it in clip B: say from which frame it is "
    "visible, or that it is not visible. Then judge clip B ONLY from what clip B shows at its frame 16 -- including the ego "
    "vehicle's own motion (speed toward an obstacle, turning across traffic, lane change) and any other road user. If the "
    "deciding road user is not visible yet and nothing else in clip B justifies alarm, say so: a low risk_score is then the "
    "correct answer for this moment, even though clip A was followed by a collision. If the ego vehicle's own motion already "
    "creates a dangerous situation, explain that and keep the risk high.\n\n"
)


def build_anchor_prompt(variant: str, delta: float, earlier: bool, anchor: dict) -> str:
    """E5 (frozen text) / E6 (E5 + TIMING + hazard fields). anchor = dict(label, temporal, why, summary, risk)."""
    assert variant in E56_VARIANTS, variant
    t = _e3_head(True)
    t = _rep(t, "INPUT:\n- 16 chronologically ordered dashcam frames\n- Frame 1 = oldest\n- Frame 16 = current moment\n"
                "- Sequence duration ≈ 2 seconds\n- Forward-facing ego vehicle camera\n\n" + BOX,
             E56_INPUT.format(delta=f"{delta:.1f}", dir="earlier" if earlier else "later") + BOX2 +
             REFERENCE.format(verdict="yes" if anchor["label"] else "no",
                              followed="was followed by a collision" if anchor["label"] else "continued without a collision",
                              temporal=anchor["temporal"], why=anchor["why"], summary=anchor["summary"], risk=anchor["risk"],
                              delta=f"{delta:.1f}", dir="earlier" if earlier else "later") + (TIMING if variant == "e6" else ""))
    keys = [("scene_context", '""'), ("dynamic_objects", '""'), ("temporal_analysis", '""'), ("safe_interpretation", '""'),
            ("collision_interpretation", '""'), ("box_ok", "true"), ("same_agent", "true")]
    if variant == "e6":
        keys += [("hazard_visible", '"yes|partly|no"'), ("first_visible_frame", "null")]
    keys += [("explanation", '"25-45 words"'), ("risk_score", "0"), ("collision", '"yes|no"'), ("prediction", PRED_KEY)]
    return t + WRITING + DECISION + PREDICTION + _json(keys)


HINDSIGHT = (
    "HINDSIGHT (for your analysis only):\n"
    "OUTCOME (verified): {outcome}\n"
    "Clip C shows what happened after clip B. Use it, together with the stated outcome, to identify the road user (or obstacle) "
    "involved and how the situation developed. Then go back to clip B: say whether that road user is visible in clip B and from "
    "which frame, which early cues of the outcome clip B already shows (including the ego vehicle's own motion), and which it "
    "does not show yet. Your explanation, risk_score and prediction must describe ONLY what is visible in clip B, judged as a "
    "driver would at clip B's frame 16 -- never describe events that appear only in clip C. " + CLIP_RULE + "\n\n"
)
E7_INPUT = (
    "INPUT:\n"
    "You receive TWO clips from the SAME drive. CLIP B (target) is the first 16 images: 16 chronologically ordered dashcam "
    "frames (about 2 seconds, forward-facing ego camera), frame 1 = oldest, frame 16 = current moment. CLIP C (what happened "
    "next) is the following {n_after} images: the frames right after clip B at the same spacing, up to just after the end of "
    "the drive's critical moment. All analysis steps below refer to CLIP B.\n\n"
)


def build_hindsight_prompt(n_after: int, label: int, pre_alert: bool = False) -> str:
    """E7: clip B (raw) + clip C (after-frames) + the verified outcome in text; explanation limited to clip B.
    pre_alert=True (E7-pre): clip B ends before the annotated alert, so the prompt asks whether the hazard is already visible."""
    t = _e3_head(False)
    t = _rep(t, "INPUT:\n- 16 chronologically ordered dashcam frames\n- Frame 1 = oldest\n- Frame 16 = current moment\n"
                "- Sequence duration ≈ 2 seconds\n- Forward-facing ego vehicle camera\n\n",
             E7_INPUT.format(n_after=n_after) + HINDSIGHT.format(
                 outcome=("the ego vehicle COLLIDED with another road user or object at the end of clip C." if label
                          else "NO collision happens in clip C or afterwards; the drive continues normally.")) +
                 (PRE_ALERT_NOTE if pre_alert else ""))
    keys = [("scene_context", '""'), ("dynamic_objects", '""'), ("temporal_analysis", '""'), ("safe_interpretation", '""'),
            ("collision_interpretation", '""'), ("involved_agent", '""'), ("hazard_visible", '"yes|partly|no"'),
            ("first_visible_frame", "null"), ("early_cues", '""'), ("explanation", '"25-45 words"'), ("risk_score", "0"),
            ("collision", '"yes|no"'), ("prediction", PRED_KEY)]
    return t + WRITING + DECISION + PREDICTION + _json(keys)


# ------------------------------------------------------------------ e4 (v7.1 recovery prompts)
def _e4(src: str) -> str:
    t = _rep(src, V6_INPUT_END, V6_INPUT_END + BOX)
    head, _ = t.split(V6_OUTPUT_HEAD)
    keys = [("scene_context", '""'), ("dynamic_objects", '""'), ("temporal_analysis", '""'), ("verdict_reasoning", '""'),
            ("box_ok", "true"), ("explanation", '"25-45 words"'), ("risk_score", "0"), ("collision", '"yes|no"')]
    return head + WRITING + DECISION + _json(keys)


def build_prompt(variant: str) -> str:
    assert variant in VARIANTS, variant
    return {"e1": lambda: _e12(False), "e2": lambda: _e12(True), "e3": _e3,
            "e4_tp": lambda: _e4(PROMPT_G_OPT_v7_1_TP_RECOVERY), "e4_tn": lambda: _e4(PROMPT_G_OPT_v7_1_TN_RECOVERY)}[variant]()


REQUIRED = {
    "e1": ("scene_context", "primary_agent", "gap_description", "explanation", "risk_score", "collision"),
    "e2": ("scene_context", "primary_agent", "gap_description", "box_ok", "explanation", "risk_score", "collision"),
    "e3": ("scene_context", "safe_interpretation", "collision_interpretation", "box_ok", "explanation", "risk_score", "collision"),
    "e4_tp": ("scene_context", "verdict_reasoning", "box_ok", "explanation", "risk_score", "collision"),
    "e4_tn": ("scene_context", "verdict_reasoning", "box_ok", "explanation", "risk_score", "collision"),
    "e5": ("scene_context", "temporal_analysis", "box_ok", "same_agent", "explanation", "risk_score", "collision", "prediction"),
    "e6": ("scene_context", "temporal_analysis", "box_ok", "same_agent", "hazard_visible", "first_visible_frame", "explanation",
           "risk_score", "collision", "prediction"),
    "e7": ("scene_context", "temporal_analysis", "involved_agent", "hazard_visible", "first_visible_frame", "early_cues", "explanation",
           "risk_score", "collision", "prediction"),
}
MULTI_VARIANTS = ("e5", "e6", "e7")
_DUMMY = dict(label=1, temporal="<anchor temporal analysis>", why="<anchor collision interpretation>", summary="<anchor explanation>", risk=92)


def build_multi_prompt(variant: str, **kw) -> str:
    """Dispatcher for the multi-clip prompts: e5/e6 need delta, earlier, anchor; e7 needs n_after, label."""
    if variant in E56_VARIANTS:
        return build_anchor_prompt(variant, kw["delta"], kw["earlier"], kw["anchor"])
    assert variant == "e7", variant
    return build_hindsight_prompt(kw["n_after"], kw["label"], kw.get("pre_alert", False))


def dummy_multi(variant: str) -> str:
    return build_multi_prompt(variant, delta=0.5, earlier=True, anchor=_DUMMY, n_after=10, label=1)
FORBIDDEN = ("gap_trend", "Use these exact terms", "MUST contain", "time_to_impact", "Time to impact", "agent_class",
             "TTE", "decreasing|increasing|constant")


def audit():
    """Prompt-level checks of the plan's verification section."""
    out = {}
    for v in VARIANTS:
        p = build_prompt(v)
        out[v] = {"chars": len(p), "has_box": "BOXED OBJECT" in p, "forbidden": [f for f in FORBIDDEN if re.search(r"\b" + re.escape(f) + r"\b", p)],
                  "ends_with_collision": p.rstrip().endswith('"collision": "yes|no"\n}') or p.rstrip().endswith('"yes|no"\n}'),
                  "has_decision": "DECISION (last)" in p}
    for v in MULTI_VARIANTS:
        p = dummy_multi(v)
        out[v] = {"chars": len(p), "has_box": "BOXED OBJECT" in p, "forbidden": [f for f in FORBIDDEN if re.search(r"\b" + re.escape(f) + r"\b", p)],
                  "ends_with_prediction_key": p.rstrip().endswith('"prediction": ' + PRED_KEY + '\n}'),
                  "has_decision": "DECISION (last)" in p, "has_prediction": "PREDICTION (last)" in p,
                  "unfilled_braces": re.findall(r"\{[a-z_]+\}", p)}
    # E6 differs from E5 by the TIMING block and the two hazard keys only
    a, b = dummy_multi("e5").splitlines(), dummy_multi("e6").splitlines()
    out["e5_to_e6_added_lines"] = len([l for l in b if l not in a])
    return out


if __name__ == "__main__":
    import difflib
    for v in VARIANTS:
        print("=" * 100 + f"\n{v}\n" + "=" * 100)
        print(build_prompt(v))
    print("=" * 100 + "\ne1 -> e2 diff\n" + "=" * 100)
    for l in difflib.unified_diff(build_prompt("e1").splitlines(), build_prompt("e2").splitlines(), lineterm="", n=0):
        if not l.startswith(("---", "+++", "@@")):
            print(l)
    print("=" * 100)
    for k, v in audit().items():
        print(k, v)
