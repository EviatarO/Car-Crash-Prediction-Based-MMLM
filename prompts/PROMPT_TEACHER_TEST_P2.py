"""
PROMPT_TEACHER_TEST_P2.py - pass 2 for the Nexar PRIVATE test windows: an ANCHORED self-check (plan 2026-10-17_Plan-Test-Private-Pass2-Anchored).
Replaces PROMPT_TEACHER_TEST_E8 (never run). Run name stays "E8" / "E8_pilot" so the website and the final merge read it unchanged.

Why (the user's review of pass 1, 2026-10-17): when pass 1 is wrong its text is usually close and ONE fact is misread (motion most often), or the teacher
followed the box onto the wrong road user (the box missed the hitter, or it described another car). So pass 2 is not a fresh look:
  - the teacher's OWN pass-1 answer is given as a draft, plus the true outcome (text only, the same 16 boxed frames);
  - the box is a HINT that can be wrong; every road user near the ego path in frames 12-16 is listed first (scan_all_road_users);
  - fact_check: the draft's road user / side / motion / gap are checked one by one; a fact is changed only with the frame numbers that show it;
  - hazard_visible (yes / partly / no): an honest way out when the frames do not support the outcome (no hazard visible yet, or a label problem);
  - the explanation is a minimal edit of the draft.
Only explanation / collision / prediction reach the reference text, in the SAME format as pass 1 and training; scan_all_road_users, fact_check,
hazard_visible, first_visible_frame are working fields (checked automatically, shown on the review page, dropped from the reference).
The students' answers are never shown to the teacher (that would make the references circular).

    build_p2_prompt(label, tte, draft) -> prompt text (images: the 16 boxed frames); draft = the pass-1 parsed answer or None
"""
from __future__ import annotations

from prompts.PROMPT_TEACHER_STUDY import BOX, DECISION, PRED_KEY, PREDICTION, WRITING, _e3_head, _json, _rep
from prompts.PROMPT_TEACHER_TEST_E8 import outcome_text

FACTS = ("road_user", "side", "motion", "gap")
REQUIRED_P2 = ("scan_all_road_users", "involved_agent", "box_ok", "fact_check", "hazard_visible", "first_visible_frame",
               "explanation", "risk_score", "collision", "prediction")

BOX_HINT = (
    "BOXED OBJECT (a hint that can be wrong):\n"
    "One road user is marked with a thin red box in the frames where it is visible. It is the object an automatic driving-risk "
    "model focused on, and that model is sometimes wrong: the box can sit on a parked or uninvolved road user, or miss the road "
    "user that matters. Use the box only as a hint and judge every road user yourself. Never mention the box. Set box_ok = false "
    "if the boxed object is not the involved road user.\n\n"
)
STEP4_BOXED = ("STEP 4 -- SAFETY vs CONFLICT ANALYSIS (BOXED AGENT FIRST):\n"
               "Focus on the boxed agent in frames 12-16: is it in or moving toward the ego lane, and is the gap to the ego "
               "vehicle shrinking faster than in frames 1-11?\n")
STEP4_ALL = ("STEP 4 -- SAFETY vs CONFLICT ANALYSIS (EVERY NEARBY ROAD USER):\n"
             "In frames 12-16, check every road user near the ego path, not only the boxed one: is it in or moving toward the ego "
             "lane, and is the gap to the ego vehicle shrinking faster than in frames 1-11? The last frames matter most.\n")

SECOND_LOOK = (
    "SECOND LOOK AT YOUR DRAFT:\n"
    "You already analysed these 16 frames once. Your first answer (the draft) was:\n"
    "  explanation: \"{expl}\"\n"
    "  decision: collision = {verdict} (risk_score {risk})\n\n"
    "VERIFIED OUTCOME (for your analysis only):\n"
    "{outcome}\n"
    "The draft's decision disagrees with this outcome, so the draft has probably misread something: often one fact (the road "
    "user's motion, its side, or how the gap changes at the end), or it described the wrong road user. The clip ends at frame 16 "
    "and you cannot see what happens next. Work in this order:\n"
    "1. scan_all_road_users: list every road user near the ego path in frames 12-16 -- what it is, where it is relative to the "
    "ego lane, what it does. Include road users the draft did not mention and road users outside the box.\n"
    "2. involved_agent: the road user (or obstacle) most likely involved, given the outcome. It may differ from the draft's road "
    "user and from the boxed one.\n"
    "3. fact_check: check the draft's four facts against the frames, one at a time: road_user (which road user the explanation "
    "is about), side (where it is relative to the ego lane at the end of the clip), motion (its main manoeuvre), gap (whether "
    "the distance to the ego vehicle closes, holds or opens at the end of the clip). For each fact give the draft value and the "
    "final value. Keep a fact the frames support, even though the decision changes. Change a fact ONLY when you can name the "
    "frames that show it: set changed = true and give those frame numbers (for example \"11-14\").\n"
    "4. hazard_visible: whether the 16 frames show a collision hazard -- \"yes\" (clearly), \"partly\" (early cues only) or "
    "\"no\"; first_visible_frame: the first frame where it shows (null if none). If the frames do not support the outcome -- "
    "no hazard visible yet for a clip that ends in a collision, or a clear hazard for a clip with no collision -- keep the "
    "draft's facts, say so honestly, and give the risk_score the visible scene justifies, even if the decision then still "
    "disagrees with the outcome. Never invent a road user, a motion or a gap change the frames do not show.\n"
    "5. explanation: edit the draft, changing only what a changed fact requires and keeping its wording elsewhere. It follows "
    "the writing rules below and must not mention frame numbers, the draft, the outcome statement or the box.\n"
    "6. risk_score, collision and prediction, as below. The prediction must agree with your collision answer.\n\n"
)

_FACT_KEY = '{"draft": "", "final": "", "changed": false, "frames": ""}'
FACT_CHECK_KEY = "{\n    " + ",\n    ".join(f'"{k}": {_FACT_KEY}' for k in FACTS) + "\n  }"


def _head() -> str:
    t = _rep(_e3_head(True), BOX, BOX_HINT)
    return _rep(t, STEP4_BOXED, STEP4_ALL)


def build_p2_prompt(label: int, tte: float, draft: dict | None) -> str:
    d = draft or {}
    block = SECOND_LOOK.format(expl=str(d.get("explanation") or "(none)").replace('"', "'"), verdict=d.get("collision", "none"),
                               risk=d.get("risk_score", "none"), outcome=outcome_text(label, tte))
    keys = [("scan_all_road_users", '""'), ("involved_agent", '""'), ("box_ok", "true"), ("fact_check", FACT_CHECK_KEY),
            ("hazard_visible", '"yes|partly|no"'), ("first_visible_frame", "null"), ("explanation", '"25-45 words"'),
            ("risk_score", "0"), ("collision", '"yes|no"'), ("prediction", PRED_KEY)]
    return _head() + block + WRITING + DECISION + PREDICTION + _json(keys)


if __name__ == "__main__":
    print(build_p2_prompt(1, 1.0, {"explanation": "A parked car on the right stays still while the ego vehicle passes it.",
                                   "collision": "no", "risk_score": 20})[-5200:])
