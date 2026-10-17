"""
PROMPT_TEACHER_TEST_E8.py - the second look for the Nexar PRIVATE test windows (plan 2026-10-17_Plan-Test-Private-Reasoning, step 5).

Where the blind first look (E3, byte-identical to the training run) disagrees with the test label, the teacher looks again at the SAME 16 boxed frames
and is TOLD the true outcome. The test clips end at the window, so the training hindsight prompt (E7: frames after the window) and the anchored prompt
(E6: a correct sibling window) are impossible; this one has no second clip.

The risk this prompt guards against: a label-informed teacher inventing a hazard the frames do not show. So it must (a) say whether the involved road user /
the hazard is visible in the 16 frames and from which frame (`hazard_visible` yes / partly / no, `first_visible_frame`), (b) describe ONLY what is visible, and
(c) give the risk that the visible scene justifies, which may be below 50 for a clip that ends in a collision ("hazard not visible yet").
These texts are REFERENCES for grading the student's explanations, never training targets.

    build_e8_prompt(label, tte) -> prompt text (images: the 16 boxed frames)
"""
from __future__ import annotations

from prompts.PROMPT_TEACHER_STUDY import DECISION, PRED_KEY, PREDICTION, WRITING, _e3_head, _json

REQUIRED_E8 = ("scene_context", "temporal_analysis", "box_ok", "involved_agent", "hazard_visible", "first_visible_frame", "early_cues",
               "explanation", "risk_score", "collision", "prediction")

OUTCOME_BLOCK = (
    "VERIFIED OUTCOME (for your analysis only):\n"
    "{outcome}\n"
    "The clip ends at frame 16 and you cannot see what happens next. Use the stated outcome only to decide which road user (or obstacle) is most "
    "likely involved and what to look for in the frames. Then judge the frames themselves, as a driver would at frame 16: say whether that road user "
    "is visible in these 16 frames and from which frame, which early cues of the outcome the frames already show (including the ego vehicle's own "
    "motion) and which they do not show yet. The boxed road user may or may not be the one involved.\n"
    "Your explanation, risk_score and prediction must describe ONLY what is visible in the 16 frames -- never describe or imply anything you cannot "
    "see. If the frames show no sign of the outcome yet, say so honestly and give the risk that the visible scene justifies, even if that is below "
    "50 for a clip that ends in a collision (set hazard_visible to \"no\"). If the frames do show the hazard, describe it concretely: the road user, "
    "where it is, and how the gap to the ego vehicle changes. The prediction must agree with your collision answer. Never mention the outcome "
    "statement, the hindsight, the reviewer or the box in any field.\n\n"
)


def outcome_text(label: int, tte: float) -> str:
    if label:
        return f"the ego vehicle COLLIDES with another road user or object about {tte:.1f} seconds after frame 16."
    return "NO collision happens after frame 16 or later in the drive; the drive continues normally."


def build_e8_prompt(label: int, tte: float) -> str:
    keys = [("scene_context", '""'), ("temporal_analysis", '""'), ("box_ok", "true"), ("involved_agent", '""'), ("hazard_visible", '"yes|partly|no"'),
            ("first_visible_frame", "null"), ("early_cues", '""'), ("explanation", '"25-45 words"'), ("risk_score", "0"), ("collision", '"yes|no"'),
            ("prediction", PRED_KEY)]
    return _e3_head(True) + OUTCOME_BLOCK.format(outcome=outcome_text(label, tte)) + WRITING + DECISION + PREDICTION + _json(keys)


if __name__ == "__main__":
    print(build_e8_prompt(1, 1.0)[-3300:])
