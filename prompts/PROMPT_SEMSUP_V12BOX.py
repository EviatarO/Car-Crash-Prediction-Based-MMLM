"""
PROMPT_SEMSUP_V12BOX -- boxed-object, label-blind scene explanation (pilot, plan 2026-10-08_Plan-BoxedTeacher-Pilot-rev3).

What is new compared with V12 (PROMPT_SEMSUP_V12_NEUTRAL.py, whose neutrality + ban blocks are reused unchanged):
* One road user is drawn with a thin red box in the frames where it is visible. It is the object the frozen crash model attends
  to (picked from the crash head's attention over tracked objects; the same track is boxed in the video's other windows).
* The description is about THAT object and its interaction with the ego vehicle.
* Closed lists (agent_class, position, object_motion, gap) replace V12's free-text fields (821 / 1,112 distinct phrasings).
* No colour, no time words, no time-to-impact.
* The decision `collision` yes|no is part of the output; build_prompt(order) controls whether it is asked BEFORE or AFTER the
  explanation (explanation_first | verdict_first). Everything else is word-for-word identical between the two orders.

The prompt is blind to the clip's label, to its time-to-event and to which window of a video it is.
NOTE: Gemini flash models think before they answer, so the JSON field order is a weak control for the TEACHER; the order that
matters is the STUDENT's answer order (tested later in training).
"""
from prompts.PROMPT_SEMSUP_V12_NEUTRAL import NEUTRALITY_BLOCK

AGENT_CLASS = ("car", "truck/bus", "motorcycle", "bicycle", "pedestrian", "other")
POSITION = ("ahead in ego lane", "ahead in left lane", "ahead in right lane", "crossing from left", "crossing from right",
            "oncoming", "beside left", "beside right")
OBJECT_MOTION = ("moving same direction", "braking or stopped", "turning", "cutting in", "crossing", "oncoming")
GAP = ("closing", "steady", "opening", "not visible")
ENUMS = {"agent_class": AGENT_CLASS, "position": POSITION, "object_motion": OBJECT_MOTION, "gap": GAP}

ROLE = (
    "ROLE: You are a calibrated scene-description analyst for a computer-vision training pipeline. You report what is visible: "
    "geometry and motion. You do not judge outcomes unless asked for the single yes/no decision below.\n\n"
)

INPUT = (
    "INPUT:\n"
    "- 16 chronologically ordered dashcam frames, ~2 seconds. Frame 1 = oldest, frame 16 = current moment.\n"
    "- Forward-facing camera of the ego vehicle (the car carrying the camera).\n"
    "- One road user is marked with a thin RED BOX in the frames where it is visible. It is the object that an automatic "
    "driving-risk model focused on. The box may be tiny when the object is far away, and it may be absent in early frames.\n"
    "- If the box is not on a real road user (for example on a parked object, a reflection, or empty road), set box_ok = false and "
    "describe the most relevant road user instead.\n\n"
)


def _lists():
    return (
        "agent_class: " + " | ".join(AGENT_CLASS) + "\n"
        "position: " + " | ".join(POSITION) + "\n"
        "object_motion: " + " | ".join(OBJECT_MOTION) + "\n"
        "gap: " + " | ".join(GAP) + "\n"
    )


STEP_IDENTIFY = (
    "STEP 1 -- IDENTIFY THE BOXED OBJECT. Choose EXACTLY ONE value from each list (copy the value word for word):\n"
    + _lists() +
    "position and object_motion describe the BOXED OBJECT (not the ego vehicle), position relative to the ego lane.\n\n"
)

STEP_GAP = (
    "STEP 2 -- GAP. Compare early frames (1-5), middle frames (6-11) and recent frames (12-16): does the distance between the boxed "
    "object and the ego vehicle close, stay steady or open? This is a measurement of the frames, independent of any judgment about "
    "danger; a closing gap is an ordinary event in traffic. Use gap = 'not visible' only if the object cannot be followed.\n\n"
)

EXPLANATION_RULES = (
    "EXPLANATION (25 to 40 words, complete sentences): describe the boxed object and how it interacts with the ego vehicle -- its "
    "class (use the same class word as agent_class), where it is, what it does, and how the gap changes. Describe ONLY what is visible "
    "in the frames; never invent objects or motion. Do not mention the box itself -- write about the object as a road user. "
    "Specific to THIS clip: two clips must not receive the same sentence.\n"
    "The explanation MUST NOT contain: colours; any time or duration statement; any word from these lists, for any clip:\n"
    "  OUTCOME: risk, danger, collision, crash, imminent, avoid, impact, hazard, accident\n"
    "  ALARM: about to, fails to, unable to, inevitably, will strike, too late, no time\n"
    "  REASSURANCE: safe, safely, no risk, uneventful, normal, routine, poses no, without incident\n"
    "Write it so that it reads identically in register whether or not anything happens after the last frame.\n\n"
)

DECISION_RULES = (
    "DECISION: collision = 'yes' if you judge that the ego vehicle is going to collide with something in the next moments after "
    "the last frame, otherwise 'no'. Decide from what the frames show; an ordinary closing gap alone is not a reason for 'yes'.\n\n"
)


def _schema(order: str) -> str:
    head = ('  "agent_class": "", "position": "", "object_motion": "", "gap": "", "box_ok": true, "evidence_frames": [],\n')
    expl = '  "explanation": "25-40 words"'
    dec = '  "collision": "yes|no"'
    body = head + (expl + ",\n" + dec if order == "explanation_first" else dec + ",\n" + expl) + "\n"
    return "OUTPUT -- return ONLY this JSON, no markdown fences, no extra text, keys in this order:\n{\n" + body + "}"


def build_prompt(order: str = "explanation_first") -> str:
    assert order in ("explanation_first", "verdict_first"), order
    seq = ("Work in this order: STEP 1, STEP 2, then write the EXPLANATION, and only then make the DECISION.\n\n"
           if order == "explanation_first" else
           "Work in this order: STEP 1, STEP 2, then make the DECISION first, and only then write the EXPLANATION.\n\n")
    return "".join([NEUTRALITY_BLOCK, ROLE, INPUT, STEP_IDENTIFY, STEP_GAP, EXPLANATION_RULES, DECISION_RULES, seq, _schema(order)])


BANNED = {
    "outcome": ("risk", "danger", "collision", "crash", "imminent", "avoid", "impact", "hazard", "accident"),
    "alarm": ("about to", "fails to", "unable to", "inevitably", "will strike", "too late", "no time"),
    "reassurance": ("safe", "safely", "no risk", "uneventful", "normal", "routine", "poses no", "without incident"),
    "time": ("second", "seconds", " sec", "minute", "moment"),
    "colour": ("red", "blue", "green", "white", "black", "grey", "gray", "silver", "yellow", "orange", "brown", "beige", "purple",
               "dark", "light-colored", "pink"),
}


if __name__ == "__main__":
    print(build_prompt("explanation_first"))
    print("\n" + "=" * 80 + "\n")
    print(_schema("verdict_first"))
