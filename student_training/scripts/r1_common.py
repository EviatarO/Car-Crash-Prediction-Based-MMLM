"""
r1_common.py - shared definitions for the week-1 reasoning-path experiment (plan 2026-10-05_Plan-Week1-GoNoGo-rev3).

* the 2 s / 16-frame window rule (30 fps source, stride 4)
* the WINDOW VISIBILITY RULE (memory: window-visibility-rule): a crash window is valid only if the hazard
  has already started inside it, with the same margin for every source: end >= hazard start + 8 frames (0.27 s)
  (DADA: hazard start = t_ai; Nexar: time_of_alert; margin added for Nexar 2026-10-06)
* r1 target texts (ground-truth fields only) and the two instruction questions
"""
from __future__ import annotations

import re

STRIDE, N_FR, FPS_SRC = 4, 16, 30
VIS_MARGIN_FRAMES = 8                  # hazard must be >= 8 source frames (0.27 s) into the window (all sources)
TTES = (0.5, 1.0, 1.5)

Q_PHASE1 = "Describe the motion and objects in this clip."
Q_PHASE2 = "Is a collision coming? Give: collision yes/no, time to impact, event, cause."
TAG = {"dada": "[MM-AU]", "nexar": "[Nexar]"}


def window_indices(end: int) -> list[int]:
    """16 frame numbers (1-based), every 4th, ending at `end` (60 source frames = 2.0 s)."""
    return [end - (N_FR - 1 - i) * STRIDE for i in range(N_FR)]


def clean(t) -> str:
    return str(t).replace("[CLS]", "").replace("[SEP]", "").replace("\xa0", " ").strip()


def dada_crash_visible(end: int, t_ai: int) -> bool:
    return end >= t_ai + VIS_MARGIN_FRAMES


def nexar_crash_visible(tte: float, lead_s: float) -> bool:
    """lead_s = time_of_event - time_of_alert. The window ends `tte` s before the event; the hazard is
    flagged `lead_s` s before the event; it is visible for at least the margin iff lead_s >= tte + 8/30 s."""
    return lead_s + 1e-9 >= tte + VIS_MARGIN_FRAMES / FPS_SRC


def dada_targets(label: int, tte, event: str, cause: str) -> dict:
    if label == 1:
        return {"phase1": f"{event}; {cause}",
                "phase2": f"Collision: yes. Time to impact: about {tte:.1f} s. Event: {event}. Cause: {cause}."}
    return {"phase1": None,
            "phase2": "Collision: no. Time to impact: none in view. Event: normal driving before the incident. Cause: none."}


def nexar_targets(label: int, tte, caption: str) -> dict:
    """Same schema as DADA. 'Cause: not annotated' for BOTH classes so that slot cannot leak the label."""
    cap = caption.strip().rstrip(".")
    if label == 1:
        return {"phase1": None,
                "phase2": f"Collision: yes. Time to impact: about {tte:.1f} s. Event: {cap}. Cause: not annotated."}
    return {"phase1": None,
            "phase2": f"Collision: no. Time to impact: none in view. Event: {cap}. Cause: not annotated."}


def parse_tte(s) -> float | None:
    m = re.search(r"([0-9]+\.?[0-9]*)", str(s))
    return float(m.group(1)) if m else None


def parse_phase2(text: str) -> dict:
    """Parse a generated 'Collision: ... Time to impact: ... Event: ... Cause: ...' answer."""
    out = {"collision": None, "tti": None, "event": None, "cause": None}
    m = re.search(r"collision:\s*(yes|no)", text, re.I)
    if m:
        out["collision"] = 1 if m.group(1).lower() == "yes" else 0
    m = re.search(r"time to impact:\s*(?:about\s*)?([0-9]+\.?[0-9]*)\s*s", text, re.I)
    if m:
        out["tti"] = float(m.group(1))
    m = re.search(r"event:\s*(.*?)\.\s*cause:", text, re.I | re.S)
    if m:
        out["event"] = m.group(1).strip()
    m = re.search(r"cause:\s*(.*?)\.?\s*$", text, re.I | re.S)
    if m:
        out["cause"] = m.group(1).strip()
    return out


# ----------------------------------------------------------------------------- reasoning dataset (2026-10-12)
# Nexar reasoning dataset: answer = "Reasoning: <explanation>\nCollision: yes|no.\nOutcome: <sentence>" (Run 1, "reason_first");
# Run 2 ("verdict_first") puts the verdict line first. The prompt announces the fields in the same order. The decision score is
# read at the label "Collision:" (see r1_reasoning.py); nothing in the prompt asks for it.
TAG["nexar_reasoning"] = "[Nexar]"
TAG["nexar_reasoning_shifted"] = "[Nexar]"
TAG["nexar_reasoning_test"] = "[Nexar]"          # plan 2026-10-17: the private test windows
TAG["nexar_reasoning_test_public"] = "[Nexar]"   # 2026-10-18: the public test windows
_FIELDS = {"reason_first": ("Reasoning: how it moves relative to the ego vehicle and how the gap between them changes.\n"
                            "Collision: yes or no.\nOutcome: one short sentence."),
           "verdict_first": ("Collision: yes or no.\n"
                             "Reasoning: how it moves relative to the ego vehicle and how the gap between them changes.\n"
                             "Outcome: one short sentence."),
           # plan 2026-10-15: a closed-list fact card first (r1_fact_cards.py v2: side / gap at the END of the clip, motion = main manoeuvre)
           "facts_first": ("Facts: road user, side, motion and gap.\n"
                           "Reasoning: how it moves relative to the ego vehicle and how the gap between them changes.\n"
                           "Collision: yes or no.\nOutcome: one short sentence.")}
ORDERS = tuple(_FIELDS)
_NUM = {3: "three", 4: "four"}
FACT_SLOTS = ("road_user", "side", "motion", "gap")


def q_reason(order: str = "reason_first") -> str:
    return f"Look at the road user that matters most for the ego vehicle. Answer in {_NUM[_FIELDS[order].count(chr(10)) + 1]} parts:\n" + _FIELDS[order]


def facts_line(card: dict) -> str:
    """{'road_user': 'car', 'side': 'ahead', 'motion': 'cutting_in', 'gap': 'closing'} -> 'Facts: road user = car; side = ahead; motion = cutting in; gap = closing'"""
    return "Facts: " + "; ".join(f"{s.replace('_', ' ')} = {card[s].replace('_', ' ')}" for s in FACT_SLOTS)


def reasoning_target(expl: str, pred: str, label_vis: int, order: str = "reason_first", facts: dict | None = None) -> str:
    expl, pred = expl.strip(), pred.strip()
    if not pred.endswith((".", "!", "?")):
        pred += "."
    v = f"Collision: {'yes' if label_vis else 'no'}."
    r, o = f"Reasoning: {expl}", f"Outcome: {pred}"
    if order == "facts_first":
        assert facts is not None, "facts_first needs the window's fact card"
        return "\n".join([facts_line(facts), r, v, o])
    return "\n".join([r, v, o] if order == "reason_first" else [v, r, o])


def parse_facts(text: str) -> dict | None:
    """The 'Facts:' line of a generated answer -> {slot: value with underscores} (missing slots -> None); None if there is no Facts line."""
    m = re.search(r"Facts:\s*([^\n]*)", text or "", re.I)
    if not m:
        return None
    out = {s: None for s in FACT_SLOTS}
    for part in m.group(1).split(";"):
        if "=" in part:
            k, v = (x.strip().lower().rstrip(".") for x in part.split("=", 1))
            k = k.replace(" ", "_")
            if k in out:
                out[k] = v.replace(" ", "_")
    return out


def parse_reasoning(text: str) -> dict:
    """Parse a generated answer of any order -> {explanation, collision (1/0/None), outcome, facts (dict or None)}."""
    out = {"explanation": None, "collision": None, "outcome": None, "facts": parse_facts(text)}
    m = re.search(r"Collision:\s*(yes|no)", text, re.I)
    if m:
        out["collision"] = 1 if m.group(1).lower() == "yes" else 0
    m = re.search(r"Reasoning:\s*(.*?)\s*(?=\n?\s*(?:Collision:|Outcome:)|$)", text, re.I | re.S)
    if m:
        out["explanation"] = m.group(1).strip() or None
    m = re.search(r"Outcome:\s*(.*)$", text, re.I | re.S)
    if m:
        out["outcome"] = m.group(1).strip() or None
    return out


def reasoning_group(label: int, tte, pre_alert: bool) -> str:
    return "normal" if not label else ("pre_alert" if pre_alert else f"tte{tte}")


def box_hint(boxes, frame_w=1280, frame_h=720) -> str:
    """Option 3 (2026-10-12): WHERE the road user the frozen crash model attends to is, as words (no class, no motion - those
    must come from the video features). boxes = the 16 per-frame boxes of the teacher's red box ([x1,y1,x2,y2] or None)."""
    last = next((b for b in reversed(boxes or []) if b), None)
    if last is None:
        return "No road user is marked."
    x1, y1, x2, y2 = last
    cx, bottom, area = (x1 + x2) / 2 / frame_w, y2 / frame_h, (x2 - x1) * (y2 - y1) / (frame_w * frame_h)
    side = "on the left" if cx < 1 / 3 else ("on the right" if cx > 2 / 3 else "ahead in the centre")
    dist = "very close" if bottom > 0.85 else ("close" if bottom > 0.65 else "farther away")
    size = "large" if area > 0.12 else ("medium-sized" if area > 0.025 else "small")
    return f"The road user that matters most is {side}, {dist}, and {size} in the last frame."
