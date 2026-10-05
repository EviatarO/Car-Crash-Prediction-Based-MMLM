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
