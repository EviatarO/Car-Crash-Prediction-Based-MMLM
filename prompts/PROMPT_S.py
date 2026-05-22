"""
PROMPT_S — Student prompt for the 4B InternVL collision-anticipation model.

Why a dedicated student prompt (vs. the teacher's PROMPT_G_OPT_v6_balanced)?
The teacher prompt is ~140 lines of dense, multi-section instructions. It works
on a 27B teacher that can attend across the whole instruction AND still read the
frames. The student is InternVL3.5-4B-Flash (~7x smaller); packing the long
instruction in front of 16 frames x 256 image tokens starves the visual evidence
of attention. PROMPT_S is intentionally short (~15 lines) and asks for a minimal
JSON output so the model spends its limited capacity on *seeing*, not on
reproducing a long output schema.

Output schema (minimal):
    {"verdict": "YES" | "NO", "reason": "<=150 tokens naming the cause + object>"}

CRITICAL: this exact prompt must be used at BOTH training time
(student_training/data/collision_dataset.py) and inference time
(student_training/scripts/trained_eval.py). The training reasoning labels are the
teacher reasoning compressed to <=150 tokens to match this schema.
"""

PROMPT_S = (
    "ROLE: You are an in-car driving assistant analysing 16 sequential dashcam "
    "frames from the EGO vehicle's camera (Frame 1 = earliest, Frame 16 = latest).\n\n"

    "TASK: Predict whether the EGO vehicle will be involved in a collision within "
    "~1.5 seconds AFTER the last frame.\n\n"

    "COLLISION = the EGO vehicle makes contact with another road user (car, "
    "motorcycle, bicycle, pedestrian) or a fixed object, OR another road user is "
    "on a clear unavoidable path into EGO. A close pass without contact is NOT a "
    "collision. If the evidence is ambiguous, prefer NO.\n\n"

    "KEY SIGNALS to weigh:\n"
    "- A road user's path is converging with EGO's path.\n"
    "- A vehicle ahead brakes hard, swerves, or stops suddenly.\n"
    "- A pedestrian or cyclist enters the road.\n"
    "- EGO loses control (skid, sudden steering).\n\n"

    "OUTPUT - return ONLY this JSON, no extra text:\n"
    '{"verdict": "YES" | "NO", "reason": "<one explanation, at most 150 tokens, '
    'naming the specific cause and the object involved>"}'
)
