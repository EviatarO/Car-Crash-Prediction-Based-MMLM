"""DSPy signature for APO. Output schema is FIXED across iterations.
Only the docstring (instruction) changes per iteration via `with_instructions()`.
"""

import dspy


class CollisionAnalysis(dspy.Signature):
    """Predict ego-vehicle collision risk from 16 dashcam frames. Output YES or NO with structured reasoning."""
    frames: list[dspy.Image] = dspy.InputField(desc="16 chronological dashcam frames (~2 seconds)")
    scene_context: str = dspy.OutputField(desc="lighting, weather, road type")
    ego_state: str = dspy.OutputField(desc="ego-vehicle motion: braking / accelerating / steady")
    dynamic_objects: str = dspy.OutputField(desc="moving agents: type, position, distinguishing features")
    temporal_analysis: str = dspy.OutputField(desc="how objects change between first and last frames")
    spatiotemporal_attention: str = dspy.OutputField(
        desc="WHERE in space (which object/region) and WHEN in time (which frame range) the threat is — explicit grounding"
    )
    time_to_contact: str = dspy.OutputField(desc="estimated seconds to contact, or N/A")
    collision_verdict: str = dspy.OutputField(desc="YES or NO")
    verdict_reasoning: str = dspy.OutputField(desc="consolidated rationale, max 150 words")


# The seed instruction — minimal, no protocol. ProTeGi will iterate from here.
SEED_INSTRUCTION = (
    "Predict ego-vehicle collision risk from 16 dashcam frames. "
    "Output YES or NO with structured reasoning."
)


def make_program(instruction: str) -> dspy.Predict:
    """Create a DSPy Predict module with a custom instruction (replaces docstring)."""
    sig = CollisionAnalysis.with_instructions(instruction)
    return dspy.Predict(sig)
