"""Smoke test: verify DSPy + multi-image + Gemini via OpenRouter all work together.

Sends 16 frames from clip 00319 to Gemini 3.1 Pro with a minimal instruction.
If this works, the full APO infrastructure is feasible.

Usage:
    py teacher_distillation/scripts/apo_smoke_test.py
"""

import os
import sys
from pathlib import Path

import dspy
from dotenv import load_dotenv
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


class CollisionAnalysis(dspy.Signature):
    """Predict ego-vehicle collision risk from 16 dashcam frames. Output YES or NO with structured reasoning."""
    frames: list[dspy.Image] = dspy.InputField(desc="16 chronological dashcam frames (~2 seconds)")
    scene_context: str = dspy.OutputField()
    ego_state: str = dspy.OutputField()
    dynamic_objects: str = dspy.OutputField()
    temporal_analysis: str = dspy.OutputField()
    spatiotemporal_attention: str = dspy.OutputField(
        desc="Where in space (object/region) and when in time (frame range) the threat is"
    )
    time_to_contact: str = dspy.OutputField()
    collision_verdict: str = dspy.OutputField(desc="YES or NO")
    verdict_reasoning: str = dspy.OutputField(desc="Consolidated rationale, max 150 words")


def main():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    # Configure DSPy with Gemini via OpenRouter
    print("[1/4] Configuring DSPy with Gemini via OpenRouter...", flush=True)
    gemini = dspy.LM(
        model="openrouter/google/gemini-3.1-pro-preview",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=1500,
    )
    dspy.configure(lm=gemini)

    # Load 16 frames from clip 00319
    print("[2/4] Loading 16 frames from clip 00319...", flush=True)
    clip_dir = REPO_ROOT / "dataset" / "train" / "00319"
    frame_paths = sorted(clip_dir.glob("frame_*.jpg"))
    if len(frame_paths) != 16:
        raise RuntimeError(f"Expected 16 frames, got {len(frame_paths)} in {clip_dir}")

    frames = [dspy.Image.from_PIL(Image.open(p).convert("RGB")) for p in frame_paths]
    print(f"        Loaded {len(frames)} frames", flush=True)

    # Build the program
    print("[3/4] Calling Gemini with 16 images...", flush=True)
    program = dspy.Predict(CollisionAnalysis)

    try:
        result = program(frames=frames)
    except Exception as exc:
        print(f"        FAILED: {exc!r}", flush=True)
        sys.exit(1)

    # Print result
    print("[4/4] SUCCESS! Output:", flush=True)
    print(f"        verdict        : {result.collision_verdict}")
    print(f"        words          : {len(result.verdict_reasoning.split())}")
    print(f"        scene_context  : {result.scene_context[:80]}...")
    print(f"        spatiotemporal : {result.spatiotemporal_attention[:80]}...")
    print(f"        reasoning      : {result.verdict_reasoning[:120]}...")
    print()
    print("DSPy + multi-image + OpenRouter is WORKING. APO infrastructure is feasible.")


if __name__ == "__main__":
    main()
