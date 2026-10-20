"""
r1_test_windows.py - step 1 of plan 2026-10-17_Plan-Test-Private-Reasoning: the Nexar PRIVATE test windows in the formats of the reasoning pipeline.

Every test clip is pre-cut by Nexar to END at its time-to-event (TTE 0.5 / 1.0 / 1.5 s), so it has ONE window: the last 16 frames at stride 4
(dataset/manifests/test_manifest_private.jsonl: frame_indices N-61 ... N-1). The 16 high-resolution frames are already extracted in
dataset/test/<id>_hires/. Test clips have no alert time, so there is no visibility rule: the visible-hazard label is the Nexar label, nothing is
"pre-alert", and some crash windows (mostly TTE 1.5 s) may not show the hazard yet (the second-look teacher reports `hazard_visible`).

Writes
  outputs/teacher_test_2026-10/windows.jsonl          one row per window for the box pipeline / the teacher (same fields as the training windows.jsonl)
  dataset/manifests/r1_nexar_reasoning_test.jsonl     one row per window for the reasoning model (source nexar_reasoning_test, split "test", format of
                                                      r1_nexar_reasoning_windows.jsonl; the teacher text is empty until the references exist)
and, with --check-frames N, verifies that the local frames equal the mp4 frames at those indices (the alignment check of the training windows).

    python r1_test_windows.py [--check-frames 10]
    python r1_test_windows.py --set public       (2026-10-18: the 667 PUBLIC test windows, decision-only evaluation of P2V; frames are extracted on the pod,
                                                  so no local frame check -> outputs/teacher_test_2026-10/windows_public.jsonl,
                                                  dataset/manifests/r1_nexar_reasoning_test_public.jsonl, source nexar_reasoning_test_public, ids nexarpub_<vid>)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETS = {"private": dict(man_in="test_manifest_private.jsonl", windows="windows.jsonl", man_out="r1_nexar_reasoning_test.jsonl",
                       source="nexar_reasoning_test", prefix="nexartest_", n=677),
        "public": dict(man_in="test_manifest_public.jsonl", windows="windows_public.jsonl", man_out="r1_nexar_reasoning_test_public.jsonl",
                       source="nexar_reasoning_test_public", prefix="nexarpub_", n=667)}
FRAMES = ROOT / "dataset" / "test"
OUT = ROOT / "outputs" / "teacher_test_2026-10"
SRC_VIDEOS = Path(os.environ.get("NEXAR_TEST_VIDEOS") or
                  r"C:\Users\eviatar.ohayon\Ramon Space\PycharmProjects\Thesis\Data-Centric-Crash-Prediction-Using-3LC-and-MViT\src\Nexar_DataSet\test")


def build(name="private"):
    S = SETS[name]
    MAN = ROOT / "dataset" / "manifests"
    rows = [json.loads(l) for l in open(MAN / S["man_in"], encoding="utf-8")]
    wins, man = [], []
    for r in rows:
        vid, label, hor = r["video_id"], int(r["event_occurs"]), float(r["time_before_event_s"])
        idx, fps = r["frame_indices"], float(r["fps"])
        assert len(idx) == 16 and idx[-1] == r["total_frames"] - 1, vid
        fd = f"{vid}_hires"
        if name == "private":                                   # the public frames are extracted on the pod
            assert len(list((FRAMES / fd).glob("frame_*.jpg"))) == 16, fd
        t_end = round(idx[-1] / fps, 4)
        wins.append({"set": f"test_{name}", "video_id": vid, "label": label, "horizon": hor, "frames_dir": fd, "t_end_s": t_end, "fps": fps,
                     "frame_idx": idx, "time_to_event_s": hor, "valid": True, "pre_alert": False, "floored": False, "tte_effective_s": hor, "label_vis": label,
                     "group": int(r["group"])})
        man.append({"id": f"{S['prefix']}{vid}", "source": S["source"], "split": "test", "video_key": f"test_{vid}", "frames_dir": fd, "label": label,
                    "tte": hor, "valid": True, "drop_reason": None,
                    "gt": {"explanation": "", "prediction": "", "risk": None, "label_vis": label, "pre_alert": False, "path": "test",
                           "status": "no_reference", "edited": False, "box_hint": ""}, "targets": {}})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / S["windows"]).write_text("".join(json.dumps(w) + "\n" for w in wins), encoding="utf-8")
    (MAN / S["man_out"]).write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in man), encoding="utf-8")
    c = collections.Counter((w["label"], w["horizon"]) for w in wins)
    print(f"[test] {len(wins)} {name} windows ({sum(w['label'] for w in wins)} crash / {sum(1 - w['label'] for w in wins)} normal); "
          f"(label, TTE) counts {dict(sorted(c.items()))}")
    print(f"       -> {OUT / S['windows']}\n       -> {MAN / S['man_out']}")
    assert len({w['video_id'] for w in wins}) == len(wins) == S["n"]
    return wins


def check_frames(wins, n):
    import cv2
    import numpy as np
    sample = random.Random(0).sample(wins, n)
    worst = 0.0
    for w in sample:
        cap = cv2.VideoCapture(str(SRC_VIDEOS / f"{w['video_id']}.mp4"))
        assert cap.isOpened(), f"cannot open {SRC_VIDEOS / (w['video_id'] + '.mp4')}"
        for k in (0, 7, 15):
            cap.set(cv2.CAP_PROP_POS_FRAMES, w["frame_idx"][k])
            ok, fr = cap.read()
            assert ok, (w["video_id"], k)
            jp = cv2.imread(str(FRAMES / w["frames_dir"] / f"frame_{k + 1:05d}.jpg"))
            assert jp is not None and jp.shape == fr.shape, (jp.shape if jp is not None else None, fr.shape)
            worst = max(worst, float(np.abs(jp.astype(np.int16) - fr.astype(np.int16)).mean()))
        cap.release()
    print(f"[check] {n} windows x 3 frames: local jpg vs mp4 frame, worst mean |diff| = {worst:.2f} (JPEG compression only; a wrong index gives > 20)")
    assert worst < 8.0, "frames are not aligned with the mp4 at the manifest indices"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-frames", type=int, default=0)
    ap.add_argument("--set", choices=list(SETS), default="private")
    a = ap.parse_args()
    w = build(a.set)
    if a.check_frames:
        check_frames(w, a.check_frames)
