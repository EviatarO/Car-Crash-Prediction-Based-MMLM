"""
r1_extra_setup.py - the working folder for the teacher run on the 874 extra crash windows (TTE 0.75 / 1.25 s), plan 2026-10-19_Plan-Extra-TTE-Windows-Retrain, step 1.

Working folder  W = outputs/teacher_extra_2026-10  (always run the r1_* scripts with R1_PILOT_OUT=W; the original dataset folder
outputs/teacher_dataset_2026-10 is only READ, never written).

W/windows.jsonl = the 874 new windows (set "full") + the existing visible crash windows of the same videos that the first look got right, and every video's 0.5 s window, because
  - r1_attention_boxes.py boxes the same track in every window of a video (target = top-attention track of the video's smallest-horizon window, i.e. the 0.5 s one), and
  - r1_teacher_flow.py anchors a wrong answer on a correct sibling of the same video (any horizon), and starts its E7 chain at the TTE 0.5 s window.
Copied from the original run, so nothing is paid or computed twice: tracks/<vid>.json (all 651 videos), attn/<frames_dir>.json of the existing windows, and every
runs/*.jsonl (E3 / E6 / E7 / E6C / E7PRE and their P2 files). Only the new windows are then processed. Checks that the tracks of every video cover the new windows
(box_at silently returns None outside the tracked span) and lists the videos that do not (they would need `--stage tracks`).

    python r1_extra_setup.py
"""
from __future__ import annotations

import collections
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs" / "teacher_dataset_2026-10"
W = ROOT / "outputs" / "teacher_extra_2026-10"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    new = [dict(w, set="full") for w in jl(D / "windows_extra.jsonl")]
    assert len(new) == 874
    vids = {w["video_id"] for w in new}
    e3 = {r["frames_dir"]: r for r in jl(D / "runs" / "E3.jsonl") if r.get("parsed")}
    # siblings = the existing visible windows whose first look (E3) was RIGHT (a correct anchor for the second look), plus every video's TTE 0.5 s window (the flow's
    # chain start and the boxes' target window). Existing WRONG windows are left out on purpose: the flow would re-route them (new nearest sibling) and pay for answers
    # that this run never uses.
    sib = [w for w in jl(D / "windows.jsonl") if w["video_id"] in vids and w["label"] == 1 and w["valid"] and not w["pre_alert"]
           and (w["horizon"] == 0.5 or (e3.get(w["frames_dir"]) or {}).get("parsed", {}).get("collision") == "yes")]
    W.mkdir(parents=True, exist_ok=True)
    for sub in ("tracks", "attn", "runs", "boxed", "review"):
        (W / sub).mkdir(exist_ok=True)
    rows = sib + new
    (W / "windows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    c = collections.Counter(r["horizon"] for r in rows)
    print(f"[windows] {len(rows)} in W/windows.jsonl: new {len(new)} + existing visible siblings {len(sib)} | per horizon {dict(sorted(c.items()))} | videos {len(vids)}")

    n_tr = n_miss = 0
    for v in sorted(vids):
        src = D / "tracks" / f"{v}.json"
        if src.exists():
            shutil.copy2(src, W / "tracks" / src.name)
            n_tr += 1
        else:
            n_miss += 1
    n_at = n_at_miss = 0
    for w in sib:
        src = D / "attn" / f"{w['frames_dir']}.json"
        if src.exists():
            shutil.copy2(src, W / "attn" / src.name)
            n_at += 1
        else:
            n_at_miss += 1
    for src in (D / "runs").glob("*.jsonl"):
        shutil.copy2(src, W / "runs" / src.name)
    print(f"[copied] tracks {n_tr} (missing {n_miss}) | attn of existing windows {n_at} (missing {n_at_miss}) | runs: {sorted(p.name for p in (W / 'runs').glob('*.jsonl'))}")

    bad = []
    for w in new:
        p = W / "tracks" / f"{w['video_id']}.json"
        if not p.exists():
            bad.append((w["frames_dir"], "no tracks file"))
            continue
        t = json.loads(p.read_text(encoding="utf-8"))
        t0, t1 = w["frame_idx"][0] / w["fps"], w["frame_idx"][-1] / w["fps"]
        if t0 < t["span"][0] - 0.05 or t1 > t["span"][1] + 0.05:
            bad.append((w["frames_dir"], f"window {t0:.2f}-{t1:.2f} s outside the tracked span {t['span']}"))
    print(f"[tracks] new windows outside the tracked span: {len(bad)}" + (f", e.g. {bad[:3]}" if bad else ""))
    (W / "tracks_to_redo.json").write_text(json.dumps(bad, indent=1), encoding="utf-8")
    print(f"[todo] attention for {len(new)} new windows (laptop GPU, about 34 s each) -> `R1_PILOT_OUT={W} python r1_attention_boxes.py --stage attn`")


if __name__ == "__main__":
    main()
