"""
r1_extra_merge.py - from the teacher run on the extra TTE windows (W = outputs/teacher_extra_2026-10, run with R1_PILOT_OUT=W) to the dataset and the manifests
(plan 2026-10-19_Plan-Extra-TTE-Windows-Retrain). The original dataset and manifest are only READ.

  python r1_extra_merge.py dataset    -> W/dataset_final.jsonl: one row per NEW window (874 minus the ones the teacher flags as "hazard not visible even with hindsight",
                                         which the original dataset removed too), fields as outputs/teacher_dataset_2026-10/dataset_final.jsonl; W/removed_windows.json; prints the routing
  python r1_extra_merge.py manifest   -> dataset/manifests/r1_nexar_reasoning_extra.jsonl  (the new windows, field extra = true; encoded by r1_cache_features.py with
                                         R1_REASONING_MANIFEST set to this file) and r1_nexar_reasoning_v2.jsonl (the original 4,169 rows + the new ones: the manifest of the new chains)
Both need `r1_teacher_flow.py` (flow_final.jsonl) and the outcome sentences (runs/*P2.jsonl) of W.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "website"))
W = ROOT / "outputs" / "teacher_extra_2026-10"
MAN = ROOT / "dataset" / "manifests"
NEW_TTES = (0.75, 1.25)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def new_windows():
    return [w for w in jl(W / "windows.jsonl") if w["horizon"] in NEW_TTES and w["set"] == "full" and "split" in w]


def build_dataset():
    import build_pilot_data as bp
    wins = new_windows()
    sr = bp.set_runs(W)
    flow = sr["flow"]
    rows, removed, paths, miss = [], [], collections.Counter(), []
    for w in sorted(wins, key=lambda w: (w["video_id"], -w["horizon"])):
        fd = w["frames_dir"]
        f = flow.get(fd)
        if f is None:
            miss.append(fd)
            continue
        if str(f.get("status", "ok")).startswith(("flagged", "check")):
            # flagged = hazard not visible even with hindsight; check = the final (second-look) verdict still disagrees with the visible-hazard label, so its text
            # would contradict the "Collision:" line of the training target (user review 2026-10-19: 10 such windows; the original dataset had none left)
            removed.append({"frames_dir": fd, "status": f["status"]})
            continue
        fr = f.get("final_from", "E3")
        x = bp.run_entry("E3", sr["E3"].get(fd)) if fr == "E3" else bp.run_entry("ROUTED", sr["ROUTED"].get(fd))
        if not x or not x.get("explanation") or not x.get("prediction"):
            miss.append(fd)
            continue
        paths[fr] += 1
        rows.append({"frames_dir": fd, "video_id": w["video_id"], "tte": w["horizon"], "label": 1, "pre_alert": False, "label_vis": f.get("label_vis", 1),
                     "path": fr, "status": f.get("status", "ok"), "explanation": x["explanation"], "prediction": x["prediction"], "risk": x["risk"], "edited": False,
                     "split": w["split"]})
    assert not miss, f"{len(miss)} new windows have no flow answer / no explanation / no outcome sentence, e.g. {miss[:3]}"
    (W / "dataset_final.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (W / "removed_windows.json").write_text(json.dumps({"reason": "hazard not visible even with hindsight (as in the original dataset)", "windows": removed}, indent=1), encoding="utf-8")
    c = collections.Counter((r["split"], r["tte"]) for r in rows)
    print(f"[dataset] {len(rows)} of {len(wins)} new windows kept; removed (flagged) {len(removed)}; routing {dict(paths)}; per (split, TTE) {dict(sorted(c.items()))}")
    e3 = sr["E3"]
    new_e3 = [e3[w["frames_dir"]] for w in wins if (e3.get(w["frames_dir"]) or {}).get("parsed")]
    right = sum(x["parsed"].get("collision") == "yes" for x in new_e3)
    print(f"[E3] blind first look on the new windows: {right} of {len(new_e3)} said 'yes' (the others went to the second look)")


def build_manifest():
    from r1_common import box_hint
    rows = jl(W / "dataset_final.jsonl")
    boxes = {r["frames_dir"]: r for r in jl(W / "boxes.jsonl")}
    out = []
    for x in sorted(rows, key=lambda x: x["frames_dir"]):
        assert x["explanation"] and x["prediction"] and x["label_vis"] in (0, 1), x["frames_dir"]
        assert (ROOT / "dataset" / "train" / x["frames_dir"]).is_dir(), x["frames_dir"]
        out.append({"id": f"nexar_{x['frames_dir']}", "source": "nexar_reasoning", "split": x["split"], "video_key": x["video_id"], "frames_dir": x["frames_dir"],
                    "label": 1, "tte": x["tte"], "valid": True, "drop_reason": None, "extra": True,
                    "gt": {"explanation": x["explanation"], "prediction": x["prediction"], "risk": x["risk"], "label_vis": x["label_vis"], "pre_alert": False,
                           "path": x["path"], "status": x["status"], "edited": False, "box_hint": box_hint((boxes.get(x["frames_dir"]) or {}).get("boxes"))},
                    "targets": {"reasoning": {"explanation": x["explanation"], "prediction": x["prediction"], "label_vis": x["label_vis"]}}})
    old = [json.loads(l) for l in open(MAN / "r1_nexar_reasoning_windows.jsonl", encoding="utf-8") if l.strip()]
    assert len(old) == 4169 and not any(r.get("extra") for r in old)
    old_split = {r["video_key"]: r["split"] for r in old}
    bad = [r["id"] for r in out if old_split.get(r["video_key"]) != r["split"]]
    assert not bad, f"a new window's split differs from its video's split in the original manifest: {bad[:3]}"
    assert not ({r["id"] for r in out} & {r["id"] for r in old}), "id clash"
    (MAN / "r1_nexar_reasoning_extra.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    (MAN / "r1_nexar_reasoning_v2.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in old + out), encoding="utf-8")
    c = collections.Counter((r["split"], r["tte"]) for r in out)
    print(f"[manifest] extra {len(out)} rows {dict(sorted(c.items()))}; union {len(old) + len(out)} rows -> r1_nexar_reasoning_extra.jsonl, r1_nexar_reasoning_v2.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dataset", "manifest"])
    a = ap.parse_args()
    build_dataset() if a.cmd == "dataset" else build_manifest()
