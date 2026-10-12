"""
r1_build_reasoning_manifest.py - manifest of the Nexar reasoning dataset (outputs/teacher_dataset_2026-10/dataset_final.jsonl)
in the format r1_build_window_repo.py / r1_cache_features.py read (source "nexar_reasoning").

Split (by VIDEO, never by window): validation = the videos that were the validation split of the earlier Nexar manifest
(r1_nexar_v12_windows.jsonl, split == val), so the new model stays comparable with the Phase-2 numbers (AUC 0.964, 88 %);
training = every other video of the dataset. There is no test split (the official test clips are not encoded yet).
Every dataset window is valid: the visibility rule is already inside `label_vis` (pre-alert windows have label_vis 0).

    python r1_build_reasoning_manifest.py          # writes dataset/manifests/r1_nexar_reasoning_windows.jsonl + prints the counts
"""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs" / "teacher_dataset_2026-10" / "dataset_final.jsonl"
OLD = ROOT / "dataset" / "manifests" / "r1_nexar_v12_windows.jsonl"
OUT = ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_windows.jsonl"
BOXES = ROOT / "outputs" / "teacher_dataset_2026-10" / "boxes.jsonl"          # the teacher's red box per frame (option 3: box_hint)


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    val_videos = {r["video_key"] for r in map(json.loads, open(OLD, encoding="utf-8")) if r["split"] == "val"}
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from r1_common import box_hint
    boxes = {json.loads(l)["frames_dir"]: json.loads(l) for l in open(BOXES, encoding="utf-8")}
    out = []
    for x in sorted(rows, key=lambda x: x["frames_dir"]):
        assert x["explanation"] and x["prediction"] and x["label_vis"] in (0, 1), x["frames_dir"]
        assert (ROOT / "dataset" / "train" / x["frames_dir"]).is_dir(), x["frames_dir"]
        out.append({"id": f"nexar_{x['frames_dir']}", "source": "nexar_reasoning", "split": "val" if x["video_id"] in val_videos else "train",
                    "video_key": x["video_id"], "frames_dir": x["frames_dir"], "label": x["label"], "tte": x["tte"] if x["label"] else None,
                    "valid": True, "drop_reason": None,
                    "gt": {"explanation": x["explanation"], "prediction": x["prediction"], "risk": x["risk"], "label_vis": x["label_vis"],
                           "pre_alert": x["pre_alert"], "path": x["path"], "status": x["status"], "edited": x["edited"],
                           "box_hint": box_hint((boxes.get(x["frames_dir"]) or {}).get("boxes"))},
                    "targets": {"reasoning": {"explanation": x["explanation"], "prediction": x["prediction"], "label_vis": x["label_vis"]}}})
    assert len({r["id"] for r in out}) == len(out)
    vids = collections.defaultdict(set)
    for r in out:
        vids[r["split"]].add(r["video_key"])
    assert not (vids["train"] & vids["val"])
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    for sp in ("train", "val"):
        rs = [r for r in out if r["split"] == sp]
        print(f"{sp}: {len(rs)} windows from {len(vids[sp])} videos; label_vis 1/0 = "
              f"{sum(r['gt']['label_vis'] for r in rs)}/{sum(1 - r['gt']['label_vis'] for r in rs)}; "
              f"crash {sum(r['label'] for r in rs)} (pre-alert {sum(r['gt']['pre_alert'] for r in rs)}), normal {sum(1 - r['label'] for r in rs)}")
    print(f"wrote {len(out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
