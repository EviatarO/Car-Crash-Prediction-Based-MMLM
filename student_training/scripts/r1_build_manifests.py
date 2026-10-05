"""
r1_build_manifests.py - build the two week-1 manifests (one JSONL per source, shared schema).

  dataset/manifests/r1_mmau_dada_windows.jsonl   (MM-AU / DADA-2000 part; official ArA split)
  dataset/manifests/r1_nexar_v12_windows.jsonl   (Nexar 1,761-window V12 pool; A1-compress256's own split)

Each row: id, source, split, label (1 crash / 0 no-crash), tte (0.5/1.0/1.5 or None), valid (visibility
rule), drop_reason, key fields to locate the frames, gt (raw ground truth), targets {phase1, phase2}.
Rows with valid=False are KEPT in the file (so the removed counts are auditable) but are never used for
training or evaluation of either phase.

    python student_training/scripts/r1_build_manifests.py            # prints the counts asserted in the plan
"""
import collections
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_common import (TTES, FPS_SRC, clean, dada_crash_visible, dada_targets, nexar_crash_visible,  # noqa: E402
                       nexar_targets, parse_tte, window_indices)

MAN = ROOT / "dataset" / "manifests"
V12 = ROOT / "outputs" / "semantic_captions" / "Caption_V12_Neutral_1761.jsonl"
POOL = ROOT / "outputs" / "semantic_captions" / "Caption_Train4500_Mixed_1761.jsonl"


# ------------------------------------------------------------------ DADA
def build_dada():
    sheet = pd.read_excel(ROOT / "dataset" / "public_samples" / "mmau" / "dada_text_annotations.xlsx", sheet_name="Sheet1")
    for c in ("video", "type"):
        sheet[c] = pd.to_numeric(sheet[c], errors="coerce")
    ara = {}
    for split, f in (("train", "dada_trainData.csv"), ("val", "dada_valData.csv"), ("test", "dada_testData.csv")):
        d = pd.read_csv(ROOT / "dataset" / "public_samples" / "mmau_ara" / f, encoding="utf-8-sig")
        for r in d.to_dict("records"):
            ara[(int(r["type"]), int(r["video"]))] = {
                "split": split, "question": r["question"], "answer": int(r["answer"]),
                "options": [r[f"a{i}"] for i in range(5)]}
    rows, skipped_noacc = [], 0
    for r in sheet.to_dict("records"):
        key = (int(r["type"]), int(r["video"]))
        a = ara[key]
        if int(r["whether an accident occurred (1/0)"]) != 1:
            skipped_noacc += 1                       # 17 videos with no accident: no t_ai/t_co to anchor windows
            continue
        tai, tco, tae, tot = (int(r["abnormal start frame"]), int(r["accident frame"]),
                              int(r["abnormal end frame"]), int(r["total frames"]))
        event, cause = clean(r["texts"]), str(r["causes"]).strip()
        gt = {"type": key[0], "video": key[1], "t_ai": tai, "t_co": tco, "t_ae": tae, "total": tot,
              "event": event, "cause": cause, "ara": {k: a[k] for k in ("question", "answer", "options")}}
        base = {"source": "dada", "split": a["split"], "video_key": f"t{key[0]:02d}_v{key[1]:03d}", "gt": gt}
        for tte in TTES:
            end = tco - round(tte * FPS_SRC)
            idx = window_indices(end)
            reason = None
            if idx[0] < 1:
                reason = "window starts before frame 1"
            elif not dada_crash_visible(end, tai):
                reason = "hazard not yet visible (end < t_ai + 8)"
            rows.append({**base, "id": f"dada_{base['video_key']}_tte{int(tte * 10):02d}", "label": 1, "tte": tte,
                         "end_frame": end, "valid": reason is None, "drop_reason": reason,
                         "targets": dada_targets(1, tte, event, cause)})
        end = tai - round(0.5 * FPS_SRC)
        reason = "window starts before frame 1" if window_indices(end)[0] < 1 else None
        rows.append({**base, "id": f"dada_{base['video_key']}_neg", "label": 0, "tte": None, "end_frame": end,
                     "valid": reason is None, "drop_reason": reason, "targets": dada_targets(0, None, event, cause)})
    return rows, skipped_noacc


# ------------------------------------------------------------------ Nexar
def build_nexar():
    sys.path.insert(0, str(ROOT / "student_training" / "scripts"))
    from semsup_common import clip_level_split, load_training_examples

    t = pd.read_csv(ROOT / "dataset" / "train.csv")
    p = t[t.target == 1]
    lead = {f"{int(i):05d}": float(e - a) for i, e, a in zip(p.id, p.time_of_event, p.time_of_alert)}
    v12 = {}
    for l in open(V12, encoding="utf-8"):
        r = json.loads(l)
        v12[r["frames_dir"]] = r
    ex = load_training_examples(captions_path=str(POOL))
    train, val = clip_level_split(ex, val_frac=0.2, seed=0)
    split_of = {e["frames_dir"]: "train" for e in train}
    split_of.update({e["frames_dir"]: "val" for e in val})
    rows = []
    for e in ex:
        fd = e["frames_dir"]
        r = v12[fd]
        label = int(e["label"])
        tte = parse_tte(r["requested_time_to_event"]) if label == 1 else None
        vid = str(e["video_id"]).zfill(5)
        reason = None
        if label == 1 and not nexar_crash_visible(tte, lead[vid]):
            reason = f"hazard not yet visible (alert {lead[vid]:.2f} s before the event < TTE {tte})"
        rows.append({"id": f"nexar_{fd}", "source": "nexar", "split": split_of[fd], "video_key": vid,
                     "frames_dir": fd, "label": label, "tte": tte, "valid": reason is None, "drop_reason": reason,
                     "gt": {"v12_caption": r["caption_neutral"], "lead_s": lead.get(vid)},
                     "targets": nexar_targets(label, tte, r["caption_neutral"])})
    return rows


def write(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize(rows, name):
    print(f"\n== {name}: {len(rows)} windows")
    for split in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        pos = [r for r in rs if r["label"] == 1]
        vpos = [r for r in pos if r["valid"]]
        neg = [r for r in rs if r["label"] == 0 and r["valid"]]
        per = {t: (sum(1 for r in vpos if r["tte"] == t), sum(1 for r in pos if r["tte"] == t)) for t in TTES}
        print(f"  {split:5s} videos={len({r['video_key'] for r in rs})} crash valid/all={len(vpos)}/{len(pos)} "
              f"{ {k: f'{a}/{b}' for k, (a, b) in per.items()} }  no-crash valid={len(neg)}")


def main():
    dada, skipped = build_dada()
    nex = build_nexar()
    write(dada, MAN / "r1_mmau_dada_windows.jsonl")
    write(nex, MAN / "r1_nexar_v12_windows.jsonl")
    summarize(dada, f"DADA ({skipped} no-accident videos skipped)")
    summarize(nex, "Nexar V12 pool")
    # plan assertions (numbers computed on 2026-10-05)
    vd = [r for r in dada if r["label"] == 1 and r["valid"]]
    assert len(vd) == 3299, len(vd)
    tr = [r for r in nex if r["split"] == "train" and r["label"] == 1 and r["valid"]]
    va = [r for r in nex if r["split"] == "val" and r["label"] == 1 and r["valid"]]
    assert (len(tr), len(va)) == (442, 110), (len(tr), len(va))
    assert not any(r["valid"] for r in nex + dada if r["drop_reason"])
    print("\n[ok] counts match the plan: DADA 3,299 valid crash windows; Nexar train 442 / val 110 valid crash")


if __name__ == "__main__":
    main()
