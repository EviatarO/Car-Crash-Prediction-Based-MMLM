"""
r1_data_review.py - pre-training review of the HF window + feature repos (read-only).

Writes to --out-dir:
  stats.md                      counts, balance, target-text diversity, train/val text overlap, P(collision) per TTE, answer lengths
  --samples: per window <id>.mp4 (the 16 encoder frames, 7.5 fps, x2 size) + <id>.png + <id>.txt (all fields + targets)
  <window_id>.png               contact sheet: the 16 encoder frames (4x4), frames after the hazard start / alert framed in red,
                                header = labels, times, Phase-1 / Phase-2 target text

    python r1_data_review.py --out-dir ../../outputs/r1_week1/data_review_2026-10-05
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import random
import statistics as st
import tarfile
import textwrap
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageDraw, ImageFont

O = "eviatarO-org"
R = {"nexar": f"{O}/nexar-windows", "dada": f"{O}/mmau-dada-windows"}
FEAT = f"{O}/vjepa2-a1-features"


def rows_of(src):
    return [json.loads(l) for l in open(hf_hub_download(R[src], "windows.jsonl", repo_type="dataset"), encoding="utf-8")]


def feature_scores():
    out = {}
    files = [f for f in HfApi().list_repo_files(FEAT, repo_type="dataset") if f.endswith(".jsonl")]
    for f in files:
        for l in open(hf_hub_download(FEAT, f, repo_type="dataset"), encoding="utf-8"):
            r = json.loads(l)
            out[r["id"]] = r["p_collision"]
    return out


def frame_times(r):
    """Time (s) of each of the 16 frames, and the hazard-start time used by the visibility rule."""
    if r["source"] == "dada":
        t = [i / 30 for i in r["source_frame_indices"]]
        return t, r["time_of_alert_s"]                       # t_ai / 30
    end = r["window_end_s"]
    if end is None:
        return None, None
    return [end - (15 - k) * 4 / 30 for k in range(16)], r["time_of_alert_s"]


def sheet(r, frames, p, path):
    W, H, PAD, HEAD = 256, 256, 4, 150
    img = Image.new("RGB", (4 * W + 5 * PAD, HEAD + 4 * H + 5 * PAD), (25, 25, 25))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    times, haz = frame_times(r)
    lines = [f"{r['window_id']}   split={r['split']}   label={r['label']} ({r['gt_label_text']})   TTE={r['tte_group']}   "
             f"A1 P(collision)={p if p is None else round(p, 3)}",
             f"window end {r['window_end_s']} s   alert / abnormal start {r['time_of_alert_s']} s   event {r['time_of_event_s']} s"
             f"   red frame = after hazard start"]
    for k in ("phase1", "phase2"):
        if r["r1_targets"].get(k):
            lines += textwrap.wrap(f"{k}: {r['r1_targets'][k]}", 150)
    y = 6
    for l in lines[:7]:
        d.text((8, y), l, fill=(235, 235, 235), font=font)
        y += 20
    for i, f in enumerate(frames):
        x0, y0 = PAD + (i % 4) * (W + PAD), HEAD + PAD + (i // 4) * (H + PAD)
        img.paste(Image.fromarray(f), (x0, y0))
        if times and haz is not None and times[i] >= haz:
            d.rectangle([x0, y0, x0 + W - 1, y0 + H - 1], outline=(230, 40, 40), width=4)
        if times:
            d.text((x0 + 4, y0 + 4), f"{times[i]:.2f}s", fill=(255, 255, 0), font=font)
    img.save(path)


def pick(rows, rng):
    """Per source: val windows, 1 crash per TTE + 2 no-crash, all from the first val shard (one download)."""
    shard = sorted({r["shard"] for r in rows if r["split"] == "val"})[0]
    rs = [r for r in rows if r["shard"] == shard]
    out = []
    for tte in (0.5, 1.0, 1.5):
        c = [r for r in rs if r["label"] == 1 and r["tte_group"] == tte]
        if c:
            out.append(rng.choice(c))
    neg = [r for r in rs if r["label"] == 0]
    out += rng.sample(neg, min(2, len(neg)))
    return shard, out


def stats(rows_by, scores):
    L = ["# r1 data review — stats (read-only, from the HF repos)", ""]
    for src, rows in rows_by.items():
        L += [f"## {src} ({len(rows)} valid windows)", "", "| split | crash | no-crash | crash TTE 0.5 / 1.0 / 1.5 | videos |", "|---|---|---|---|---|"]
        for sp in ("train", "val", "test"):
            rs = [r for r in rows if r["split"] == sp]
            if not rs:
                continue
            c = collections.Counter(r["tte_group"] for r in rs if r["label"] == 1)
            L.append(f"| {sp} | {sum(r['label'] for r in rs)} | {sum(1 - r['label'] for r in rs)} | {c[0.5]} / {c[1.0]} / {c[1.5]} | "
                     f"{len({r['video_id'] for r in rs})} |")
        vids = {sp: {r["video_id"] for r in rows if r["split"] == sp} for sp in ("train", "val", "test")}
        L.append(f"\nvideo overlap train∩val = {len(vids['train'] & vids['val'])}, train∩test = {len(vids['train'] & vids['test'])}, "
                 f"val∩test = {len(vids['val'] & vids['test'])} (must all be 0)")
        # target text diversity
        key = "phase1" if src == "dada" else "phase2"
        for sp in ("train", "val"):
            t = [r["r1_targets"][key] for r in rows if r["split"] == sp and r["label"] == 1 and r["r1_targets"].get(key)]
            if not t:
                continue
            cnt = collections.Counter(t)
            top = cnt.most_common(1)[0]
            words = [len(x.split()) for x in t]
            L.append(f"- {sp} crash {key} targets: {len(t)} windows, **{len(cnt)} distinct texts**; most frequent text used {top[1]}× "
                     f"({top[1] / len(t):.0%}); words median {st.median(words)} (min {min(words)}, max {max(words)})")
        tr = {r["r1_targets"][key] for r in rows if r["split"] == "train" and r["label"] == 1 and r["r1_targets"].get(key)}
        va = [r["r1_targets"][key] for r in rows if r["split"] == "val" and r["label"] == 1 and r["r1_targets"].get(key)]
        if va:
            L.append(f"- val crash targets whose exact text also appears in train: {sum(x in tr for x in va)}/{len(va)} "
                     f"({sum(x in tr for x in va) / len(va):.0%})")
        if src == "dada":
            ev = collections.Counter(r["event"] for r in rows if r["label"] == 1)
            ca = collections.Counter(r["cause"] for r in rows if r["label"] == 1)
            L.append(f"- distinct events {len(ev)}, distinct causes {len(ca)}; top events: "
                     + "; ".join(f"'{k}' {v}" for k, v in ev.most_common(4)))
        # P(collision)
        L.append("\n| P(collision) from the features | n | median | share ≥ 0.5 |\n|---|---|---|---|")
        for lab, tte in ((1, 0.5), (1, 1.0), (1, 1.5), (0, None)):
            p = [scores[r["window_id"]] for r in rows if r["label"] == lab and r["tte_group"] == tte and r["window_id"] in scores]
            if p:
                L.append(f"| {'crash TTE ' + str(tte) if lab else 'no-crash'} | {len(p)} | {st.median(p):.2f} | {np.mean(np.array(p) >= 0.5):.0%} |")
        L.append("")
    return "\n".join(L)


def txt_card(r, p, path):
    times, haz = frame_times(r)
    vis = sum(t >= haz for t in times) if times and haz is not None else None
    L = ["=" * 78, f"dataset: {'Nexar (V12 teacher pool)' if r['source'] == 'nexar' else 'MM-AU / DADA-2000 (human labels from a fixed list)'}",
         f"HF repo: {R[r['source']]}   shard: {r['shard']}", f"window_id: {r['window_id']}   video: {r['video_id']}   split: {r['split']}",
         f"label: {r['label']} ({r['gt_label_text']})   TTE group: {r['tte_group']}", "=" * 78,
         "FILES: <id>.mp4 = the 16 frames exactly as the encoder receives them (256x256, full frame squashed), shown at 7.5 fps, x2;",
         "       red border = frame at/after the hazard start; <id>.png = the same 16 frames as a grid.", "",
         "--- timing ---",
         f"alert / abnormal start: {r['time_of_alert_s']} s   event (collision): {r['time_of_event_s']} s   window end: {r['window_end_s']} s"]
    if r["label"] == 1:
        tta = r["time_of_event_s"] - r["time_of_alert_s"]
        L.append(f"time to alert (event - alert) = {tta:.3f} s  >=  TTE {r['tte_group']} s : {tta >= r['tte_group'] - 1e-6}  (window visibility rule)")
    if times:
        L.append(f"frame times (s): {', '.join(f'{t:.2f}' for t in times)}")
        if vis is not None:
            L.append(f"hazard already started in {vis} of the 16 frames")
    if r.get("source_frame_indices"):
        L.append(f"source frame numbers (30 fps): {r['source_frame_indices']}")
    L += ["", f"--- frozen encoder crash score (A1-compress256, full crash path) ---", f"P(collision) = {p if p is None else round(p, 3)}", "",
          f"--- reasoning text ({r['reasoning_kind']}) ---", r["reasoning_text"]]
    if r["source"] == "dada":
        L += [f"event: {r['event']}", f"cause: {r['cause']}", "", f"--- ArA question (evaluation only): {r['ara']['question']} ---"]
        L += [f"  {'*' if i == r['ara']['answer'] else ' '} a{i}: {o}" for i, o in enumerate(r["ara"]["options"])]
    L += ["", "--- training targets (what the language model must output) ---",
          f"Phase 1 (alignment, question 'Describe the motion and objects in this clip.'): "
          f"{r['r1_targets']['phase1'] or '(not used in Phase 1: ' + ('Nexar is Phase 2 only' if r['source'] == 'nexar' else 'no-crash window') + ')'}",
          f"Phase 2 (SFT, question 'Is a collision coming? Give: collision yes/no, time to impact, event, cause.'): {r['r1_targets']['phase2']}",
          "", f"preprocess: {r['preprocess']}"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def video(r, frames, path):
    import cv2
    times, haz = frame_times(r)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (512, 512))
    for i, f in enumerate(frames):
        im = cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), (512, 512), interpolation=cv2.INTER_NEAREST)
        if times and haz is not None and times[i] >= haz:
            cv2.rectangle(im, (0, 0), (511, 511), (40, 40, 230), 8)
        if times:
            cv2.putText(im, f"{times[i]:.2f}s  frame {i + 1}/16", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        vw.write(im)
    vw.release()


def export_samples(spec, out, scores):
    """spec: {source: [window_id or 'auto_clear', ...]} -> out/<source>/<id>.{mp4,png,txt}"""
    for src, ids in spec.items():
        rows = {r["window_id"]: r for r in rows_of(src)}
        chosen = []
        for i in ids:
            if i == "auto_clear":                 # a val crash window the encoder flags, hazard visible >= 0.5 s
                c = sorted(w for w, r in rows.items() if r["split"] == "val" and r["label"] == 1 and scores.get(w, 0) >= 0.9
                           and r["window_end_s"] - r["time_of_alert_s"] >= 0.5 and r["shard"] == "shards/val-0000.tar")
                i = random.Random(0).choice(c)
            chosen.append(rows[i])
        d = out / src
        d.mkdir(parents=True, exist_ok=True)
        for shard in sorted({r["shard"] for r in chosen}):
            want = {r["window_id"]: r for r in chosen if r["shard"] == shard}
            with tarfile.open(hf_hub_download(R[src], shard, repo_type="dataset")) as tf:
                for m in tf:
                    w = m.name[:-4]
                    if w in want:
                        fr = np.load(io.BytesIO(tf.extractfile(m).read()))["frames"]
                        sheet(want[w], fr, scores.get(w), d / f"{w}.png")
                        video(want[w], fr, d / f"{w}.mp4")
                        txt_card(want[w], scores.get(w), d / f"{w}.txt")
                        print("sample", src, w, "P =", scores.get(w))
        yield src, [r["window_id"] for r in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--samples", nargs="*", default=None, help="source:id,id,... (id 'auto_clear' = a clear encoder-flagged crash)")
    args = ap.parse_args()
    if args.samples:
        spec = {x.split(":")[0]: x.split(":")[1].split(",") for x in args.samples}
        for src, ids in export_samples(spec, Path(args.out_dir), feature_scores()):
            print(src, ids)
        return
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows_by = {s: rows_of(s) for s in ("dada", "nexar")}
    scores = feature_scores()
    (out / "stats.md").write_text(stats(rows_by, scores), encoding="utf-8")
    print((out / "stats.md").read_text(encoding="utf-8"))
    for src, rows in rows_by.items():
        shard, chosen = pick(rows, rng)
        tp = hf_hub_download(R[src], shard, repo_type="dataset")
        want = {r["window_id"]: r for r in chosen}
        with tarfile.open(tp) as tf:
            for m in tf:
                wid = m.name[:-4]
                if wid in want:
                    fr = np.load(io.BytesIO(tf.extractfile(m).read()))["frames"]
                    assert fr.shape == (16, 256, 256, 3) and fr.dtype == np.uint8
                    sheet(want[wid], fr, scores.get(wid), out / f"{wid}.png")
                    print("sheet", wid)


if __name__ == "__main__":
    main()
