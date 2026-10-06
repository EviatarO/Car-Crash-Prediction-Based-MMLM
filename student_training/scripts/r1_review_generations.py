"""
r1_review_generations.py - turn the gates' generations.jsonl files into one human-readable review file.

For every generated validation window: the 16 encoder frames (png grid), the correct text (target), the text the model
wrote with the real video, whether the event / cause part matches exactly, and the encoder crash score.

    python r1_review_generations.py --run-dir ../../outputs/r1_week1/pod_phase1_2026-10-06 --phase 1
      -> <run-dir>/review/phase1_validation_outputs.md, .csv, frames/<window_id>.png
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from r1_data_review import R, feature_scores, rows_of, sheet

GROUPS = [  # (gates folder, source, title, what it is)
    ("gates_phase1_dada", "dada", "DADA validation — crash windows the encoder flags (A1 P(collision) ≥ 0.5)",
     "Same kind of windows as Phase-1 training, but from validation videos the model never saw."),
    ("gates_phase1_dada_lowp", "dada", "DADA validation — crash windows the encoder does NOT flag (A1 P(collision) < 0.5)",
     "Never used in training or for choosing the checkpoint."),
    ("gates_phase1_nexar_zeroshot", "nexar", "Nexar validation — zero-shot (Phase 1 never trained on Nexar)",
     "Target = the V12 teacher caption. Phase 1 only learned the DADA phrase list, so exact matches are impossible; "
     "read it as 'is the DADA phrase it picks a sensible description of this Nexar clip?'."),
]


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", str(t).lower()).strip()


def split_ec(t):
    p = [x.strip() for x in str(t).split(";", 1)]
    return p[0], (p[1] if len(p) > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--phase", type=int, default=1)
    args = ap.parse_args()
    run = Path(args.run_dir)
    out = run / "review"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    scores = feature_scores()
    rows = {s: {r["window_id"]: r for r in rows_of(s)} for s in ("dada", "nexar")}

    gens = {g: [json.loads(l) for l in open(run / g / "generations.jsonl", encoding="utf-8")] for g, *_ in GROUPS}
    # frames: download each needed shard once
    need = {}
    for g, src, *_ in GROUPS:
        for x in gens[g]:
            need.setdefault((src, rows[src][x["id"]]["shard"]), set()).add(x["id"])
    for (src, shard), ids in sorted(need.items()):
        with tarfile.open(hf_hub_download(R[src], shard, repo_type="dataset")) as tf:
            for m in tf:
                w = m.name[:-4]
                if w in ids and not (out / "frames" / f"{w}.png").exists():
                    fr = np.load(io.BytesIO(tf.extractfile(m).read()))["frames"]
                    sheet(rows[src][w], fr, scores.get(w), out / "frames" / f"{w}.png")
        print("frames from", src, shard, len(ids))

    blank = {g: sorted({x["blank"] for x in gens[g]}) for g, *_ in GROUPS}
    md = [f"# Phase {args.phase} — model outputs on validation windows", "",
          "How to read this file:",
          "- **Correct text** = the ground-truth text we trained the model to write (DADA: `event; cause` from the human annotation; "
          "Nexar: the V12 teacher caption).",
          "- **Model output** = what the model wrote after seeing this window's video (greedy decoding, max 80 tokens), "
          "question: \"Describe the motion and objects in this clip.\"",
          "- **Event ✓ / Cause ✓** = the part before / after the `;` is word-for-word identical to the correct event / cause "
          "(ignoring capital letters and punctuation). A different but reasonable wording counts as ✗.",
          "- **A1 P** = the frozen crash model's probability of collision for this window (0–1).",
          "- **Frames** = link to the 16 frames the encoder saw (red border = after the hazard started).",
          "- Each group shows the 50 windows the gates script picked at random (seed 0) from that validation group — not every window.",
          ""]
    csv_rows = []
    for g, src, title, what in GROUPS:
        G = gens[g]
        md += [f"## {title}", "", what, ""]
        if src == "dada":
            ev = [norm(split_ec(x["real"])[0]) == norm(split_ec(x["target"])[0]) for x in G]
            ca = [norm(split_ec(x["real"])[1]) == norm(split_ec(x["target"])[1]) for x in G]
            md += [f"Event ✓ in **{sum(ev)}/{len(G)}**, cause ✓ in **{sum(ca)}/{len(G)}**, both ✓ in "
                   f"**{sum(a and b for a, b in zip(ev, ca))}/{len(G)}**.", ""]
            md += ["| # | window (frames) | TTE | A1 P | correct event | model event | ✓ | correct cause | model cause | ✓ |",
                   "|---|---|---|---|---|---|---|---|---|---|"]
            for k, x in enumerate(G, 1):
                te, tc = split_ec(x["target"])
                me, mc = split_ec(x["real"])
                p = scores.get(x["id"])
                md.append(f"| {k} | [{x['id']}](frames/{x['id']}.png) | {x['tte']} | {p:.2f} | {te} | {me} | {'✓' if ev[k-1] else '✗'} "
                          f"| {tc} | {mc} | {'✓' if ca[k-1] else '✗'} |")
                csv_rows.append([g, x["id"], x["tte"], p, te, me, int(ev[k-1]), tc, mc, int(ca[k-1]), x["blank"]])
        else:
            md += ["| # | window (frames) | label | TTE | A1 P | correct text (V12 caption) | model output |", "|---|---|---|---|---|---|---|"]
            for k, x in enumerate(G, 1):
                p = scores.get(x["id"])
                md.append(f"| {k} | [{x['id']}](frames/{x['id']}.png) | {'crash' if x['label'] else 'no-crash'} | {x['tte']} | {p:.2f} "
                          f"| {x['target']} | {x['real']} |")
                csv_rows.append([g, x["id"], x["tte"], p, x["target"], x["real"], "", "", "", "", x["blank"]])
        md += ["", f"Model output with a **blank video** (all-zero features), identical for all {len(G)} windows of this group "
                   f"({len(blank[g])} distinct): _{blank[g][0][:300]}…_", ""]
    (out / f"phase{args.phase}_validation_outputs.md").write_text("\n".join(md), encoding="utf-8")
    with open(out / f"phase{args.phase}_validation_outputs.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group", "window_id", "tte", "a1_p", "correct_event_or_text", "model_event_or_output", "event_ok",
                    "correct_cause", "model_cause", "cause_ok", "blank_video_output"])
        w.writerows(csv_rows)
    print("wrote", out / f"phase{args.phase}_validation_outputs.md")


if __name__ == "__main__":
    main()
