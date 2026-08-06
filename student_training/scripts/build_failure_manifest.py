"""build_failure_manifest.py -- reconstructs the manifest of the 587 windows
the frozen A0 scorer got wrong in the train4500-inference pipeline.

WHY THIS EXISTS
------------------
The original per-row failures.jsonl was never pulled off the RunPod pod
before it was terminated (2026-08-01). The failure identities survive in
outputs/train4500_inference/monitor_train4500_coverage.xlsx's model_verdict
column ("correct"/"wrong" per (video_id, TTE)), which this script joins back
to dataset/manifests/train4500_hires.jsonl (the full 4,446-window manifest,
which carries frames_dir/frame_indices/event_occurs for every window) to
reconstruct a Stage-A-schema manifest for exactly the 587 failures.

Verified during exploration (2026-08-02): the join on (video_id,
horizon_label) matches 587/587 with zero unmatched rows, and every row's
frames_dir already resolves to 16 extracted frames on disk (no new
extraction needed - these frames were already pulled for the original
train4500-inference scoring run).

Usage:
    python build_failure_manifest.py                    # writes the full 587
    python build_failure_manifest.py --split             # + pos/neg splits
    python build_failure_manifest.py --sample 30 --seed 0 --stratify 18,12
        # stratified pilot manifest: 18 negatives + 12 positives
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import openpyxl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

MONITOR_XLSX = PROJECT_ROOT / "outputs" / "train4500_inference" / "monitor_train4500_coverage.xlsx"
TRAIN4500_MANIFEST = PROJECT_ROOT / "dataset" / "manifests" / "train4500_hires.jsonl"
FRAMES_ROOT = PROJECT_ROOT / "dataset" / "train"
OUT_DIR = PROJECT_ROOT / "dataset" / "manifests"
OUT_ALL = OUT_DIR / "train4500_failures_587.jsonl"
OUT_POS = OUT_DIR / "train4500_failures_pos_269.jsonl"
OUT_NEG = OUT_DIR / "train4500_failures_neg_318.jsonl"


def load_manifest_index() -> dict:
    """(video_id, horizon_label) -> manifest row, from the full 4,446-window
    train4500 manifest."""
    idx = {}
    for line in open(TRAIN4500_MANIFEST, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        idx[(row["video_id"], row["horizon_label"])] = row
    return idx


def load_verdict_keys(verdict: str) -> list:
    """[(video_id, horizon_label), ...] for every row the monitor xlsx marks
    model_verdict == `verdict` ('wrong' or 'correct')."""
    wb = openpyxl.load_workbook(MONITOR_XLSX, read_only=True)
    ws = wb["train4500"]
    keys = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        vid, gt_verdict, tte, model_verdict = row[0], row[1], row[2], row[3]
        if not vid or not model_verdict:
            continue
        if str(model_verdict).strip().lower() == verdict:
            keys.append((str(vid), str(tte)))
    return keys


def build_rows_for_verdict(verdict: str) -> list:
    idx = load_manifest_index()
    keys = load_verdict_keys(verdict)
    unmatched = [k for k in keys if k not in idx]
    if unmatched:
        raise RuntimeError(
            f"{len(unmatched)}/{len(keys)} failure keys did not match the train4500 "
            f"manifest (join on video_id, horizon_label). Refusing to silently drop "
            f"rows - first few unmatched: {unmatched[:5]}"
        )
    rows = [idx[k] for k in keys]

    # verify frames exist on disk for every row - these should already be
    # extracted (reused from the original train4500-inference scoring run),
    # so a missing frame here is a real problem worth failing loudly on.
    missing_frames = []
    for r in rows:
        frame_dir = FRAMES_ROOT / r["frames_dir"]
        paths = [frame_dir / f"frame_{i:05d}.jpg" for i in range(1, 17)]
        if not all(p.exists() for p in paths):
            missing_frames.append(r["video_id"])
    if missing_frames:
        raise RuntimeError(
            f"{len(missing_frames)} failure windows have missing frames on disk: "
            f"{missing_frames[:10]}{'...' if len(missing_frames) > 10 else ''}"
        )

    return rows


def write_jsonl(rows: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verdict", default="wrong", choices=["wrong", "correct"],
                     help="'wrong' (default) = the 587 A0-failure windows this script was "
                          "originally built for. 'correct' = the complementary pool A0 "
                          "already gets right (3,859 windows) - used to sample additional "
                          "balanced windows to mix into the hard-only 587 pool.")
    ap.add_argument("--split", action="store_true",
                     help="also write pos/neg split manifests")
    ap.add_argument("--sample", type=int, default=0,
                     help="write a stratified pilot manifest of this size instead")
    ap.add_argument("--stratify", default="18,12",
                     help="neg,pos counts for --sample (default 18 FP / 12 FN)")
    ap.add_argument("--balanced-sample", type=int, default=0,
                     help="write a class-BALANCED random sample of this total size "
                          "(N/2 positive + N/2 negative) instead - e.g. --balanced-sample "
                          "1174 with --verdict correct draws 587 pos + 587 neg from the "
                          "A0-correct pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                     help="override output path for --sample/--balanced-sample")
    args = ap.parse_args()

    rows = build_rows_for_verdict(args.verdict)
    pos_rows = [r for r in rows if r["event_occurs"] == 1]
    neg_rows = [r for r in rows if r["event_occurs"] == 0]

    print(f"Reconstructed {len(rows)} '{args.verdict}' windows "
          f"({len(pos_rows)} positive, {len(neg_rows)} negative) "
          f"from {MONITOR_XLSX.name} joined to {TRAIN4500_MANIFEST.name}.")
    print(f"Distinct videos: {len({r['video_id'] for r in rows})}")

    if args.balanced_sample:
        n_each = args.balanced_sample // 2
        if args.balanced_sample % 2 != 0:
            raise ValueError(f"--balanced-sample {args.balanced_sample} must be even")
        if n_each > len(pos_rows) or n_each > len(neg_rows):
            raise ValueError(f"requested {n_each}/class but only {len(pos_rows)} positive / "
                              f"{len(neg_rows)} negative available in the '{args.verdict}' pool")
        rng = random.Random(args.seed)
        sample = rng.sample(pos_rows, n_each) + rng.sample(neg_rows, n_each)
        out_path = Path(args.out) if args.out else \
            OUT_DIR / f"train4500_{args.verdict}_balanced_{args.balanced_sample}.jsonl"
        write_jsonl(sample, out_path)
        print(f"Wrote balanced sample ({n_each} pos + {n_each} neg, seed={args.seed}): {out_path}")
        return

    if args.sample:
        n_neg, n_pos = (int(x) for x in args.stratify.split(","))
        if n_neg + n_pos != args.sample:
            raise ValueError(f"--stratify {args.stratify} sums to {n_neg+n_pos}, "
                              f"not --sample {args.sample}")
        rng = random.Random(args.seed)
        sample = rng.sample(neg_rows, n_neg) + rng.sample(pos_rows, n_pos)
        out_path = Path(args.out) if args.out else OUT_DIR / f"train4500_failures_pilot_{args.sample}.jsonl"
        write_jsonl(sample, out_path)
        print(f"Wrote stratified pilot ({n_neg} neg + {n_pos} pos, seed={args.seed}): {out_path}")
        return

    write_jsonl(rows, OUT_ALL)
    print(f"Wrote {OUT_ALL}")
    if args.split:
        write_jsonl(pos_rows, OUT_POS)
        write_jsonl(neg_rows, OUT_NEG)
        print(f"Wrote {OUT_POS}")
        print(f"Wrote {OUT_NEG}")


if __name__ == "__main__":
    main()
