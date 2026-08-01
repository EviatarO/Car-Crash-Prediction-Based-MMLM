"""
run_train4500_pipeline.py
===========================
LOCAL half of the train4500-inference pipeline (deliverable 2b of
~/.claude/plans/but-if-b-a1-it-woolly-metcalfe.md). Extracts the 4,446-window
train manifest in chunks (~500 videos/chunk) and, after each chunk, prints the
exact commands to push that chunk to the pod and score it - it does NOT SSH,
rsync, or touch RunPod itself. That boundary is deliberate: scoring needs GPU
credit and pod access this script does not have, and per the e4 stage-gating
convention (and the user's explicit instruction) the GPU step is a
stop-and-review checkpoint, not something to run unattended.

What this script actually automates (all local, all safe to re-run):
  1. Extraction per chunk, via semsup_extract_promptbakeoff_frames.py
     --manifest --chunk-size --chunk-index --workers (subprocess call - reuses
     that script's sequential-decode extraction and idempotent skip logic
     rather than reimplementing it).
  2. Writes each chunk's own sub-manifest JSONL to
     dataset/manifests/train4500_chunks/chunk_{NN}.jsonl - the pod-side
     scorer reads this directly with e4_stageA_badas_open_eval.py --split
     Train, one manifest per chunk so each chunk scores to its own output
     file (the scorer opens --output with mode "w" - a single shared file
     across chunks would truncate the previous chunk's results).
  3. Prints (does not run) the rsync command for that chunk. rsync -avz is
     already incremental/idempotent, so the simplest correct choice is one
     recurring "sync the whole dataset/train tree" command reused every
     chunk (matches the existing RUNPOD_E4_STAGEA_RUN.sh recipe) rather than
     computing a chunk-filtered file list, which would be more fragile for
     no real benefit - rsync only transfers what changed since last sync.

Checkpoint after chunk 1 (--stop-after-chunk, default 1): the point of
pipelining is to see the real failure rate on real data before committing to
all ~9 chunks. This script does not compute that number itself (it has no
scores yet - scoring is the pod's job); it prints the exact commands to score
chunk 1's manifest and run evaluate_metrics.py against it, and instructs the
user to compare that chunk's error rate against A0's known 23.6% test error
rate (TP/FP/TN/FN = 308/130/209/30) before continuing.

Usage:
    python student_training/scripts/run_train4500_pipeline.py \\
        --manifest dataset/manifests/train4500_hires.jsonl \\
        --chunk-size 500 --workers 8 --stop-after-chunk 1
Then, after reviewing chunk 1's pod results, re-run with a higher
--start-chunk to continue (extraction is idempotent - safe to re-run from 0).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACT_SCRIPT = PROJECT_ROOT / "student_training" / "scripts" / "semsup_extract_promptbakeoff_frames.py"
CHUNK_MANIFEST_DIR = PROJECT_ROOT / "dataset" / "manifests" / "train4500_chunks"
POD_REPO = "/workspace/MMLM_AI"
POD_STAGE_DIR = f"{POD_REPO}/outputs/train4500_inference"


def load_manifest(path: Path) -> list:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def class_interleaved_video_order(rows: list) -> list:
    """Mirrors semsup_extract_promptbakeoff_frames.py's chunking order exactly,
    so this script's chunk boundaries and the extractor's --chunk-index agree
    on which videos belong to which chunk. See that script's docstring for
    why a plain sort is wrong here (positives and negatives occupy disjoint
    raw-id ranges in train.csv, verified: pos 0-1039, neg 1040-2139)."""
    by_video = {}
    for r in rows:
        by_video.setdefault(r["video_id"], []).append(r)
    pos_ids = sorted(v for v, rs in by_video.items() if rs[0]["event_occurs"] == 1)
    neg_ids = sorted(v for v, rs in by_video.items() if rs[0]["event_occurs"] == 0)
    order = []
    for a, b in zip(pos_ids, neg_ids):
        order.append(a)
        order.append(b)
    order.extend(pos_ids[len(neg_ids):])
    order.extend(neg_ids[len(pos_ids):])
    return order, by_video


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(PROJECT_ROOT / "dataset" / "manifests" / "train4500_hires.jsonl"))
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2,
                     help="8 concurrent cv2/FFmpeg processes exhausted this machine's free RAM "
                          "(only ~2.8GB free at the time, out of 32GB total) and caused a "
                          "DLL-load/OutOfMemory failure mid-chunk-0; 2 workers completed the same "
                          "500-video chunk with 0 failures in 376s. Raise this only after checking "
                          "free RAM (PowerShell: Get-CimInstance Win32_OperatingSystem | Select "
                          "FreePhysicalMemory) is comfortably above what N workers x (OpenCV+FFmpeg "
                          "process overhead) would need.")
    ap.add_argument("--start-chunk", type=int, default=0)
    ap.add_argument("--stop-after-chunk", type=int, default=1,
                     help="extract+prepare through this chunk index (inclusive) then stop and "
                          "print the checkpoint instructions; 0-based, default 1 means chunks 0-1")
    ap.add_argument("--dry-run", action="store_true", help="print the chunk plan, extract nothing")
    args = ap.parse_args()

    rows = load_manifest(Path(args.manifest))
    order, by_video = class_interleaved_video_order(rows)
    n_chunks = (len(order) + args.chunk_size - 1) // args.chunk_size
    print(f"Manifest: {len(rows)} rows, {len(order)} distinct videos, "
          f"{n_chunks} chunks of ~{args.chunk_size} videos each.")

    CHUNK_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    last_chunk = min(args.stop_after_chunk, n_chunks - 1)
    for chunk_idx in range(args.start_chunk, last_chunk + 1):
        chunk_videos = order[chunk_idx * args.chunk_size:(chunk_idx + 1) * args.chunk_size]
        chunk_rows = [r for v in chunk_videos for r in by_video[v]]
        n_pos = sum(1 for r in chunk_rows if r["event_occurs"] == 1)
        n_neg = sum(1 for r in chunk_rows if r["event_occurs"] == 0)
        print(f"\n{'='*70}\nCHUNK {chunk_idx}/{n_chunks-1}: {len(chunk_videos)} videos, "
              f"{len(chunk_rows)} rows ({n_pos} pos / {n_neg} neg)\n{'='*70}")

        chunk_manifest_path = CHUNK_MANIFEST_DIR / f"chunk_{chunk_idx:02d}.jsonl"
        with open(chunk_manifest_path, "w", encoding="utf-8") as f:
            for r in chunk_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Sub-manifest written: {chunk_manifest_path} ({len(chunk_rows)} rows)")

        if args.dry_run:
            print("  [dry-run] skipping extraction.")
            continue

        cmd = [sys.executable, str(EXTRACT_SCRIPT), "--manifest", args.manifest,
               "--chunk-size", str(args.chunk_size), "--chunk-index", str(chunk_idx),
               "--workers", str(args.workers)]
        print(f"  Extracting: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print(f"  EXTRACTION FAILED (exit {result.returncode}) on chunk {chunk_idx} - stopping.",
                  file=sys.stderr)
            sys.exit(1)

        print(f"\n  --- Chunk {chunk_idx} extracted. NOT syncing/scoring automatically. ---")
        print(f"  1. Push new frames to the pod (rsync is incremental - safe to re-run every chunk):")
        print(f"       rsync -avz --progress 'dataset/train/' \\")
        print(f"           root@<POD_IP>:{POD_REPO}/dataset/train/ -e 'ssh -p <PORT>'")
        print(f"  2. Push this chunk's manifest:")
        print(f"       rsync -avz --progress '{chunk_manifest_path}' \\")
        print(f"           root@<POD_IP>:{POD_REPO}/dataset/manifests/train4500_chunks/ -e 'ssh -p <PORT>'")
        print(f"  3. On the pod, score this chunk (see RUNPOD_TRAIN4500_STAGEA.sh for the full loop):")
        print(f"       HF_HOME=/root/.cache/huggingface python student_training/scripts/"
              f"e4_stageA_badas_open_eval.py \\")
        print(f"           --config student_training/configs/e4_stageA.yaml \\")
        print(f"           --manifest dataset/manifests/train4500_chunks/chunk_{chunk_idx:02d}.jsonl \\")
        print(f"           --frames_root dataset/train --split Train \\")
        print(f"           --output {POD_STAGE_DIR}/scores_chunk_{chunk_idx:02d}.jsonl")

    if last_chunk == args.stop_after_chunk:
        print(f"\n{'='*70}\nCHECKPOINT: stopped after chunk {last_chunk} as requested "
              f"(--stop-after-chunk {args.stop_after_chunk}).")
        print(f"Score chunk {last_chunk} on the pod, then run evaluate_metrics.py on its output and")
        print(f"compare against A0's known test-set error rate: TP/FP/TN/FN = 308/130/209/30,")
        print(f"23.6% error, false-positive dominated. A wildly different number on this chunk means")
        print(f"a manifest/preprocessing bug, not a real finding - stop and diagnose before continuing.")
        print(f"\nTo continue after reviewing: re-run with --start-chunk {last_chunk+1} "
              f"--stop-after-chunk {n_chunks-1}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
