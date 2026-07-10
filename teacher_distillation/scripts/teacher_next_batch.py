"""
teacher_next_batch.py
=====================
Emit the next balanced batch of UNRUN clips for a teacher stage, so we never re-pull a clip
already in the aggregate (Teacher_Reasoning_All_Clips.jsonl).

  test  batch: draws from the two test manifests (complete rows, ready to run), balanced by
               event_occurs (TP/TN 50/50). Test TTE distribution is fixed by the manifest.
  train batch: draws from train.csv (1,500 videos). Each video -> 3 windows keyed
               requested_time_to_event in {0.5,1.0,1.5}. Balanced TP/TN 50/50 AND, within TP,
               equal across the three TTE. t_seconds = time_of_event - TTE for positives
               (frame_indices are filled downstream by the HiRes windowing/extraction step).

Usage:
  python teacher_distillation/scripts/teacher_next_batch.py --dataset train --n 500 \
      --out outputs/teacher_reasoning/stages/train_batch_next.todo.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
_TR_DIR = REPO_ROOT / "outputs" / "teacher_reasoning"


def _all_jsonl(dataset: str) -> Path:
    tag = "Test" if dataset == "test" else "Train"
    return _TR_DIR / f"Teacher_Reasoning_{tag}_All_Clips.jsonl"
TEST_PRIV = REPO_ROOT / "dataset" / "manifests" / "test_manifest_hires.jsonl"
TEST_PUB = REPO_ROOT / "dataset" / "manifests" / "test_manifest_public_hires.jsonl"
TRAIN_CSV = Path(r"C:\Users\eviatar.ohayon\Ramon Space\PycharmProjects\Thesis"
                 r"\Data-Centric-Crash-Prediction-Using-3LC-and-MViT\src\Nexar_DataSet\train.csv")

TTES = [0.5, 1.0, 1.5]


def _norm_vid(v) -> str:
    s = str(v).strip()
    try:
        return f"{int(s):05d}"
    except ValueError:
        return s


def _tte_label(x) -> str:
    try:
        return f"TTE_{float(x):.1f}"
    except (ValueError, TypeError):
        s = str(x).strip()
        return {"MID": "TTE_0.5", "MID-4": "TTE_1.0", "MID-8": "TTE_1.5"}.get(s, s)


def _resolve_tte(rec: dict) -> str:
    hz = str(rec.get("horizon_label") or "").strip()
    if hz.startswith("TTE_"):
        return _tte_label(hz.replace("TTE_", ""))
    return _tte_label(rec.get("requested_time_to_event"))


def _done_keys(dataset: str) -> Set[Tuple[str, str]]:
    """(video_id, TTE_label) already run for this dataset."""
    done: Set[Tuple[str, str]] = set()
    path = _all_jsonl(dataset)
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        done.add((_norm_vid(r.get("video_id")), _resolve_tte(r)))
    return done


def _load_jsonl(p: Path) -> List[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _next_test(n: int, split_filter: str = "both") -> List[dict]:
    done = _done_keys("test")
    sources = []
    if split_filter in ("private", "both"):
        sources.append((TEST_PRIV, "private"))
    if split_filter in ("public", "both"):
        sources.append((TEST_PUB, "public"))
    pos, neg = [], []
    for path, split in sources:
        for r in _load_jsonl(path):
            vid = _norm_vid(r["video_id"])
            tte = _tte_label(r.get("time_before_event_s"))
            if (vid, tte) in done:
                continue
            occ = int(r.get("event_occurs", 0))
            fi = r.get("frame_indices") or []
            fps = float(r.get("fps", 30.0)) or 30.0
            # Enrich so the row is a ready-to-run teacher replay record.
            row = dict(r)
            row["video_id"] = vid
            row["split"] = split
            row["gt_verdict"] = "YES" if occ == 1 else "NO"
            row["target"] = occ
            row["requested_time_to_event"] = r.get("time_before_event_s")
            row["t_seconds"] = round(int(fi[-1]) / fps, 3) if fi else None
            # keep frames_dir ("<id>_hires") so the teacher can locate HiRes frames
            row.setdefault("frames_dir", f"{vid}_hires")
            (pos if occ == 1 else neg).append(row)
    half = n // 2
    # Stratified proportional sample within each class so all TTE horizons appear
    # (test TTE shape is inherited from the manifest, not forced equal).
    return _strat_by_tte(pos, half) + _strat_by_tte(neg, half)


def _strat_by_tte(rows: List[dict], k: int) -> List[dict]:
    """Take k rows from `rows`, allocated across TTE buckets proportional to bucket size
    (largest-remainder rounding so the parts sum to exactly k)."""
    if k >= len(rows):
        return rows[:k]
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        buckets[_tte_label(r.get("requested_time_to_event"))].append(r)
    total = len(rows)
    quotas = {b: k * len(v) / total for b, v in buckets.items()}
    alloc = {b: int(q) for b, q in quotas.items()}
    # distribute the remainder to the largest fractional parts
    rem = k - sum(alloc.values())
    for b, _ in sorted(quotas.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[:rem]:
        alloc[b] += 1
    out: List[dict] = []
    for b, v in buckets.items():
        out.extend(v[:alloc[b]])
    return out


def _next_train(n: int) -> List[dict]:
    done = _done_keys("train")
    rows = list(csv.DictReader(TRAIN_CSV.open(encoding="utf-8-sig")))
    pos_by_tte: Dict[float, List[dict]] = defaultdict(list)
    neg: List[dict] = []
    for r in rows:
        vid = _norm_vid(r["id"])
        target = int(r["target"])
        toe = float(r["time_of_event"]) if r.get("time_of_event") not in ("", None) else None
        for tte in TTES:
            if (vid, _tte_label(tte)) in done:
                continue
            rec = {
                "video_id": vid, "target": target,
                "gt_verdict": "YES" if target == 1 else "NO",
                "requested_time_to_event": tte,
                "t_seconds": (round(toe - tte, 3) if (target == 1 and toe is not None) else None),
                "time_of_event": toe,
                "note": "frame_indices filled downstream by HiRes windowing",
            }
            if target == 1:
                pos_by_tte[tte].append(rec)
            else:
                neg.append(rec)

    half = n // 2               # TP total
    per_tte = half // len(TTES)  # equal across the 3 TTE
    tp: List[dict] = []
    for tte in TTES:
        tp.extend(pos_by_tte[tte][:per_tte])
    tn = neg[:half]
    return tp + tn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["test", "train"])
    ap.add_argument("--n", type=int, required=True, help="batch size (split ~50/50 TP/TN)")
    ap.add_argument("--split", default="both", choices=["private", "public", "both"],
                    help="test only: restrict to one manifest (default both)")
    ap.add_argument("--out", required=True, help="output batch JSONL (.todo)")
    args = ap.parse_args()

    batch = (_next_test(args.n, args.split) if args.dataset == "test"
             else _next_train(args.n))
    out = Path(args.out if Path(args.out).is_absolute() else REPO_ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in batch:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_tp = sum(1 for r in batch if r.get("gt_verdict") == "YES")
    n_tn = len(batch) - n_tp
    from collections import Counter
    tte_dist = Counter(_tte_label(r.get("requested_time_to_event") or r.get("time_before_event_s"))
                       for r in batch if r.get("gt_verdict") == "YES")
    print(f"Batch: {len(batch)} clips  (TP={n_tp}, TN={n_tn})  -> {out}")
    print(f"  TP by TTE: {dict(tte_dist)}")


if __name__ == "__main__":
    main()
