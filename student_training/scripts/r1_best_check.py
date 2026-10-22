"""
r1_best_check.py - the best check point of a run by VALIDATION decision AUC (plan 2026-10-20). Reads <run dir>/train_log.jsonl, where whole-number "epoch" records are epoch ends
(checkpoint epoch_XX.pt) and fractional ones are mid-epoch checks (checkpoint check_<frac>.pt). Ties go to the earlier point. Validation only, never the test.

  python r1_best_check.py <run dir>            prints one line:  <checkpoint file name> <epoch fraction> <AUC>      (the table goes to stderr)
  python r1_best_check.py <run dir> --after F  also exit code 0 if the best point lies after epoch F (e.g. 1.0), 3 if at or before F: the R2 -> R1 rule of the plan
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def points(run_dir):
    """[(epoch fraction, checkpoint file name, decision AUC, validation loss)] in training order."""
    out = []
    for l in open(Path(run_dir) / "train_log.jsonl", encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        d = r.get("decision") or {}
        auc = next((d[k]["auc"] for k in ("first", "before") if k in d and d[k].get("auc") is not None), None)
        ep = r["epoch"]
        name = f"check_{ep:.2f}.pt" if r.get("mid") else f"epoch_{int(ep):02d}.pt"
        vl = next((v.get("loss_real") for k, v in r.items() if k.startswith("val_") and isinstance(v, dict)), None)
        if auc is not None:
            out.append((float(ep), name, float(auc), vl))
    return out


def best(run_dir):
    pts = points(run_dir)
    assert pts, "no decision AUC in the train log"
    return max(pts, key=lambda p: (round(p[2], 6), -p[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--after", type=float, default=None)
    a = ap.parse_args()
    b = best(a.run_dir)
    print("epoch fraction | checkpoint | validation decision AUC | validation loss", file=sys.stderr)
    for ep, name, auc, vl in points(a.run_dir):
        print(f"  {ep:5.2f} | {name:14s} | {auc:.4f} | {vl if vl is None else round(vl, 3)}" + ("   <- best" if name == b[1] else ""), file=sys.stderr)
    print(f"{b[1]} {b[0]:g} {b[2]:.4f}")
    if a.after is not None:
        sys.exit(0 if b[0] > a.after + 1e-9 else 3)


if __name__ == "__main__":
    main()
