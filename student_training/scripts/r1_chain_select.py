"""
r1_chain_select.py - the checkpoints of a P2V chain to evaluate (plan 2026-10-19). Runs with no early stop, so every epoch has a checkpoint (epoch_XX.pt) and a line in
train_log.jsonl with the validation decision AUC (read at "Collision:", validation = val full). The selection uses VALIDATION ONLY, never the test.

  python r1_chain_select.py <P2V dir> [--k 3]     prints the epoch numbers (best AUC first), one line "e1 e2 e3", and a table to stderr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(a.run_dir) / "train_log.jsonl", encoding="utf-8") if l.strip()]
    sc = []
    for r in rows:
        d = (r.get("decision") or {}).get("first") or {}
        if "auc" in d:
            sc.append((d["auc"], r["epoch"], r["val_nexar_reasoning"]["loss_real"], r["train_loss"]))
    assert sc, "no decision AUC in the train log"
    sc.sort(key=lambda x: (-x[0], x[1]))                       # ties: the earlier epoch
    print("epoch | val decision AUC | val loss | train loss", file=sys.stderr)
    for auc, ep, vl, tl in sorted(sc, key=lambda x: x[1]):
        print(f"  {ep:2d}  | {auc:.4f} | {vl:.3f} | {tl:.3f}" + ("   <- top" if (auc, ep, vl, tl) in sc[:a.k] else ""), file=sys.stderr)
    print(" ".join(str(ep) for _, ep, _, _ in sc[:a.k]))


if __name__ == "__main__":
    main()
