"""
semsup_caption_geometry.py
============================
Gate 1: caption-set embedding geometry, computed from SigLIP TEXT embeddings
only - no video, no GPU required (CPU is fine and is the default here, unlike
semsup_common.load_siglip()'s default of 'cuda'). See PLAN: prompt-bakeoff-
harness (2026-07-27).

This is the highest value-per-cost gate: it directly targets the known failure
mode from the original B1 null result. A predictor that ignores the video
entirely can only "cheat" a semantic loss if the caption embeddings themselves
are clustered (anisotropic) - see docs_agents/EXPERIMENTS.md's A-1 finding,
where the collapse floor for the incumbent 267-caption set was
1 - ||mean(E)|| = 1 - 0.8648 = 0.1352, and the real trained run beat it by only
0.53% of the available range.

Anisotropy ||mean(E)|| on the incumbent set is 0.8547 (measured directly on the
caption text, distinct from the 0.8648 control-cosine figure computed inside a
training run) - reproducing that number here is this script's own calibration
test (plan verification step 3).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "student_training" / "scripts"))


def load_captions(path: Path) -> list:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def embed_captions(rows: list, siglip_model, tokenizer, device: str) -> np.ndarray:
    from semsup_common import siglip_text_embed
    texts = [r["caption"] for r in rows]
    # batch to keep memory bounded on CPU
    feats = []
    bs = 32
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        emb = siglip_text_embed(batch, siglip_model, tokenizer, device)
        feats.append(emb.cpu().numpy())
    return np.concatenate(feats, axis=0)


def anisotropy(E: np.ndarray) -> float:
    """||mean of L2-normalized embeddings||. 1.0 = all identical direction
    (total collapse), 0.0 = perfectly isotropic (uniform on the sphere)."""
    return float(np.linalg.norm(E.mean(axis=0)))


def mean_pairwise_cosine(E: np.ndarray) -> float:
    sims = E @ E.T
    n = E.shape[0]
    off_diag_sum = sims.sum() - np.trace(sims)
    return float(off_diag_sum / (n * (n - 1))) if n > 1 else float("nan")


def effective_rank(E: np.ndarray) -> float:
    """Participation ratio of the singular value spectrum: (sum(s^2))^2 / sum(s^4).
    Ranges from 1 (all variance in one direction) to min(n, d) (isotropic)."""
    Ec = E - E.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Ec, compute_uv=False)
    s2 = s ** 2
    if s2.sum() == 0:
        return 0.0
    return float((s2.sum() ** 2) / (s2 ** 2).sum())


def nn_purity(E: np.ndarray, group_labels: list) -> float:
    """Fraction of rows whose single nearest neighbour (by cosine, excluding
    self) shares the same group label. Higher = embeddings cluster by group."""
    sims = E @ E.T
    np.fill_diagonal(sims, -np.inf)
    nn_idx = sims.argmax(axis=1)
    labels = np.asarray(group_labels)
    matches = labels[nn_idx] == labels
    return float(matches.mean())


def centroid_separation(E: np.ndarray, binary_labels: list) -> float:
    labels = np.asarray(binary_labels)
    pos = E[labels == 1]
    neg = E[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    cp, cn = pos.mean(axis=0), neg.mean(axis=0)
    cos = float(np.dot(cp, cn) / (np.linalg.norm(cp) * np.linalg.norm(cn) + 1e-12))
    return 1.0 - cos  # cosine DISTANCE between class centroids


def analyze_arm(path: Path, label: str, siglip_model, tokenizer, device: str) -> dict:
    rows = load_captions(path)
    E = embed_captions(rows, siglip_model, tokenizer, device)

    class_labels = [1 if str(r.get("gt_verdict", "")).upper() == "YES" else 0 for r in rows]
    tte_labels = [str(r.get("requested_time_to_event", r.get("horizon_label", "?"))) for r in rows]

    result = {
        "label": label, "path": str(path), "n": len(rows),
        "anisotropy": anisotropy(E),
        "mean_pairwise_cosine": mean_pairwise_cosine(E),
        "effective_rank": effective_rank(E),
        "nn_purity_by_class": nn_purity(E, class_labels),
        "nn_purity_by_tte": nn_purity(E, tte_labels),
        "centroid_separation_pos_neg": centroid_separation(E, class_labels),
    }
    return result


def print_comparison(results: list):
    print("=" * 100)
    print("GATE 1: caption-set embedding geometry")
    print("=" * 100)
    header = f"{'arm':12s} {'n':>5s} {'anisotropy':>11s} {'mean_cos':>9s} {'eff_rank':>9s} " \
             f"{'nn_pur_cls':>11s} {'nn_pur_tte':>11s} {'centroid_sep':>13s}"
    print(header)
    for r in results:
        print(f"{r['label']:12s} {r['n']:5d} {r['anisotropy']:11.4f} "
              f"{r['mean_pairwise_cosine']:9.4f} {r['effective_rank']:9.2f} "
              f"{r['nn_purity_by_class']:11.4f} {r['nn_purity_by_tte']:11.4f} "
              f"{r['centroid_separation_pos_neg']:13.4f}")
    print()
    print("Lower anisotropy / mean_pairwise_cosine = less collapse risk (higher is worse).")
    print("centroid_separation_pos_neg: high separation may mean useful structure OR label")
    print("leakage into caption_neutral - disambiguate only against arm_c, not read alone.")
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True,
                     help="one or more caption-schema JSONL files (each row needs 'caption', "
                          "'gt_verdict', and 'requested_time_to_event' or 'horizon_label')")
    ap.add_argument("--labels", nargs="+", default=None,
                     help="labels for each --inputs entry (default: filename stem)")
    ap.add_argument("--siglip-model", default="google/siglip-base-patch16-224")
    ap.add_argument("--device", default="cpu", help="Gate 1 is designed to run free on CPU")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "semantic_captions" /
                                          "promptbakeoff" / "geometry.json"))
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    labels = args.labels if args.labels else [p.stem for p in paths]
    if len(labels) != len(paths):
        raise SystemExit("--labels must match --inputs in count")

    from semsup_common import load_siglip
    siglip_model, tokenizer = load_siglip(model_id=args.siglip_model, device=args.device)

    results = [analyze_arm(p, l, siglip_model, tokenizer, args.device)
               for p, l in zip(paths, labels)]
    print_comparison(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
