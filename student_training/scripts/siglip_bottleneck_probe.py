#!/usr/bin/env python3
"""Locate WHERE crash-relevant caption information is lost on the way to the trunk.

THE QUESTION
------------
Measured on V13 (n=4446, GroupKFold by video_id, base rate 0.500):

    caption TEXT (tfidf 1-2gram) -> crash     AUC = 0.7775
    5 structured fields          -> crash     AUC = 0.7268

So the captions genuinely carry crash signal. Yet every semantic-supervision arm that
consumed those captions through SigLIP + InfoNCE produced no crash-AP gain, and the
crash/semantic gradient cosine sat at ~0.00 (sign-flipping, |cos| 0.01-0.08).

Something between "text worth AUC 0.78" and "gradient reaching the LoRA trunk"
destroys the signal. There are two candidate stages, and this script separates them:

    stage 1   text  --SigLIP text encoder-->  512-d embedding
    stage 2   embedding  --InfoNCE vs bank-->  gradient on the trunk

This probe measures stage 1 ONLY, by training a linear classifier directly on the
frozen SigLIP embeddings and asking whether the crash label is still linearly
decodable.

    if AUC(SigLIP embed) ~= AUC(tfidf)   -> stage 1 is fine; InfoNCE (stage 2) is the
                                            culprit, and a different objective on the
                                            SAME embeddings could still work.
    if AUC(SigLIP embed) << AUC(tfidf)   -> the SigLIP text encoder itself discards the
                                            crash-relevant content. No amount of loss
                                            engineering downstream can recover it, and
                                            the target space must change (or be dropped).

That is a genuine fork with different next actions, which is why it is worth the run.
A linear probe is the right instrument: it is an UPPER bound on what the linear InfoNCE
similarity head could exploit, so a low number is decisive, not merely suggestive.

CORPORA
-------
Run on all three (V10, V12, V13). They differ in exactly the way that matters:
  V10  leaky      - contains outcome wording; the tfidf number will be inflated, which
                    makes it the positive control (if SigLIP cannot even keep V10's
                    signal, the encoder is definitively the bottleneck).
  V12  de-leaked  - the corpus every headline semantic arm was trained on.
  V13  causal     - richest, and the one whose fields measure AUC 0.727 standalone.

Runs fine on CPU (SigLIP-base's text tower is ~110M params, and only short captions
are embedded) - pass --device cpu. No pod required.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semsup_common import load_siglip, siglip_text_embed  # noqa: E402

CAPTION_FIELDS = ("caption_neutral", "caption")   # V12/V13 use the first, V10 the second


def pick_caption(row):
    for f in CAPTION_FIELDS:
        v = row.get(f)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def load_labels(path):
    """frames_dir -> 'YES'/'NO'.

    Labels come from a separate scored file because the caption corpora are produced
    BLIND (no gt_verdict field) by design - that blindness is the anti-leakage
    mechanism, so joining externally is required, not a workaround.
    """
    lab = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("gt_verdict"):
                lab[r["frames_dir"]] = r["gt_verdict"]
    return lab


def cv_scores(X, y, groups, n_splits=5):
    p = cross_val_predict(LogisticRegression(max_iter=5000), X, y,
                          groups=groups, cv=GroupKFold(n_splits=n_splits),
                          method="predict_proba")[:, 1]
    return roc_auc_score(y, p), average_precision_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True, help="caption corpus .jsonl")
    ap.add_argument("--labels", default="../../outputs/semtest200/A0_full4446.jsonl",
                    help="any .jsonl carrying frames_dir + gt_verdict")
    ap.add_argument("--siglip-model", default="google/siglip-base-patch16-224")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--skip-siglip", action="store_true",
                    help="text-baseline only; used to verify the load/join path against "
                         "an already-known number before spending time on embeddings")
    ap.add_argument("--out", default=None, help="optional JSON summary path")
    args = ap.parse_args()

    lab = load_labels(args.labels)
    rows = []
    for line in open(args.captions, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        cap = pick_caption(r)
        gt = r.get("gt_verdict") or lab.get(r.get("frames_dir"))
        if cap and gt in ("YES", "NO"):
            rows.append((cap, 1 if gt == "YES" else 0, r.get("video_id", r["frames_dir"])))
    if not rows:
        raise SystemExit(f"no usable rows in {args.captions} (need a caption field + a label)")

    caps = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])
    groups = np.array([r[2] for r in rows])
    print(f"corpus  : {args.captions}")
    print(f"n={len(y)}  base_rate={y.mean():.3f}  unique_videos={len(set(groups))}")
    if len(set(y)) < 2:
        raise SystemExit("single-class corpus, nothing to probe")

    # ---- Reference: does the raw TEXT carry crash signal at all? ----
    # Grouped by video_id so a video's other windows cannot leak its label.
    T = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000).fit_transform(caps)
    t_auc, t_ap = cv_scores(T, y, groups, args.folds)
    print(f"  [text  tfidf 1-2gram ] AUC={t_auc:.4f}  AP={t_ap:.4f}")

    if args.skip_siglip:
        return

    # ---- The actual question: does SigLIP's embedding still carry it? ----
    print(f"[load] SigLIP: {args.siglip_model} on {args.device}")
    model, tok = load_siglip(args.siglip_model, args.device)
    embs = []
    for i in range(0, len(caps), args.batch):
        embs.append(siglip_text_embed(caps[i:i + args.batch], model, tok, args.device)
                    .float().cpu().numpy())
        if (i // args.batch) % 10 == 0:
            print(f"    embedded {min(i + args.batch, len(caps))}/{len(caps)}", flush=True)
    E = np.concatenate(embs, axis=0)
    # Embeddings are already L2-normalized; StandardScaler puts every dim on equal
    # footing so LogisticRegression's single C is not silently per-dimension.
    Es = StandardScaler().fit_transform(E)
    e_auc, e_ap = cv_scores(Es, y, groups, args.folds)
    print(f"  [SigLIP text embed   ] AUC={e_auc:.4f}  AP={e_ap:.4f}")

    # ---- Distinctiveness, reported alongside because it is the other half of the
    # story: an embedding space where every caption looks alike cannot separate them. ----
    S = E @ E.T
    n = len(E)
    off = (S.sum() - np.trace(S)) / (n * (n - 1))
    print(f"  [mean cross-caption cosine] {off:.4f}   distinctiveness={1 - off:.4f}")

    retention = e_auc / t_auc if t_auc > 0 else float("nan")
    print()
    print(f"VERDICT  SigLIP retains {retention:6.1%} of the text's crash-AUC "
          f"({e_auc:.4f} / {t_auc:.4f})")
    # 0.90 is a judgement call, stated so it is arguable rather than hidden: below it,
    # the encoder is losing more than measurement noise on a 5-fold probe of this size.
    if retention < 0.90:
        print("         -> stage 1 (SigLIP text encoder) is a REAL bottleneck. Changing the "
              "loss cannot recover what the encoder already discarded; change the target "
              "space or drop the embedding route.")
    else:
        print("         -> stage 1 is fine. The signal survives into the embedding, so the "
              "loss/objective (stage 2, InfoNCE) is where it is lost - a different "
              "objective on these same embeddings is still worth testing.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "captions": args.captions, "n": int(len(y)), "base_rate": float(y.mean()),
            "text_auc": float(t_auc), "text_ap": float(t_ap),
            "siglip_auc": float(e_auc), "siglip_ap": float(e_ap),
            "retention": float(retention),
            "mean_cross_cosine": float(off), "distinctiveness": float(1 - off),
        }, indent=2), encoding="utf-8")
        print(f"[ok] wrote {args.out}")


if __name__ == "__main__":
    main()
